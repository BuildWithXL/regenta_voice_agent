"""
Ways to feed audio into the STT pipeline:

  - MicrophoneStream: pulls audio live from your computer's microphone.
  - FileAudioStream:  reads a WAV file and yields it in small chunks,
                       paced in real time so it behaves like a live call.
  - PhoneAudioStream: audio of a live Plivo call (see plivo_server.py).

All of them yield raw 16-bit PCM bytes, matching what the
Speech-to-Text streaming API expects.
"""

import queue
import sys
import time
import wave

from config import CHANNELS, CHUNK_SIZE_FRAMES, SAMPLE_RATE_HZ


class MicrophoneStream:
    """Opens the default microphone and yields raw PCM16 audio chunks."""

    def __init__(self, sample_rate=SAMPLE_RATE_HZ, chunk_frames=CHUNK_SIZE_FRAMES):
        self._sample_rate = sample_rate
        self._chunk_frames = chunk_frames
        self._buff = queue.Queue()
        self._stream = None
        self.closed = True
        self.muted = False

    def __enter__(self):
        import sounddevice as sd      # only needed locally; servers have no audio device

        self.closed = False
        self._stream = sd.RawInputStream(
            samplerate=self._sample_rate,
            blocksize=self._chunk_frames,
            channels=CHANNELS,
            dtype="int16",
            callback=self._fill_buffer,
        )
        self._stream.start()
        time.sleep(0.3)  # let the mic driver settle before we start feeding audio to Flux
        # Drain whatever got captured during that settle window (often just a startup pop/click)
        while True:
            try:
                self._buff.get(block=False)
            except queue.Empty:
                break
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.closed = True
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
        self._buff.put(None)

    def _fill_buffer(self, in_data, frame_count, time_info, status):
        if status:
            print(f"[mic status] {status}", file=sys.stderr)
        if self.muted:
            # Send silence instead of nothing, so the STT clock keeps running
            # and nothing said while the AI talks reaches the recogniser.
            self._buff.put(bytes(len(in_data)))
        else:
            self._buff.put(bytes(in_data))

    def generator(self):
        """Yields chunks of raw audio until the stream is closed (Ctrl+C)."""
        while not self.closed:
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]
            # Drain anything else that's queued up so we don't fall behind.
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break
            yield b"".join(data)


class FileAudioStream:
    """
    Reads a 16-bit PCM WAV file and yields it in small chunks, sleeping
    between chunks so the file streams in at roughly real-time speed --
    this simulates a live call for testing purposes instead of dumping
    the whole file at once.
    """

    def __init__(self, file_path, chunk_frames=CHUNK_SIZE_FRAMES):
        self._file_path = file_path
        self._chunk_frames = chunk_frames

    def _validate_and_open(self):
        wf = wave.open(self._file_path, "rb")
        if wf.getsampwidth() != 2:
            sys.exit(
                f"'{self._file_path}' is not 16-bit PCM audio. Convert it first, e.g.:\n"
                f"  ffmpeg -i input.mp3 -ar {SAMPLE_RATE_HZ} -ac 1 -sample_fmt s16 output.wav"
            )
        if wf.getframerate() != SAMPLE_RATE_HZ:
            print(
                f"[warning] '{self._file_path}' is {wf.getframerate()}Hz, "
                f"expected {SAMPLE_RATE_HZ}Hz. Recognition quality may suffer. "
                f"Recommended fix:\n"
                f"  ffmpeg -i {self._file_path} -ar {SAMPLE_RATE_HZ} -ac 1 "
                f"-sample_fmt s16 fixed.wav",
                file=sys.stderr,
            )
        if wf.getnchannels() != CHANNELS:
            print(
                f"[warning] '{self._file_path}' has {wf.getnchannels()} channels, "
                f"expected {CHANNELS} (mono). Recognition quality may suffer.",
                file=sys.stderr,
            )
        return wf

    def generator(self):
        wf = self._validate_and_open()
        try:
            frame_duration_s = self._chunk_frames / float(wf.getframerate())
            while True:
                chunk = wf.readframes(self._chunk_frames)
                if not chunk:
                    break
                yield chunk
                # Pace playback so it arrives roughly like a live call
                # instead of all at once.
                time.sleep(frame_duration_s)
        finally:
            wf.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class PhoneAudioStream:
    """Audio from a live phone call (plivo_server.py pushes each media frame
    in with put()). Same interface as MicrophoneStream: muted, generator()."""

    def __init__(self):
        self._buff = queue.Queue()
        self.closed = False
        self.muted = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def put(self, pcm: bytes) -> None:
        # While the AI talks, send silence so the STT clock keeps running.
        self._buff.put(bytes(len(pcm)) if self.muted else pcm)

    def close(self) -> None:
        self.closed = True
        self._buff.put(None)

    def generator(self):
        """Yields raw PCM16 chunks until the call ends."""
        while not self.closed:
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]
            while True:
                try:
                    chunk = self._buff.get(block=False)
                except queue.Empty:
                    break
                if chunk is None:
                    yield b"".join(data)
                    return
                data.append(chunk)
            yield b"".join(data)
