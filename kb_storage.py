"""
kb_storage.py - where the generated knowledge files live in production.

  KB_STORAGE=local   (default) kb_build.py writes the files to KB_LOCAL_DIR
                     (./hotel_kb) and the app reads them from there.
                     For development on your laptop.
  KB_STORAGE=s3      kb_build.py builds in KB_BUILD_DIR (./.kb_build) and
                     uploads to the bucket (AWS S3, or any S3-compatible
                     store); every app instance downloads the files into
                     KB_CACHE_DIR at startup and on a timer. The app serves
                     from memory, never from the bucket, so a bucket outage
                     can't affect a live call.

Switching between the two is purely a matter of environment variables.

Env vars (s3 mode):
  KB_S3_BUCKET        required
  KB_S3_PREFIX        default "hotel_kb/"
  KB_S3_ENDPOINT_URL  optional (R2 / MinIO)
  KB_S3_REGION        optional
  KB_CACHE_DIR        default ./.kb_cache   (app: downloaded copy)
  KB_BUILD_DIR        default ./.kb_build   (kb_build.py working copy)
  credentials         standard AWS env vars / instance role (boto3 default
                      chain). IAM: the app needs only s3:GetObject on the
                      prefix; the build job also needs s3:PutObject,
                      s3:DeleteObject and s3:ListBucket.

Consistency: a manifest.json (sha256 of every file + a combined hash) is
uploaded LAST on publish. Readers download the files it lists, verify every
checksum, and only then swap the new copy into place - so a half-finished
publish is never loaded.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

MANIFEST = "manifest.json"
_last_sync_msg: str | None = None
_HERE = Path(__file__).resolve().parent


def backend() -> str:
    return os.getenv("KB_STORAGE", "local").strip().lower()


def local_dir() -> Path:
    return Path(os.getenv("KB_LOCAL_DIR") or _HERE / "hotel_kb")


def cache_dir() -> Path:
    return Path(os.getenv("KB_CACHE_DIR") or _HERE / ".kb_cache")


def build_dir() -> Path:
    return Path(os.getenv("KB_BUILD_DIR") or _HERE / ".kb_build")


def output_dir() -> Path:
    """Where kb_build.py assembles the files."""
    return local_dir() if backend() != "s3" else build_dir()


def _bucket() -> str:
    b = os.getenv("KB_S3_BUCKET")
    if not b:
        raise RuntimeError("KB_STORAGE=s3 but KB_S3_BUCKET is not set")
    return b


def _prefix() -> str:
    p = os.getenv("KB_S3_PREFIX", "hotel_kb/")
    return p if p.endswith("/") else p + "/"


def _s3():
    import boto3   # pip install boto3 (only needed in s3 mode)
    return boto3.client(
        "s3",
        endpoint_url=os.getenv("KB_S3_ENDPOINT_URL") or None,
        region_name=os.getenv("KB_S3_REGION") or None,
    )


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------
def write_manifest(kb_dir: Path) -> dict:
    kb_dir = Path(kb_dir)
    files = {
        p.relative_to(kb_dir).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(kb_dir.rglob("*.md"))
    }
    combined = hashlib.sha256(
        json.dumps(files, sort_keys=True).encode("utf-8")).hexdigest()
    existing = manifest_hash(kb_dir)
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hash": combined,
        "files": files,
    }
    if existing != combined:           # don't churn git when nothing changed
        (kb_dir / MANIFEST).write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8")
    return manifest


def manifest_hash(kb_dir: Path) -> str | None:
    try:
        return json.loads((Path(kb_dir) / MANIFEST).read_text(encoding="utf-8")).get("hash")
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------------------
# publish (build job)
# ---------------------------------------------------------------------------
def publish(src: Path) -> None:
    src = Path(src)
    if backend() != "s3":
        print("KB_STORAGE is not 's3' - nothing to upload "
              f"(files stay in {src}).")
        return
    manifest = json.loads((src / MANIFEST).read_text(encoding="utf-8"))
    s3, bucket, prefix = _s3(), _bucket(), _prefix()

    try:
        remote = json.loads(
            s3.get_object(Bucket=bucket, Key=prefix + MANIFEST)["Body"].read())
        if remote.get("hash") == manifest["hash"]:
            print(f"s3://{bucket}/{prefix} is already up to date")
            return
    except Exception:
        pass                      # nothing published yet (or unreadable): upload

    for rel in sorted(manifest["files"]):
        s3.put_object(Bucket=bucket, Key=prefix + rel,
                      Body=(src / rel).read_bytes(),
                      ContentType="text/markdown; charset=utf-8")
    # manifest LAST: readers only see the new set once all files are up.
    s3.put_object(Bucket=bucket, Key=prefix + MANIFEST,
                  Body=(src / MANIFEST).read_bytes(),
                  ContentType="application/json")

    keep = {prefix + r for r in manifest["files"]} | {prefix + MANIFEST}
    removed = 0
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"] not in keep:
                s3.delete_object(Bucket=bucket, Key=obj["Key"])
                removed += 1
    print(f"published {len(manifest['files'])} file(s) to "
          f"s3://{bucket}/{prefix} (removed {removed} stale)")


# ---------------------------------------------------------------------------
# sync (app side)
# ---------------------------------------------------------------------------
def _sync_down(cache: Path) -> bool:
    s3, bucket, prefix = _s3(), _bucket(), _prefix()
    raw = s3.get_object(Bucket=bucket, Key=prefix + MANIFEST)["Body"].read()
    remote = json.loads(raw)
    if manifest_hash(cache) == remote.get("hash"):
        return False

    tmp = cache.parent / f".{cache.name}_tmp"
    old = cache.parent / f".{cache.name}_old"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    try:
        for rel, sha in remote["files"].items():
            if rel.startswith("/") or ".." in Path(rel).parts:
                raise RuntimeError(f"unsafe path in manifest: {rel}")
            body = s3.get_object(Bucket=bucket, Key=prefix + rel)["Body"].read()
            if hashlib.sha256(body).hexdigest() != sha:
                raise RuntimeError(f"checksum mismatch for {rel} (publish in progress?)")
            dest = tmp / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(body)
        (tmp / MANIFEST).write_bytes(raw)

        shutil.rmtree(old, ignore_errors=True)
        if cache.exists():
            cache.rename(old)
        tmp.rename(cache)
        shutil.rmtree(old, ignore_errors=True)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return True


def pull_current(dest: Path) -> None:
    """Build job: start from the currently published files so unchanged
    files stay byte-identical (no timestamp churn, no needless client reloads)."""
    if backend() != "s3":
        return
    try:
        _sync_down(Path(dest))
    except Exception as exc:
        print(f"  (no published copy to start from: {exc})")


def ensure_local() -> Path:
    """Blocking. Returns a local directory holding the current KB files.
    Call via asyncio.to_thread from async code."""
    if backend() != "s3":
        return local_dir()
    cache = cache_dir()
    try:
        _sync_down(cache)
    except Exception as exc:
        if manifest_hash(cache) is None:
            raise RuntimeError(f"KB sync failed and there is no cached copy: {exc}") from exc
        msg = f"  [kb] sync from storage failed ({exc}); using the cached copy"
        global _last_sync_msg
        if msg != _last_sync_msg:          # don't repeat the same line every refresh
            print(msg)
            _last_sync_msg = msg
        return cache
    _last_sync_msg = None
    return cache