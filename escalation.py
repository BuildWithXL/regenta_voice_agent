"""
Escalation rule engine. ai_response.py runs check_escalation() on every caller
turn (and again after tools run) before deciding to keep trying or hand over.

Phrase matching covers English, Hindi (Devanagari) and romanized Hindi.
"""
from dataclasses import dataclass
import re

MAX_CLARIFICATION_FAILURES = 3

_EN_VERB = r"(?:talk|speak|connect|transfer|put me through|get me|call)"
_EN_HUMAN = r"(?:human|person|agent|representative|manager|supervisor|someone|customer care|customer service)"
_HI_VERB = r"(?:baat|bat|bol|jod|connect|transfer|बात|बोल|जोड़|जोड|कनेक्ट|ट्रांसफर)"
_HI_HUMAN = r"(?:insaan|insan|admi|aadmi|agent|manager|customer care|इंसान|आदमी|एजेंट|मैनेजर|सुपरवाइज़र|कस्टमर केयर|किसी)"

_HUMAN_EN = re.compile(rf"\b{_EN_VERB}\b.{{0,25}}\b{_EN_HUMAN}\b|\b{_EN_HUMAN}\b.{{0,25}}\b{_EN_VERB}\b")
_HUMAN_HI = re.compile(rf"{_HI_VERB}.{{0,30}}{_HI_HUMAN}|{_HI_HUMAN}.{{0,30}}{_HI_VERB}")

LEGAL_RE = re.compile(
    r"\b(lawyer|sue|sued|legal action|attorney|consumer court|police|fraud|scam)\b"
    r"|वकील|कानूनी|कोर्ट|पुलिस|धोखा|धोखाधड़ी"
)
# Clear payment disputes only; a plain refund question is answered normally.
PAYMENT_RE = re.compile(
    r"\b(charged twice|double charged|double charge|overcharged|wrongly charged|wrong charge|"
    r"unauthori[sz]ed (?:charge|transaction)|chargeback)\b"
    r"|दो बार (?:पैसे )?कट|ज्यादा चार्ज|ज़्यादा चार्ज|paise do baar kat"
)


@dataclass
class EscalationResult:
    should_escalate: bool
    reason: str | None = None
    urgency: str = "normal"


def check_escalation(
    session,
    user_message: str,
    ai_confidence: float | None = None,
    payment_dispute: bool = False,
    identity_mismatch: bool = False,
    backend_failure: bool = False,
    sentiment: str | None = None,  # from a sentiment classifier, if you add one
) -> EscalationResult:
    text = (user_message or "").lower()

    if _HUMAN_EN.search(text) or _HUMAN_HI.search(text):
        return EscalationResult(True, "customer_requested_human")

    if LEGAL_RE.search(text):
        return EscalationResult(True, "legal_complaint", urgency="high")

    if payment_dispute or PAYMENT_RE.search(text):
        return EscalationResult(True, "payment_dispute")

    if identity_mismatch:
        return EscalationResult(True, "sensitive_account_issue")

    if backend_failure:
        return EscalationResult(True, "system_unavailable")

    if ai_confidence is not None and ai_confidence < 0.5 and session.clarification_failures >= 1:
        return EscalationResult(True, "low_confidence_after_clarification")

    if session.clarification_failures >= MAX_CLARIFICATION_FAILURES:
        return EscalationResult(True, "repeated_misunderstanding")

    if sentiment in ("escalated", "negative"):
        return EscalationResult(True, "customer_frustration")

    return EscalationResult(False)
