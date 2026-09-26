import re

_POLICY_TERMS = re.compile(
    r"\b(policy|policies|disput(?:e|es|ing)|unauthori[sz]ed|unrecognized|"
    r"unfamiliar|compromis(?:e|ed)|suspicious|"
    r"don't recognize|do not recognize|pending|posted|reversed|"
    r"freez(?:e|es|ing)|frozen|"
    r"unfreeze|replacement|replace|"
    r"protected action|confirmation|escalat(?:e|ion)|human support)\b",
    re.IGNORECASE,
)
_INFORMATIONAL_FORM = re.compile(
    r"\b(how long|how does|how do|what happens|what should|"
    r"what(?:'s| is) the process|when can|when should|can i|am i able|"
    r"should i|does|whether|why|what is .*policy|tell me about|explain)\b",
    re.IGNORECASE,
)
_EXPLICIT_ACTION = re.compile(
    r"(?:^|[.!?]\s+)"
    r"(?:(?:please|kindly)\s+|(?:can|could|would|will)\s+you\s+|"
    r"i\s+(?:want|need|would\s+like)\s+(?:you\s+)?to\s+|help\s+me\s+)?"
    r"(?:"
    r"(?:freeze|unfreeze|replace)\s+"
    r"(?:(?:my|this|the)\s+)?(?:debit\s+|credit\s+)?card"
    r"(?:\s+(?:ending(?:\s+in)?\s+)?\d{4})?"
    r"|(?:file|open|create|submit|start)\s+(?:a\s+)?dispute\b"
    r"|dispute\s+(?:my|this|the)\s+(?:charge|purchase|transaction)\b"
    r")",
    re.IGNORECASE,
)


def is_policy_question(text: str) -> bool:
    normalized = " ".join(text.split())
    return bool(
        _POLICY_TERMS.search(normalized)
        and (
            _INFORMATIONAL_FORM.search(normalized)
            or "policy" in normalized.lower()
        )
        and not _EXPLICIT_ACTION.search(normalized)
    )
