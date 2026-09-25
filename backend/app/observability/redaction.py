import re
from collections.abc import Mapping, Sequence
from typing import Any

REDACTED = "<redacted>"
REDACTED_IDENTIFIER = "<redacted-id>"

_SECRET_KEYS = {
    "api_key",
    "auth_token",
    "authorization",
    "bearer_token",
    "cookie",
    "demo_pin",
    "livekit_secret",
    "password",
    "pin",
    "raw_audio",
    "raw_auth_token",
    "secret",
    "session_token",
    "transcript",
    "user_text",
}
_SENSITIVE_SUFFIXES = ("_api_key", "_password", "_secret", "_token")
_INTERNAL_ID_KEYS = {
    "account_id",
    "card_id",
    "customer_id",
    "resource_id",
    "transaction_id",
}
_BEARER = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+")
_LONG_NUMBER = re.compile(r"(?<!\d)\d{8,19}(?!\d)")
_UUID = re.compile(
    r"(?i)\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}\b"
)
_LIKELY_API_KEY = re.compile(
    r"\b(?:gsk|sk|lkapi)_[A-Za-z0-9_-]{12,}\b",
    re.IGNORECASE,
)
_PIN_VALUE = re.compile(r"(?i)(\bpin\s*[:=]?\s*)\d{4,12}\b")


def _redact_string(value: str) -> str:
    value = _BEARER.sub("Bearer <redacted>", value)
    value = _LIKELY_API_KEY.sub(REDACTED, value)
    value = _PIN_VALUE.sub(r"\1<redacted>", value)
    value = _UUID.sub(REDACTED_IDENTIFIER, value)
    return _LONG_NUMBER.sub(REDACTED, value)


def redact_metadata(value: Any, *, key: str | None = None) -> Any:
    """Return a JSON-safe, centrally redacted copy of trace metadata."""
    normalized_key = key.casefold() if key else None
    if normalized_key is not None:
        if normalized_key in _SECRET_KEYS or normalized_key.endswith(
            _SENSITIVE_SUFFIXES
        ):
            return REDACTED
        if normalized_key in _INTERNAL_ID_KEYS:
            return REDACTED_IDENTIFIER

    if isinstance(value, Mapping):
        return {
            str(item_key): redact_metadata(item_value, key=str(item_key))
            for item_key, item_value in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return [redact_metadata(item) for item in value]
    if isinstance(value, bytes):
        return REDACTED
    if isinstance(value, str):
        return _redact_string(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _redact_string(str(value))
