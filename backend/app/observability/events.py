from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.app.observability.redaction import redact_metadata


class TraceStatus(StrEnum):
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"
    INFO = "info"


class TraceEvent(BaseModel):
    event_name: str = Field(min_length=3, max_length=128)
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )
    trace_id: str = Field(min_length=16, max_length=128)
    session_id: str = Field(min_length=1, max_length=128)
    turn_id: str = Field(min_length=16, max_length=128)
    component: str = Field(min_length=1, max_length=64)
    status: TraceStatus
    duration_ms: float | None = Field(default=None, ge=0)
    error_category: str | None = Field(default=None, max_length=64)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True, extra="forbid")

    @field_validator("timestamp")
    @classmethod
    def timestamp_must_be_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")
        return value

    @field_validator("metadata", mode="before")
    @classmethod
    def redact_event_metadata(cls, value: Any) -> dict[str, Any]:
        redacted = redact_metadata(value or {})
        if not isinstance(redacted, dict):
            raise TypeError("metadata must be a mapping")
        return redacted
