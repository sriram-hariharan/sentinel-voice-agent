import re
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from secrets import token_urlsafe

TRACE_ID_HEADER = "X-SentinelVoice-Trace-ID"
TURN_ID_HEADER = "X-SentinelVoice-Turn-ID"

_CORRELATION_ID = re.compile(r"^[A-Za-z0-9_-]{16,128}$")


@dataclass(frozen=True)
class TraceContext:
    trace_id: str
    session_id: str
    turn_id: str


class VoiceCorrelationQueue:
    """FIFO handoff from bounded STT completion to the finalized voice turn."""

    def __init__(self) -> None:
        self._items: deque[TraceContext] = deque()

    def create(self, *, session_id: str) -> TraceContext:
        return build_trace_context(session_id=session_id)

    def record(self, context: TraceContext) -> None:
        self._items.append(context)

    def consume(self) -> TraceContext | None:
        return self._items.popleft() if self._items else None


_trace_context: ContextVar[TraceContext | None] = ContextVar(
    "sentinelvoice_trace_context",
    default=None,
)


def new_correlation_id() -> str:
    """Return an opaque URL-safe identifier with no embedded authority."""
    return token_urlsafe(18)


def validate_correlation_id(value: str, *, field_name: str) -> str:
    normalized = value.strip()
    if not _CORRELATION_ID.fullmatch(normalized):
        raise ValueError(
            f"{field_name} must be an opaque 16-128 character identifier"
        )
    return normalized


def build_trace_context(
    *,
    session_id: str,
    trace_id: str | None = None,
    turn_id: str | None = None,
) -> TraceContext:
    if not session_id.strip():
        raise ValueError("session_id must not be empty")
    return TraceContext(
        trace_id=validate_correlation_id(
            trace_id or new_correlation_id(),
            field_name="trace_id",
        ),
        session_id=session_id,
        turn_id=validate_correlation_id(
            turn_id or new_correlation_id(),
            field_name="turn_id",
        ),
    )


def current_trace_context() -> TraceContext | None:
    return _trace_context.get()


@contextmanager
def trace_scope(
    *,
    session_id: str,
    trace_id: str | None = None,
    turn_id: str | None = None,
) -> Iterator[TraceContext]:
    context = build_trace_context(
        session_id=session_id,
        trace_id=trace_id,
        turn_id=turn_id,
    )
    token: Token[TraceContext | None] = _trace_context.set(context)
    try:
        yield context
    finally:
        _trace_context.reset(token)
