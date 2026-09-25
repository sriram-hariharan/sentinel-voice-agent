import logging
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Any, Protocol, Self

from backend.app.observability.context import current_trace_context
from backend.app.observability.events import TraceEvent, TraceStatus

logger = logging.getLogger("sentinelvoice.trace")


class TraceSink(Protocol):
    def emit(self, event: TraceEvent) -> None: ...


class LoggingTraceSink:
    """Emit one stable JSON object per Python log record."""

    def emit(self, event: TraceEvent) -> None:
        logger.info(event.model_dump_json())


class InMemoryTraceSink:
    def __init__(self) -> None:
        self.events: list[TraceEvent] = []

    def emit(self, event: TraceEvent) -> None:
        self.events.append(event)


_default_sink = LoggingTraceSink()
_trace_sink: ContextVar[TraceSink] = ContextVar(
    "sentinelvoice_trace_sink",
    default=_default_sink,
)


@contextmanager
def use_trace_sink(sink: TraceSink) -> Iterator[TraceSink]:
    token: Token[TraceSink] = _trace_sink.set(sink)
    try:
        yield sink
    finally:
        _trace_sink.reset(token)


def emit_trace_event(
    event_name: str,
    *,
    component: str,
    status: TraceStatus = TraceStatus.INFO,
    duration_ms: float | None = None,
    error_category: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> TraceEvent | None:
    context = current_trace_context()
    if context is None:
        return None
    event = TraceEvent(
        event_name=event_name,
        trace_id=context.trace_id,
        session_id=context.session_id,
        turn_id=context.turn_id,
        component=component,
        status=status,
        duration_ms=duration_ms,
        error_category=error_category,
        metadata=metadata or {},
    )
    _trace_sink.get().emit(event)
    return event


def safe_error_category(exc: BaseException) -> str:
    name = type(exc).__name__.casefold()
    if "malformedmodeloutput" in name:
        return "malformed_model_output"
    if "timeout" in name:
        return "provider_timeout"
    if "authentication" in name or "authorization" in name:
        return "authorization_denied"
    if "validation" in name:
        return "validation_error"
    if "retrieval" in name or "policysearch" in name:
        return "retrieval_error"
    if "tool" in name:
        return "tool_error"
    if "provider" in name:
        return "provider_error"
    return "internal_error"


class TraceSpan:
    def __init__(
        self,
        name: str,
        *,
        component: str,
        metadata: dict[str, Any] | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.name = name
        self.component = component
        self.metadata = dict(metadata or {})
        self._clock = clock
        self._started_at: float | None = None

    def set_metadata(self, **metadata: Any) -> None:
        self.metadata.update(metadata)

    def __enter__(self) -> Self:
        self._started_at = self._clock()
        emit_trace_event(
            f"{self.name}.started",
            component=self.component,
            status=TraceStatus.STARTED,
            metadata=self.metadata,
        )
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        del exc_type, traceback
        started_at = self._started_at
        duration_ms = (
            max(0.0, (self._clock() - started_at) * 1000)
            if started_at is not None
            else 0.0
        )
        if exc is None:
            emit_trace_event(
                f"{self.name}.completed",
                component=self.component,
                status=TraceStatus.COMPLETED,
                duration_ms=duration_ms,
                metadata=self.metadata,
            )
        else:
            emit_trace_event(
                f"{self.name}.failed",
                component=self.component,
                status=TraceStatus.FAILED,
                duration_ms=duration_ms,
                error_category=safe_error_category(exc),
                metadata=self.metadata,
            )
        return False

    async def __aenter__(self) -> Self:
        return self.__enter__()

    async def __aexit__(self, exc_type, exc, traceback) -> bool:
        return self.__exit__(exc_type, exc, traceback)


def trace_span(
    name: str,
    *,
    component: str,
    metadata: dict[str, Any] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> TraceSpan:
    return TraceSpan(
        name,
        component=component,
        metadata=metadata,
        clock=clock,
    )
