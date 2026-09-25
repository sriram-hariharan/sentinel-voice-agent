"""Application-native tracing, metrics, usage, and redaction primitives."""

from backend.app.observability.context import (
    TraceContext,
    current_trace_context,
    new_correlation_id,
    trace_scope,
)
from backend.app.observability.events import TraceEvent, TraceStatus
from backend.app.observability.tracing import (
    InMemoryTraceSink,
    LoggingTraceSink,
    emit_trace_event,
    trace_span,
    use_trace_sink,
)

__all__ = [
    "InMemoryTraceSink",
    "LoggingTraceSink",
    "TraceContext",
    "TraceEvent",
    "TraceStatus",
    "current_trace_context",
    "emit_trace_event",
    "new_correlation_id",
    "trace_scope",
    "trace_span",
    "use_trace_sink",
]
