import logging

TRACE_LOGGER_NAME = "sentinelvoice.trace"

# These libraries can include provider request bodies, multipart audio, prompt
# content, or authorization headers in DEBUG records. Application-owned safe
# telemetry remains available through ``sentinelvoice.trace``.
_SENSITIVE_THIRD_PARTY_LOGGERS = (
    "groq",
    "groq._base_client",
    "httpx",
    "httpcore",
    "hpack",
    "h2",
)
_HANDLER_MARKER = "_sentinelvoice_trace_console"


def configure_runtime_logging() -> None:
    """Apply safe, idempotent logging defaults for API and voice processes."""
    for logger_name in _SENSITIVE_THIRD_PARTY_LOGGERS:
        third_party_logger = logging.getLogger(logger_name)
        third_party_logger.disabled = False
        third_party_logger.setLevel(logging.WARNING)

    trace_logger = logging.getLogger(TRACE_LOGGER_NAME)
    trace_logger.disabled = False
    trace_logger.setLevel(logging.INFO)
    trace_logger.propagate = False

    if not any(
        getattr(handler, _HANDLER_MARKER, False)
        for handler in trace_logger.handlers
    ):
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter("%(message)s"))
        setattr(handler, _HANDLER_MARKER, True)
        trace_logger.addHandler(handler)
