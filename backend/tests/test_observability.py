import io
import json
import logging
from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError

from backend.app.agent.orchestrator import AgentOrchestrator
from backend.app.agent.resource_resolver import ResourceResolution
from backend.app.conversation.state import ConversationState
from backend.app.observability.context import (
    current_trace_context,
    trace_scope,
)
from backend.app.observability.events import TraceEvent, TraceStatus
from backend.app.observability.logging import configure_runtime_logging
from backend.app.observability.metrics import (
    nearest_rank_percentile,
    summarize_numeric,
)
from backend.app.observability.summaries import summarize_turn
from backend.app.observability.tracing import (
    BoundedInMemoryTraceSink,
    InMemoryTraceSink,
    LoggingTraceSink,
    trace_span,
    use_trace_sink,
)
from backend.app.observability.usage import (
    CostEstimator,
    PricingCatalog,
    PricingRate,
    UsageRecord,
)
from backend.app.providers.llm import LLMResponse, LLMUsage


class NoopResourceResolver:
    async def resolve(self, **kwargs) -> ResourceResolution:
        return ResourceResolution()


class UsageLLM:
    provider = "groq"
    model = "openai/gpt-oss-20b"

    async def generate(self, *, messages, tools=None) -> LLMResponse:
        del messages, tools
        return LLMResponse(
            content="Safe response.",
            model=self.model,
            usage=LLMUsage(
                prompt_tokens=17,
                completion_tokens=5,
                total_tokens=22,
            ),
        )


def test_runtime_logging_blocks_provider_debug_payloads_and_emits_traces(
) -> None:
    configure_runtime_logging()
    trace_logger = logging.getLogger("sentinelvoice.trace")
    handlers_before = list(trace_logger.handlers)

    configure_runtime_logging()

    assert trace_logger.level == logging.INFO
    assert trace_logger.propagate is False
    assert trace_logger.handlers == handlers_before
    owned_handlers = [
        handler
        for handler in trace_logger.handlers
        if getattr(handler, "_sentinelvoice_trace_console", False)
    ]
    assert len(owned_handlers) == 1
    for logger_name in (
        "groq",
        "groq._base_client",
        "httpx",
        "httpcore",
        "hpack",
        "h2",
    ):
        provider_logger = logging.getLogger(logger_name)
        assert not provider_logger.isEnabledFor(logging.DEBUG)
        assert provider_logger.isEnabledFor(logging.WARNING)

    provider_logger = logging.getLogger("groq._base_client")
    unsafe_output = io.StringIO()
    unsafe_handler = logging.StreamHandler(unsafe_output)
    provider_logger.addHandler(unsafe_handler)
    try:
        provider_logger.debug(
            "Request options audio=%r authorization=%s",
            b"raw-audio-bytes",
            "Bearer raw-token",
        )
    finally:
        provider_logger.removeHandler(unsafe_handler)
    assert unsafe_output.getvalue() == ""

    handler = owned_handlers[0]
    assert isinstance(handler, logging.StreamHandler)
    original_stream = handler.stream
    captured = io.StringIO()
    handler.setStream(captured)
    try:
        LoggingTraceSink().emit(
            TraceEvent(
                event_name="logging.test",
                trace_id="trace_1234567890123456",
                session_id="session-1",
                turn_id="turn_12345678901234567",
                component="test",
                status=TraceStatus.INFO,
            )
        )
    finally:
        handler.setStream(original_stream)
    assert captured.getvalue().count('"event_name":"logging.test"') == 1


def test_trace_context_generates_ids_and_nests_safely() -> None:
    assert current_trace_context() is None
    with trace_scope(session_id="session-1") as outer:
        assert len(outer.trace_id) >= 16
        assert len(outer.turn_id) >= 16
        assert current_trace_context() == outer
        with trace_scope(
            session_id="session-1",
            trace_id=outer.trace_id,
            turn_id=outer.turn_id,
        ) as inner:
            assert inner == outer
            assert current_trace_context() == outer
        assert current_trace_context() == outer
    assert current_trace_context() is None


def test_trace_event_serializes_and_redacts_secrets() -> None:
    event = TraceEvent(
        event_name="tool.execution.completed",
        trace_id="trace_1234567890123456",
        session_id="session-1",
        turn_id="turn_12345678901234567",
        component="tools",
        status=TraceStatus.COMPLETED,
        metadata={
            "pin": "1234",
            "api_key": "plain-secret-value",
            "auth_token": "plain-token-value",
            "freeform_key": "gsk_this-is-an-obvious-secret",
            "authorization": "Bearer raw-token-value",
            "card_number": "4111111111111111",
            "account_number": "123456789",
            "freeform": "resource 11111111-1111-4111-8111-111111111111",
            "prompt_tokens": 17,
            "nested": {"customer_id": "customer-internal-id"},
        },
    )
    payload = event.model_dump_json()
    metadata = json.loads(payload)["metadata"]

    assert '"pin":"1234"' not in payload
    assert "gsk_this-is-an-obvious-secret" not in payload
    assert "raw-token-value" not in payload
    assert "4111111111111111" not in payload
    assert metadata["account_number"] == "<redacted>"
    assert "11111111-1111-4111-8111-111111111111" not in payload
    assert "customer-internal-id" not in payload
    assert '"prompt_tokens":17' in payload
    assert metadata["pin"] == "<redacted>"


def test_trace_span_records_deterministic_duration_and_failure_category() -> None:
    sink = InMemoryTraceSink()
    clock_values = iter([10.0, 10.125, 20.0, 20.050])
    with use_trace_sink(sink), trace_scope(session_id="session-1"):
        with trace_span(
            "rag.retrieval",
            component="rag",
            clock=lambda: next(clock_values),
        ):
            pass
        with pytest.raises(TimeoutError), trace_span(
            "tool.execution",
            component="tools",
            clock=lambda: next(clock_values),
            metadata={"password": "never-log-me"},
        ):
            raise TimeoutError("secret stack detail")

    assert [event.event_name for event in sink.events] == [
        "rag.retrieval.started",
        "rag.retrieval.completed",
        "tool.execution.started",
        "tool.execution.failed",
    ]
    assert sink.events[1].duration_ms == pytest.approx(125.0)
    assert sink.events[3].duration_ms == pytest.approx(50.0)
    assert sink.events[3].error_category == "provider_timeout"
    assert sink.events[3].metadata["password"] == "<redacted>"


@pytest.mark.asyncio
async def test_llm_usage_is_captured_in_trace_metadata() -> None:
    sink = InMemoryTraceSink()
    orchestrator = AgentOrchestrator(
        llm=UsageLLM(),
        resource_resolver=NoopResourceResolver(),
    )
    with use_trace_sink(sink), trace_scope(session_id="session-1"):
        await orchestrator.handle_text_turn(
            user_text="Hello",
            state=ConversationState(session_id="session-1"),
            db=object(),  # type: ignore[arg-type]
        )

    completed = next(
        event
        for event in sink.events
        if event.event_name == "llm.request.completed"
    )
    assert completed.metadata["prompt_tokens"] == 17
    assert completed.metadata["completion_tokens"] == 5
    assert completed.metadata["total_tokens"] == 22


def test_cost_estimator_calculates_known_rate_and_reports_unknown() -> None:
    catalog = PricingCatalog(
        [
            PricingRate(
                provider="fake",
                model="model-a",
                operation="llm",
                unit="input_token",
                usd_per_unit=Decimal("0.01"),
                effective_date=date(2026, 1, 1),
                source_note="test fixture",
            )
        ]
    )
    estimator = CostEstimator(catalog)
    known = estimator.estimate(
        [
            UsageRecord(
                provider="fake",
                model="model-a",
                operation="llm",
                quantities={"input_token": 3},
            )
        ]
    )
    unknown = estimator.estimate(
        [
            UsageRecord(
                provider="fake",
                model="unknown",
                operation="llm",
                quantities={"input_token": 3},
            )
        ]
    )

    assert known.estimated_usd == Decimal("0.03")
    assert known.unavailable == ()
    assert unknown.estimated_usd is None
    assert unknown.unavailable == ("fake/unknown/llm/input_token",)


def test_percentiles_use_documented_nearest_rank_method() -> None:
    values = list(range(1, 11))

    assert nearest_rank_percentile(values, 50) == 5
    assert nearest_rank_percentile(values, 90) == 9
    assert nearest_rank_percentile(values, 95) == 10
    summary = summarize_numeric(values)
    assert summary.count == 10
    assert summary.minimum == 1
    assert summary.maximum == 10
    assert summary.mean == 5.5


def test_turn_summary_aggregates_tools_retrieval_latency_usage_and_cost() -> None:
    common = {
        "trace_id": "trace_1234567890123456",
        "session_id": "session-1",
        "turn_id": "turn_12345678901234567",
    }
    events = [
        TraceEvent(
            **common,
            event_name="rag.retrieval.completed",
            component="rag",
            status=TraceStatus.COMPLETED,
            duration_ms=2,
            metadata={"retrieval_result_count": 3},
        ),
        TraceEvent(
            **common,
            event_name="tool.execution.completed",
            component="tools",
            status=TraceStatus.COMPLETED,
            duration_ms=4,
            metadata={"tool_name": "get_account_balance"},
        ),
        TraceEvent(
            **common,
            event_name="llm.request.completed",
            component="llm",
            status=TraceStatus.COMPLETED,
            duration_ms=5,
            metadata={
                "provider": "groq",
                "model": "openai/gpt-oss-20b",
                "prompt_tokens": 10,
                "completion_tokens": 5,
            },
        ),
        TraceEvent(
            **common,
            event_name="llm.request.completed",
            component="llm",
            status=TraceStatus.COMPLETED,
            duration_ms=7,
            metadata={
                "provider": "groq",
                "model": "openai/gpt-oss-20b",
                "prompt_tokens": 4,
                "completion_tokens": 2,
            },
        ),
        TraceEvent(
            **common,
            event_name="tool.execution.failed",
            component="tools",
            status=TraceStatus.FAILED,
            duration_ms=1,
            error_category="tool_error",
        ),
        TraceEvent(
            **common,
            event_name="agent.turn.completed",
            component="api",
            status=TraceStatus.COMPLETED,
            duration_ms=20,
        ),
    ]

    summary = summarize_turn(events)

    assert summary.tool_calls == ["get_account_balance"]
    assert summary.retrieval_count == 3
    assert summary.status == "completed"
    assert summary.latency_ms["llm.request"] == 12
    assert summary.estimated_cost_usd is not None
    assert summary.errors == ["tool_error"]


def test_trace_event_rejects_naive_timestamp() -> None:
    from datetime import datetime

    with pytest.raises(ValidationError, match="timezone-aware"):
        TraceEvent(
            event_name="agent.turn.started",
            timestamp=datetime(2026, 1, 1),  # noqa: DTZ001 - validation test
            trace_id="trace_1234567890123456",
            session_id="session-1",
            turn_id="turn_12345678901234567",
            component="agent",
            status=TraceStatus.STARTED,
        )


def test_bounded_trace_sink_evicts_oldest_events() -> None:
    sink = BoundedInMemoryTraceSink(max_events=2)

    common = {
        "trace_id": "trace_1234567890123456",
        "session_id": "session-1",
        "turn_id": "turn_12345678901234567",
        "component": "test",
        "status": TraceStatus.COMPLETED,
    }

    sink.emit(TraceEvent(**common, event_name="test.first"))
    sink.emit(TraceEvent(**common, event_name="test.second"))
    sink.emit(TraceEvent(**common, event_name="test.third"))

    assert [
        event.event_name
        for event in sink.events_for_session("session-1")
    ] == ["test.second", "test.third"]
