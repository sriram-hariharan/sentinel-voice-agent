from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field

from backend.app.observability.events import TraceEvent
from backend.app.observability.metrics import NumericSummary, latency_by_stage
from backend.app.observability.usage import (
    OFFICIAL_GROQ_PRICING,
    CostEstimator,
    UsageRecord,
)


class TurnTraceSummary(BaseModel):
    trace_id: str
    turn_id: str
    status: str
    outcome: str | None = None
    tool_calls: list[str] = Field(default_factory=list)
    retrieval_count: int = 0
    policy_sources: list[str] = Field(default_factory=list)
    latency_ms: dict[str, float] = Field(default_factory=dict)
    usage: list[UsageRecord] = Field(default_factory=list)
    estimated_cost_usd: Decimal | None = None
    cost_unavailable: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    model_config = ConfigDict(frozen=True)


class SessionTraceSummary(BaseModel):
    session_id: str
    turn_count: int
    successful_tasks: int | None = None
    latency: dict[str, NumericSummary] = Field(default_factory=dict)
    usage: list[UsageRecord] = Field(default_factory=list)
    estimated_total_cost_usd: Decimal | None = None
    estimated_cost_per_success_usd: Decimal | None = None
    cost_unavailable: list[str] = Field(default_factory=list)

    model_config = ConfigDict(frozen=True)


def _usage_from_events(events: list[TraceEvent]) -> list[UsageRecord]:
    usage: list[UsageRecord] = []
    for event in events:
        metadata = event.metadata
        if event.event_name == "llm.request.completed":
            usage.append(
                UsageRecord(
                    provider=str(metadata.get("provider", "unknown")),
                    model=str(metadata.get("model", "unknown")),
                    operation="llm",
                    quantities={
                        "input_token": float(metadata.get("prompt_tokens", 0)),
                        "output_token": float(
                            metadata.get("completion_tokens", 0)
                        ),
                    },
                )
            )
        elif event.event_name == "stt.completed":
            usage.append(
                UsageRecord(
                    provider=str(metadata.get("provider", "unknown")),
                    model=str(metadata.get("model", "unknown")),
                    operation="stt",
                    quantities={
                        "audio_second": float(metadata.get("audio_seconds", 0))
                    },
                )
            )
        elif event.event_name == "tts.completed":
            usage.append(
                UsageRecord(
                    provider=str(metadata.get("provider", "unknown")),
                    model=str(metadata.get("model", "unknown")),
                    operation="tts",
                    quantities={
                        "character": float(metadata.get("character_count", 0))
                    },
                )
            )
    return usage


def summarize_turn(events: list[TraceEvent]) -> TurnTraceSummary:
    if not events:
        raise ValueError("at least one trace event is required")
    trace_ids = {event.trace_id for event in events}
    turn_ids = {event.turn_id for event in events}
    if len(trace_ids) != 1 or len(turn_ids) != 1:
        raise ValueError("turn summary events must share trace_id and turn_id")

    usage = _usage_from_events(events)
    estimate = CostEstimator(OFFICIAL_GROQ_PRICING).estimate(usage)
    failed = [event for event in events if event.status == "failed"]
    agent_terminal = next(
        (
            event
            for event in reversed(events)
            if event.event_name
            in {"agent.turn.completed", "agent.turn.failed"}
        ),
        None,
    )
    retrieval_count = 0
    policy_sources: list[str] = []
    for event in events:
        if event.event_name == "rag.retrieval.completed":
            retrieval_count = int(
                event.metadata.get("retrieval_result_count", 0)
            )
            for source in event.metadata.get("policy_source_slugs", []):
                source_name = str(source)
                if source_name not in policy_sources:
                    policy_sources.append(source_name)
    latency_ms: dict[str, float] = {}
    for event in events:
        if (
            event.duration_ms is None
            or not event.event_name.endswith(".completed")
        ):
            continue
        stage = event.event_name.removesuffix(".completed")
        latency_ms[stage] = latency_ms.get(stage, 0.0) + event.duration_ms

    status = "info"
    if agent_terminal is not None:
        status = agent_terminal.status.value
    elif failed:
        status = "failed"
    elif any(event.status == "completed" for event in events):
        status = "completed"

    return TurnTraceSummary(
        trace_id=next(iter(trace_ids)),
        turn_id=next(iter(turn_ids)),
        status=status,
        outcome=(
            str(agent_terminal.metadata.get("turn_status"))
            if agent_terminal is not None
            and agent_terminal.metadata.get("turn_status") is not None
            else None
        ),
        tool_calls=[
            str(event.metadata.get("tool_name"))
            for event in events
            if event.event_name == "tool.execution.completed"
        ],
        retrieval_count=retrieval_count,
        policy_sources=policy_sources,
        latency_ms=latency_ms,
        usage=usage,
        estimated_cost_usd=estimate.estimated_usd,
        cost_unavailable=list(estimate.unavailable),
        errors=[
            event.error_category or "unknown_error" for event in failed
        ],
    )


def summarize_session(
    events: list[TraceEvent],
    *,
    successful_tasks: int | None = None,
) -> SessionTraceSummary:
    if not events:
        raise ValueError("at least one trace event is required")
    session_ids = {event.session_id for event in events}
    if len(session_ids) != 1:
        raise ValueError("session summary events must share session_id")
    usage = _usage_from_events(events)
    estimate = CostEstimator(OFFICIAL_GROQ_PRICING).estimate(usage)
    per_success = None
    if (
        estimate.estimated_usd is not None
        and successful_tasks is not None
        and successful_tasks > 0
    ):
        per_success = estimate.estimated_usd / successful_tasks
    return SessionTraceSummary(
        session_id=next(iter(session_ids)),
        turn_count=len({event.turn_id for event in events}),
        successful_tasks=successful_tasks,
        latency=latency_by_stage(events),
        usage=usage,
        estimated_total_cost_usd=estimate.estimated_usd,
        estimated_cost_per_success_usd=per_success,
        cost_unavailable=list(estimate.unavailable),
    )
