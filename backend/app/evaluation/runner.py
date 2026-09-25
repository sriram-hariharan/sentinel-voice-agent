import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from backend.app.agent.orchestrator import AgentOrchestrator
from backend.app.agent.resource_resolver import ResourceResolution
from backend.app.conversation.state import (
    AuthenticationLevel,
    ConversationState,
)
from backend.app.evaluation.graders import (
    calculate_evaluation_metrics,
    grade_scenario,
)
from backend.app.evaluation.models import (
    AgentScenario,
    EvaluationLLMResponse,
    EvaluationReport,
    ObservedToolCall,
    ScenarioDataset,
    ScenarioObservation,
    ScenarioResult,
)
from backend.app.observability.context import trace_scope
from backend.app.observability.metrics import latency_by_stage
from backend.app.observability.redaction import redact_metadata
from backend.app.observability.tracing import (
    InMemoryTraceSink,
    safe_error_category,
    trace_span,
    use_trace_sink,
)
from backend.app.observability.usage import (
    OFFICIAL_GROQ_PRICING,
    CostEstimator,
    UsageRecord,
)
from backend.app.providers.llm import LLMResponse, LLMToolCall, LLMUsage
from backend.app.rag.models import PolicyChunk, RetrievedPolicyChunk
from backend.app.tools.errors import ToolError, ToolTimeoutError
from backend.app.tools.executor import ToolExecutor
from backend.app.tools.registry import TOOL_REGISTRY, RegisteredTool

EVALUATION_MODEL = "openai/gpt-oss-20b"
EVALUATION_PROVIDER = "groq"
EVALUATION_DATE = date(2026, 9, 24)
CUSTOMER_ID = UUID("11111111-1111-4111-8111-111111111111")


class EvaluationToolOutput(BaseModel):
    tool_name: str
    ok: bool = True
    payload: dict[str, Any] = Field(default_factory=dict)


class EvaluationLLM:
    provider = EVALUATION_PROVIDER
    model = EVALUATION_MODEL

    def __init__(self, responses: list[EvaluationLLMResponse]) -> None:
        self._responses = list(responses)
        self.requested_calls: list[ObservedToolCall] = []

    async def generate(self, *, messages, tools=None) -> LLMResponse:
        del messages, tools
        if not self._responses:
            raise RuntimeError("evaluation LLM response queue is empty")
        response = self._responses.pop(0)
        tool_calls = [
            LLMToolCall(
                id=f"eval-call-{index}",
                name=call.name,
                arguments=call.arguments,
            )
            for index, call in enumerate(response.tool_calls, start=1)
        ]
        self.requested_calls.extend(
            ObservedToolCall(name=call.name, arguments=call.arguments)
            for call in response.tool_calls
        )
        return LLMResponse(
            content=response.content,
            tool_calls=tool_calls,
            model=self.model,
            finish_reason="stop" if not tool_calls else "tool_calls",
            usage=LLMUsage(
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                total_tokens=(
                    response.prompt_tokens + response.completion_tokens
                ),
            ),
        )


class EvaluationPolicyRetriever:
    def __init__(self, scenario: AgentScenario) -> None:
        self._turns = {turn.user: turn for turn in scenario.turns}

    async def retrieve(self, *, query, search, top_k):
        del search
        turn = self._turns[query]
        if turn.retrieval_failure:
            raise RuntimeError("simulated retrieval failure")
        return [
            RetrievedPolicyChunk(
                chunk=PolicyChunk(
                    chunk_id=f"{source.policy_id}:evaluation:00",
                    policy_id=source.policy_id,
                    title=source.title,
                    version="1.0",
                    effective_date=EVALUATION_DATE,
                    section=source.section,
                    chunk_index=0,
                    content=source.content,
                ),
                keyword_rank=index,
                vector_rank=index,
                keyword_score=1.0,
                vector_similarity=0.9,
                rrf_score=2 / (60 + index),
            )
            for index, source in enumerate(turn.policy_sources[:top_k], start=1)
        ]


class EvaluationResourceResolver:
    """Keeps provider decisions deterministic while orchestration stays real."""

    async def resolve(self, **kwargs) -> ResourceResolution:
        del kwargs
        return ResourceResolution()


class SyntheticToolRuntime:
    def __init__(self, failures: dict[str, str]) -> None:
        self.failures = failures
        self.calls: list[ObservedToolCall] = []

    async def handle(self, tool_name, request, context, session):
        del context, session
        failure = self.failures.get(tool_name)
        if failure == "timeout":
            raise ToolTimeoutError("simulated timeout")
        if failure == "tool_error":
            raise ToolError("simulated tool failure")
        arguments = request.model_dump(mode="json")
        self.calls.append(
            ObservedToolCall(name=tool_name, arguments=arguments)
        )
        return EvaluationToolOutput(
            tool_name=tool_name,
            payload={"arguments_accepted": True},
        )

    def executor(self) -> ToolExecutor:
        registry: dict[str, RegisteredTool] = {}
        for name, registered in TOOL_REGISTRY.items():
            async def handler(request, context, session, *, tool_name=name):
                return await self.handle(
                    tool_name,
                    request,
                    context,
                    session,
                )

            registry[name] = RegisteredTool(
                definition=registered.definition,
                input_model=registered.input_model,
                handler=handler,
            )
        return ToolExecutor(registry=registry)


def _correlation_id(prefix: str, scenario_id: str, index: int) -> str:
    digest = hashlib.sha256(
        f"{prefix}:{scenario_id}:{index}".encode()
    ).hexdigest()[:24]
    return f"eval_{digest}"


def _initial_state(scenario: AgentScenario) -> ConversationState:
    initial = scenario.initial_session
    return ConversationState(
        session_id=f"eval-{scenario.scenario_id}",
        customer_id=CUSTOMER_ID if initial.authenticated else None,
        authentication_level=(
            AuthenticationLevel.AUTHENTICATED
            if initial.authenticated
            else AuthenticationLevel.UNAUTHENTICATED
        ),
        active_intent=initial.active_intent,
        active_account_id=(
            UUID(initial.active_account_id)
            if initial.active_account_id
            else None
        ),
        active_card_id=(
            UUID(initial.active_card_id) if initial.active_card_id else None
        ),
        active_transaction_id=(
            UUID(initial.active_transaction_id)
            if initial.active_transaction_id
            else None
        ),
    )


async def run_scenario(
    scenario: AgentScenario,
) -> tuple[ScenarioResult, list]:
    llm = EvaluationLLM(
        [response for turn in scenario.turns for response in turn.llm_responses]
    )
    runtime = SyntheticToolRuntime(scenario.tool_failure)
    orchestrator = AgentOrchestrator(
        llm=llm,
        tool_executor=runtime.executor(),
        resource_resolver=EvaluationResourceResolver(),
        policy_retriever=EvaluationPolicyRetriever(scenario),
    )
    state = _initial_state(scenario)
    sink = InMemoryTraceSink()
    responses: list[str] = []
    executed_tools: list[str] = []
    final_status: str | None = None
    exception_category: str | None = None
    interrupted = False

    with use_trace_sink(sink):
        for index, turn in enumerate(scenario.turns, start=1):
            if turn.interrupt_before:
                interrupted = True
                state.interrupt(
                    preserve_pending_confirmation=(
                        turn.preserve_pending_confirmation
                    )
                )
            trace_id = _correlation_id("trace", scenario.scenario_id, index)
            turn_id = _correlation_id("turn", scenario.scenario_id, index)
            try:
                with trace_scope(
                    session_id=state.session_id,
                    trace_id=trace_id,
                    turn_id=turn_id,
                ), trace_span(
                    "agent.turn",
                    component="evaluation",
                    metadata={"evaluation_scenario": scenario.scenario_id},
                ):
                    result = await orchestrator.handle_text_turn(
                        user_text=turn.user,
                        state=state,
                        db=object(),  # type: ignore[arg-type]
                    )
                responses.append(result.text)
                executed_tools.extend(result.executed_tools)
                final_status = result.status.value
            except Exception as exc:  # noqa: BLE001 - scenario records failures
                exception_category = safe_error_category(exc)
                final_status = "FAILED"
                break

    policy_ids = sorted(
        {
            str(policy_id)
            for event in sink.events
            if event.event_name == "rag.retrieval.completed"
            for policy_id in event.metadata.get("policy_source_slugs", [])
        }
    )
    observation = ScenarioObservation(
        requested_tools=llm.requested_calls,
        executed_tools=executed_tools,
        executed_tool_arguments={
            call.name: call.arguments for call in runtime.calls
        },
        final_turn_status=final_status,
        pending_action=(
            state.pending_action.action if state.pending_action else None
        ),
        policy_ids=policy_ids,
        responses=responses,
        escalated=state.escalation_status == "ESCALATED",
        interrupted=interrupted,
        exception_category=exception_category,
    )
    return (
        grade_scenario(
            scenario,
            observation,
            trace_event_count=len(sink.events),
        ),
        sink.events,
    )


def _usage_records(events: list) -> list[UsageRecord]:
    records: list[UsageRecord] = []
    for event in events:
        if event.event_name != "llm.request.completed":
            continue
        records.append(
            UsageRecord(
                provider=str(event.metadata.get("provider", "unknown")),
                model=str(event.metadata.get("model", "unknown")),
                operation="llm",
                quantities={
                    "input_token": float(event.metadata.get("prompt_tokens", 0)),
                    "output_token": float(
                        event.metadata.get("completion_tokens", 0)
                    ),
                },
            )
        )
    return records


async def run_evaluation(dataset: ScenarioDataset) -> EvaluationReport:
    results: list[ScenarioResult] = []
    events = []
    for scenario in dataset.scenarios:
        result, scenario_events = await run_scenario(scenario)
        results.append(result)
        events.extend(scenario_events)

    metrics = calculate_evaluation_metrics(list(dataset.scenarios), results)
    usage_records = _usage_records(events)
    estimate = CostEstimator(OFFICIAL_GROQ_PRICING).estimate(usage_records)
    prompt_tokens = sum(
        record.quantities.get("input_token", 0) for record in usage_records
    )
    completion_tokens = sum(
        record.quantities.get("output_token", 0) for record in usage_records
    )
    successful = sum(result.passed for result in results)
    total_cost = estimate.estimated_usd
    return EvaluationReport(
        evaluation_version=dataset.evaluation_version,
        generated_at=datetime.now(UTC),
        timing_label="offline deterministic evaluation timings",
        scenario_count=len(results),
        passed=successful,
        failed=len(results) - successful,
        scenario_results=results,
        metrics=metrics,
        latency=latency_by_stage(events),
        usage={
            "llm_input_tokens": prompt_tokens,
            "llm_output_tokens": completion_tokens,
        },
        pricing_catalog_version=OFFICIAL_GROQ_PRICING.version,
        estimated_cost_usd=(str(total_cost) if total_cost is not None else None),
        estimated_cost_per_session_usd=(
            str(total_cost / len(results))
            if total_cost is not None and results
            else None
        ),
        estimated_cost_per_successful_task_usd=(
            str(total_cost / successful)
            if total_cost is not None and successful
            else None
        ),
        cost_unavailable=list(estimate.unavailable),
        failures=[result.scenario_id for result in results if not result.passed],
    )


def write_json_report(report: EvaluationReport, path: Path) -> None:
    safe_payload = redact_metadata(report.model_dump(mode="json"))
    path.write_text(
        json.dumps(safe_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
