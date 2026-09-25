import json
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.app.observability.metrics import NumericSummary


class EvaluationToolCall(BaseModel):
    name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True, extra="forbid")


class EvaluationLLMResponse(BaseModel):
    content: str = ""
    tool_calls: tuple[EvaluationToolCall, ...] = ()
    prompt_tokens: int = Field(default=20, ge=0)
    completion_tokens: int = Field(default=10, ge=0)

    model_config = ConfigDict(frozen=True, extra="forbid")


class EvaluationPolicySource(BaseModel):
    policy_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    section: str = Field(min_length=1)
    content: str = Field(min_length=1)

    model_config = ConfigDict(frozen=True, extra="forbid")


class EvaluationTurn(BaseModel):
    user: str = Field(min_length=1)
    llm_responses: tuple[EvaluationLLMResponse, ...] = ()
    policy_sources: tuple[EvaluationPolicySource, ...] = ()
    retrieval_failure: bool = False
    interrupt_before: bool = False
    preserve_pending_confirmation: bool = False

    model_config = ConfigDict(frozen=True, extra="forbid")


class InitialSession(BaseModel):
    authenticated: bool = False
    active_intent: str | None = None
    active_account_id: str | None = None
    active_card_id: str | None = None
    active_transaction_id: str | None = None

    model_config = ConfigDict(frozen=True, extra="forbid")


class ScenarioExpectation(BaseModel):
    requested_tools: tuple[str, ...] = ()
    executed_tools: tuple[str, ...] = ()
    forbidden_tools: tuple[str, ...] = ()
    tool_arguments: dict[str, dict[str, Any]] = Field(default_factory=dict)
    final_turn_status: str | None = None
    pending_action: str | None = None
    policy_ids: tuple[str, ...] = ()
    response_contains: tuple[str, ...] = ()
    escalation_required: bool = False
    unauthorized_action_attempt: bool = False
    confirmation_required: bool = False
    confirmation_should_execute: bool = False
    interruption_expected: bool = False
    expected_exception: bool = False
    expected_error_category: str | None = None

    model_config = ConfigDict(frozen=True, extra="forbid")

    @model_validator(mode="after")
    def error_category_requires_exception(self) -> "ScenarioExpectation":
        if (
            self.expected_error_category is not None
            and not self.expected_exception
        ):
            raise ValueError(
                "expected_error_category requires expected_exception=true"
            )
        return self


class AgentScenario(BaseModel):
    scenario_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    tags: tuple[str, ...] = Field(min_length=1)
    initial_session: InitialSession = Field(default_factory=InitialSession)
    turns: tuple[EvaluationTurn, ...] = Field(min_length=1)
    tool_failure: dict[str, Literal["tool_error", "timeout"]] = Field(
        default_factory=dict
    )
    expected: ScenarioExpectation

    model_config = ConfigDict(frozen=True, extra="forbid")


class ScenarioDataset(BaseModel):
    evaluation_version: str = Field(min_length=1)
    scenarios: tuple[AgentScenario, ...] = Field(min_length=1)

    model_config = ConfigDict(frozen=True, extra="forbid")

    @model_validator(mode="after")
    def scenario_ids_are_unique(self) -> "ScenarioDataset":
        ids = [scenario.scenario_id for scenario in self.scenarios]
        if len(ids) != len(set(ids)):
            raise ValueError("scenario IDs must be unique")
        return self


class ObservedToolCall(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)


class ScenarioObservation(BaseModel):
    requested_tools: list[ObservedToolCall] = Field(default_factory=list)
    executed_tools: list[str] = Field(default_factory=list)
    executed_tool_arguments: dict[str, dict[str, Any]] = Field(
        default_factory=dict
    )
    final_turn_status: str | None = None
    pending_action: str | None = None
    policy_ids: list[str] = Field(default_factory=list)
    responses: list[str] = Field(default_factory=list)
    escalated: bool = False
    interrupted: bool = False
    exception_category: str | None = None

    model_config = ConfigDict(frozen=True)


class GradeCheck(BaseModel):
    name: str
    passed: bool
    critical: bool = True
    detail: str = ""

    model_config = ConfigDict(frozen=True)


class ScenarioResult(BaseModel):
    scenario_id: str
    description: str
    tags: tuple[str, ...]
    passed: bool
    checks: list[GradeCheck]
    observation: ScenarioObservation
    trace_event_count: int = Field(ge=0)

    model_config = ConfigDict(frozen=True)


class SafetyMetrics(BaseModel):
    unauthorized_action_attempts: int = 0
    unauthorized_actions_executed: int = 0
    confirmation_required_actions: int = 0
    confirmation_compliant_actions: int = 0
    cross_customer_attempts_blocked: int = 0
    prompt_injection_attempts_blocked: int = 0

    model_config = ConfigDict(frozen=True)


class EvaluationMetrics(BaseModel):
    task_success_rate: float
    tool_selection_accuracy: float
    tool_argument_accuracy: float
    unauthorized_action_rate: float
    confirmation_compliance: float
    escalation_precision: float
    escalation_recall: float
    interruption_recovery_rate: float
    policy_source_accuracy: float
    safety: SafetyMetrics

    model_config = ConfigDict(frozen=True)


class EvaluationReport(BaseModel):
    evaluation_version: str
    generated_at: datetime
    timing_label: str
    scenario_count: int
    passed: int
    failed: int
    scenario_results: list[ScenarioResult]
    metrics: EvaluationMetrics
    latency: dict[str, NumericSummary]
    usage: dict[str, float]
    pricing_catalog_version: str
    estimated_cost_usd: str | None
    estimated_cost_per_session_usd: str | None
    estimated_cost_per_successful_task_usd: str | None
    cost_unavailable: list[str]
    failures: list[str]

    model_config = ConfigDict(frozen=True, extra="forbid")


def load_agent_scenarios(path: Path) -> ScenarioDataset:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ScenarioDataset.model_validate(payload)
