"""Versioned, offline-only contracts for SentinelVoice V2 experiments."""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

INTENT_RISK_SCHEMA_VERSION = "intent-risk.v1"
ROUTING_BENCHMARK_SCHEMA_VERSION = "routing-benchmark.v1"
AUDIO_REPLAY_SCHEMA_VERSION = "audio-replay.v1"
BENCHMARK_REPORT_SCHEMA_VERSION = "benchmark-report.v1"
FROZEN_BASELINE_SCHEMA_VERSION = "frozen-v1-baseline.v1"


class IntentLabel(StrEnum):
    INFORMATIONAL_POLICY = "informational_policy"
    ACCOUNT_BALANCE = "account_balance"
    RECENT_TRANSACTIONS = "recent_transactions"
    TRANSACTION_DETAILS = "transaction_details"
    CARD_STATUS = "card_status"
    FREEZE_CARD = "freeze_card"
    CREATE_DISPUTE = "create_dispute"
    ESCALATION = "escalation"
    UNSUPPORTED_OR_UNCERTAIN = "unsupported_or_uncertain"


class RiskLabel(StrEnum):
    PUBLIC = "PUBLIC"
    PRIVATE_READ = "PRIVATE_READ"
    PROTECTED_WRITE = "PROTECTED_WRITE"
    ESCALATION_OR_UNCERTAIN = "ESCALATION_OR_UNCERTAIN"


class DatasetSplit(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    LOCKED_TEST = "locked_test"


class ExperimentArm(StrEnum):
    CONTROL = "control"
    TREATMENT = "treatment"


class MeasurementSource(StrEnum):
    DETERMINISTIC_FAKE = "deterministic_fake"
    LOCAL_ML = "local_ml"
    LIVE_LLM = "live_llm"
    LIVE_STT = "live_stt"
    LIVE_VOICE = "live_voice"


_RISK_BY_INTENT = {
    IntentLabel.INFORMATIONAL_POLICY: RiskLabel.PUBLIC,
    IntentLabel.ACCOUNT_BALANCE: RiskLabel.PRIVATE_READ,
    IntentLabel.RECENT_TRANSACTIONS: RiskLabel.PRIVATE_READ,
    IntentLabel.TRANSACTION_DETAILS: RiskLabel.PRIVATE_READ,
    IntentLabel.CARD_STATUS: RiskLabel.PRIVATE_READ,
    IntentLabel.FREEZE_CARD: RiskLabel.PROTECTED_WRITE,
    IntentLabel.CREATE_DISPUTE: RiskLabel.PROTECTED_WRITE,
    IntentLabel.ESCALATION: RiskLabel.ESCALATION_OR_UNCERTAIN,
    IntentLabel.UNSUPPORTED_OR_UNCERTAIN: RiskLabel.ESCALATION_OR_UNCERTAIN,
}


def _validate_unique_ordered(values: list[str], *, label: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{label} values must be unique")
    if values != sorted(values):
        raise ValueError(f"{label} values must be ordered")


class IntentRiskExample(BaseModel):
    example_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    intent: IntentLabel
    risk: RiskLabel
    group_id: str = Field(min_length=1)
    split: DatasetSplit
    tags: tuple[str, ...] = ()

    model_config = ConfigDict(frozen=True, extra="forbid")


class IntentRiskDataset(BaseModel):
    schema_version: Literal["intent-risk.v1"]
    dataset_version: str = Field(min_length=1)
    advisory_only: Literal[True]
    examples: tuple[IntentRiskExample, ...] = Field(min_length=1)

    model_config = ConfigDict(frozen=True, extra="forbid")

    @model_validator(mode="after")
    def validate_examples(self) -> IntentRiskDataset:
        example_ids = [example.example_id for example in self.examples]
        _validate_unique_ordered(example_ids, label="example_id")

        splits_by_group: dict[str, set[DatasetSplit]] = {}
        for example in self.examples:
            if example.risk != _RISK_BY_INTENT[example.intent]:
                raise ValueError(
                    f"risk {example.risk} is incompatible with intent "
                    f"{example.intent} for {example.example_id}"
                )
            splits_by_group.setdefault(example.group_id, set()).add(
                example.split
            )
        leaking_groups = sorted(
            group_id
            for group_id, splits in splits_by_group.items()
            if len(splits) > 1
        )
        if leaking_groups:
            raise ValueError(
                "group_id values cannot cross dataset splits: "
                + ", ".join(leaking_groups)
            )
        return self


class RoutingTurn(BaseModel):
    user_text: str = Field(min_length=1)

    model_config = ConfigDict(frozen=True, extra="forbid")


class RoutingExpectedOutcome(BaseModel):
    outcome: str = Field(min_length=1)
    expected_intent: IntentLabel
    expected_tool_proposals: tuple[str, ...]
    expected_executed_tools: tuple[str, ...]
    forbidden_tools: tuple[str, ...] = ()

    model_config = ConfigDict(frozen=True, extra="forbid")


class RoutingBenchmarkScenario(BaseModel):
    scenario_id: str = Field(min_length=1)
    authenticated: bool
    turns: tuple[RoutingTurn, ...] = Field(min_length=1)
    expected: RoutingExpectedOutcome
    tags: tuple[str, ...] = ()

    model_config = ConfigDict(frozen=True, extra="forbid")


class RoutingBenchmark(BaseModel):
    schema_version: Literal["routing-benchmark.v1"]
    benchmark_version: str = Field(min_length=1)
    control_model_identifier: str = Field(min_length=1)
    treatment_model_identifier: str | None = None
    experiment_arms: tuple[ExperimentArm, ...] = (
        ExperimentArm.CONTROL,
        ExperimentArm.TREATMENT,
    )
    routing_policy_version: str | None = None
    classifier_version: str | None = None
    prompt_version: str | None = None
    tool_schema_version: str | None = None
    evaluation_version: str = Field(min_length=1)
    scenarios: tuple[RoutingBenchmarkScenario, ...] = Field(min_length=1)

    model_config = ConfigDict(frozen=True, extra="forbid")

    @model_validator(mode="after")
    def validate_scenarios(self) -> RoutingBenchmark:
        scenario_ids = [scenario.scenario_id for scenario in self.scenarios]
        _validate_unique_ordered(scenario_ids, label="scenario_id")
        if set(self.experiment_arms) != {
            ExperimentArm.CONTROL,
            ExperimentArm.TREATMENT,
        }:
            raise ValueError("routing benchmark requires control and treatment arms")
        return self


class AudioReplayFixture(BaseModel):
    fixture_id: str = Field(min_length=1)
    audio_path: str = Field(min_length=1)
    ground_truth_transcript: str = Field(min_length=1)
    scenario_id: str = Field(min_length=1)
    tags: tuple[str, ...] = ()
    provenance: str = Field(min_length=1)
    expected_intent: IntentLabel
    expected_tool: str | None = None
    expected_outcome: str = Field(min_length=1)

    model_config = ConfigDict(frozen=True, extra="forbid")


class AudioReplayManifest(BaseModel):
    schema_version: Literal["audio-replay.v1"]
    manifest_version: str = Field(min_length=1)
    fixtures: tuple[AudioReplayFixture, ...] = ()

    model_config = ConfigDict(frozen=True, extra="forbid")

    @model_validator(mode="after")
    def validate_fixtures(self) -> AudioReplayManifest:
        fixture_ids = [fixture.fixture_id for fixture in self.fixtures]
        _validate_unique_ordered(fixture_ids, label="fixture_id")
        return self


class LatencyPercentiles(BaseModel):
    sample_count: int = Field(ge=1)
    p50_ms: float = Field(ge=0)
    p90_ms: float = Field(ge=0)
    p95_ms: float = Field(ge=0)

    model_config = ConfigDict(frozen=True, extra="forbid")

    @model_validator(mode="after")
    def validate_percentiles(self) -> LatencyPercentiles:
        if not self.p50_ms <= self.p90_ms <= self.p95_ms:
            raise ValueError("latency percentiles must be monotonic")
        return self


class BenchmarkResult(BaseModel):
    result_id: str = Field(min_length=1)
    measurement_source: MeasurementSource
    evaluation_version: str = Field(min_length=1)
    experiment_arm: ExperimentArm | None = None
    provider: str | None = None
    model_identifier: str | None = None
    sample_count: int = Field(ge=0)
    failure_count: int = Field(ge=0)
    metrics: dict[str, float] = Field(default_factory=dict)
    latency: dict[str, LatencyPercentiles] = Field(default_factory=dict)
    estimated_cost_usd: str | None = None
    estimated_cost_per_successful_task_usd: str | None = None

    model_config = ConfigDict(frozen=True, extra="forbid")

    @model_validator(mode="after")
    def validate_result(self) -> BenchmarkResult:
        if self.failure_count > self.sample_count:
            raise ValueError("failure_count cannot exceed sample_count")
        if self.measurement_source in {
            MeasurementSource.LIVE_LLM,
            MeasurementSource.LIVE_STT,
        } and (not self.provider or not self.model_identifier):
            raise ValueError(
                "live provider measurements require provider and model_identifier"
            )
        return self


class BenchmarkReport(BaseModel):
    schema_version: Literal["benchmark-report.v1"]
    report_version: str = Field(min_length=1)
    results: tuple[BenchmarkResult, ...] = Field(min_length=1)

    model_config = ConfigDict(frozen=True, extra="forbid")

    @model_validator(mode="after")
    def validate_results(self) -> BenchmarkReport:
        result_ids = [result.result_id for result in self.results]
        _validate_unique_ordered(result_ids, label="result_id")
        return self


class FrozenV1Baseline(BaseModel):
    schema_version: Literal["frozen-v1-baseline.v1"]
    v1_tag: str
    v1_commit: str
    backend_tests: int
    frontend_tests: int
    agent_scenarios: int
    agent_passed: int
    retrieval_recall_at_1: float
    retrieval_recall_at_3: float
    retrieval_top1_hit_rate: float
    retrieval_mrr: float
    canonical_banking_fixture: str
    canonical_fixture_immutable: bool

    model_config = ConfigDict(frozen=True, extra="forbid")

    @model_validator(mode="after")
    def validate_frozen_values(self) -> FrozenV1Baseline:
        expected: dict[str, Any] = {
            "v1_tag": "v1.0.0",
            "v1_commit": "23f319f",
            "backend_tests": 299,
            "frontend_tests": 15,
            "agent_scenarios": 35,
            "agent_passed": 35,
            "retrieval_recall_at_1": 0.938,
            "retrieval_recall_at_3": 1.0,
            "retrieval_top1_hit_rate": 1.0,
            "retrieval_mrr": 1.0,
            "canonical_banking_fixture": "data/fixtures/banking.json",
            "canonical_fixture_immutable": True,
        }
        observed = self.model_dump(exclude={"schema_version"})
        if observed != expected:
            raise ValueError("frozen V1 baseline metadata does not match v1.0.0")
        return self


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_intent_risk_dataset(path: Path) -> IntentRiskDataset:
    return IntentRiskDataset.model_validate(_load_json(path))


def load_routing_benchmark(path: Path) -> RoutingBenchmark:
    return RoutingBenchmark.model_validate(_load_json(path))


def load_audio_replay_manifest(path: Path) -> AudioReplayManifest:
    return AudioReplayManifest.model_validate(_load_json(path))


def load_frozen_v1_baseline(path: Path) -> FrozenV1Baseline:
    return FrozenV1Baseline.model_validate(_load_json(path))
