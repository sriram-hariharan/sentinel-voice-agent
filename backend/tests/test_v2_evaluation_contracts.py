import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from backend.app.evaluation.v2_contracts import (
    AudioReplayManifest,
    BenchmarkReport,
    BenchmarkResult,
    FrozenV1Baseline,
    IntentRiskDataset,
    IntentRiskExample,
    MeasurementSource,
    RoutingBenchmark,
    load_audio_replay_manifest,
    load_frozen_v1_baseline,
    load_intent_risk_dataset,
    load_routing_benchmark,
)

V2_DATA_DIR = Path("data/evals/v2")
INTENT_DATASET_PATH = V2_DATA_DIR / "intent_risk_seed.json"
ROUTING_BENCHMARK_PATH = V2_DATA_DIR / "routing_benchmark.json"
AUDIO_MANIFEST_PATH = V2_DATA_DIR / "audio_replay_manifest.json"
BASELINE_PATH = V2_DATA_DIR / "frozen_v1_baseline.json"


def _example_payload(**overrides):
    payload = {
        "example_id": "example_001",
        "text": "What is my balance?",
        "intent": "account_balance",
        "risk": "PRIVATE_READ",
        "group_id": "balance_group_01",
        "split": "train",
        "tags": [],
    }
    payload.update(overrides)
    return payload


def _fixture_payload(**overrides):
    payload = {
        "fixture_id": "fixture_001",
        "audio_path": "audio/fixture_001.wav",
        "ground_truth_transcript": "What is my balance?",
        "scenario_id": "balance_001",
        "tags": [],
        "provenance": "synthetic test fixture",
        "expected_intent": "account_balance",
        "expected_tool": "get_account_balance",
        "expected_outcome": "Return the selected account balance.",
    }
    payload.update(overrides)
    return payload


def test_versioned_v2_artifacts_load() -> None:
    intent_dataset = load_intent_risk_dataset(INTENT_DATASET_PATH)
    routing_benchmark = load_routing_benchmark(ROUTING_BENCHMARK_PATH)
    audio_manifest = load_audio_replay_manifest(AUDIO_MANIFEST_PATH)
    baseline = load_frozen_v1_baseline(BASELINE_PATH)

    assert len(intent_dataset.examples) == 54
    assert len(routing_benchmark.scenarios) == 12
    assert audio_manifest.fixtures == ()
    assert baseline.v1_commit == "23f319f"


def test_intent_dataset_has_expected_split_and_label_coverage() -> None:
    dataset = load_intent_risk_dataset(INTENT_DATASET_PATH)
    split_counts: dict[str, int] = {}
    for example in dataset.examples:
        split_counts[example.split.value] = (
            split_counts.get(example.split.value, 0) + 1
        )

    assert split_counts == {
        "train": 27,
        "validation": 9,
        "locked_test": 18,
    }
    assert len({example.intent for example in dataset.examples}) == 9
    assert len({example.risk for example in dataset.examples}) == 4


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("intent", "transfer_money"),
        ("risk", "AUTHORIZED"),
        ("split", "test"),
    ],
)
def test_invalid_intent_risk_example_enums_are_rejected(
    field: str,
    value: str,
) -> None:
    with pytest.raises(ValidationError):
        IntentRiskExample.model_validate(
            _example_payload(**{field: value})
        )


def test_incompatible_intent_and_risk_are_rejected() -> None:
    with pytest.raises(ValidationError, match="incompatible with intent"):
        IntentRiskDataset.model_validate(
            {
                "schema_version": "intent-risk.v1",
                "dataset_version": "test",
                "advisory_only": True,
                "examples": [_example_payload(risk="PROTECTED_WRITE")],
            }
        )


def test_duplicate_example_ids_are_rejected() -> None:
    example = _example_payload()
    with pytest.raises(ValidationError, match="example_id values must be unique"):
        IntentRiskDataset.model_validate(
            {
                "schema_version": "intent-risk.v1",
                "dataset_version": "test",
                "advisory_only": True,
                "examples": [example, example],
            }
        )


def test_group_ids_cannot_cross_dataset_splits() -> None:
    with pytest.raises(ValidationError, match="cannot cross dataset splits"):
        IntentRiskDataset.model_validate(
            {
                "schema_version": "intent-risk.v1",
                "dataset_version": "test",
                "advisory_only": True,
                "examples": [
                    _example_payload(example_id="example_001"),
                    _example_payload(
                        example_id="example_002",
                        split="locked_test",
                    ),
                ],
            }
        )


def test_duplicate_routing_scenario_ids_are_rejected() -> None:
    payload = json.loads(ROUTING_BENCHMARK_PATH.read_text(encoding="utf-8"))
    payload["scenarios"] = [payload["scenarios"][0], payload["scenarios"][0]]

    with pytest.raises(ValidationError, match="scenario_id values must be unique"):
        RoutingBenchmark.model_validate(payload)


def test_routing_tool_proposals_and_executions_are_explicit() -> None:
    benchmark = load_routing_benchmark(ROUTING_BENCHMARK_PATH)
    expected = {
        "account_balance": (
            ("get_account_balance",),
            ("get_account_balance",),
        ),
        "ambiguous_transaction": ((), ()),
        "card_status": (("get_card_status",), ("get_card_status",)),
        "create_dispute": (("create_dispute",), ()),
        "escalation": (("escalate_to_human",), ("escalate_to_human",)),
        "freeze_card": (("freeze_card",), ()),
        "informational_policy": ((), ()),
        "policy_versus_action": ((), ()),
        "recent_transactions": (
            ("get_recent_transactions",),
            ("get_recent_transactions",),
        ),
        "transaction_correction": (
            ("get_transaction_details",),
            ("get_transaction_details",),
        ),
        "transaction_details": (
            ("get_transaction_details",),
            ("get_transaction_details",),
        ),
        "unsupported_or_uncertain": ((), ()),
    }

    observed = {
        scenario.scenario_id: (
            scenario.expected.expected_tool_proposals,
            scenario.expected.expected_executed_tools,
        )
        for scenario in benchmark.scenarios
    }
    assert observed == expected


@pytest.mark.parametrize(
    "missing_field",
    ["expected_tool_proposals", "expected_executed_tools"],
)
def test_routing_scenario_requires_both_tool_expectations(
    missing_field: str,
) -> None:
    payload = json.loads(ROUTING_BENCHMARK_PATH.read_text(encoding="utf-8"))
    del payload["scenarios"][0]["expected"][missing_field]

    with pytest.raises(ValidationError):
        RoutingBenchmark.model_validate(payload)


def test_routing_scenario_requires_expected_outcome() -> None:
    payload = json.loads(ROUTING_BENCHMARK_PATH.read_text(encoding="utf-8"))
    del payload["scenarios"][0]["expected"]["outcome"]

    with pytest.raises(ValidationError):
        RoutingBenchmark.model_validate(payload)


def test_duplicate_audio_fixture_ids_are_rejected() -> None:
    fixture = _fixture_payload()
    with pytest.raises(ValidationError, match="fixture_id values must be unique"):
        AudioReplayManifest.model_validate(
            {
                "schema_version": "audio-replay.v1",
                "manifest_version": "test",
                "fixtures": [fixture, fixture],
            }
        )


def test_audio_fixture_requires_expected_outcome() -> None:
    fixture = _fixture_payload()
    del fixture["expected_outcome"]

    with pytest.raises(ValidationError):
        AudioReplayManifest.model_validate(
            {
                "schema_version": "audio-replay.v1",
                "manifest_version": "test",
                "fixtures": [fixture],
            }
        )


def test_invalid_measurement_source_is_rejected() -> None:
    with pytest.raises(ValidationError):
        BenchmarkResult.model_validate(
            {
                "result_id": "result_001",
                "measurement_source": "offline_milliseconds",
                "evaluation_version": "test",
                "sample_count": 1,
                "failure_count": 0,
            }
        )


def test_live_provider_measurement_requires_provider_and_model() -> None:
    with pytest.raises(ValidationError, match="require provider and model"):
        BenchmarkResult(
            result_id="result_001",
            measurement_source=MeasurementSource.LIVE_LLM,
            evaluation_version="test",
            sample_count=1,
            failure_count=0,
        )


@pytest.mark.parametrize(
    ("model", "payload"),
    [
        (
            IntentRiskDataset,
            {
                "schema_version": "intent-risk.v2",
                "dataset_version": "test",
                "advisory_only": True,
                "examples": [_example_payload()],
            },
        ),
        (
            BenchmarkReport,
            {
                "schema_version": "benchmark-report.v2",
                "report_version": "test",
                "results": [
                    {
                        "result_id": "result_001",
                        "measurement_source": "deterministic_fake",
                        "evaluation_version": "test",
                        "sample_count": 1,
                        "failure_count": 0,
                    }
                ],
            },
        ),
    ],
)
def test_unsupported_schema_versions_are_rejected(model, payload) -> None:
    with pytest.raises(ValidationError):
        model.model_validate(payload)


def test_frozen_baseline_metadata_cannot_drift() -> None:
    payload = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    payload["agent_passed"] = 34

    with pytest.raises(ValidationError, match="does not match v1.0.0"):
        FrozenV1Baseline.model_validate(payload)


def test_artifact_loading_preserves_declared_deterministic_order() -> None:
    first = load_intent_risk_dataset(INTENT_DATASET_PATH)
    second = load_intent_risk_dataset(INTENT_DATASET_PATH)
    first_ids = [example.example_id for example in first.examples]

    assert first == second
    assert first_ids == sorted(first_ids)
