from __future__ import annotations

import ast
import io
import json
from pathlib import Path
from typing import Any

import pytest

from scripts import analyze_v2c4_selection_probe_postmortem as analysis

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    ROOT / "data/evals/v2/ml/v2c4_selection_probe_postmortem_config.json"
)
SCRIPT_PATH = ROOT / "scripts/analyze_v2c4_selection_probe_postmortem.py"


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def example(ordinal: int, intent: str, risk: str) -> dict[str, Any]:
    return {
        "example_id": f"fixture:{ordinal}",
        "intent": intent,
        "risk": risk,
        "design_lane": "fixture_lane",
        "family_id": f"family:{ordinal}",
        "group_id": f"group:{ordinal}",
        "lineage_id": f"lineage:{ordinal}",
        "tags": ["fixture"],
        "text_sha256": f"{ordinal:064x}",
        "normalized_text_sha256": f"{ordinal + 100:064x}",
    }


def test_config_marks_probe_consumed_and_diagnostic_only() -> None:
    config = load_config()
    analysis.validate_config(config)
    assert config["analysis_role"] == "consumed_v2c4_model_selection_diagnostic"
    assert config["source_data_role"] == "v2c4_model_selection_probe"
    assert config["probe_consumed"] is True
    assert config["eligibility"] == {
        "model_selection_eligible": False,
        "training_eligible": False,
        "threshold_selection_eligible": False,
        "final_acceptance_evidence": False,
    }
    assert config["candidate_c_allowed"] is False
    assert config["final_holdout_accessed"] is False


def test_exact_frozen_artifact_and_report_hashes() -> None:
    config = load_config()
    frozen = config["frozen_inputs"]
    assert {name: spec["sha256"] for name, spec in frozen.items()} == (
        analysis.EXPECTED_HASHES
    )
    for name, spec in frozen.items():
        assert spec["path"] == analysis.EXPECTED_PATHS[name]
        assert analysis.sha256_file(analysis.repository_path(spec["path"])) == (
            spec["sha256"]
        )


def test_final_holdout_paths_are_rejected() -> None:
    for filename in analysis.FORBIDDEN_INPUT_FILENAMES:
        with pytest.raises(ValueError, match="final holdout input is forbidden"):
            analysis.guard_path(ROOT / "data/evals/v2/ml" / filename)


def test_reproduction_requires_exact_committed_metrics() -> None:
    gold = ["account_balance", "unsupported_or_uncertain"]
    predictions = {
        "baseline": ["account_balance", "unsupported_or_uncertain"],
        "candidate_a": ["account_balance", "account_balance"],
        "candidate_b": ["unsupported_or_uncertain", "unsupported_or_uncertain"],
    }
    metrics = {
        name: analysis.train_a.calculate_metrics(
            gold, labels, analysis.train_a.EXPECTED_INTENTS
        )
        for name, labels in predictions.items()
    }
    candidate_a_report = {
        "baseline_metrics": metrics["baseline"],
        "candidate_a_metrics": metrics["candidate_a"],
    }
    candidate_b_report = {
        "baseline_metrics": metrics["baseline"],
        "candidate_a_metrics": metrics["candidate_a"],
        "candidate_b_metrics": metrics["candidate_b"],
        "selection_decision": {
            "decision": "stop_for_human_review",
            "selected_candidate": None,
            "candidate_c_created_or_allowed": False,
        },
        "final_acceptance_evidence": False,
        "final_holdout_accessed": False,
    }
    reproduced, _ = analysis.verify_reproduction(
        gold,
        predictions,
        candidate_a_report,
        candidate_b_report,
    )
    assert reproduced["status"] == "passed"
    assert reproduced["committed_per_example_prediction_sequence_available"] is False

    candidate_b_report["candidate_b_metrics"] = metrics["baseline"]
    with pytest.raises(ValueError, match="candidate_b metrics differ"):
        analysis.verify_reproduction(
            gold,
            predictions,
            candidate_a_report,
            candidate_b_report,
        )


def test_candidate_a_failure_bucket_logic() -> None:
    rows = [
        example(1, "freeze_card", "PROTECTED_WRITE"),
        example(2, "create_dispute", "PROTECTED_WRITE"),
        example(3, "account_balance", "PRIVATE_READ"),
        example(4, "unsupported_or_uncertain", "ESCALATION_OR_UNCERTAIN"),
    ]
    summary = analysis.summarize_failures(
        rows,
        [
            "account_balance",
            "freeze_card",
            "create_dispute",
            "account_balance",
        ],
    )
    assert summary == {
        "protected_total": 2,
        "protected_recognized": 1,
        "protected_misses": 1,
        "protected_wrong_subtype": 1,
        "nonprotected_total": 2,
        "protected_false_positives": 1,
        "unsupported_total": 1,
        "unsupported_correct": 0,
        "unsupported_false_supported": 1,
    }


def test_candidate_b_failure_bucket_logic_matches_same_definitions() -> None:
    flags = analysis.failure_flags("unsupported_or_uncertain", "freeze_card")
    assert flags == {
        "protected_miss": False,
        "protected_false_positive": True,
        "unsupported_false_supported": True,
        "protected_wrong_subtype": False,
    }
    config = load_config()
    assert config["expected_frozen_failures"]["candidate_b"] == {
        "protected_total": 60,
        "protected_recognized": 48,
        "protected_misses": 12,
        "nonprotected_total": 210,
        "protected_false_positives": 9,
        "unsupported_total": 30,
        "unsupported_correct": 11,
        "unsupported_false_supported": 19,
    }


@pytest.mark.parametrize(
    ("gold", "trace", "expected"),
    [
        (
            "unsupported_or_uncertain",
            {
                "final_prediction": "account_balance",
                "stage_1_prediction": "supported_current_request",
                "stage_2_prediction": "nonprotected",
            },
            "stage_1",
        ),
        (
            "freeze_card",
            {
                "final_prediction": "unsupported_or_uncertain",
                "stage_1_prediction": "unsupported_or_uncertain",
                "stage_2_prediction": None,
            },
            "stage_1",
        ),
        (
            "freeze_card",
            {
                "final_prediction": "account_balance",
                "stage_1_prediction": "supported_current_request",
                "stage_2_prediction": "nonprotected",
            },
            "stage_2",
        ),
        (
            "account_balance",
            {
                "final_prediction": "freeze_card",
                "stage_1_prediction": "supported_current_request",
                "stage_2_prediction": "protected_write",
            },
            "stage_2",
        ),
        (
            "freeze_card",
            {
                "final_prediction": "create_dispute",
                "stage_1_prediction": "supported_current_request",
                "stage_2_prediction": "protected_write",
            },
            "stage_3a",
        ),
        (
            "account_balance",
            {
                "final_prediction": "card_status",
                "stage_1_prediction": "supported_current_request",
                "stage_2_prediction": "nonprotected",
            },
            "stage_3b",
        ),
    ],
)
def test_hierarchy_first_failure_stage(
    gold: str, trace: dict[str, Any], expected: str
) -> None:
    assert analysis.primary_failure_stage(gold, trace) == expected


@pytest.mark.parametrize(
    ("a_correct", "b_correct", "same", "expected"),
    [
        (True, True, True, "both_correct"),
        (False, False, True, "both_wrong_same_prediction"),
        (False, False, False, "both_wrong_different_prediction"),
        (True, False, False, "candidate_a_correct_candidate_b_wrong"),
        (False, True, False, "candidate_a_wrong_candidate_b_correct"),
    ],
)
def test_a_vs_b_transition_categories(
    a_correct: bool, b_correct: bool, same: bool, expected: str
) -> None:
    assert analysis.transition_category(a_correct, b_correct, same) == expected


def test_raw_text_is_rejected_from_persisted_report() -> None:
    valid = {"raw_text_persisted": False, "per_example_comparison": []}
    analysis.validate_no_raw_text(valid)
    with pytest.raises(ValueError, match="raw text field"):
        analysis.validate_no_raw_text(
            {"raw_text_persisted": False, "records": [{"text": "secret"}]}
        )


def test_review_queue_stdout_may_join_source_text() -> None:
    report = {
        "schema_version": analysis.SCHEMA_VERSION,
        "analysis_role": analysis.ANALYSIS_ROLE,
        "selected_v2c4_candidate": None,
        "candidate_c_allowed": False,
        "final_holdout_accessed": False,
        "final_acceptance_evidence": False,
        "raw_text_persisted": False,
        "review_queue": [
            {
                "example_id": "fixture:1",
                "gold_intent": "freeze_card",
                "risk": "PROTECTED_WRITE",
                "design_lane": "fixture_lane",
                "tags": ["fixture"],
                "candidate_a_prediction": "account_balance",
                "candidate_b_prediction": "unsupported_or_uncertain",
                "candidate_b_primary_failure_stage": "stage_1",
                "review_reasons": ["candidate_a:protected_miss"],
                "text_sha256": "1" * 64,
                "normalized_text_sha256": "2" * 64,
            }
        ],
    }
    source = [
        {
            "example_id": "fixture:1",
            "intent": "freeze_card",
            "text": "Please freeze the card.",
            "text_sha256": "1" * 64,
        }
    ]
    stream = io.StringIO()
    joined = analysis.print_review_queue(
        report=report,
        examples=source,
        stream=stream,
    )
    assert joined[0]["text"] == "Please freeze the card."
    assert "Please freeze the card." in stream.getvalue()
    assert "text" not in report["review_queue"][0]


def test_decision_scores_are_explicitly_nonprobabilistic() -> None:
    config = load_config()
    assert config["decision_scores_are_calibrated_probabilities"] is False
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "not calibrated probabilities" in source


def test_report_serialization_is_deterministic() -> None:
    first = {"b": 2, "a": {"d": 4, "c": 3}}
    second = {"a": {"c": 3, "d": 4}, "b": 2}
    assert analysis.stable_json_bytes(first) == analysis.stable_json_bytes(second)


def test_analyzer_contains_no_fitting_or_tuning_api_calls() -> None:
    tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
    forbidden_calls = {
        "fit",
        "fit_predict",
        "fit_transform",
        "partial_fit",
        "calibrate",
        "GridSearchCV",
        "RandomizedSearchCV",
    }
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute):
            found.add(node.func.attr)
        elif isinstance(node.func, ast.Name):
            found.add(node.func.id)
    assert not (found & forbidden_calls)


def test_candidate_c_and_final_acceptance_remain_prohibited() -> None:
    config = load_config()
    policy = config["findings_use_policy"]
    assert policy["candidate_c_prohibited"] is True
    assert policy["may_tune_or_create_another_v2c4_candidate"] is False
    assert policy["may_inform_v2c5_taxonomy_or_discovery_hypotheses"] is True
    assert policy["start_v2c5_implementation"] is False
    assert config["eligibility"]["final_acceptance_evidence"] is False


def test_v2c5_hypothesis_inputs_do_not_make_taxonomy_decisions() -> None:
    rows = [
        {
            "intent": "unsupported_or_uncertain",
            "design_lane": "ambiguous_or_fragment",
            "candidate_a": {"correct": False},
            "candidate_b": {
                "correct": False,
                "hierarchy": {"primary_failure_stage": "stage_1"},
            },
        },
        {
            "intent": "freeze_card",
            "design_lane": "boundary_case",
            "candidate_a": {"correct": True},
            "candidate_b": {
                "correct": False,
                "hierarchy": {"primary_failure_stage": "stage_2"},
            },
        },
    ]
    result = analysis.v2c5_hypothesis_inputs(rows)
    assert result["taxonomy_or_intent_decisions_made"] is False
    assert result["v2c5_implementation_started"] is False
    assert result["candidate_c_justified_or_allowed"] is False
    assert result["human_adjudication_required_for_future_taxonomy_changes"] is True
