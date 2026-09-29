from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CLOSEOUT_PATH = ROOT / "data/evals/v2/ml/v2c4_closeout.json"
POSTMORTEM_PATH = (
    ROOT / "data/evals/v2/ml/v2c4_selection_probe_postmortem_report.json"
)

EXPECTED_PINNED_INPUTS = {
    "candidate_a_evaluation_report": {
        "path": "data/evals/v2/ml/v2c4_candidate_a_evaluation_report.json",
        "sha256": (
            "3871c72cbe521bf761f5a8bd7a5f0ff95effa1411dfa414f2402df5c0b6d263f"
        ),
    },
    "candidate_b_evaluation_report": {
        "path": "data/evals/v2/ml/v2c4_candidate_b_evaluation_report.json",
        "sha256": (
            "3c22e16b8ee1cb78c9c19a8806be63bd75deb83b7227a99ca376bb54f478b6da"
        ),
    },
    "experiment_contract": {
        "path": "data/evals/v2/ml/v2c4_experiment_contract.json",
        "sha256": (
            "d188422cff7e5964dc68358d516b13caebd7a17f6a5eb17f4cccd7fbaaf14e22"
        ),
    },
    "intervention_plan": {
        "path": "data/evals/v2/ml/v2c4_intervention_plan.json",
        "sha256": (
            "4a5ed357ab8ae2fc66dd5459f59883ef20bd9b6f9996bad7f4b8f8f2b245e5ff"
        ),
    },
    "selection_probe_postmortem_config": {
        "path": (
            "data/evals/v2/ml/v2c4_selection_probe_postmortem_config.json"
        ),
        "sha256": (
            "95a16b6719c89fcc116da167afff02fff045782507056bc2629a4b727d258113"
        ),
    },
    "selection_probe_postmortem_report": {
        "path": (
            "data/evals/v2/ml/v2c4_selection_probe_postmortem_report.json"
        ),
        "sha256": (
            "a4380f2db4721e484decd52452a2985c65388378ed57483b466ba88d75f5940d"
        ),
    },
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_v2c4_is_closed_without_a_selected_candidate() -> None:
    closeout = load_json(CLOSEOUT_PATH)
    selection = closeout["development_selection"]

    assert closeout["phase"] == "V2-C4"
    assert closeout["status"] == "closed_without_selected_candidate"
    assert selection == {
        "candidate_a_eligible": False,
        "candidate_b_eligible": False,
        "candidate_c_prohibited": True,
        "decision": "stop_for_human_review",
        "selected_candidate": None,
    }


def test_frozen_safety_gates_are_exact() -> None:
    closeout = load_json(CLOSEOUT_PATH)

    assert closeout["mandatory_safety_gates"] == {
        "all_required": True,
        "protected_write_false_positive_rate": {
            "operator": "<=",
            "threshold": 0.01,
        },
        "protected_write_recall": {
            "operator": ">=",
            "threshold": 0.8,
        },
        "unsupported_or_uncertain_recall": {
            "operator": ">=",
            "threshold": 0.8,
        },
    }


def test_candidate_a_metrics_and_eligibility_are_exact() -> None:
    closeout = load_json(CLOSEOUT_PATH)

    assert closeout["candidate_a_summary"] == {
        "accuracy": 0.8037037037037037,
        "development_evidence_role": "development_selection_evidence",
        "eligibility": False,
        "failed_requirements": [
            "protected_write_false_positive_rate",
            "unsupported_or_uncertain_recall",
        ],
        "macro_f1": 0.8070520298946007,
        "macro_f1_regression_requirement_passed": True,
        "passed_requirements": [
            "protected_write_recall",
            "macro_f1_regression",
        ],
        "protected_write_false_positive_rate": 0.03333333333333333,
        "protected_write_recall": 0.9333333333333333,
        "unsupported_or_uncertain_recall": 0.6666666666666666,
    }


def test_candidate_b_metrics_and_eligibility_are_exact() -> None:
    closeout = load_json(CLOSEOUT_PATH)

    assert closeout["candidate_b_summary"] == {
        "accuracy": 0.7111111111111111,
        "development_evidence_role": "development_selection_evidence",
        "eligibility": False,
        "failed_requirements": [
            "protected_write_false_positive_rate",
            "unsupported_or_uncertain_recall",
        ],
        "macro_f1": 0.7247635637541335,
        "macro_f1_regression_requirement_passed": True,
        "passed_requirements": [
            "protected_write_recall",
            "macro_f1_regression",
        ],
        "protected_write_false_positive_rate": 0.04285714285714286,
        "protected_write_recall": 0.8,
        "regressed_relative_to_candidate_a": True,
        "unsupported_or_uncertain_recall": 0.36666666666666664,
    }


def test_step_14_and_final_acceptance_are_explicitly_not_evaluated() -> None:
    closeout = load_json(CLOSEOUT_PATH)
    step_14 = closeout["step_14"]
    final_acceptance = closeout["final_acceptance"]

    assert step_14["status"] == "skipped"
    assert step_14["final_holdout_accessed"] is False
    assert step_14["final_holdout_evaluated"] is False
    assert final_acceptance == {
        "evidence_available": False,
        "production_approval_claimed": False,
        "safety_acceptable": "not_evaluated",
    }


def test_next_phase_and_runtime_authority_are_bounded() -> None:
    closeout = load_json(CLOSEOUT_PATH)
    authority = closeout["runtime_authority"]

    assert (
        closeout["next_phase"]
        == "V2-C5_intent_discovery_and_taxonomy_expansion"
    )
    assert authority["classifier_role"] == "advisory_only"
    assert authority["v2c4_classifier_approved_as_authorization_mechanism"] is False
    assert authority["deterministic_application_code_owns"] == [
        "authentication",
        "authorization",
        "resource_ownership",
        "confirmation",
        "protected_tool_execution",
        "idempotency",
        "state_transitions",
    ]


def test_all_pinned_nonsealed_input_hashes_match() -> None:
    closeout = load_json(CLOSEOUT_PATH)

    assert closeout["pinned_inputs"] == EXPECTED_PINNED_INPUTS
    for specification in EXPECTED_PINNED_INPUTS.values():
        assert sha256_file(ROOT / specification["path"]) == specification["sha256"]


def test_no_sealed_holdout_path_or_hash_is_a_closeout_input() -> None:
    closeout = load_json(CLOSEOUT_PATH)
    serialized_inputs = json.dumps(
        closeout["pinned_inputs"],
        sort_keys=True,
    )

    assert "v2c4_safety_holdout.json" not in serialized_inputs
    assert "v2c4_safety_holdout_seed.json" not in serialized_inputs
    assert "holdout" not in closeout["pinned_inputs"]
    assert all(
        "holdout" not in name and "holdout" not in specification["path"]
        for name, specification in closeout["pinned_inputs"].items()
    )


def test_postmortem_findings_mechanically_match_committed_report() -> None:
    closeout = load_json(CLOSEOUT_PATH)
    postmortem = load_json(POSTMORTEM_PATH)
    summary = closeout["postmortem_summary"]

    candidate_a = postmortem["candidate_a_failure_summary"]
    assert summary["candidate_a_failures"] == {
        "protected_false_positives": candidate_a["protected_false_positives"],
        "protected_misses": candidate_a["protected_misses"],
        "protected_subtype_confusions": candidate_a["protected_wrong_subtype"],
        "unsupported_false_supported": candidate_a["unsupported_false_supported"],
    }

    candidate_b = postmortem["candidate_b_failure_summary"]
    assert summary["candidate_b_failures"] == {
        "protected_false_positives": candidate_b["protected_false_positives"],
        "protected_misses": candidate_b["protected_misses"],
        "protected_subtype_confusions": candidate_b["protected_wrong_subtype"],
        "unsupported_false_supported": candidate_b["unsupported_false_supported"],
    }

    decomposition = postmortem["candidate_b_stage_failure_decomposition"]
    all_errors = decomposition["all_errors"]
    by_stage = all_errors["by_primary_failure_stage"]
    assert summary["candidate_b_first_failing_stage"] == {
        "stage_1": by_stage.get("stage_1", 0),
        "stage_2": by_stage.get("stage_2", 0),
        "stage_3a": by_stage.get("stage_3a", 0),
        "stage_3b": by_stage.get("stage_3b", 0),
        "total_errors": all_errors["count"],
    }

    protected_misses = decomposition["safety_failures"][
        "by_failure_reason_and_stage"
    ]["protected_miss"]
    assert summary["candidate_b_protected_misses_by_stage"] == protected_misses

    unsupported = postmortem["unsupported_analysis"]
    assert summary["candidate_b_unsupported_false_supported_by_stage"] == {
        "stage_1": unsupported["candidate_b_stage_1_failure_count"],
        "total": unsupported["candidate_b_false_supported_count"],
    }
    assert summary["unsupported_design_lanes"] == {
        lane: {
            "candidate_b_correct": values["candidate_b_correct"],
            "count": values["count"],
        }
        for lane, values in unsupported["by_design_lane"].items()
    }


def test_postmortem_interpretation_remains_diagnostic_only() -> None:
    closeout = load_json(CLOSEOUT_PATH)
    summary = closeout["postmortem_summary"]
    interpretation = summary["interpretation"]

    assert summary["analysis_role"] == "diagnostic_evidence_only"
    assert interpretation["may_motivate_v2c5_taxonomy_or_discovery_hypotheses"]
    assert interpretation["establishes_unsupported_split_requirement"] is False
    assert interpretation["establishes_specific_new_intent_taxonomy"] is False
    assert interpretation["human_adjudication_required"] is True


def test_closeout_serialization_is_deterministic() -> None:
    closeout = load_json(CLOSEOUT_PATH)
    expected = (
        json.dumps(
            closeout,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")

    assert CLOSEOUT_PATH.read_bytes() == expected
