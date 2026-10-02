from __future__ import annotations

import copy
import inspect
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from scripts import run_v2c6_targeted_remediation_failure_analysis as runner

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def contract() -> dict[str, Any]:
    return json.loads(runner.DEFAULT_PATHS.contract.read_text(encoding="utf-8"))


def prediction(
    record_id: str,
    gold: str,
    predicted: str,
    *,
    family: str | None = None,
    development: bool = True,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "gold_intent": gold,
        "predicted_intent": predicted,
    }
    if development:
        row["example_id"] = record_id
        row["fold_id"] = "group_cv_fold_0"
    else:
        row["record_id"] = record_id
        row["source_family_id"] = family
    return row


def safety_rows() -> list[dict[str, Any]]:
    return [
        prediction("p1", "cancel_transfer", "freeze_card"),
        prediction("p2", "close_account", "account_balance"),
        prediction("n1", "account_balance", "freeze_card"),
        prediction("n2", "card_status", "card_status"),
        prediction(
            "u1",
            "unsupported_or_uncertain",
            "create_dispute",
        ),
        prediction(
            "u2",
            "unsupported_or_uncertain",
            "account_balance",
        ),
        prediction(
            "u3",
            "unsupported_or_uncertain",
            "unsupported_or_uncertain",
        ),
    ]


def test_exact_contract_lineage_is_required(contract: dict[str, Any]) -> None:
    assert runner.sha256_file(runner.DEFAULT_PATHS.contract) == (
        runner.EXPECTED_CONTRACT_SHA256
    )
    runner.validate_contract(contract)

    changed = copy.deepcopy(contract)
    changed["contract_status"]["next_required"] = "invented_step"
    with pytest.raises(ValueError, match="does not authorize"):
        runner.validate_contract(changed)


def test_exact_completed_result_hash_and_identity_are_required(
    contract: dict[str, Any],
) -> None:
    specification = contract["source_artifacts"][
        "targeted_remediation_model_selection_results"
    ]
    assert runner.sha256_file(runner.DEFAULT_PATHS.results) == (
        specification["sha256"]
    )
    results = json.loads(runner.DEFAULT_PATHS.results.read_text(encoding="utf-8"))
    assert results["schema_version"] == specification["schema_version"]
    assert results["execution_status"] == "COMPLETED"


def test_contract_hash_mismatch_fails_before_other_inputs(tmp_path: Path) -> None:
    changed_contract = tmp_path / "contract.json"
    changed_contract.write_text("{}\n", encoding="utf-8")
    paths = replace(runner.DEFAULT_PATHS, contract=changed_contract)

    with pytest.raises(ValueError, match="contract SHA-256 mismatch"):
        runner.load_preconditions(paths, require_outputs_absent=False)


def test_source_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}\n", encoding="utf-8")
    paths = replace(runner.DEFAULT_PATHS, repository_root=tmp_path)

    with pytest.raises(ValueError, match="artifact SHA-256 mismatch"):
        runner.validate_binding(
            artifact,
            {"path": "artifact.json", "sha256": "0" * 64},
            paths,
            "artifact",
        )


def test_no_acceptable_candidate_prerequisite_is_fail_closed(
    contract: dict[str, Any],
) -> None:
    results = json.loads(runner.DEFAULT_PATHS.results.read_text(encoding="utf-8"))
    manifest = json.loads(
        runner.DEFAULT_PATHS.results_manifest.read_text(encoding="utf-8")
    )
    runner.validate_input_state(contract, results, manifest)

    changed = copy.deepcopy(results)
    changed["selection"]["selection_status"] = "SELECTED"
    with pytest.raises(ValueError, match="NO_ACCEPTABLE_CANDIDATE"):
        runner.validate_input_state(contract, changed, manifest)


def test_exact_four_candidates_and_fresh_families(
    contract: dict[str, Any],
) -> None:
    assert contract["candidate_coverage"]["candidate_count"] == 4
    assert contract["candidate_coverage"]["candidate_ids"] == [
        "BGE_SMALL_LINEAR_SVC__C=4.0__class_weight=none",
        "WORD_CHAR_TFIDF_LINEAR_SVC__C=1.0__class_weight=none",
        "HYBRID_BGE_TFIDF_LINEAR_SVC__C=1.0__class_weight=none",
        "HIERARCHICAL_TFIDF_LINEAR_SVC__C=1.0__class_weight=none",
    ]
    assert contract["consumed_evidence_governance"][
        "fresh_evaluation_source_family_ids"
    ] == [
        "v2c6_r2_eval_sf1_independent_casework",
        "v2c6_r2_eval_sf2_independent_naturalistic",
    ]


def test_protected_false_positive_definition_is_exact() -> None:
    result = runner.protected_false_positive_analysis(
        safety_rows(),
        {"cancel_transfer", "close_account", "create_dispute", "freeze_card"},
    )

    assert result["non_protected_gold_count"] == 5
    assert result["protected_false_positive_count"] == 2
    assert result["protected_false_positive_rate"] == 0.4
    assert result["counts_by_true_intent"] == {
        "account_balance": 1,
        "unsupported_or_uncertain": 1,
    }
    assert result["counts_by_predicted_protected_intent"] == {
        "create_dispute": 1,
        "freeze_card": 1,
    }


def test_unsupported_miss_definition_is_exact() -> None:
    result = runner.unsupported_miss_analysis(
        safety_rows(),
        {"cancel_transfer", "close_account", "create_dispute", "freeze_card"},
    )

    assert result["unsupported_gold_count"] == 3
    assert result["unsupported_correct_count"] == 1
    assert result["unsupported_miss_count"] == 2
    assert result["unsupported_recall"] == pytest.approx(1 / 3)
    assert result["counts_by_predicted_intent"] == {
        "account_balance": 1,
        "create_dispute": 1,
    }
    assert result["protected_destination_count"] == 1
    assert result["non_protected_destination_count"] == 1


def test_protected_recall_miss_definition_is_aggregate_protected() -> None:
    result = runner.protected_recall_miss_analysis(
        safety_rows(),
        {"cancel_transfer", "close_account", "create_dispute", "freeze_card"},
    )

    assert result["protected_gold_count"] == 2
    assert result["protected_prediction_as_protected_count"] == 1
    assert result["protected_recall_miss_count"] == 1
    assert result["protected_recall"] == 0.5
    assert result["true_protected_intent_by_predicted_intent_matrix"][
        "close_account"
    ] == {"account_balance": 1}


def test_required_metric_denominators_fail_closed() -> None:
    with pytest.raises(ValueError, match="denominator must be positive"):
        runner.unsupported_miss_analysis(
            [prediction("n1", "account_balance", "account_balance")], set()
        )


def test_directional_hard_negative_accounting_is_exact() -> None:
    records = {
        "a1": {
            "intent": "cancel_transfer",
            "boundary_target": "transfer_pending",
            "is_hard_negative": True,
        },
        "a2": {
            "intent": "cancel_transfer",
            "boundary_target": "transfer_pending",
            "is_hard_negative": True,
        },
        "b1": {
            "intent": "transfer_pending",
            "boundary_target": "cancel_transfer",
            "is_hard_negative": True,
        },
    }
    rows = [
        prediction("a1", "cancel_transfer", "transfer_pending"),
        prediction("a2", "cancel_transfer", "account_balance"),
        prediction("b1", "transfer_pending", "cancel_transfer"),
    ]

    result = runner.hard_negative_directional_analysis(
        rows,
        records,
        [["cancel_transfer", "transfer_pending"]],
        id_field="example_id",
    )["boundaries"][0]

    assert result["true_a_total"] == 2
    assert result["true_b_total"] == 1
    assert result["true_a_predicted_b_count"] == 1
    assert result["true_b_predicted_a_count"] == 1
    assert result["true_a_predicted_b_rate"] == 0.5
    assert result["true_b_predicted_a_rate"] == 1.0
    assert result["other_error_count"] == 1


def failure_key(record_id: str) -> runner.FailureKey:
    return (
        runner.GROUP_SCOPE,
        None,
        record_id,
        "protected_false_positive",
    )


def test_pairwise_overlap_and_jaccard_math() -> None:
    candidates = ["bge", "tfidf", "hybrid", "hierarchical"]
    sets = {
        "bge": {failure_key("shared"), failure_key("first_three"), failure_key("b")},
        "tfidf": {
            failure_key("shared"),
            failure_key("first_three"),
            failure_key("t"),
        },
        "hybrid": {
            failure_key("shared"),
            failure_key("first_three"),
            failure_key("h"),
        },
        "hierarchical": {failure_key("shared"), failure_key("x")},
    }

    result = runner.overlap_analysis(sets, candidates)

    assert result["intersection_shared_by_all_four_candidates"]["count"] == 1
    assert result["intersection_shared_by_bge_tfidf_and_hybrid"]["count"] == 2
    first_pair = result["pairwise"][0]
    assert first_pair["intersection_count"] == 2
    assert first_pair["union_count"] == 4
    assert first_pair["jaccard"] == 0.5


def test_zero_union_jaccard_is_null() -> None:
    candidates = ["a", "b", "c", "d"]
    result = runner.overlap_analysis(
        {candidate: set() for candidate in candidates}, candidates
    )
    assert all(row["jaccard"] is None for row in result["pairwise"])


def test_fresh_family_and_grouped_cv_scopes_are_isolated() -> None:
    families = ["family_a", "family_b"]
    candidate = {
        "group_aware_cv": {
            "predictions": [prediction("development", "account_balance", "card_status")]
        },
        "fresh_source_evaluation": {
            "pooled_predictions": [
                prediction(
                    "fresh_a",
                    "account_balance",
                    "card_status",
                    family="family_a",
                    development=False,
                ),
                prediction(
                    "fresh_b",
                    "card_status",
                    "card_status",
                    family="family_b",
                    development=False,
                ),
            ]
        },
    }

    scopes = runner.prediction_scopes(candidate, families)

    assert [row["example_id"] for row in scopes[runner.GROUP_SCOPE]] == [
        "development"
    ]
    assert [row["record_id"] for row in scopes["family_a"]] == ["fresh_a"]
    assert [row["record_id"] for row in scopes["family_b"]] == ["fresh_b"]
    assert len(scopes[runner.POOLED_FRESH_SCOPE]) == 2


def test_hierarchical_stage_unavailability_is_recorded(
    contract: dict[str, Any],
) -> None:
    candidate = {
        "group_aware_cv": {"predictions": []},
        "fresh_source_evaluation": {"pooled_predictions": []},
    }
    result = runner.hierarchical_analysis(contract, candidate)

    assert result["stage_level_predictions_persisted"] is False
    assert result["exact_stage_1_error_attribution_available"] is False
    assert result["limitation"] == (
        "Stage-level predictions were not persisted, so exact Stage-1 error "
        "attribution is unavailable."
    )
    assert result["model_reconstructed"] is False
    assert result["model_rerun"] is False


def test_unexpected_hierarchical_stage_data_fails_closed(
    contract: dict[str, Any],
) -> None:
    candidate = {
        "group_aware_cv": {"stage_1_predictions": []},
        "fresh_source_evaluation": {},
    }
    with pytest.raises(ValueError, match="availability changed"):
        runner.hierarchical_analysis(contract, candidate)


def test_contract_permitted_hierarchical_stage_data_is_analyzed(
    contract: dict[str, Any],
) -> None:
    available_contract = copy.deepcopy(contract)
    definition = available_contract["analysis_dimensions"][
        "hierarchical_architecture_diagnosis"
    ]
    definition["stage_level_predictions_persisted"] = True
    definition["exact_stage_1_error_attribution_available"] = True
    candidate = {
        "group_aware_cv": {
            "stage_1_predictions": [
                {"gold_label": "supported", "predicted_label": "supported"},
                {
                    "gold_label": "unsupported",
                    "predicted_label": "supported",
                },
            ]
        },
        "fresh_source_evaluation": {
            "stage_1_predictions": [
                {
                    "gold_label": "unsupported",
                    "predicted_label": "unsupported",
                }
            ]
        },
    }

    result = runner.hierarchical_analysis(available_contract, candidate)

    assert result["stage_level_predictions_persisted"] is True
    assert result["exact_stage_1_error_attribution_available"] is True
    assert result["stage_1_analysis"][runner.GROUP_SCOPE][
        "prediction_count"
    ] == 2
    assert result["stage_1_analysis"][runner.GROUP_SCOPE]["correct_count"] == 1
    assert result["model_reconstructed"] is False
    assert result["model_rerun"] is False


def _classification_scope(
    *,
    fp_count: int,
    fp_rate: float,
    unsupported_misses: int,
    unsupported_recall: float,
) -> dict[str, Any]:
    return {
        "protected_false_positives": {
            "protected_false_positive_count": fp_count,
            "protected_false_positive_rate": fp_rate,
        },
        "unsupported_misses": {
            "unsupported_miss_count": unsupported_misses,
            "unsupported_recall": unsupported_recall,
        },
    }


def test_evidence_classification_is_noncausal_and_not_selection() -> None:
    analyses = {
        "candidate": {
            runner.GROUP_SCOPE: _classification_scope(
                fp_count=1,
                fp_rate=0.01,
                unsupported_misses=1,
                unsupported_recall=0.9,
            ),
            runner.POOLED_FRESH_SCOPE: _classification_scope(
                fp_count=2,
                fp_rate=0.02,
                unsupported_misses=2,
                unsupported_recall=0.7,
            ),
        }
    }
    overlaps = {
        runner.GROUP_SCOPE: {
            "intersection_shared_by_all_four_candidates": {"count": 1},
            "failures_unique_to_each_candidate": {
                "candidate": {"count": 1}
            },
        }
    }

    result = runner.classify_evidence(analyses, overlaps)

    assert {row["category"] for row in result} == {
        "development_distribution_boundary_weakness",
        "fresh_source_generalization_weakness",
        "architecture_specific_weakness",
        "cross_architecture_shared_weakness",
    }
    assert all(row["causal_claim"] is False for row in result)
    assert all("selected" not in row for row in result)


def test_final_holdout_access_is_rejected_before_read() -> None:
    with pytest.raises(PermissionError, match="final holdout access"):
        runner.read_bytes(
            runner.DEFAULT_PATHS.prohibited_holdout,
            runner.DEFAULT_PATHS,
        )


def test_runner_has_no_ml_fitting_or_inference_imports() -> None:
    source = inspect.getsource(runner)
    assert "from sklearn" not in source
    assert "import sklearn" not in source
    assert "FastEmbed" not in source
    assert "fit_predict" not in source
    assert ".predict(" not in source
    assert ".fit(" not in source


def test_create_once_output_behavior(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    runner.durable_create(path, b"first\n")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        runner.durable_create(path, b"second\n")
    assert path.read_bytes() == b"first\n"


def _synthetic_inputs(contract: dict[str, Any]) -> runner.AnalysisInputs:
    families = contract["consumed_evidence_governance"][
        "fresh_evaluation_source_family_ids"
    ]
    group_rows = [
        prediction("d1", "cancel_transfer", "freeze_card"),
        prediction("d2", "account_balance", "freeze_card"),
        prediction(
            "d3", "unsupported_or_uncertain", "unsupported_or_uncertain"
        ),
        prediction("d4", "card_status", "card_status"),
    ]
    development_records = {
        row["example_id"]: {
            "intent": row["gold_intent"],
            "is_hard_negative": False,
            "boundary_target": None,
        }
        for row in group_rows
    }
    fresh_rows: list[dict[str, Any]] = []
    fresh_records: dict[str, dict[str, Any]] = {}
    for family_index, family in enumerate(families):
        definitions = [
            ("cancel_transfer", "freeze_card"),
            ("account_balance", "freeze_card"),
            ("unsupported_or_uncertain", "unsupported_or_uncertain"),
            ("card_status", "card_status"),
        ]
        for index, (gold, predicted) in enumerate(definitions):
            record_id = f"f{family_index}-{index}"
            row = prediction(
                record_id,
                gold,
                predicted,
                family=family,
                development=False,
            )
            fresh_rows.append(row)
            fresh_records[record_id] = {
                "intent": gold,
                "source_family_id": family,
                "is_hard_negative": False,
                "boundary_target": None,
            }
    candidate = {
        "group_aware_cv": {"predictions": group_rows},
        "fresh_source_evaluation": {"pooled_predictions": fresh_rows},
    }
    candidates = {
        candidate_id: copy.deepcopy(candidate)
        for candidate_id in contract["candidate_coverage"]["candidate_ids"]
    }
    return runner.AnalysisInputs(
        contract=contract,
        contract_sha256=runner.EXPECTED_CONTRACT_SHA256,
        results={},
        results_manifest={},
        development_records=development_records,
        fresh_records=fresh_records,
        candidate_results=candidates,
        prediction_coverage={
            "missing_prediction_count": 0,
            "duplicate_prediction_count": 0,
        },
    )


def test_result_and_manifest_governance_validation(
    contract: dict[str, Any],
) -> None:
    inputs = _synthetic_inputs(contract)
    result = runner.build_results(inputs)
    result_bytes = runner.stable_json_bytes(result)
    manifest = runner.build_manifest(inputs, result_bytes)

    assert result["schema_version"] == runner.RESULT_SCHEMA_VERSION
    assert result["next_required"] is None
    assert {
        "protected_false_positive_analysis",
        "unsupported_miss_analysis",
        "protected_recall_miss_analysis",
        "hard_negative_boundary_analysis",
        "cross_candidate_overlap",
        "fresh_family_comparison",
        "grouped_cv_vs_fresh_evaluation",
        "hierarchical_analysis",
        "evidence_classification",
    }.issubset(result)
    assert result["governance"]["candidate_selected"] is False
    assert result["governance"]["candidate_ranked"] is False
    assert result["governance"]["step29i_authorized"] is False
    assert result["governance"]["raw_utterance_text_persisted"] is False
    assert manifest["analysis_result"]["sha256"] == runner.sha256_bytes(
        result_bytes
    )
    assert manifest["failure_analysis_executed"] is True
    assert manifest["models_run"] is False
    assert manifest["inference_performed"] is False
    assert manifest["next_required"] is None

    changed_manifest = copy.deepcopy(manifest)
    changed_manifest["step29i_authorized"] = True
    with pytest.raises(ValueError, match="prohibited manifest governance"):
        runner.validate_manifest(changed_manifest, inputs, result_bytes)


def test_tracked_result_rejects_raw_text(contract: dict[str, Any]) -> None:
    inputs = _synthetic_inputs(contract)
    payload = runner.build_results(inputs)
    payload["text"] = "forbidden"

    with pytest.raises(ValueError, match="raw-text field"):
        runner.validate_results(payload, inputs)


def test_preflight_writes_nothing_and_runs_no_prediction_analysis(
    contract: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    inputs = _synthetic_inputs(contract)
    paths = replace(
        runner.DEFAULT_PATHS,
        result=tmp_path / "result.json",
        result_manifest=tmp_path / "manifest.json",
    )
    monkeypatch.setattr(
        runner,
        "load_preconditions",
        lambda *_args, **_kwargs: inputs,
    )
    monkeypatch.setattr(
        runner,
        "build_results",
        lambda *_args, **_kwargs: pytest.fail(
            "preflight must not execute prediction-level analysis"
        ),
    )

    result = runner.preflight(paths)

    assert result["prediction_level_analysis_executed"] is False
    assert result["files_written"] is False
    assert result["models_run"] is False
    assert result["inference_performed"] is False
    assert result["step29i_authorized"] is False
    assert result["next_required"] is None
    assert list(tmp_path.iterdir()) == []


def test_parser_modes_are_mutually_exclusive() -> None:
    parser = runner.build_parser()
    assert parser.parse_args(["--preflight"]).preflight is True
    assert parser.parse_args(["--run"]).run is True
    with pytest.raises(SystemExit):
        parser.parse_args(["--preflight", "--run"])
