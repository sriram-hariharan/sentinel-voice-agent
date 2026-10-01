from __future__ import annotations

import inspect
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scripts import run_v2c6_source_aware_model_selection as runner

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = (
    ROOT / "data/evals/v2/ml/v2c6_source_aware_model_selection_contract.json"
)
FINAL_HOLDOUT_PATH = ROOT / "data/evals/v2/ml/v2c5_final_holdout.json"


@pytest.fixture(scope="module")
def contract() -> dict[str, Any]:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def inputs() -> runner.LoadedInputs:
    return runner.load_and_validate_inputs()


def synthetic_group_examples(contract: dict[str, Any]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for label in contract["taxonomy"]["intent_label_order"]:
        for group_number in range(5):
            text = f"{label}|{group_number}"
            examples.append(
                {
                    "example_id": f"{label}-{group_number}",
                    "group_id": f"{label}-group-{group_number}",
                    "intent": label,
                    "text": text,
                }
            )
    return examples


def synthetic_source_examples(contract: dict[str, Any]) -> list[dict[str, Any]]:
    labels = contract["taxonomy"]["intent_label_order"]
    examples = [
        {
            "example_id": f"inherited-{index:04d}",
            "group_id": f"inherited-group-{index:04d}",
            "intent": labels[index % len(labels)],
            "text": f"inherited text {index}",
        }
        for index in range(8198)
    ]
    for family_index, family in enumerate(
        contract["source_family_holdout"]["source_families"], start=1
    ):
        for offset in range(270):
            example_id = f"sf{family_index}-{offset:04d}"
            examples.append(
                {
                    "example_id": example_id,
                    "record_id": example_id,
                    "group_id": f"{example_id}-group",
                    "intent": labels[offset % len(labels)],
                    "text": f"source family text {family_index} {offset}",
                    "source_family_id": family,
                    "v2c6_lineage": runner.REMEDIATION_LINEAGE,
                }
            )
    return examples


def selection_result(
    candidate_id: str,
    *,
    eligible: bool = True,
    worst: float = 0.7,
    mean: float = 0.8,
    pooled_source: float = 0.81,
    group_macro: float = 0.82,
    supported_to_unsupported: float = 0.1,
    protected_fpr: float = 0.005,
    protected_recall: float = 0.9,
    unsupported_recall: float = 0.9,
    complexity_rank: int = 1,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "eligible": eligible,
        "ineligibility_reasons": [] if eligible else ["failed"],
        "configuration": {"candidate_id": candidate_id},
        "selection_metrics": {
            "worst_family_primary_8_macro_f1": worst,
            "mean_family_primary_8_macro_f1": mean,
            "pooled_source_family_primary_8_macro_f1": pooled_source,
            "pooled_group_cv_macro_f1_16": group_macro,
            "pooled_source_family_supported_to_unsupported_rate": (
                supported_to_unsupported
            ),
            "pooled_source_family_protected_false_positive_rate": protected_fpr,
            "pooled_source_family_protected_recall": protected_recall,
            "pooled_source_family_unsupported_recall": unsupported_recall,
            "candidate_complexity_rank": complexity_rank,
            "candidate_id": candidate_id,
        },
    }


def patch_artifact_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ml_path = tmp_path / "data/evals/v2/ml"
    monkeypatch.setattr(runner, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(runner, "ML_DIRECTORY", ml_path)
    monkeypatch.setattr(
        runner,
        "RESULTS_PATH",
        ml_path / "v2c6_source_aware_model_selection_results.json",
    )
    monkeypatch.setattr(
        runner,
        "RESULTS_MANIFEST_PATH",
        ml_path / "v2c6_source_aware_model_selection_results.manifest.json",
    )
    monkeypatch.setattr(
        runner,
        "BGE_CACHE_PATH",
        ml_path / "local/v2c6_source_aware_model_selection_bge_cache.npz",
    )
    monkeypatch.setattr(
        runner,
        "BGE_CACHE_MANIFEST_PATH",
        ml_path
        / "local/v2c6_source_aware_model_selection_bge_cache.manifest.json",
    )


def test_exact_contract_and_dataset_sha_are_required(
    contract: dict[str, Any],
) -> None:
    runner.validate_contract(contract)
    assert runner.EXPECTED_CONTRACT_SHA256 == runner.sha256_file(CONTRACT_PATH)
    assert contract["frozen_input_lineage"]["dataset"]["sha256"] == (
        "d7f78d7a76799f47bfdc3c1291505d1b964b9d143245d2d353931e8cb4f4a493"
    )


def test_changed_contract_identity_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    contract: dict[str, Any],
) -> None:
    changed = deepcopy(contract)
    changed["phase"] = "changed"
    path = tmp_path / "contract.json"
    path.write_bytes(runner.stable_json_bytes(changed))
    monkeypatch.setattr(runner, "CONTRACT_PATH", path)

    with pytest.raises(ValueError, match="contract SHA mismatch"):
        runner.load_contract()


def test_step29f_freeze_state_is_required(
    inputs: runner.LoadedInputs,
) -> None:
    changed = deepcopy(inputs.freeze)
    changed["freeze_status"] = "NOT_FROZEN"

    with pytest.raises(ValueError, match="freeze_status"):
        runner.validate_freeze(changed, inputs.contract)


def test_step29f_next_required_is_required(
    inputs: runner.LoadedInputs,
) -> None:
    changed = deepcopy(inputs.freeze)
    changed["next_required"] = "changed"

    with pytest.raises(ValueError, match="next_required"):
        runner.validate_freeze(changed, inputs.contract)


def test_exact_six_candidate_definitions_resolve_from_contract(
    contract: dict[str, Any],
) -> None:
    candidates = contract["candidate_search_space"]["candidates"]

    assert tuple(row["candidate_id"] for row in candidates) == (
        runner.EXPECTED_CANDIDATE_IDS
    )
    assert len(candidates) == 6
    for candidate in candidates:
        configuration = runner.candidate_configuration(contract, candidate)
        assert configuration["representation"] == contract["representations"][
            candidate["representation_id"]
        ]
        assert configuration["classifier"]["parameters"] == {
            **contract["classifiers"][candidate["classifier_id"]][
                "fixed_parameters"
            ],
            **candidate["hyperparameters"],
        }


def test_missing_candidate_representation_is_rejected(
    contract: dict[str, Any],
) -> None:
    changed = deepcopy(contract)
    del changed["representations"]["BGE_SMALL"]

    with pytest.raises(ValueError, match="representation definitions"):
        runner.validate_contract(changed)


def test_candidate_hyperparameter_mismatch_is_rejected(
    contract: dict[str, Any],
) -> None:
    changed = deepcopy(contract)
    changed["candidate_search_space"]["candidates"][0]["hyperparameters"]["C"] = 1.0

    with pytest.raises(ValueError, match="hyperparameters conflict"):
        runner.validate_contract(changed)


def test_group_fold_generation_is_deterministic_and_complete(
    contract: dict[str, Any],
) -> None:
    examples = synthetic_group_examples(contract)
    first = runner.build_group_folds(examples, contract)
    second = runner.build_group_folds(examples, contract)

    assert first == second
    assert len(first) == 5
    validation = [index for fold in first for index in fold.test_indices]
    assert sorted(validation) == list(range(len(examples)))
    assert len(validation) == len(set(validation))


def test_group_overlap_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    contract: dict[str, Any],
) -> None:
    examples = synthetic_group_examples(contract)
    examples[0]["group_id"] = examples[1]["group_id"]

    class LeakingSplitter:
        def __init__(self, **_: Any) -> None:
            pass

        def split(self, *_: Any) -> Any:
            yield np.asarray([0, 2]), np.asarray([1])

    monkeypatch.setattr(runner, "StratifiedGroupKFold", LeakingSplitter)
    with pytest.raises(ValueError, match="group leakage"):
        runner.build_group_folds(examples, contract)


def test_historical_mixed_label_group_remains_atomic(
    contract: dict[str, Any],
) -> None:
    examples = synthetic_group_examples(contract)
    shared_group = "historical-mixed-group"
    examples[0]["group_id"] = shared_group
    examples[5]["group_id"] = shared_group
    folds = runner.build_group_folds(examples, contract)
    validation_fold_by_index = {
        index: fold.partition_id for fold in folds for index in fold.test_indices
    }

    assert examples[0]["intent"] != examples[5]["intent"]
    assert validation_fold_by_index[0] == validation_fold_by_index[5]


def test_exact_three_source_family_rounds_and_counts(
    contract: dict[str, Any],
) -> None:
    examples = synthetic_source_examples(contract)
    rounds = runner.build_source_family_rounds(examples, contract)

    assert len(rounds) == 3
    assert {len(row.train_indices) for row in rounds} == {8738}
    assert {len(row.test_indices) for row in rounds} == {270}


def test_held_out_source_family_is_absent_from_training(
    contract: dict[str, Any],
) -> None:
    examples = synthetic_source_examples(contract)
    rounds = runner.build_source_family_rounds(examples, contract)

    for round_ in rounds:
        assert all(
            examples[index].get("source_family_id")
            != round_.held_out_source_family_id
            for index in round_.train_indices
        )


def test_all_810_remediation_rows_are_tested_once(
    contract: dict[str, Any],
) -> None:
    examples = synthetic_source_examples(contract)
    rounds = runner.build_source_family_rounds(examples, contract)
    tested = [index for round_ in rounds for index in round_.test_indices]

    assert len(tested) == 810
    assert len(set(tested)) == 810
    assert set(tested) == set(range(8198, 9008))


def test_split_audit_is_deterministic_text_free(
    inputs: runner.LoadedInputs,
) -> None:
    first = runner.build_evaluation_splits(inputs.examples, inputs.contract)
    second = runner.build_evaluation_splits(inputs.examples, inputs.contract)

    assert runner.stable_json_bytes(first.audit) == runner.stable_json_bytes(
        second.audit
    )
    runner.validate_text_free(first.audit, inputs.examples)
    assert first.audit["group_cv"]["pooled_validation_count"] == 9008
    assert first.audit["source_family_holdout"]["pooled_test_count"] == 810


def test_same_split_object_is_reused_for_every_candidate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    inputs: runner.LoadedInputs,
) -> None:
    patch_artifact_paths(monkeypatch, tmp_path)
    split_sentinel = runner.EvaluationSplits((), (), {"audit_sha256": "a" * 64})
    build_calls: list[str] = []
    observed_split_ids: list[int] = []
    monkeypatch.setattr(
        runner,
        "build_evaluation_splits",
        lambda *_: build_calls.append("built") or split_sentinel,
    )
    monkeypatch.setattr(runner, "prepare_bge_cache", lambda *_: ({}, {}))

    def fake_evaluate(
        candidate: dict[str, Any],
        _: runner.LoadedInputs,
        splits: runner.EvaluationSplits,
        matrices: Any,
    ) -> dict[str, Any]:
        observed_split_ids.append(id(splits))
        return {"candidate_id": candidate["candidate_id"]}

    monkeypatch.setattr(runner, "evaluate_candidate", fake_evaluate)
    monkeypatch.setattr(
        runner,
        "build_results_payload",
        lambda *_: {"selection": {"selected_candidate_id": None}},
    )
    monkeypatch.setattr(runner, "build_results_manifest", lambda *_: {})

    runner.run_model_selection(inputs)

    assert build_calls == ["built"]
    assert observed_split_ids == [id(split_sentinel)] * 6


def test_group_cv_produces_one_prediction_per_record(
    monkeypatch: pytest.MonkeyPatch,
    contract: dict[str, Any],
) -> None:
    examples = synthetic_group_examples(contract)
    folds = runner.build_group_folds(examples, contract)
    splits = runner.EvaluationSplits(folds, (), {"audit_sha256": "a" * 64})

    def predict_gold(
        candidate: dict[str, Any],
        partition: runner.Partition,
        records: list[dict[str, Any]],
        frozen_contract: dict[str, Any],
        matrices: Any,
    ) -> np.ndarray:
        return np.asarray(
            [records[index]["intent"] for index in partition.test_indices],
            dtype=object,
        )

    monkeypatch.setattr(runner, "_fit_and_predict_partition", predict_gold)
    candidate = contract["candidate_search_space"]["candidates"][4]

    result = runner._group_cv_result(
        candidate,
        examples,
        splits,
        contract,
        None,
    )

    assert result["prediction_count"] == len(examples)
    assert len({row["example_id"] for row in result["predictions"]}) == len(
        examples
    )
    assert result["metrics"]["macro_f1_16"] == pytest.approx(1.0)


def test_tfidf_is_fitted_only_on_training_partition(
    monkeypatch: pytest.MonkeyPatch,
    contract: dict[str, Any],
) -> None:
    examples = [
        {"text": "training one", "intent": "account_balance"},
        {"text": "training two", "intent": "card_status"},
        {"text": "validation only", "intent": "account_balance"},
    ]
    partition = runner.Partition("fold", (0, 1), (2,))
    fitted: list[list[str]] = []
    transformed: list[list[str]] = []

    class RecordingRepresentation:
        def fit_transform(self, texts: list[str]) -> np.ndarray:
            fitted.append(list(texts))
            return np.ones((len(texts), 1))

        def transform(self, texts: list[str]) -> np.ndarray:
            transformed.append(list(texts))
            return np.ones((len(texts), 1))

    class SyntheticClassifier:
        def fit(self, features: Any, labels: Any) -> SyntheticClassifier:
            return self

        def predict(self, features: Any) -> np.ndarray:
            return np.asarray(["account_balance"] * len(features))

    monkeypatch.setattr(
        runner, "build_representation", lambda *_: RecordingRepresentation()
    )
    monkeypatch.setattr(runner, "build_classifier", lambda *_: SyntheticClassifier())
    candidate = contract["candidate_search_space"]["candidates"][4]

    runner._fit_and_predict_partition(candidate, partition, examples, contract, None)

    assert fitted == [["training one", "training two"]]
    assert transformed == [["validation only"]]
    assert "validation only" not in fitted[0]


def test_tfidf_parameters_are_explicit_and_contract_driven(
    contract: dict[str, Any],
) -> None:
    representation = runner.build_representation("WORD_TFIDF", contract)
    parameters = representation.get_params(deep=False)

    assert parameters["ngram_range"] == (1, 2)
    assert parameters["min_df"] == 2
    assert parameters["sublinear_tf"] is True
    assert parameters["norm"] == "l2"
    assert parameters["smooth_idf"] is True
    assert parameters["use_idf"] is True


def synthetic_cache_inputs(
    contract: dict[str, Any], examples: list[dict[str, Any]]
) -> runner.LoadedInputs:
    return runner.LoadedInputs(
        contract=contract,
        freeze={},
        dataset={},
        examples=tuple(examples),
        combined_manifest={},
        remediation_manifest={},
        contract_sha256=runner.EXPECTED_CONTRACT_SHA256,
    )


def write_cache_fixture(
    inputs: runner.LoadedInputs,
    splits: runner.EvaluationSplits,
) -> dict[str, Any]:
    specifications = runner._expected_cache_partitions(inputs.examples, splits)
    matrices: dict[str, np.ndarray] = {}
    for specification in specifications:
        matrix = np.zeros(
            (specification["row_count"], 384),
            dtype=np.float32,
        )
        matrix[:, 0] = 1.0
        matrices[specification["matrix_key"]] = matrix
    runner.BGE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(runner.BGE_CACHE_PATH, **matrices)
    manifest = runner.expected_bge_cache_manifest(
        inputs,
        splits,
        cache_file_sha256=runner.sha256_file(runner.BGE_CACHE_PATH),
        fastembed_version="synthetic",
    )
    runner.BGE_CACHE_MANIFEST_PATH.write_bytes(runner.stable_json_bytes(manifest))
    return manifest


def test_bge_cache_identity_and_partition_matrices(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    contract: dict[str, Any],
) -> None:
    patch_artifact_paths(monkeypatch, tmp_path)
    examples = synthetic_group_examples(contract)[:4]
    partition = runner.Partition("group_cv_fold_0", (0, 1), (2, 3))
    splits = runner.EvaluationSplits((partition,), (), {"audit_sha256": "a" * 64})
    synthetic = synthetic_cache_inputs(contract, examples)
    write_cache_fixture(synthetic, splits)

    matrices, manifest = runner.validate_bge_cache(synthetic, splits)

    assert manifest["dataset_sha256"] == contract["frozen_input_lineage"][
        "dataset"
    ]["sha256"]
    assert manifest["representation_id"] == "BGE_SMALL"
    assert manifest["dimensions"] == 384
    assert manifest["l2_normalized"] is True
    assert manifest["partition_transforms_generated_separately"] is True
    assert len(matrices) == 2


def test_stale_bge_cache_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    contract: dict[str, Any],
) -> None:
    patch_artifact_paths(monkeypatch, tmp_path)
    examples = synthetic_group_examples(contract)[:4]
    partition = runner.Partition("group_cv_fold_0", (0, 1), (2, 3))
    splits = runner.EvaluationSplits((partition,), (), {"audit_sha256": "a" * 64})
    synthetic = synthetic_cache_inputs(contract, examples)
    manifest = write_cache_fixture(synthetic, splits)
    manifest["dataset_sha256"] = "0" * 64
    runner.BGE_CACHE_MANIFEST_PATH.write_bytes(runner.stable_json_bytes(manifest))

    with pytest.raises(ValueError, match="stale or incompatible"):
        runner.validate_bge_cache(synthetic, splits)


def test_exact_primary_eight_excludes_two_nonprimary_intents(
    contract: dict[str, Any],
) -> None:
    primary = contract["taxonomy"]["primary_remediation_8_intents"]

    assert len(primary) == 8
    assert "lost_or_stolen_phone" not in primary
    assert "passcode_recovery" not in primary


def test_protected_recall_counts_any_protected_prediction(
    contract: dict[str, Any],
) -> None:
    metrics = runner.safety_metrics(
        [
            "cancel_transfer",
            "close_account",
            "account_balance",
            "unsupported_or_uncertain",
        ],
        [
            "freeze_card",
            "close_account",
            "account_balance",
            "unsupported_or_uncertain",
        ],
        contract["safety_gates"]["protected_intents"],
    )

    assert metrics["protected_recall"] == pytest.approx(1.0)


def test_protected_false_positive_rate_is_exact(
    contract: dict[str, Any],
) -> None:
    metrics = runner.safety_metrics(
        [
            "cancel_transfer",
            "account_balance",
            "card_status",
            "unsupported_or_uncertain",
            "recent_transactions",
        ],
        [
            "cancel_transfer",
            "freeze_card",
            "card_status",
            "unsupported_or_uncertain",
            "create_dispute",
        ],
        contract["safety_gates"]["protected_intents"],
    )

    assert metrics["protected_false_positive_rate"] == pytest.approx(0.5)


def test_unsupported_recall_is_exact(contract: dict[str, Any]) -> None:
    metrics = runner.safety_metrics(
        [
            "unsupported_or_uncertain",
            "unsupported_or_uncertain",
            "account_balance",
            "cancel_transfer",
        ],
        [
            "unsupported_or_uncertain",
            "account_balance",
            "account_balance",
            "freeze_card",
        ],
        contract["safety_gates"]["protected_intents"],
    )

    assert metrics["unsupported_recall"] == pytest.approx(0.5)


def test_safety_metrics_rejects_a_required_zero_denominator(
    contract: dict[str, Any],
) -> None:
    with pytest.raises(
        ValueError,
        match="metric denominator is zero: unsupported_recall",
    ):
        runner.safety_metrics(
            ["cancel_transfer", "account_balance"],
            ["cancel_transfer", "account_balance"],
            contract["safety_gates"]["protected_intents"],
        )


def safety_scope_result(
    protected_recall: float = 0.8,
    protected_fpr: float = 0.01,
    unsupported_recall: float = 0.8,
) -> dict[str, float]:
    return {
        "protected_recall": protected_recall,
        "protected_false_positive_rate": protected_fpr,
        "unsupported_recall": unsupported_recall,
    }


def test_safety_gate_boundaries_are_inclusive(contract: dict[str, Any]) -> None:
    group = {"metrics": {"safety": safety_scope_result()}}
    source = {
        "pooled_metrics": {"safety": safety_scope_result()},
        "rounds": [
            {
                "held_out_source_family_id": family,
                "metrics": {"safety": safety_scope_result()},
            }
            for family in contract["source_family_holdout"]["source_families"]
        ],
    }

    result = runner.apply_mandatory_safety_gates(group, source, contract)

    assert result["all_gates_pass"] is True
    assert len(result["gates"]) == 15
    assert {row["scope"] for row in result["gates"]} == {
        "pooled_group_aware_cv_predictions",
        "pooled_source_family_holdout_predictions",
        (
            "individual_source_family_holdout:"
            "v2c6_sf1_definition_direct"
        ),
        (
            "individual_source_family_holdout:"
            "v2c6_sf2_scenario_narrative"
        ),
        (
            "individual_source_family_holdout:"
            "v2c6_sf3_boundary_conversational"
        ),
    }


def test_failed_individual_family_gate_makes_candidate_ineligible(
    contract: dict[str, Any],
) -> None:
    group = {"metrics": {"safety": safety_scope_result()}}
    rounds = [
        {
            "held_out_source_family_id": family,
            "metrics": {"safety": safety_scope_result()},
        }
        for family in contract["source_family_holdout"]["source_families"]
    ]
    rounds[1]["metrics"]["safety"]["protected_recall"] = 0.799
    source = {
        "pooled_metrics": {"safety": safety_scope_result()},
        "rounds": rounds,
    }
    gates = runner.apply_mandatory_safety_gates(group, source, contract)
    eligible, reasons = runner.determine_candidate_eligibility(
        gates,
        completed_group_folds=5,
        completed_source_rounds=3,
        group_prediction_count=9008,
        source_prediction_count=810,
    )

    assert eligible is False
    assert reasons == [
        (
            "mandatory_safety_gate_failed:individual_source_family_holdout:"
            "v2c6_sf2_scenario_narrative:protected_recall"
        )
    ]


def test_incomplete_prediction_coverage_is_infrastructure_failure() -> None:
    gates = {"gates": [], "all_gates_pass": True}

    with pytest.raises(RuntimeError, match="prediction_coverage_exact"):
        runner.determine_candidate_eligibility(
            gates,
            completed_group_folds=5,
            completed_source_rounds=3,
            group_prediction_count=9007,
            source_prediction_count=810,
        )


def test_nonfinite_metrics_are_rejected() -> None:
    with pytest.raises(ValueError, match="non-finite metric"):
        runner._require_finite_metrics({"macro_f1": float("nan")})


def test_hard_negative_diagnostics_preserve_exact_ten_pairs(
    contract: dict[str, Any],
) -> None:
    pairs = contract["hard_negative_diagnostics"]["required_pairs"]
    records: list[dict[str, Any]] = []
    predictions: list[str] = []
    for index, (side_a, side_b) in enumerate(pairs):
        records.append(
            {
                "example_id": f"a-{index}",
                "intent": side_a,
                "boundary_target": side_b,
                "is_hard_negative": True,
            }
        )
        records.append(
            {
                "example_id": f"b-{index}",
                "intent": side_b,
                "boundary_target": side_a,
                "is_hard_negative": True,
            }
        )
        predictions.extend([side_a, side_b])

    diagnostics = runner.hard_negative_diagnostics(records, predictions, pairs)

    assert diagnostics["pair_count"] == 10
    assert [row["pair"] for row in diagnostics["pairs"]] == pairs
    assert diagnostics["aggregate_hard_negative_accuracy"] == pytest.approx(1.0)
    assert diagnostics["mandatory_safety_gate"] is False


def test_unsupported_boundary_diagnostics_are_exact(
    contract: dict[str, Any],
) -> None:
    metrics = runner.unsupported_boundary_metrics(
        ["account_blocked", "cancel_transfer", "unsupported_or_uncertain"],
        [
            "unsupported_or_uncertain",
            "cancel_transfer",
            "account_blocked",
        ],
        contract["taxonomy"]["primary_remediation_8_intents"],
    )

    assert metrics["supported_to_unsupported_rate"] == pytest.approx(0.5)
    assert metrics["unsupported_to_supported_rate"] == pytest.approx(1.0)


def test_lexicographic_selection_starts_with_worst_family(
    contract: dict[str, Any],
) -> None:
    baseline = selection_result("baseline", worst=0.7, mean=0.99)
    robust = selection_result("robust", worst=0.71, mean=0.8)

    selected = runner.select_candidate([baseline, robust], contract)

    assert selected["selected_candidate_id"] == "robust"


def test_selection_follows_every_frozen_metric_before_complexity(
    contract: dict[str, Any],
) -> None:
    baseline = selection_result("baseline")
    alternatives = [
        selection_result("mean", mean=0.81),
        selection_result("pooled", pooled_source=0.82),
        selection_result("group", group_macro=0.83),
        selection_result("boundary", supported_to_unsupported=0.09),
        selection_result("fpr", protected_fpr=0.004),
        selection_result("protected", protected_recall=0.91),
        selection_result("unsupported", unsupported_recall=0.91),
    ]

    for alternative in alternatives:
        selected = runner.select_candidate([baseline, alternative], contract)
        assert selected["selected_candidate_id"] == alternative["candidate_id"]


def test_numerical_tie_tolerance_and_lexical_tie_break(
    contract: dict[str, Any],
) -> None:
    lexical = selection_result("a-candidate", worst=0.7)
    within_tolerance = selection_result("z-candidate", worst=0.7 + 5e-13)
    outside_tolerance = selection_result("z-candidate", worst=0.7 + 2e-12)

    assert runner.select_candidate([within_tolerance, lexical], contract)[
        "selected_candidate_id"
    ] == "a-candidate"
    assert runner.select_candidate([outside_tolerance, lexical], contract)[
        "selected_candidate_id"
    ] == "z-candidate"


def test_exact_complexity_order_breaks_ties(contract: dict[str, Any]) -> None:
    lower = selection_result("lower", complexity_rank=1)
    higher = selection_result("higher", complexity_rank=2)

    selected = runner.select_candidate([higher, lower], contract)

    assert selected["selected_candidate_id"] == "lower"
    assert contract["selection_rule"]["complexity_order_low_to_high"] == [
        runner.EXPECTED_CANDIDATE_IDS[4],
        runner.EXPECTED_CANDIDATE_IDS[5],
        runner.EXPECTED_CANDIDATE_IDS[2],
        runner.EXPECTED_CANDIDATE_IDS[3],
        runner.EXPECTED_CANDIDATE_IDS[0],
        runner.EXPECTED_CANDIDATE_IDS[1],
    ]


def test_no_acceptable_candidate_never_forces_a_winner(
    contract: dict[str, Any],
) -> None:
    selection = runner.select_candidate(
        [selection_result("failed", eligible=False)], contract
    )

    assert selection["selection_status"] == "NO_ACCEPTABLE_CANDIDATE"
    assert selection["selected_candidate_id"] is None
    assert selection["selected_candidate"] is None
    assert selection["winner_forced"] is False
    assert selection["next_required"] is None
    assert selection["next_required_contract_ambiguity"]


def test_no_weighted_score_or_threshold_tuning_path() -> None:
    source = inspect.getsource(runner)

    assert "GridSearchCV" not in source
    assert "RandomizedSearchCV" not in source
    assert "predict_proba" not in source
    assert "def weighted_score" not in source
    assert '"threshold_tuning_performed": False' in source


def test_preflight_performs_no_model_work_or_writes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    inputs: runner.LoadedInputs,
) -> None:
    patch_artifact_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(
        runner,
        "load_fastembed_model",
        lambda *_: pytest.fail("preflight loaded FastEmbed"),
    )
    monkeypatch.setattr(
        runner,
        "fit_classifier",
        lambda *_: pytest.fail("preflight fit a classifier"),
    )

    report = runner.preflight(inputs)

    assert report["embeddings_generated"] is False
    assert report["classifier_fitting_performed"] is False
    assert report["inference_performed"] is False
    assert report["model_selection_performed"] is False
    assert report["model_training_performed"] is False
    assert report["files_written"] is False
    assert list(tmp_path.rglob("*")) == []


def test_result_serialization_rejects_raw_text(
    contract: dict[str, Any],
) -> None:
    examples = synthetic_group_examples(contract)[:2]
    runner.validate_text_free({"safe": "metadata"}, examples)

    with pytest.raises(ValueError, match="raw development text"):
        runner.validate_text_free({"diagnostic": examples[0]["text"]}, examples)


def test_final_holdout_read_and_hash_are_prohibited() -> None:
    with pytest.raises(PermissionError, match="never access"):
        runner.read_json(FINAL_HOLDOUT_PATH)
    with pytest.raises(PermissionError, match="never access"):
        runner.sha256_file(FINAL_HOLDOUT_PATH)


def test_result_manifest_keeps_governance_false() -> None:
    payload = {
        "contract": {"path": "contract.json", "sha256": "a" * 64},
        "frozen_inputs": {},
        "split_audit": {"audit_sha256": "b" * 64},
        "candidate_count": 6,
        "completed_candidate_count": 6,
        "selection": {
            "selection_status": "NO_ACCEPTABLE_CANDIDATE",
            "selected_candidate_id": None,
        },
        "next_required": None,
    }

    manifest = runner.build_results_manifest(payload, b"{}\n")

    assert manifest["final_full_development_model_fitting_performed"] is False
    assert manifest["model_selection_performed"] is True
    assert manifest["model_training_performed"] is False
    assert manifest["final_holdout_accessed"] is False
    assert manifest["final_holdout_evaluated"] is False
    assert manifest["threshold_tuning_performed"] is False
    assert manifest["runtime_behavior_changed"] is False
    assert manifest["final_model_acceptance_claimed"] is False


def test_result_artifacts_are_create_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    inputs: runner.LoadedInputs,
) -> None:
    patch_artifact_paths(monkeypatch, tmp_path)
    runner.RESULTS_PATH.parent.mkdir(parents=True)
    runner.RESULTS_PATH.write_text("existing\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing overwrite"):
        runner.run_model_selection(inputs)


def test_stable_json_serialization_is_deterministic() -> None:
    first = runner.stable_json_bytes({"b": 2, "a": [3, 1]})
    second = runner.stable_json_bytes({"a": [3, 1], "b": 2})

    assert first == second
    assert runner.sha256_bytes(first) == runner.sha256_bytes(second)
