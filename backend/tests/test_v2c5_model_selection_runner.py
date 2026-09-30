from __future__ import annotations

import inspect
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scripts import run_v2c5_model_selection as runner

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "data/evals/v2/ml/v2c5_model_selection_contract.json"
FINAL_HOLDOUT_PATH = ROOT / "data/evals/v2/ml/v2c5_final_holdout.json"


@pytest.fixture(scope="module")
def contract() -> dict[str, Any]:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def synthetic_examples(contract: dict[str, Any]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for label in contract["taxonomy"]["intent_label_order"]:
        for group_number in range(5):
            for variant in range(2):
                text = f"{label}|{group_number}|{variant}"
                examples.append(
                    {
                        "example_id": f"{label}-{group_number}-{variant}",
                        "group_id": f"{label}-group-{group_number}",
                        "intent": label,
                        "text": text,
                        "text_sha256": runner.sha256_bytes(text.encode("utf-8")),
                        "data_role": "development",
                    }
                )
    return examples


def direct_folds(examples: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray(
        [int(row["group_id"].rsplit("-", 1)[1]) for row in examples],
        dtype=np.int8,
    )


def selection_result(
    candidate_id: str,
    *,
    mean: float = 0.9,
    worst: float = 0.8,
    protected_fpr: float = 0.005,
    unsupported_recall: float = 0.9,
    latency: float = 1.0,
    eligible: bool = True,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "configuration": {"candidate_id": candidate_id},
        "aggregate_metrics": {
            "mean_fold_macro_f1": mean,
            "worst_fold_macro_f1": worst,
        },
        "safety_metrics": {
            "protected_write_false_positive_rate": protected_fpr,
            "unsupported_or_uncertain_recall": unsupported_recall,
        },
        "safety_gate_results": {"all_gates_pass": eligible},
        "timing": {"local_prediction_latency_ms_per_example": latency},
    }


def patch_fold_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, contract: dict[str, Any]
) -> None:
    contract_path = tmp_path / "data/evals/v2/ml/v2c5_model_selection_contract.json"
    runner_path = tmp_path / runner.RUNNER_RELATIVE_PATH
    contract_path.parent.mkdir(parents=True)
    runner_path.parent.mkdir(parents=True)
    contract_path.write_bytes(runner.stable_json_bytes(contract))
    runner_path.write_text("# synthetic runner\n", encoding="utf-8")
    monkeypatch.setattr(runner, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(runner, "ML_DIRECTORY", contract_path.parent)
    monkeypatch.setattr(runner, "CONTRACT_PATH", contract_path)
    monkeypatch.setattr(
        runner,
        "FOLDS_PATH",
        contract_path.parent / "v2c5_model_selection_folds.json",
    )
    monkeypatch.setattr(
        runner,
        "FOLDS_MANIFEST_PATH",
        contract_path.parent / "v2c5_model_selection_folds.manifest.json",
    )
    monkeypatch.setattr(
        runner,
        "RESULTS_PATH",
        contract_path.parent / "v2c5_model_selection_results.json",
    )
    monkeypatch.setattr(
        runner,
        "RESULTS_MANIFEST_PATH",
        contract_path.parent / "v2c5_model_selection_results.manifest.json",
    )
    monkeypatch.setattr(
        runner,
        "BGE_CACHE_PATH",
        contract_path.parent / "local/v2c5_model_selection_bge_cache.npz",
    )
    monkeypatch.setattr(
        runner,
        "BGE_CACHE_MANIFEST_PATH",
        contract_path.parent / "local/v2c5_model_selection_bge_cache.manifest.json",
    )


def test_contract_expands_exactly_27_candidates_across_five_families(
    contract: dict[str, Any],
) -> None:
    runner.validate_contract(contract)
    candidates = runner.expand_candidates(contract)
    assert len(candidates) == 27
    assert len({row["candidate_id"] for row in candidates}) == 27
    assert {row["family_id"] for row in candidates} == {
        "WORD_TFIDF_LINEAR_SVC",
        "CHAR_TFIDF_LINEAR_SVC",
        "WORD_CHAR_TFIDF_LINEAR_SVC",
        "BGE_SMALL_LINEAR_SVC",
        "BGE_SMALL_LOGISTIC_REGRESSION",
    }


def test_historical_bge_balanced_linear_svc_c4_occurs_once(
    contract: dict[str, Any],
) -> None:
    historical = "BGE_SMALL_LINEAR_SVC__C=4.0__class_weight=balanced"
    assert [row["candidate_id"] for row in runner.expand_candidates(contract)].count(
        historical
    ) == 1


def test_candidate_matrix_is_driven_by_contract(contract: dict[str, Any]) -> None:
    changed = deepcopy(contract)
    changed["candidate_matrix"]["families"][0]["hyperparameter_grid"]["C"] = [0.5]
    changed["candidate_matrix"]["families"][0]["configuration_count"] = 2
    changed["candidate_matrix"]["candidate_count"] = 23
    candidates = runner.expand_candidates(changed)
    assert candidates[0]["hyperparameters"]["C"] == 0.5
    assert len(candidates) == 23


def test_fold_assignment_is_deterministic_text_free_and_complete(
    contract: dict[str, Any],
) -> None:
    examples = synthetic_examples(contract)
    first = runner.build_fold_payload(examples, contract)
    second = runner.build_fold_payload(examples, contract)
    assert runner.stable_json_bytes(first) == runner.stable_json_bytes(second)
    assert len(first["assignments"]) == len(examples)
    assert len({row["example_id"] for row in first["assignments"]}) == len(examples)
    assert all("text" not in row for row in first["assignments"])
    assert first["integrity"]["each_record_validation_assignment_count"] == 1
    assert first["integrity"]["group_leakage_count"] == 0


def test_prepare_folds_writes_deterministic_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    contract: dict[str, Any],
) -> None:
    patch_fold_paths(monkeypatch, tmp_path, contract)
    examples = synthetic_examples(contract)
    runner.prepare_folds(examples, contract)
    first_fold_bytes = runner.FOLDS_PATH.read_bytes()
    first_manifest_bytes = runner.FOLDS_MANIFEST_PATH.read_bytes()
    runner.prepare_folds(examples, contract)
    assert runner.FOLDS_PATH.read_bytes() == first_fold_bytes
    assert runner.FOLDS_MANIFEST_PATH.read_bytes() == first_manifest_bytes


def test_group_leakage_is_rejected(contract: dict[str, Any]) -> None:
    examples = synthetic_examples(contract)
    payload = runner.build_fold_payload(examples, contract)
    first_group = examples[0]["group_id"]
    rows = [row for row in payload["assignments"] if row["group_id"] == first_group]
    rows[1]["validation_fold"] = (rows[0]["validation_fold"] + 1) % 5
    payload["assignment_sha256"] = runner.sha256_bytes(
        runner.stable_json_bytes(payload["assignments"])
    )
    with pytest.raises(ValueError, match="crosses validation folds"):
        runner.validate_fold_payload(payload, examples, contract)


def test_group_aware_feasibility_allows_a_group_with_multiple_labels(
    contract: dict[str, Any],
) -> None:
    examples = synthetic_examples(contract)
    examples[0]["group_id"] = examples[10]["group_id"]
    counts = runner.validate_group_feasibility(examples, contract)
    assert set(counts) == set(contract["taxonomy"]["intent_label_order"])


def test_duplicate_or_missing_validation_assignment_is_rejected(
    contract: dict[str, Any],
) -> None:
    examples = synthetic_examples(contract)
    payload = runner.build_fold_payload(examples, contract)
    payload["assignments"][-1] = deepcopy(payload["assignments"][0])
    payload["assignment_sha256"] = runner.sha256_bytes(
        runner.stable_json_bytes(payload["assignments"])
    )
    with pytest.raises(ValueError, match="duplicate validation assignment"):
        runner.validate_fold_payload(payload, examples, contract)


def test_check_folds_writes_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    contract: dict[str, Any],
) -> None:
    patch_fold_paths(monkeypatch, tmp_path, contract)
    examples = synthetic_examples(contract)
    runner.prepare_folds(examples, contract)
    before = {
        runner.FOLDS_PATH: runner.FOLDS_PATH.read_bytes(),
        runner.FOLDS_MANIFEST_PATH: runner.FOLDS_MANIFEST_PATH.read_bytes(),
    }
    runner.check_folds(examples, contract)
    assert {path: path.read_bytes() for path in before} == before


def test_run_refuses_absent_folds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    contract: dict[str, Any],
) -> None:
    patch_fold_paths(monkeypatch, tmp_path, contract)
    with pytest.raises(FileNotFoundError, match="fold artifact pair"):
        runner.run_model_selection(synthetic_examples(contract), contract)


def test_tfidf_is_fit_only_on_each_training_fold(
    monkeypatch: pytest.MonkeyPatch, contract: dict[str, Any]
) -> None:
    examples = synthetic_examples(contract)
    fitted_batches: list[list[str]] = []

    class RecordingRepresentation:
        def fit_transform(self, texts: list[str]) -> np.ndarray:
            fitted_batches.append(list(texts))
            return np.asarray(texts, dtype=object).reshape(-1, 1)

        def transform(self, texts: list[str]) -> np.ndarray:
            return np.asarray(texts, dtype=object).reshape(-1, 1)

    class LabelFromTextClassifier:
        def fit(self, features: np.ndarray, labels: np.ndarray) -> LabelFromTextClassifier:
            return self

        def predict(self, features: np.ndarray) -> np.ndarray:
            return np.asarray([str(row[0]).split("|", 1)[0] for row in features])

    monkeypatch.setattr(runner, "build_representation", lambda *args: RecordingRepresentation())
    monkeypatch.setattr(runner, "build_classifier", lambda *args: LabelFromTextClassifier())
    monkeypatch.setattr(runner, "serialized_model_size", lambda model: 1)
    candidate = runner.expand_candidates(contract)[0]
    result = runner.run_candidate_cv(
        candidate,
        examples,
        direct_folds(examples),
        contract,
        None,
        None,
    )
    assert len(fitted_batches) == 5
    for fold, fitted in enumerate(fitted_batches):
        validation_texts = {
            row["text"]
            for row, assigned in zip(examples, direct_folds(examples), strict=True)
            if assigned == fold
        }
        assert validation_texts.isdisjoint(fitted)
        assert len(fitted) + len(validation_texts) == len(examples)
    fold_macro_f1 = [row["macro_f1"] for row in result["fold_metrics"]]
    assert result["aggregate_metrics"]["mean_fold_macro_f1"] == pytest.approx(
        np.mean(fold_macro_f1)
    )
    assert result["aggregate_metrics"]["worst_fold_macro_f1"] == pytest.approx(
        min(fold_macro_f1)
    )


def write_cache_fixture(
    examples: list[dict[str, Any]],
    contract: dict[str, Any],
    embeddings: np.ndarray,
) -> dict[str, Any]:
    runner.BGE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        runner.BGE_CACHE_PATH,
        embeddings=embeddings,
        example_ids=np.asarray(runner.ordered_example_ids(examples)),
        text_sha256=np.asarray(runner.ordered_text_hashes(examples)),
    )
    manifest = runner.expected_bge_cache_manifest(
        examples,
        contract,
        cache_sha256=runner.sha256_file(runner.BGE_CACHE_PATH),
        generation_elapsed_seconds=1.25,
        fastembed_version="synthetic",
    )
    runner.BGE_CACHE_MANIFEST_PATH.write_bytes(runner.stable_json_bytes(manifest))
    return manifest


def test_bge_cache_lineage_row_order_and_384_dimensions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    contract: dict[str, Any],
) -> None:
    patch_fold_paths(monkeypatch, tmp_path, contract)
    examples = synthetic_examples(contract)[:3]
    embeddings = np.zeros((3, 384), dtype=np.float32)
    embeddings[:, 0] = 1.0
    write_cache_fixture(examples, contract, embeddings)
    loaded, _ = runner.validate_bge_cache(examples, contract)
    assert loaded.shape == (3, 384)

    with np.load(runner.BGE_CACHE_PATH, allow_pickle=False) as cached:
        reversed_ids = cached["example_ids"][::-1]
    np.savez_compressed(
        runner.BGE_CACHE_PATH,
        embeddings=embeddings,
        example_ids=reversed_ids,
        text_sha256=np.asarray(runner.ordered_text_hashes(examples)),
    )
    manifest = json.loads(runner.BGE_CACHE_MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["cache_file_sha256"] = runner.sha256_file(runner.BGE_CACHE_PATH)
    runner.BGE_CACHE_MANIFEST_PATH.write_bytes(runner.stable_json_bytes(manifest))
    with pytest.raises(ValueError, match="example row order"):
        runner.validate_bge_cache(examples, contract)


def test_stale_bge_cache_and_wrong_dimensions_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    contract: dict[str, Any],
) -> None:
    patch_fold_paths(monkeypatch, tmp_path, contract)
    examples = synthetic_examples(contract)[:2]
    embeddings = np.zeros((2, 384), dtype=np.float32)
    embeddings[:, 0] = 1.0
    manifest = write_cache_fixture(examples, contract, embeddings)
    manifest["development_dataset_sha256"] = "0" * 64
    runner.BGE_CACHE_MANIFEST_PATH.write_bytes(runner.stable_json_bytes(manifest))
    with pytest.raises(ValueError, match="stale or mismatched"):
        runner.validate_bge_cache(examples, contract)

    wrong = np.zeros((2, 383), dtype=np.float32)
    wrong[:, 0] = 1.0
    write_cache_fixture(examples, contract, wrong)
    with pytest.raises(ValueError, match="shape mismatch"):
        runner.validate_bge_cache(examples, contract)


def test_safety_metrics_use_exact_protected_write_semantics(
    contract: dict[str, Any],
) -> None:
    protected = contract["taxonomy"]["protected_write_intents"]
    gold = [
        "cancel_transfer",
        "close_account",
        "account_balance",
        "unsupported_or_uncertain",
    ]
    predicted = [
        "close_account",
        "close_account",
        "freeze_card",
        "unsupported_or_uncertain",
    ]
    metrics = runner.safety_metrics(gold, predicted, protected)
    assert protected == [
        "cancel_transfer",
        "close_account",
        "create_dispute",
        "freeze_card",
    ]
    assert metrics["protected_write_false_positive_rate"] == pytest.approx(0.5)
    assert metrics["exact_protected_write_recall"] == pytest.approx(0.5)
    assert metrics["unsupported_or_uncertain_recall"] == pytest.approx(1.0)


def test_three_gate_boundaries_are_inclusive(contract: dict[str, Any]) -> None:
    metrics = {
        "protected_write_false_positive_rate": 0.01,
        "exact_protected_write_recall": 0.8,
        "unsupported_or_uncertain_recall": 0.8,
    }
    assert runner.apply_safety_gates(metrics, contract)["all_gates_pass"] is True
    for field, failed in (
        ("protected_write_false_positive_rate", 0.0100001),
        ("exact_protected_write_recall", 0.7999999),
        ("unsupported_or_uncertain_recall", 0.7999999),
    ):
        changed = dict(metrics)
        changed[field] = failed
        result = runner.apply_safety_gates(changed, contract)
        assert result["all_gates_pass"] is False
        assert len(result["gates"]) == 3


def test_pooled_metrics_include_mean_worst_and_taxonomy_subsets(
    contract: dict[str, Any],
) -> None:
    labels = contract["taxonomy"]["intent_label_order"]
    metrics = runner.pooled_classification_metrics(labels, labels, contract)
    assert metrics["pooled_oof_macro_f1"] == pytest.approx(1.0)
    assert metrics["balanced_accuracy"] == pytest.approx(1.0)
    assert metrics["new_seven_intent_macro_f1"] == pytest.approx(1.0)
    assert metrics["historical_nine_label_macro_f1"] == pytest.approx(1.0)
    assert list(metrics["per_intent"]) == labels


def test_selection_uses_frozen_order(contract: dict[str, Any]) -> None:
    baseline = selection_result("baseline")
    higher_mean = selection_result("higher-mean", mean=0.91, worst=0.1)
    assert runner.select_candidate([baseline, higher_mean], contract)[
        "selected_candidate_id"
    ] == "higher-mean"

    better_worst = selection_result("better-worst", worst=0.81)
    assert runner.select_candidate([baseline, better_worst], contract)[
        "selected_candidate_id"
    ] == "better-worst"

    better_fpr = selection_result("better-fpr", protected_fpr=0.004)
    assert runner.select_candidate([baseline, better_fpr], contract)[
        "selected_candidate_id"
    ] == "better-fpr"

    better_unsupported = selection_result("better-unsupported", unsupported_recall=0.91)
    assert runner.select_candidate([baseline, better_unsupported], contract)[
        "selected_candidate_id"
    ] == "better-unsupported"

    lower_latency = selection_result("lower-latency", latency=0.9)
    assert runner.select_candidate([baseline, lower_latency], contract)[
        "selected_candidate_id"
    ] == "lower-latency"


def test_selection_tolerance_and_lexical_final_tie(contract: dict[str, Any]) -> None:
    almost_higher = selection_result("z-candidate", mean=0.9 + 5e-13)
    lexical = selection_result("a-candidate", mean=0.9)
    assert runner.select_candidate([almost_higher, lexical], contract)[
        "selected_candidate_id"
    ] == "a-candidate"
    clearly_higher = selection_result("z-candidate", mean=0.9 + 2e-12)
    assert runner.select_candidate([clearly_higher, lexical], contract)[
        "selected_candidate_id"
    ] == "z-candidate"


def test_selection_handles_no_candidate_and_one_selected(
    contract: dict[str, Any],
) -> None:
    failure = runner.select_candidate(
        [selection_result("failed", eligible=False)], contract
    )
    assert failure["selected_candidate_id"] is None
    assert failure["model_selection_success"] is False
    assert failure["step23_permitted"] is False
    success = runner.select_candidate([selection_result("selected")], contract)
    assert success["selected_candidate_id"] == "selected"
    assert success["model_selection_success"] is True
    assert success["step23_permitted"] is True


def test_partial_candidate_matrix_is_rejected(contract: dict[str, Any]) -> None:
    candidates = runner.expand_candidates(contract)
    partial = [{"candidate_id": row["candidate_id"]} for row in candidates[:-1]]
    with pytest.raises(ValueError, match="complete ordered 27-candidate"):
        runner.build_results_payload([], {"assignment_sha256": "x"}, partial, contract)


def test_oof_prediction_digest_is_deterministic(contract: dict[str, Any]) -> None:
    examples = synthetic_examples(contract)[:4]
    predictions = [row["intent"] for row in examples]
    first = runner.oof_prediction_sha256(examples, predictions)
    second = runner.oof_prediction_sha256(examples, predictions)
    assert first == second
    changed = list(predictions)
    changed[-1] = "unsupported_or_uncertain"
    assert runner.oof_prediction_sha256(examples, changed) != first


def test_result_artifact_rejects_raw_text(contract: dict[str, Any]) -> None:
    examples = synthetic_examples(contract)[:2]
    runner.validate_text_free({"candidate_id": "safe"}, examples)
    with pytest.raises(ValueError, match="raw development text"):
        runner.validate_text_free({"diagnostic": examples[0]["text"]}, examples)


def test_result_manifest_keeps_holdout_and_runtime_governance_false(
    contract: dict[str, Any],
) -> None:
    payload = {
        "contract": {"path": "contract.json", "sha256": "a" * 64},
        "source_development_dataset": contract["source_artifacts"][
            "expanded_development_dataset"
        ],
        "source_artifacts": contract["source_artifacts"],
        "fold_artifact": {"path": "folds.json", "sha256": "b" * 64},
        "candidate_count": 27,
        "completed_candidate_count": 27,
        "development_record_count": 8198,
        "selection": {
            "model_selection_success": False,
            "selected_candidate_id": None,
            "step23_permitted": False,
        },
    }
    manifest = runner.build_results_manifest(payload, b"{}\n")
    assert manifest["final_holdout_accessed"] is False
    assert manifest["final_holdout_evaluated"] is False
    assert manifest["final_holdout_inference_performed"] is False
    assert manifest["threshold_tuning_performed"] is False
    assert manifest["runtime_behavior_changed"] is False
    assert manifest["final_model_acceptance_claimed"] is False


def test_check_results_writes_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    contract: dict[str, Any],
) -> None:
    patch_fold_paths(monkeypatch, tmp_path, contract)
    result = {"selection": {"selected_candidate_id": None}}
    manifest = {"manifest": "synthetic"}
    runner.RESULTS_PATH.write_bytes(runner.stable_json_bytes(result))
    runner.RESULTS_MANIFEST_PATH.write_bytes(runner.stable_json_bytes(manifest))
    monkeypatch.setattr(runner, "check_folds", lambda *args: ({}, np.asarray([])))
    monkeypatch.setattr(runner, "validate_results_payload", lambda *args: None)
    monkeypatch.setattr(runner, "build_results_manifest", lambda *args: manifest)
    before = {
        runner.RESULTS_PATH: runner.RESULTS_PATH.read_bytes(),
        runner.RESULTS_MANIFEST_PATH: runner.RESULTS_MANIFEST_PATH.read_bytes(),
    }
    runner.check_results([], contract)
    assert {path: path.read_bytes() for path in before} == before


def test_latency_protocol_has_one_warmup_and_timed_value() -> None:
    calls: list[str] = []
    clock_values = iter([10.0, 10.5])

    def predict() -> list[str]:
        calls.append("predict")
        return ["a", "b"]

    latency = runner.measure_local_prediction_latency(
        predict,
        2,
        warmup_runs=1,
        clock=lambda: next(clock_values),
    )
    assert calls == ["predict", "predict"]
    assert latency == pytest.approx(250.0)


def test_holdout_dataset_opening_and_hashing_are_prohibited() -> None:
    with pytest.raises(PermissionError, match="must never open or hash"):
        runner.read_json(FINAL_HOLDOUT_PATH)
    with pytest.raises(PermissionError, match="must never open or hash"):
        runner.sha256_file(FINAL_HOLDOUT_PATH)


def test_preflight_reads_holdout_manifest_only(
    monkeypatch: pytest.MonkeyPatch, contract: dict[str, Any]
) -> None:
    opened: list[Path] = []
    original = runner.read_json

    def recording_read(path: Path) -> dict[str, Any]:
        opened.append(path.resolve())
        if path.resolve() == FINAL_HOLDOUT_PATH.resolve():
            raise AssertionError("sealed holdout dataset was opened")
        return original(path)

    monkeypatch.setattr(runner, "read_json", recording_read)
    outcome = runner.preflight(contract)
    assert outcome["final_holdout_accessed"] is False
    assert FINAL_HOLDOUT_PATH.resolve() not in opened
    assert (
        ROOT / "data/evals/v2/ml/v2c5_final_holdout.manifest.json"
    ).resolve() in opened


def test_runner_contains_no_threshold_tuning_or_holdout_inference_path() -> None:
    source = inspect.getsource(runner)
    assert "GridSearchCV" not in source
    assert "RandomizedSearchCV" not in source
    assert '"threshold_tuning_performed": False' in source
    assert '"final_holdout_inference_performed": False' in source
    assert "predict_proba" not in source
