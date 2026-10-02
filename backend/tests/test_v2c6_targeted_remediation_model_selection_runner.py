from __future__ import annotations

import inspect
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy import sparse

from scripts import run_v2c6_targeted_remediation_model_selection as runner

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = (
    ROOT / "data/evals/v2/ml/v2c6_targeted_remediation_design_contract.json"
)
STEP29G_CONTRACT_PATH = (
    ROOT / "data/evals/v2/ml/v2c6_source_aware_model_selection_contract.json"
)
FINAL_HOLDOUT_PATH = ROOT / "data/evals/v2/ml/v2c5_final_holdout.json"


@pytest.fixture(scope="module")
def contract() -> dict[str, Any]:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def step29g_contract() -> dict[str, Any]:
    return json.loads(STEP29G_CONTRACT_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def inputs() -> runner.LoadedInputs:
    return runner.load_and_validate_inputs()


def synthetic_group_examples(
    step29g_contract: dict[str, Any],
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for label in step29g_contract["taxonomy"]["intent_label_order"]:
        for group_number in range(5):
            examples.append(
                {
                    "example_id": f"{label}-{group_number}",
                    "group_id": f"{label}-group-{group_number}",
                    "intent": label,
                    "text": f"{label} sample {group_number}",
                }
            )
    return examples


def synthetic_fresh_examples() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family in runner.FRESH_FAMILY_IDS:
        for intent in runner.PRIMARY_INTENTS:
            for index in range(40):
                record_id = f"{family}:{intent}:{index:02d}"
                text = f"fresh {family} {intent} {index}"
                rows.append(
                    {
                        "record_id": record_id,
                        "group_id": f"{record_id}:group",
                        "intent": intent,
                        "text": text,
                        "text_sha256": runner.sha256_bytes(text.encode("utf-8")),
                        "source_family_id": family,
                        "dataset_role": "fresh_source_evaluation",
                        "excluded_from_candidate_fitting": True,
                        "not_training_data": True,
                    }
                )
    return rows


def synthetic_inputs(
    contract: dict[str, Any], step29g_contract: dict[str, Any]
) -> runner.LoadedInputs:
    development = synthetic_group_examples(step29g_contract)
    for row in development:
        row["text_sha256"] = runner.sha256_bytes(row["text"].encode("utf-8"))
        row["data_role"] = "development"
    return runner.LoadedInputs(
        contract=contract,
        step29g_contract=step29g_contract,
        development_dataset={},
        fresh_dataset={},
        development_examples=tuple(development),
        fresh_examples=tuple(synthetic_fresh_examples()),
        development_manifest={},
        fresh_manifest={},
        contract_sha256=runner.EXPECTED_CONTRACT_SHA256,
        development_sha256=runner.EXPECTED_DEVELOPMENT_SHA256,
        fresh_sha256=runner.EXPECTED_FRESH_SHA256,
    )


def patch_output_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ml_path = tmp_path / "data/evals/v2/ml"
    monkeypatch.setattr(runner, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(runner, "ML_DIRECTORY", ml_path)
    monkeypatch.setattr(
        runner,
        "RESULTS_PATH",
        ml_path / "v2c6_targeted_remediation_model_selection_results.json",
    )
    monkeypatch.setattr(
        runner,
        "RESULTS_MANIFEST_PATH",
        ml_path
        / "v2c6_targeted_remediation_model_selection_results.manifest.json",
    )
    monkeypatch.setattr(
        runner,
        "BGE_CACHE_PATH",
        ml_path / "local/v2c6_targeted_remediation_model_selection_bge_cache.npz",
    )
    monkeypatch.setattr(
        runner,
        "BGE_CACHE_MANIFEST_PATH",
        ml_path
        / "local/v2c6_targeted_remediation_model_selection_bge_cache.manifest.json",
    )


def safety_scope(
    *,
    protected_recall: float = 0.8,
    protected_fpr: float = 0.01,
    unsupported_recall: float = 0.8,
) -> dict[str, float]:
    return {
        "protected_recall": protected_recall,
        "protected_false_positive_rate": protected_fpr,
        "unsupported_recall": unsupported_recall,
    }


def gate_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    group = {"metrics": {"safety": safety_scope()}}
    fresh = {
        "families": [
            {
                "source_family_id": family,
                "metrics": {"safety": safety_scope()},
            }
            for family in runner.FRESH_FAMILY_IDS
        ],
        "pooled_metrics": {"safety": safety_scope()},
    }
    return group, fresh


def selection_result(
    candidate_id: str,
    *,
    eligible: bool = True,
    worst: float = 0.70,
    mean: float = 0.75,
    pooled: float = 0.76,
    group_macro: float = 0.80,
    protected_fpr: float = 0.005,
    protected_recall: float = 0.90,
    unsupported_recall: float = 0.90,
    complexity_rank: int = 1,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "eligible": eligible,
        "ineligibility_reasons": [] if eligible else ["failed"],
        "configuration": {"candidate_id": candidate_id},
        "selection_metrics": {
            "worst_fresh_family_primary_8_macro_f1": worst,
            "mean_fresh_family_primary_8_macro_f1": mean,
            "pooled_fresh_primary_8_macro_f1": pooled,
            "pooled_group_cv_macro_f1_16": group_macro,
            "pooled_fresh_protected_false_positive_rate": protected_fpr,
            "pooled_fresh_protected_recall": protected_recall,
            "pooled_fresh_unsupported_recall": unsupported_recall,
            "model_complexity_rank": complexity_rank,
            "candidate_id": candidate_id,
        },
    }


def test_exact_step29h_c_contract_identity_is_required(
    contract: dict[str, Any],
) -> None:
    runner.validate_contract(contract)
    assert runner.sha256_file(CONTRACT_PATH) == runner.EXPECTED_CONTRACT_SHA256


def test_exact_four_candidate_definitions_are_required(
    contract: dict[str, Any],
) -> None:
    assert tuple(
        row["candidate_id"]
        for row in contract["candidate_search_space"]["candidates"]
    ) == runner.EXPECTED_CANDIDATE_IDS
    changed = deepcopy(contract)
    changed["candidate_search_space"]["candidates"].reverse()
    with pytest.raises(ValueError, match="candidate definitions or ordering"):
        runner.validate_contract(changed)


def test_balanced_or_alternative_c_candidate_is_rejected(
    contract: dict[str, Any],
) -> None:
    changed = deepcopy(contract)
    changed["candidate_search_space"]["candidates"][0]["class_weight"] = "balanced"
    with pytest.raises(ValueError, match="candidate definitions"):
        runner.validate_contract(changed)

    changed = deepcopy(contract)
    changed["candidate_search_space"]["candidates"][1]["C"] = 2.0
    with pytest.raises(ValueError, match="candidate definitions"):
        runner.validate_contract(changed)


def test_frozen_dataset_schemas_counts_and_families_are_exact(
    inputs: runner.LoadedInputs,
) -> None:
    assert inputs.development_dataset["schema_version"] == runner.DEVELOPMENT_SCHEMA_VERSION
    assert len(inputs.development_examples) == 9608
    assert inputs.fresh_dataset["schema_version"] == runner.FRESH_SCHEMA_VERSION
    assert len(inputs.fresh_examples) == 640
    counts = {family: 0 for family in runner.FRESH_FAMILY_IDS}
    for row in inputs.fresh_examples:
        counts[row["source_family_id"]] += 1
        assert row["excluded_from_candidate_fitting"] is True
    assert counts == dict.fromkeys(runner.FRESH_FAMILY_IDS, 320)


def test_changed_development_or_fresh_population_is_rejected(
    inputs: runner.LoadedInputs,
) -> None:
    changed_development = dict(inputs.development_dataset)
    changed_development["example_count"] = 9607
    with pytest.raises(ValueError, match="development population"):
        runner.validate_development_dataset(
            changed_development, inputs.step29g_contract
        )

    changed_fresh = deepcopy(inputs.fresh_dataset)
    changed_fresh["examples"][0]["source_family_id"] = "changed"
    with pytest.raises(ValueError, match="unexpected fresh source family"):
        runner.validate_fresh_dataset(
            changed_fresh, inputs.development_examples
        )


def test_final_holdout_permission_change_is_rejected(
    contract: dict[str, Any],
) -> None:
    changed = deepcopy(contract)
    changed["final_holdout_policy"]["existing_final_holdout_access_permitted"] = True
    with pytest.raises(ValueError, match="final-holdout policy"):
        runner.validate_contract(changed)


def test_final_holdout_access_is_prohibited() -> None:
    with pytest.raises(PermissionError, match="never access"):
        runner.read_json(FINAL_HOLDOUT_PATH)
    with pytest.raises(PermissionError, match="never access"):
        runner.sha256_file(FINAL_HOLDOUT_PATH)


def test_group_folds_are_deterministic_complete_and_group_atomic(
    contract: dict[str, Any], step29g_contract: dict[str, Any]
) -> None:
    examples = synthetic_group_examples(step29g_contract)
    first = runner.build_group_folds(examples, contract)
    second = runner.build_group_folds(examples, contract)
    assert first == second
    assert len(first) == 5
    validation = [index for fold in first for index in fold.validation_indices]
    assert sorted(validation) == list(range(len(examples)))
    assert len(validation) == len(set(validation))
    for fold in first:
        train_groups = {examples[index]["group_id"] for index in fold.train_indices}
        validation_groups = {
            examples[index]["group_id"] for index in fold.validation_indices
        }
        assert train_groups.isdisjoint(validation_groups)


def test_group_leakage_fails_loudly(
    monkeypatch: pytest.MonkeyPatch,
    contract: dict[str, Any],
    step29g_contract: dict[str, Any],
) -> None:
    examples = synthetic_group_examples(step29g_contract)
    examples[0]["group_id"] = examples[1]["group_id"]

    class LeakingSplitter:
        def __init__(self, **_: Any) -> None:
            pass

        def split(self, *_: Any) -> Any:
            yield np.asarray([0, 2]), np.asarray([1])

    monkeypatch.setattr(runner, "StratifiedGroupKFold", LeakingSplitter)
    with pytest.raises(ValueError, match="group leakage"):
        runner.build_group_folds(examples, contract)


def test_split_audit_excludes_every_fresh_row(
    contract: dict[str, Any], step29g_contract: dict[str, Any]
) -> None:
    synthetic = synthetic_inputs(contract, step29g_contract)
    splits = runner.build_evaluation_splits(
        synthetic.development_examples,
        synthetic.fresh_examples,
        contract,
    )
    assert splits.audit["fold_count"] == 5
    assert splits.audit["pooled_validation_count"] == len(
        synthetic.development_examples
    )
    assert splits.audit["fresh_record_count"] == 640
    assert splits.audit["fresh_records_in_group_cv"] == 0
    runner.validate_text_free(splits.audit, synthetic.development_examples)


def test_same_frozen_folds_are_reused_for_all_candidates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    contract: dict[str, Any],
    step29g_contract: dict[str, Any],
) -> None:
    synthetic = synthetic_inputs(contract, step29g_contract)
    patch_output_paths(monkeypatch, tmp_path)
    split_sentinel = runner.EvaluationSplits((), {"audit_sha256": "a" * 64})
    observed: list[int] = []
    monkeypatch.setattr(runner, "build_evaluation_splits", lambda *_: split_sentinel)
    monkeypatch.setattr(runner, "prepare_bge_cache", lambda *_: ({}, {}))

    def fake_evaluate(
        candidate: dict[str, Any],
        _: runner.LoadedInputs,
        splits: runner.EvaluationSplits,
        matrices: Any,
    ) -> dict[str, Any]:
        observed.append(id(splits))
        return {"candidate_id": candidate["candidate_id"]}

    monkeypatch.setattr(runner, "evaluate_candidate", fake_evaluate)
    monkeypatch.setattr(
        runner,
        "build_results_payload",
        lambda *_: {"phase": "test", "selection": {"selection_status": "test"}},
    )
    monkeypatch.setattr(runner, "build_results_manifest", lambda *_: {})
    runner.run_model_selection(synthetic)
    assert observed == [id(split_sentinel)] * 4


def test_tfidf_fits_only_training_partition(
    monkeypatch: pytest.MonkeyPatch, step29g_contract: dict[str, Any]
) -> None:
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
            return np.asarray(["account_balance"] * len(features), dtype=object)

    monkeypatch.setattr(
        runner, "build_representation", lambda *_: RecordingRepresentation()
    )
    monkeypatch.setattr(runner, "build_classifier", lambda *_args, **_kwargs: SyntheticClassifier())
    train = [
        {"text": "training one", "intent": "account_balance"},
        {"text": "training two", "intent": "card_status"},
    ]
    fresh = [{"text": "fresh only", "intent": "account_balance"}]
    runner.fit_predict_candidate(
        runner.EXPECTED_CANDIDATE_IDS[1], train, fresh, step29g_contract
    )
    assert fitted == [["training one", "training two"]]
    assert transformed == [["fresh only"]]
    assert "fresh only" not in fitted[0]


def test_full_development_candidate_fit_is_used_once_for_both_families(
    monkeypatch: pytest.MonkeyPatch,
    contract: dict[str, Any],
    step29g_contract: dict[str, Any],
) -> None:
    synthetic = synthetic_inputs(contract, step29g_contract)
    calls: list[tuple[int, int]] = []

    def predict_gold(
        candidate_id: str,
        train_rows: Any,
        evaluation_rows: Any,
        frozen_contract: Any,
        **_: Any,
    ) -> np.ndarray:
        calls.append((len(train_rows), len(evaluation_rows)))
        return np.asarray([row["intent"] for row in evaluation_rows], dtype=object)

    monkeypatch.setattr(runner, "fit_predict_candidate", predict_gold)
    result = runner._fresh_evaluation_result(
        runner.EXPECTED_CANDIDATE_IDS[0],
        synthetic,
        np.zeros((len(synthetic.development_examples), 384), dtype=np.float32),
        np.zeros((640, 384), dtype=np.float32),
    )
    assert calls == [(len(synthetic.development_examples), 640)]
    assert result["full_development_fit_count"] == 1
    assert result["fresh_records_used_for_fitting"] == 0
    assert result["same_fitted_candidate_used_for_both_families"] is True
    assert [row["prediction_count"] for row in result["families"]] == [320, 320]
    assert result["pooled_prediction_count"] == 640


def test_word_char_tfidf_parameters_reuse_frozen_step29g(
    step29g_contract: dict[str, Any],
) -> None:
    representation = runner.build_representation("WORD_CHAR_TFIDF", step29g_contract)
    parameters = representation.transformer_list[0][1].get_params(deep=False)
    assert parameters["ngram_range"] == (1, 2)
    assert parameters["min_df"] == 2
    assert parameters["sublinear_tf"] is True
    assert representation.transformer_list[1][1].get_params(deep=False)[
        "ngram_range"
    ] == (3, 5)


def test_linear_svc_parameters_are_exact_and_unweighted(
    step29g_contract: dict[str, Any],
) -> None:
    bge_classifier = runner.build_classifier(step29g_contract, c_value=4.0)
    tfidf_classifier = runner.build_classifier(step29g_contract, c_value=1.0)
    assert bge_classifier.C == 4.0
    assert tfidf_classifier.C == 1.0
    assert bge_classifier.class_weight is None
    assert tfidf_classifier.class_weight is None
    assert bge_classifier.random_state == 20260930


def test_hybrid_is_sparse_concatenated_and_row_l2_normalized() -> None:
    tfidf = sparse.csr_matrix([[3.0, 0.0], [0.0, 4.0]])
    bge = np.asarray([[0.0, 4.0], [3.0, 0.0]], dtype=np.float32)
    combined = runner._hybrid_features(tfidf, bge)
    assert sparse.issparse(combined)
    assert combined.shape == (2, 4)
    assert np.linalg.norm(combined.toarray(), axis=1) == pytest.approx([1.0, 1.0])


def test_hierarchical_routing_and_partition_local_fits_are_exact(
    monkeypatch: pytest.MonkeyPatch, step29g_contract: dict[str, Any]
) -> None:
    fitted_texts: list[list[str]] = []
    transformed_texts: list[list[str]] = []

    class RecordingRepresentation:
        def fit_transform(self, texts: list[str]) -> np.ndarray:
            fitted_texts.append(list(texts))
            return np.ones((len(texts), 1))

        def transform(self, texts: list[str]) -> np.ndarray:
            transformed_texts.append(list(texts))
            return np.ones((len(texts), 1))

    class SyntheticClassifier:
        def __init__(self, predictions: list[str]) -> None:
            self.predictions = predictions

        def fit(self, features: Any, labels: Any) -> SyntheticClassifier:
            return self

        def predict(self, features: Any) -> np.ndarray:
            return np.asarray(self.predictions[: len(features)], dtype=object)

    classifiers = iter(
        [
            SyntheticClassifier([runner.UNSUPPORTED_INTENT, runner.SUPPORTED_BINARY_LABEL]),
            SyntheticClassifier(["cancel_transfer", "close_account"]),
        ]
    )
    monkeypatch.setattr(
        runner, "build_representation", lambda *_: RecordingRepresentation()
    )
    monkeypatch.setattr(
        runner, "build_classifier", lambda *_args, **_kwargs: next(classifiers)
    )
    train = [
        {"text": "unsupported train", "intent": runner.UNSUPPORTED_INTENT},
        {"text": "cancel train", "intent": "cancel_transfer"},
        {"text": "close train", "intent": "close_account"},
    ]
    evaluation = [
        {"text": "first evaluation", "intent": runner.UNSUPPORTED_INTENT},
        {"text": "second evaluation", "intent": "close_account"},
    ]
    predicted = runner.fit_predict_candidate(
        runner.EXPECTED_CANDIDATE_IDS[3],
        train,
        evaluation,
        step29g_contract,
    )
    assert predicted.tolist() == [runner.UNSUPPORTED_INTENT, "close_account"]
    assert fitted_texts == [
        ["unsupported train", "cancel train", "close train"],
        ["cancel train", "close train"],
    ]
    assert transformed_texts == [
        ["first evaluation", "second evaluation"],
        ["first evaluation", "second evaluation"],
    ]


def test_no_balanced_weight_calibration_or_threshold_path() -> None:
    source = inspect.getsource(runner)
    assert "class_weight=\"balanced\"" not in source
    assert "CalibratedClassifierCV" not in source
    assert "predict_proba" not in source
    assert "GridSearchCV" not in source
    assert "RandomizedSearchCV" not in source


def test_all_four_scopes_and_all_three_gates_are_required(
    contract: dict[str, Any],
) -> None:
    group, fresh = gate_inputs()
    result = runner.apply_mandatory_safety_gates(group, fresh, contract)
    assert result["required_scope_count"] == 4
    assert result["gate_count_per_scope"] == 3
    assert len(result["gates"]) == 12
    assert {row["scope"] for row in result["gates"]} == {
        "pooled_group_aware_cv",
        *runner.FRESH_FAMILY_IDS,
        "pooled_fresh_evaluation",
    }


@pytest.mark.parametrize(
    ("protected_false_positive_count", "expected_pass"),
    [(1, True), (2, False)],
)
def test_fresh_family_protected_fp_allowance_is_exact(
    contract: dict[str, Any],
    protected_false_positive_count: int,
    expected_pass: bool,
) -> None:
    gold = ["cancel_transfer"] * 160 + [runner.UNSUPPORTED_INTENT] * 40 + [
        "account_blocked"
    ] * 120
    predicted = list(gold)
    for index in range(protected_false_positive_count):
        predicted[200 + index] = "freeze_card"
    metrics = runner.safety_metrics(gold, predicted)
    assert metrics["non_protected_gold_count"] == 160
    group, fresh = gate_inputs()
    fresh["families"][0]["metrics"]["safety"] = metrics
    result = runner.apply_mandatory_safety_gates(group, fresh, contract)
    gate = next(
        row
        for row in result["gates"]
        if row["scope"] == runner.FRESH_FAMILY_IDS[0]
        and row["metric"] == "protected_false_positive_rate"
    )
    assert gate["passed"] is expected_pass


def test_failing_any_gate_makes_candidate_ineligible(
    contract: dict[str, Any],
) -> None:
    group, fresh = gate_inputs()
    fresh["families"][1]["metrics"]["safety"]["unsupported_recall"] = 0.79
    gates = runner.apply_mandatory_safety_gates(group, fresh, contract)
    eligible, reasons = runner.determine_candidate_eligibility(
        gates,
        completed_group_folds=5,
        group_prediction_count=9608,
        fresh_prediction_count=640,
        fresh_family_prediction_counts=[320, 320],
    )
    assert eligible is False
    assert reasons == [
        f"mandatory_safety_gate_failed:{runner.FRESH_FAMILY_IDS[1]}:unsupported_recall"
    ]


def test_incomplete_prediction_coverage_is_infrastructure_failure() -> None:
    with pytest.raises(RuntimeError, match="prediction_coverage_exact"):
        runner.determine_candidate_eligibility(
            {"gates": []},
            completed_group_folds=5,
            group_prediction_count=9607,
            fresh_prediction_count=640,
            fresh_family_prediction_counts=[320, 320],
        )


@pytest.mark.parametrize(
    ("overrides", "winner"),
    [
        ({"worst": 0.71}, "alternative"),
        ({"mean": 0.76}, "alternative"),
        ({"pooled": 0.77}, "alternative"),
        ({"group_macro": 0.81}, "alternative"),
        ({"protected_fpr": 0.004}, "alternative"),
        ({"protected_recall": 0.91}, "alternative"),
        ({"unsupported_recall": 0.91}, "alternative"),
        ({"complexity_rank": 0}, "alternative"),
    ],
)
def test_lexicographic_selection_applies_each_numeric_criterion_in_order(
    contract: dict[str, Any], overrides: dict[str, Any], winner: str
) -> None:
    baseline = selection_result("baseline")
    alternative = selection_result("alternative", **overrides)
    assert runner.select_candidate([baseline, alternative], contract)[
        "selected_candidate_id"
    ] == winner


def test_lexical_candidate_id_is_the_final_tie_breaker(
    contract: dict[str, Any],
) -> None:
    first = selection_result("z-candidate")
    second = selection_result("a-candidate")
    assert runner.select_candidate([first, second], contract)[
        "selected_candidate_id"
    ] == "a-candidate"


def test_ineligible_candidate_can_never_win(contract: dict[str, Any]) -> None:
    ineligible = selection_result("ineligible", eligible=False, worst=1.0)
    eligible = selection_result("eligible", worst=0.1)
    assert runner.select_candidate([ineligible, eligible], contract)[
        "selected_candidate_id"
    ] == "eligible"


def test_no_acceptable_candidate_never_forces_or_weakens(
    contract: dict[str, Any],
) -> None:
    selection = runner.select_candidate(
        [selection_result("failed", eligible=False)], contract
    )
    assert selection["selection_status"] == "NO_ACCEPTABLE_CANDIDATE"
    assert selection["selected_candidate"] is None
    assert selection["selected_candidate_id"] is None
    assert selection["winner_forced"] is False
    assert selection["gates_weakened"] is False
    assert selection["step29i_authorized"] is False


def write_synthetic_cache(inputs: runner.LoadedInputs) -> dict[str, Any]:
    matrices: dict[str, np.ndarray] = {}
    for key, count in (
        ("development", len(inputs.development_examples)),
        ("fresh_evaluation", len(inputs.fresh_examples)),
    ):
        matrix = np.zeros((count, 384), dtype=np.float32)
        matrix[:, 0] = 1.0
        matrices[key] = matrix
    runner.BGE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(runner.BGE_CACHE_PATH, **matrices)
    manifest = runner.expected_bge_cache_manifest(
        inputs,
        cache_file_sha256=runner.sha256_file(runner.BGE_CACHE_PATH),
        fastembed_version="synthetic",
    )
    runner.BGE_CACHE_MANIFEST_PATH.write_bytes(runner.stable_json_bytes(manifest))
    return manifest


def test_bge_cache_is_bound_to_both_ordered_populations(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    contract: dict[str, Any],
    step29g_contract: dict[str, Any],
) -> None:
    inputs = synthetic_inputs(contract, step29g_contract)
    patch_output_paths(monkeypatch, tmp_path)
    manifest = write_synthetic_cache(inputs)
    matrices, validated = runner.validate_bge_cache(inputs)
    assert validated == manifest
    assert [row["matrix_key"] for row in manifest["populations"]] == [
        "development",
        "fresh_evaluation",
    ]
    assert matrices["development"].shape[0] == len(inputs.development_examples)
    assert matrices["fresh_evaluation"].shape[0] == 640


def test_stale_step29g_or_other_population_cache_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    contract: dict[str, Any],
    step29g_contract: dict[str, Any],
) -> None:
    inputs = synthetic_inputs(contract, step29g_contract)
    patch_output_paths(monkeypatch, tmp_path)
    manifest = write_synthetic_cache(inputs)
    manifest["development_dataset_sha256"] = "0" * 64
    runner.BGE_CACHE_MANIFEST_PATH.write_bytes(runner.stable_json_bytes(manifest))
    with pytest.raises(ValueError, match="stale or incompatible"):
        runner.validate_bge_cache(inputs)


def test_result_artifacts_are_create_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    contract: dict[str, Any],
    step29g_contract: dict[str, Any],
) -> None:
    inputs = synthetic_inputs(contract, step29g_contract)
    patch_output_paths(monkeypatch, tmp_path)
    runner.RESULTS_PATH.parent.mkdir(parents=True)
    runner.RESULTS_PATH.write_text("existing\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing overwrite"):
        runner.run_model_selection(inputs)


def test_historical_step29h_result_paths_are_untouched() -> None:
    historical = {
        "v2c6_source_aware_model_selection_results.json",
        "v2c6_source_aware_model_selection_results.manifest.json",
    }
    assert runner.RESULTS_PATH.name not in historical
    assert runner.RESULTS_MANIFEST_PATH.name not in historical


def test_preflight_does_no_model_work_and_writes_nothing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    contract: dict[str, Any],
    step29g_contract: dict[str, Any],
) -> None:
    inputs = synthetic_inputs(contract, step29g_contract)
    patch_output_paths(monkeypatch, tmp_path)
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
    assert report["files_written"] is False
    assert report["fresh_evaluation_used_for_fitting"] is False
    assert report["step29i_authorized"] is False
    assert list(tmp_path.rglob("*")) == []


def test_results_reject_raw_development_or_fresh_text(
    contract: dict[str, Any], step29g_contract: dict[str, Any]
) -> None:
    inputs = synthetic_inputs(contract, step29g_contract)
    records = (*inputs.development_examples, *inputs.fresh_examples)
    runner.validate_text_free({"safe": "metadata"}, records)
    with pytest.raises(ValueError, match="raw example text"):
        runner.validate_text_free({"diagnostic": records[-1]["text"]}, records)


def test_result_manifest_keeps_runtime_and_step29i_governance_false() -> None:
    payload = {
        "phase": "test",
        "contract": {},
        "frozen_inputs": {},
        "split_audit": {"audit_sha256": "a" * 64},
        "candidate_count": 4,
        "completed_candidate_count": 4,
        "selection": {
            "selection_status": "NO_ACCEPTABLE_CANDIDATE",
            "selected_candidate_id": None,
        },
    }
    manifest = runner.build_results_manifest(payload, b"{}\n")
    assert manifest["winner_forced"] is False
    assert manifest["gates_weakened"] is False
    assert manifest["fresh_evaluation_used_for_fitting"] is False
    assert manifest["final_holdout_accessed"] is False
    assert manifest["runtime_behavior_changed"] is False
    assert manifest["step29i_authorized"] is False


def test_stable_json_serialization_is_deterministic() -> None:
    first = runner.stable_json_bytes({"b": 2, "a": [3, 1]})
    second = runner.stable_json_bytes({"a": [3, 1], "b": 2})
    assert first == second
    assert runner.sha256_bytes(first) == runner.sha256_bytes(second)
