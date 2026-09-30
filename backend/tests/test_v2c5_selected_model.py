from __future__ import annotations

import inspect
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from sklearn.svm import LinearSVC

from scripts import prepare_v2c5_selected_model as preparation

ROOT = Path(__file__).resolve().parents[2]
ML_DIRECTORY = ROOT / "data/evals/v2/ml"
CONTRACT_PATH = ML_DIRECTORY / "v2c5_model_selection_contract.json"
RESULTS_PATH = ML_DIRECTORY / "v2c5_model_selection_results.json"
RESULTS_MANIFEST_PATH = (
    ML_DIRECTORY / "v2c5_model_selection_results.manifest.json"
)
FINAL_HOLDOUT_PATH = ML_DIRECTORY / "v2c5_final_holdout.json"
FINAL_HOLDOUT_MANIFEST_PATH = (
    ML_DIRECTORY / "v2c5_final_holdout.manifest.json"
)


@pytest.fixture(scope="module")
def contract() -> dict[str, Any]:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def results() -> dict[str, Any]:
    return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def results_manifest() -> dict[str, Any]:
    return json.loads(RESULTS_MANIFEST_PATH.read_text(encoding="utf-8"))


def synthetic_inputs(contract: dict[str, Any]) -> dict[str, Any]:
    labels = contract["taxonomy"]["intent_label_order"]
    examples = [
        {
            "example_id": f"synthetic:{index:02d}",
            "intent": label,
            "text": f"synthetic text {index}",
        }
        for index, label in enumerate(labels)
    ]
    return {
        "contract": contract,
        "examples": examples,
        "intent_counts": dict.fromkeys(labels, 1),
        "selected_configuration": preparation.expected_selected_configuration(
            contract
        ),
    }


def test_frozen_results_supply_the_selected_candidate(
    contract: dict[str, Any], results: dict[str, Any]
) -> None:
    selected = preparation.validate_selected_candidate(results, contract)
    assert selected == results["selection"]["selected_candidate_configuration"]
    assert selected["candidate_id"] == (
        "BGE_SMALL_LINEAR_SVC__C=4.0__class_weight=none"
    )
    assert results["selection"]["eligible_candidate_count"] == 13
    assert results["selection"]["step23_permitted"] is True


def test_successful_model_selection_is_required(
    contract: dict[str, Any], results: dict[str, Any]
) -> None:
    changed = deepcopy(results)
    changed["selection"]["model_selection_success"] = False
    with pytest.raises(ValueError, match="successful Step 22"):
        preparation.validate_selected_candidate(changed, contract)


def test_unexpected_selected_id_or_configuration_is_rejected(
    contract: dict[str, Any], results: dict[str, Any]
) -> None:
    changed_id = deepcopy(results)
    changed_id["selection"]["selected_candidate_id"] = "unexpected"
    with pytest.raises(ValueError, match="unexpected candidate"):
        preparation.validate_selected_candidate(changed_id, contract)

    changed_configuration = deepcopy(results)
    changed_configuration["selection"]["selected_candidate_configuration"][
        "classifier"
    ]["parameters"]["C"] = 1.0
    with pytest.raises(ValueError, match="differs from the frozen contract"):
        preparation.validate_selected_candidate(changed_configuration, contract)


def test_expected_selected_configuration_is_exact(
    contract: dict[str, Any], results: dict[str, Any]
) -> None:
    configuration = preparation.validate_selected_candidate(results, contract)
    assert configuration["hyperparameters"] == {
        "C": 4.0,
        "class_weight": None,
    }
    assert configuration["classifier"]["parameters"] == (
        preparation.EXPECTED_CLASSIFIER_PARAMETERS
    )
    assert configuration["representation"] == preparation.EXPECTED_REPRESENTATION
    assert configuration["representation"] == {
        "dimensions": 384,
        "embedding_method": "passage_embed",
        "fine_tuning": False,
        "implementation": "FastEmbed",
        "l2_normalized": True,
        "model_identifier": "BAAI/bge-small-en-v1.5",
    }


def test_real_preflight_uses_all_8198_rows_and_writes_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def prohibited(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("preflight attempted model work or a write")

    monkeypatch.setattr(preparation, "write_bytes_atomically", prohibited)
    monkeypatch.setattr(preparation.joblib, "dump", prohibited)
    monkeypatch.setattr(
        preparation.model_selection, "prepare_bge_cache", prohibited
    )
    report = preparation.preflight()
    assert report["development_record_count"] == 8198
    assert report["intent_count"] == 16
    assert report["fit_performed"] is False
    assert report["embeddings_generated"] is False
    assert report["files_written"] is False
    assert report["final_holdout_accessed"] is False
    assert report["selection_step23_permitted"] is True
    assert "step23_permitted" not in report


def test_full_fit_uses_one_classifier_and_no_cv(
    monkeypatch: pytest.MonkeyPatch, contract: dict[str, Any]
) -> None:
    inputs = synthetic_inputs(contract)
    labels = contract["taxonomy"]["intent_label_order"]
    embeddings = np.zeros((len(labels), 384), dtype=np.float32)
    embeddings[:, 0] = 1.0
    fit_batches: list[tuple[np.ndarray, np.ndarray]] = []

    class RecordingLinearSVC(LinearSVC):
        def fit(
            self, features: np.ndarray, target: np.ndarray
        ) -> RecordingLinearSVC:
            fit_batches.append((features.copy(), target.copy()))
            self.classes_ = np.asarray(sorted(set(target.tolist())), dtype=object)
            self.n_features_in_ = features.shape[1]
            self.coef_ = np.zeros((len(self.classes_), features.shape[1]))
            self.intercept_ = np.zeros(len(self.classes_))
            return self

    monkeypatch.setattr(preparation, "LinearSVC", RecordingLinearSVC)
    classifier = preparation.fit_full_development_classifier(
        embeddings,
        inputs["examples"],
        inputs["selected_configuration"],
        labels,
    )
    assert isinstance(classifier, RecordingLinearSVC)
    assert len(fit_batches) == 1
    assert fit_batches[0][0].shape == (16, 384)
    assert fit_batches[0][1].tolist() == labels
    fit_source = inspect.getsource(preparation.fit_full_development_classifier)
    assert "StratifiedGroupKFold" not in fit_source
    assert "select_candidate" not in fit_source
    assert "threshold" not in fit_source


def test_embedding_input_requires_exact_384_dimensions(
    contract: dict[str, Any]
) -> None:
    inputs = synthetic_inputs(contract)
    embeddings = np.zeros((16, 383), dtype=np.float32)
    with pytest.raises(ValueError, match="embedding shape"):
        preparation.validate_embedding_input(
            embeddings,
            inputs["examples"],
            inputs["selected_configuration"],
        )


def test_classifier_requires_exact_16_frozen_labels(
    contract: dict[str, Any]
) -> None:
    configuration = preparation.expected_selected_configuration(contract)
    classifier = LinearSVC(**configuration["classifier"]["parameters"])
    classifier.classes_ = np.asarray(
        contract["taxonomy"]["intent_label_order"][:-1], dtype=object
    )
    classifier.n_features_in_ = 384
    classifier.coef_ = np.zeros((15, 384))
    classifier.intercept_ = np.zeros(15)
    with pytest.raises(ValueError, match="frozen label order"):
        preparation.validate_fitted_classifier(
            classifier,
            configuration,
            contract["taxonomy"]["intent_label_order"],
        )


def test_compatible_bge_cache_is_reused(
    monkeypatch: pytest.MonkeyPatch, contract: dict[str, Any]
) -> None:
    inputs = synthetic_inputs(contract)
    embeddings = np.zeros((16, 384), dtype=np.float32)
    manifest = {"cache": "compatible"}
    calls: list[tuple[Any, Any]] = []

    def reuse(examples: Any, received_contract: Any) -> tuple[Any, Any]:
        calls.append((examples, received_contract))
        return embeddings, manifest

    monkeypatch.setattr(preparation.model_selection, "prepare_bge_cache", reuse)
    loaded_embeddings, loaded_manifest = (
        preparation.load_or_create_development_embeddings(inputs)
    )
    assert loaded_embeddings is embeddings
    assert loaded_manifest is manifest
    assert calls == [(inputs["examples"], contract)]


def test_stale_bge_cache_rejection_is_not_bypassed(
    monkeypatch: pytest.MonkeyPatch, contract: dict[str, Any]
) -> None:
    inputs = synthetic_inputs(contract)

    def reject(*args: Any, **kwargs: Any) -> None:
        raise ValueError("stale or mismatched BGE cache")

    monkeypatch.setattr(preparation.model_selection, "prepare_bge_cache", reject)
    with pytest.raises(ValueError, match="stale or mismatched"):
        preparation.load_or_create_development_embeddings(inputs)


def patch_manifest_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ml_directory = tmp_path / "data/evals/v2/ml"
    script_path = tmp_path / preparation.SCRIPT_RELATIVE_PATH
    ml_directory.mkdir(parents=True)
    script_path.parent.mkdir(parents=True)
    script_path.write_text("# synthetic preparation script\n", encoding="utf-8")
    monkeypatch.setattr(preparation, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(preparation, "ML_DIRECTORY", ml_directory)
    monkeypatch.setattr(preparation, "CONTRACT_PATH", ml_directory / "contract.json")
    monkeypatch.setattr(preparation, "RESULTS_PATH", ml_directory / "results.json")
    monkeypatch.setattr(
        preparation,
        "RESULTS_MANIFEST_PATH",
        ml_directory / "results.manifest.json",
    )
    monkeypatch.setattr(
        preparation,
        "MODEL_ARTIFACT_PATH",
        ml_directory / "local/selected.joblib",
    )
    monkeypatch.setattr(
        preparation,
        "MODEL_MANIFEST_PATH",
        ml_directory / "selected.manifest.json",
    )
    preparation.CONTRACT_PATH.write_text("{}\n", encoding="utf-8")
    preparation.RESULTS_PATH.write_text("{}\n", encoding="utf-8")
    preparation.RESULTS_MANIFEST_PATH.write_text("{}\n", encoding="utf-8")
    preparation.MODEL_ARTIFACT_PATH.parent.mkdir(parents=True)
    preparation.MODEL_ARTIFACT_PATH.write_bytes(b"trusted local classifier")


def test_manifest_persists_classifier_hash_and_is_text_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    contract: dict[str, Any],
) -> None:
    patch_manifest_paths(monkeypatch, tmp_path)
    inputs = synthetic_inputs(contract)
    inputs["examples"] = [
        {"text": f"private development text {index}"} for index in range(8198)
    ]
    cache_lineage = {"cache_sha256": "a" * 64, "dimensions": 384}
    manifest = preparation.build_model_manifest(inputs, cache_lineage)
    assert manifest["local_classifier_artifact"]["sha256"] == (
        preparation.sha256_file(preparation.MODEL_ARTIFACT_PATH)
    )
    assert manifest["development_record_count"] == 8198
    preparation.validate_text_free_manifest(manifest, inputs["examples"])
    serialized = preparation.stable_json_bytes(manifest)
    assert b"private development text" not in serialized
    assert manifest["runtime_behavior_changed"] is False
    assert manifest["final_model_acceptance_claimed"] is False
    assert manifest["step23_permitted"] is True
    assert manifest["next_required"] == (
        "v2c5_once_only_final_holdout_evaluation"
    )


def test_final_holdout_dataset_opening_and_hashing_are_prohibited() -> None:
    with pytest.raises(PermissionError, match="must never open or hash"):
        preparation.read_json(FINAL_HOLDOUT_PATH)
    with pytest.raises(PermissionError, match="must never open or hash"):
        preparation.sha256_file(FINAL_HOLDOUT_PATH)


def test_preflight_reads_only_the_final_holdout_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_read_json = preparation.read_json
    observed_paths: list[Path] = []

    def recording_read_json(path: Path) -> dict[str, Any]:
        observed_paths.append(path.resolve())
        return original_read_json(path)

    monkeypatch.setattr(preparation, "read_json", recording_read_json)
    report = preparation.preflight()
    assert report["final_holdout_accessed"] is False
    assert FINAL_HOLDOUT_MANIFEST_PATH.resolve() in observed_paths
    assert FINAL_HOLDOUT_PATH.resolve() not in observed_paths


def test_check_revalidates_cache_and_writes_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    contract: dict[str, Any],
) -> None:
    inputs = synthetic_inputs(contract)
    artifact_path = tmp_path / "selected.joblib"
    manifest_path = tmp_path / "selected.manifest.json"
    artifact_path.write_bytes(b"artifact")
    cache_lineage = {"cache_sha256": "a" * 64}
    manifest = {
        "embedding_cache_lineage": cache_lineage,
        "local_classifier_artifact": {"sha256": "b" * 64},
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    cache_checks: list[tuple[Any, Any]] = []

    def validate_cache(examples: Any, received_contract: Any) -> tuple[Any, Any]:
        cache_checks.append((examples, received_contract))
        return np.empty((0, 384), dtype=np.float32), {"cache": "valid"}

    def prohibited(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("check attempted a write")

    monkeypatch.setattr(preparation, "MODEL_ARTIFACT_PATH", artifact_path)
    monkeypatch.setattr(preparation, "MODEL_MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(preparation, "load_preconditions", lambda: inputs)
    monkeypatch.setattr(preparation, "validate_model_manifest", lambda *args: None)
    monkeypatch.setattr(
        preparation.model_selection, "validate_bge_cache", validate_cache
    )
    monkeypatch.setattr(
        preparation, "embedding_cache_lineage", lambda *args: cache_lineage
    )
    monkeypatch.setattr(preparation.joblib, "load", lambda path: {})
    monkeypatch.setattr(
        preparation, "validate_artifact_payload", lambda *args: None
    )
    monkeypatch.setattr(preparation, "write_bytes_atomically", prohibited)
    monkeypatch.setattr(preparation.joblib, "dump", prohibited)
    before = {path: path.read_bytes() for path in (artifact_path, manifest_path)}
    report = preparation.check_selected_model()
    assert cache_checks == [(inputs["examples"], contract)]
    assert report["files_written"] is False
    assert {path: path.read_bytes() for path in before} == before


def test_cli_exposes_only_preflight_fit_and_check_modes() -> None:
    parser = preparation.build_parser()
    assert parser.parse_args(["--preflight"]).preflight is True
    assert parser.parse_args(["--fit"]).fit is True
    assert parser.parse_args(["--check"]).check is True
    with pytest.raises(SystemExit):
        parser.parse_args(["--preflight", "--fit"])
