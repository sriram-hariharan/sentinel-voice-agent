from __future__ import annotations

import json
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from sklearn.svm import LinearSVC

from scripts import run_v2c5_final_evaluation as runner

ROOT = Path(__file__).resolve().parents[2]
REAL_CONTRACT_PATH = ROOT / "data/evals/v2/ml/v2c5_final_evaluation_contract.json"


class RecordingLinearSVC(LinearSVC):
    prediction_values: np.ndarray
    predict_calls: int

    def predict(self, features: np.ndarray) -> np.ndarray:
        self.predict_calls += 1
        return self.prediction_values.copy()

    def fit(self, features: Any, labels: Any) -> RecordingLinearSVC:
        raise AssertionError("final evaluation must never fit or refit a classifier")


@dataclass
class SyntheticEvaluation:
    artifact_payload: dict[str, Any]
    classifier: RecordingLinearSVC
    contract: dict[str, Any]
    holdout_bytes: bytes
    holdout_payload: dict[str, Any]
    paths: runner.EvaluationPaths
    taxonomy: dict[str, Any]

    def artifact_loader(self, path: Path) -> dict[str, Any]:
        assert path == self.paths.classifier_artifact
        return self.artifact_payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(runner.stable_json_bytes(payload))


def synthetic_examples(
    labels: list[str], risk_by_intent: dict[str, str]
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for label in labels:
        for index in range(40):
            text = f"synthetic holdout text {label} {index:02d}"
            examples.append(
                {
                    "data_role": "final_holdout",
                    "example_id": f"synthetic:{len(examples):04d}",
                    "intent": label,
                    "risk": risk_by_intent[label],
                    "text": text,
                }
            )
    return examples


def make_classifier(labels: list[str]) -> RecordingLinearSVC:
    classifier = RecordingLinearSVC(**runner.EXPECTED_CLASSIFIER["parameters"])
    classifier.classes_ = np.asarray(labels, dtype=object)
    classifier.n_features_in_ = 384
    classifier.coef_ = np.zeros((16, 384), dtype=np.float64)
    classifier.intercept_ = np.zeros(16, dtype=np.float64)
    classifier.prediction_values = np.asarray([], dtype=object)
    classifier.predict_calls = 0
    return classifier


@pytest.fixture
def synthetic(tmp_path: Path) -> SyntheticEvaluation:
    ml_root = tmp_path / "data/evals/v2/ml"
    paths = runner.EvaluationPaths(
        repository_root=tmp_path,
        contract=ml_root / "v2c5_final_evaluation_contract.json",
        selected_model_manifest=ml_root / "v2c5_selected_model.manifest.json",
        classifier_artifact=ml_root / "local/v2c5_selected_classifier.joblib",
        final_holdout_manifest=ml_root / "v2c5_final_holdout.manifest.json",
        final_holdout_contract=ml_root / "v2c5_final_holdout_contract.json",
        final_holdout_dataset=ml_root / "v2c5_final_holdout.json",
        model_selection_results=ml_root / "v2c5_model_selection_results.json",
        taxonomy=ml_root / "v2c5_taxonomy_freeze.json",
        state=ml_root / "v2c5_final_evaluation_state.json",
        results=ml_root / "v2c5_final_evaluation_results.json",
        results_manifest=(
            ml_root / "v2c5_final_evaluation_results.manifest.json"
        ),
        runner=tmp_path / runner.SCRIPT_RELATIVE_PATH,
    )
    ml_root.mkdir(parents=True)
    paths.runner.parent.mkdir(parents=True)
    paths.runner.write_text("# synthetic final-evaluation runner\n", encoding="utf-8")
    paths.classifier_artifact.parent.mkdir(parents=True)
    paths.classifier_artifact.write_bytes(b"synthetic trusted-local classifier")
    paths.model_selection_results.write_bytes(b'{"synthetic":true}\n')
    paths.final_holdout_contract.write_bytes(b'{"synthetic":true}\n')

    contract = json.loads(REAL_CONTRACT_PATH.read_text(encoding="utf-8"))
    labels = contract["model_under_test"]["class_labels"]
    protected_labels = contract["safety_gates"]["protected_write_intents"]
    protected = set(protected_labels)
    risk_by_intent = {
        label: "PROTECTED_WRITE" if label in protected else "PRIVATE_READ"
        for label in labels
    }
    taxonomy = {
        "final_taxonomy_frozen": True,
        "intent_label_order": labels,
        "protected_write_intents": protected_labels,
        "risk_by_intent": risk_by_intent,
        "schema_version": "v2c5-taxonomy-freeze.v1",
    }
    write_json(paths.taxonomy, taxonomy)

    examples = synthetic_examples(labels, risk_by_intent)
    holdout_payload = {
        "example_count": 640,
        "examples": examples,
        "schema_version": runner.HOLDOUT_SCHEMA_VERSION,
    }
    holdout_bytes = runner.stable_json_bytes(holdout_payload)
    holdout_manifest = {
        "counts": {
            "intent_counts": dict.fromkeys(labels, 40),
            "non_protected_count": 480,
            "protected_write_positive_count": 160,
            "total_count": 640,
        },
        "dataset": {
            "path": runner.display_path(paths.final_holdout_dataset, paths),
            "sha256": runner.sha256_bytes(holdout_bytes),
        },
        "execution_status": {
            "classifier_evaluation_performed": False,
            "final_holdout_evaluated": False,
            "final_holdout_frozen": True,
        },
        "schema_version": "v2c5-final-holdout-manifest.v1",
        "taxonomy_intent_labels": labels,
    }
    write_json(paths.final_holdout_manifest, holdout_manifest)

    classifier_specification = {
        "path": runner.display_path(paths.classifier_artifact, paths),
        "runtime_eligible": False,
        "schema_version": runner.SELECTED_ARTIFACT_SCHEMA_VERSION,
        "sha256": runner.sha256_file(paths.classifier_artifact),
        "size_bytes": paths.classifier_artifact.stat().st_size,
        "trusted_local": True,
    }
    model_selection_specification = {
        "path": runner.display_path(paths.model_selection_results, paths),
        "sha256": runner.sha256_file(paths.model_selection_results),
    }
    development_specification = {
        "path": "data/evals/v2/ml/synthetic_development.json",
        "sha256": "d" * 64,
    }
    taxonomy_specification = {
        "path": runner.display_path(paths.taxonomy, paths),
        "sha256": runner.sha256_file(paths.taxonomy),
    }
    selected_manifest = {
        "class_labels": labels,
        "classifier_config": runner.EXPECTED_CLASSIFIER,
        "classifier_training_performed": True,
        "development_dataset": development_specification,
        "final_holdout_accessed": False,
        "final_holdout_evaluated": False,
        "final_holdout_inference_performed": False,
        "final_model_acceptance_claimed": False,
        "fit_completed": True,
        "local_classifier_artifact": classifier_specification,
        "representation_config": runner.EXPECTED_REPRESENTATION,
        "runtime_behavior_changed": False,
        "schema_version": runner.SELECTED_MODEL_SCHEMA_VERSION,
        "selected_candidate_id": runner.EXPECTED_CANDIDATE_ID,
        "source_model_selection_results": model_selection_specification,
        "step23_permitted": True,
        "taxonomy": taxonomy_specification,
        "threshold_tuning_performed": False,
    }
    write_json(paths.selected_model_manifest, selected_manifest)

    sources = contract["source_artifacts"]
    sources["development_dataset"] = development_specification
    sources["final_holdout_contract"] = {
        "path": runner.display_path(paths.final_holdout_contract, paths),
        "sha256": runner.sha256_file(paths.final_holdout_contract),
    }
    sources["final_holdout_dataset_declaration"] = {
        "hash_source": runner.display_path(paths.final_holdout_manifest, paths),
        "opened_or_hashed_by_step23a": False,
        "path": runner.display_path(paths.final_holdout_dataset, paths),
        "sha256": runner.sha256_bytes(holdout_bytes),
    }
    sources["final_holdout_manifest"] = {
        "path": runner.display_path(paths.final_holdout_manifest, paths),
        "sha256": runner.sha256_file(paths.final_holdout_manifest),
    }
    sources["model_selection_results"] = model_selection_specification
    sources["selected_classifier_artifact"] = classifier_specification
    sources["selected_model_manifest"] = {
        "path": runner.display_path(paths.selected_model_manifest, paths),
        "sha256": runner.sha256_file(paths.selected_model_manifest),
    }
    sources["taxonomy"] = taxonomy_specification
    write_json(paths.contract, contract)

    classifier = make_classifier(labels)
    artifact_payload = {
        "artifact_schema_version": runner.SELECTED_ARTIFACT_SCHEMA_VERSION,
        "classifier": classifier,
        "classifier_config": runner.EXPECTED_CLASSIFIER,
        "classifier_training_performed": True,
        "development_dataset_sha256": development_specification["sha256"],
        "final_holdout_accessed": False,
        "final_holdout_evaluated": False,
        "final_holdout_inference_performed": False,
        "final_model_acceptance_claimed": False,
        "fit_completed": True,
        "intent_label_order": labels,
        "representation": runner.EXPECTED_REPRESENTATION,
        "runtime_authority": False,
        "runtime_behavior_changed": False,
        "selected_candidate_id": runner.EXPECTED_CANDIDATE_ID,
        "source_model_selection_results_sha256": model_selection_specification[
            "sha256"
        ],
        "taxonomy_sha256": taxonomy_specification["sha256"],
        "threshold_tuning_performed": False,
        "trusted_local_artifact": True,
    }
    return SyntheticEvaluation(
        artifact_payload=artifact_payload,
        classifier=classifier,
        contract=contract,
        holdout_bytes=holdout_bytes,
        holdout_payload=holdout_payload,
        paths=paths,
        taxonomy=taxonomy,
    )


def initialize(synthetic: SyntheticEvaluation) -> dict[str, Any]:
    runner.initialize_state(
        synthetic.paths,
        artifact_loader=synthetic.artifact_loader,
    )
    return runner.read_json_object(synthetic.paths.state)


def perfect_predictions(synthetic: SyntheticEvaluation) -> np.ndarray:
    return np.asarray(
        [record["intent"] for record in synthetic.holdout_payload["examples"]],
        dtype=object,
    )


def normalized_embeddings(count: int = 640, dimensions: int = 384) -> np.ndarray:
    values = np.zeros((count, dimensions), dtype=np.float32)
    values[:, 0] = 1.0
    return values


def test_preflight_never_opens_final_holdout(
    synthetic: SyntheticEvaluation,
) -> None:
    assert not synthetic.paths.final_holdout_dataset.exists()
    report = runner.preflight(
        synthetic.paths,
        artifact_loader=synthetic.artifact_loader,
    )
    assert report["status"] == "ready"
    assert report["state"] == "absent"
    assert report["final_holdout_accessed"] is False
    assert report["embeddings_generated"] is False
    assert report["inference_performed"] is False
    assert report["files_written"] is False
    assert not synthetic.paths.final_holdout_dataset.exists()


def test_initialize_state_is_deterministic_and_contains_exact_lineage(
    synthetic: SyntheticEvaluation,
) -> None:
    first = initialize(synthetic)
    first_bytes = synthetic.paths.state.read_bytes()
    assert first["state"] == "not_started"
    assert first["evaluation_attempt_count"] == 0
    assert first["source_lineage"] == runner.source_lineage(
        synthetic.contract, synthetic.paths
    )
    assert first["source_lineage"] == {
        "classifier_artifact_sha256": synthetic.contract["source_artifacts"][
            "selected_classifier_artifact"
        ]["sha256"],
        "declared_final_holdout_dataset_sha256": runner.sha256_bytes(
            synthetic.holdout_bytes
        ),
        "development_dataset_sha256": synthetic.contract["source_artifacts"][
            "development_dataset"
        ]["sha256"],
        "final_evaluation_contract_sha256": runner.sha256_file(
            synthetic.paths.contract
        ),
        "final_holdout_contract_sha256": runner.sha256_file(
            synthetic.paths.final_holdout_contract
        ),
        "final_holdout_manifest_sha256": runner.sha256_file(
            synthetic.paths.final_holdout_manifest
        ),
        "model_selection_results_sha256": runner.sha256_file(
            synthetic.paths.model_selection_results
        ),
        "runner_sha256": runner.sha256_file(synthetic.paths.runner),
        "selected_model_manifest_sha256": runner.sha256_file(
            synthetic.paths.selected_model_manifest
        ),
        "taxonomy_sha256": runner.sha256_file(synthetic.paths.taxonomy),
    }
    synthetic.paths.state.unlink()
    initialize(synthetic)
    assert synthetic.paths.state.read_bytes() == first_bytes


def test_initialize_refuses_existing_or_incompatible_state(
    synthetic: SyntheticEvaluation,
) -> None:
    initialize(synthetic)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        runner.initialize_state(
            synthetic.paths,
            artifact_loader=synthetic.artifact_loader,
        )
    changed = runner.read_json_object(synthetic.paths.state)
    changed["source_lineage"]["taxonomy_sha256"] = "0" * 64
    write_json(synthetic.paths.state, changed)
    with pytest.raises(ValueError, match="state lineage"):
        runner.initialize_state(
            synthetic.paths,
            artifact_loader=synthetic.artifact_loader,
        )


def test_evaluate_requires_initialized_not_started_state(
    synthetic: SyntheticEvaluation,
) -> None:
    holdout_calls = 0

    def prohibited_loader(path: Path) -> bytes:
        nonlocal holdout_calls
        holdout_calls += 1
        raise AssertionError("evaluation opened holdout without initialized state")

    with pytest.raises(FileNotFoundError, match="initialized.*state"):
        runner.evaluate(
            synthetic.paths,
            artifact_loader=synthetic.artifact_loader,
            holdout_loader=prohibited_loader,
        )
    assert holdout_calls == 0


@pytest.mark.parametrize("state_name", ["started", "completed", "failed_after_access"])
def test_evaluate_refuses_every_consumed_state(
    synthetic: SyntheticEvaluation,
    state_name: str,
) -> None:
    keyword_arguments: dict[str, Any] = {}
    if state_name == "completed":
        keyword_arguments = {
            "final_holdout_accessed": True,
            "final_holdout_evaluated": True,
            "final_holdout_inference_performed": True,
            "results": {
                "results_manifest_sha256": "a" * 64,
                "results_sha256": "b" * 64,
            },
        }
    elif state_name == "failed_after_access":
        keyword_arguments = {
            "failure": {
                "exception_type": "RuntimeError",
                "message_sha256": "c" * 64,
                "stage": "synthetic",
            },
            "final_holdout_accessed": True,
        }
    state = runner.build_state_payload(
        synthetic.contract,
        synthetic.paths,
        state_name,
        **keyword_arguments,
    )
    write_json(synthetic.paths.state, state)
    holdout_calls = 0

    def prohibited_loader(path: Path) -> bytes:
        nonlocal holdout_calls
        holdout_calls += 1
        raise AssertionError("consumed state reopened holdout")

    with pytest.raises(RuntimeError, match=f"state={state_name}"):
        runner.evaluate(
            synthetic.paths,
            artifact_loader=synthetic.artifact_loader,
            holdout_loader=prohibited_loader,
        )
    assert holdout_calls == 0


def test_durable_started_state_precedes_first_holdout_open(
    synthetic: SyntheticEvaluation,
) -> None:
    initialize(synthetic)
    synthetic.classifier.prediction_values = perfect_predictions(synthetic)
    observed_states: list[str] = []

    def observing_loader(path: Path) -> bytes:
        assert path == synthetic.paths.final_holdout_dataset
        state = runner.read_json_object(synthetic.paths.state)
        observed_states.append(state["state"])
        assert state["state"] == "started"
        assert state["evaluation_attempt_count"] == 1
        return synthetic.holdout_bytes

    runner.evaluate(
        synthetic.paths,
        artifact_loader=synthetic.artifact_loader,
        holdout_loader=observing_loader,
        embedder=lambda texts, representation: normalized_embeddings(),
    )
    assert observed_states == ["started"]
    assert runner.read_json_object(synthetic.paths.state)["state"] == "completed"


def test_failure_before_started_leaves_holdout_unopened_and_state_not_started(
    synthetic: SyntheticEvaluation,
) -> None:
    initialize(synthetic)
    synthetic.paths.classifier_artifact.write_bytes(b"changed artifact")
    holdout_calls = 0

    def prohibited_loader(path: Path) -> bytes:
        nonlocal holdout_calls
        holdout_calls += 1
        return synthetic.holdout_bytes

    with pytest.raises(ValueError, match="artifact hash"):
        runner.evaluate(
            synthetic.paths,
            artifact_loader=synthetic.artifact_loader,
            holdout_loader=prohibited_loader,
        )
    assert holdout_calls == 0
    assert runner.read_json_object(synthetic.paths.state)["state"] == "not_started"


def test_failure_after_access_records_failed_after_access(
    synthetic: SyntheticEvaluation,
) -> None:
    initialize(synthetic)

    def failing_loader(path: Path) -> bytes:
        assert runner.read_json_object(synthetic.paths.state)["state"] == "started"
        raise OSError("synthetic access failure")

    with pytest.raises(OSError, match="synthetic access failure"):
        runner.evaluate(
            synthetic.paths,
            artifact_loader=synthetic.artifact_loader,
            holdout_loader=failing_loader,
        )
    state = runner.read_json_object(synthetic.paths.state)
    assert state["state"] == "failed_after_access"
    assert state["final_holdout_accessed"] is True
    assert state["final_holdout_inference_performed"] is False
    assert state["failure"]["stage"] == "opening_final_holdout"
    assert "synthetic access failure" not in json.dumps(state)


def test_exact_holdout_sha_is_validated_after_started_transition(
    synthetic: SyntheticEvaluation,
) -> None:
    initialize(synthetic)
    with pytest.raises(ValueError, match="hash differs"):
        runner.evaluate(
            synthetic.paths,
            artifact_loader=synthetic.artifact_loader,
            holdout_loader=lambda path: synthetic.holdout_bytes + b" ",
        )
    state = runner.read_json_object(synthetic.paths.state)
    assert state["state"] == "failed_after_access"
    assert state["failure"]["stage"] == "validating_final_holdout_hash"


def test_exact_holdout_population_and_unique_ids_are_validated(
    synthetic: SyntheticEvaluation,
) -> None:
    records = runner.validate_holdout_payload(
        synthetic.holdout_payload,
        synthetic.contract,
        synthetic.taxonomy,
    )
    assert len(records) == 640
    assert len({record["example_id"] for record in records}) == 640
    assert Counter(record["intent"] for record in records) == Counter(
        synthetic.contract["holdout"]["intent_counts"]
    )

    missing = deepcopy(synthetic.holdout_payload)
    missing["examples"].pop()
    missing["example_count"] = 639
    with pytest.raises(ValueError, match="record count"):
        runner.validate_holdout_payload(
            missing, synthetic.contract, synthetic.taxonomy
        )

    duplicate = deepcopy(synthetic.holdout_payload)
    duplicate["examples"][1]["example_id"] = duplicate["examples"][0][
        "example_id"
    ]
    with pytest.raises(ValueError, match="duplicate"):
        runner.validate_holdout_payload(
            duplicate, synthetic.contract, synthetic.taxonomy
        )


def test_protected_and_non_protected_populations_are_validated(
    synthetic: SyntheticEvaluation,
) -> None:
    changed = deepcopy(synthetic.contract)
    changed["safety_gates"]["protected_write_intents"] = [
        "cancel_transfer",
        "close_account",
        "create_dispute",
    ]
    with pytest.raises(ValueError, match="protected-write population"):
        runner.validate_holdout_payload(
            synthetic.holdout_payload,
            changed,
            synthetic.taxonomy,
        )


def test_embedding_validation_requires_exact_640_by_384_l2_matrix(
    synthetic: SyntheticEvaluation,
) -> None:
    representation = synthetic.contract["model_under_test"]["representation"]
    runner.validate_embedding_matrix(normalized_embeddings(), 640, representation)
    with pytest.raises(ValueError, match="embedding shape"):
        runner.validate_embedding_matrix(
            normalized_embeddings(dimensions=383),
            640,
            representation,
        )
    non_normalized = normalized_embeddings()
    non_normalized[:, 0] = 2.0
    with pytest.raises(ValueError, match="not L2 normalized"):
        runner.validate_embedding_matrix(non_normalized, 640, representation)


def test_evaluate_embeds_once_predicts_once_and_never_fits(
    synthetic: SyntheticEvaluation,
) -> None:
    initialize(synthetic)
    synthetic.classifier.prediction_values = perfect_predictions(synthetic)
    embedding_calls = 0

    def fake_embedder(
        texts: list[str], representation: dict[str, Any]
    ) -> np.ndarray:
        nonlocal embedding_calls
        embedding_calls += 1
        assert len(texts) == 640
        assert representation == runner.EXPECTED_REPRESENTATION
        return normalized_embeddings()

    runner.evaluate(
        synthetic.paths,
        artifact_loader=synthetic.artifact_loader,
        holdout_loader=lambda path: synthetic.holdout_bytes,
        embedder=fake_embedder,
    )
    assert embedding_calls == 1
    assert synthetic.classifier.predict_calls == 1
    assert runner.read_json_object(synthetic.paths.state)["state"] == "completed"


def test_exact_metric_and_safety_formulas(
    synthetic: SyntheticEvaluation,
) -> None:
    gold = perfect_predictions(synthetic).tolist()
    predicted = list(gold)
    predicted[0] = "freeze_card"
    protected_index = gold.index("cancel_transfer")
    predicted[protected_index] = "close_account"
    unsupported_index = gold.index("unsupported_or_uncertain")
    predicted[unsupported_index] = "account_balance"

    metrics = runner.classification_metrics(gold, predicted, synthetic.contract)
    safety = runner.safety_metrics(gold, predicted, synthetic.contract)
    assert metrics["accuracy"] == pytest.approx(637 / 640)
    assert metrics["confusion_matrix"]["label_order"] == synthetic.contract[
        "model_under_test"
    ]["class_labels"]
    assert safety["protected_write_false_positive_count"] == 1
    assert safety["protected_write_false_positive_rate"] == pytest.approx(1 / 480)
    assert safety["exact_protected_write_correct_count"] == 159
    assert safety["exact_protected_write_recall"] == pytest.approx(159 / 160)
    assert safety["unsupported_or_uncertain_correct_count"] == 39
    assert safety["unsupported_or_uncertain_recall"] == pytest.approx(39 / 40)


def test_all_gate_pass_and_single_gate_failure_behavior(
    synthetic: SyntheticEvaluation,
) -> None:
    perfect = {
        "exact_protected_write_recall": 1.0,
        "protected_write_false_positive_rate": 0.0,
        "unsupported_or_uncertain_recall": 1.0,
    }
    passed = runner.apply_safety_gates(perfect, synthetic.contract)
    assert passed["all_gates_pass"] is True
    assert all(gate["passed"] for gate in passed["gates"].values())

    failed_metrics = dict(perfect)
    failed_metrics["protected_write_false_positive_rate"] = 0.0100001
    failed = runner.apply_safety_gates(failed_metrics, synthetic.contract)
    assert failed["all_gates_pass"] is False
    assert failed["gates"]["protected_write_false_positive_rate"][
        "threshold"
    ] == 0.01
    assert failed["gates"]["protected_write_false_positive_rate"][
        "passed"
    ] is False
    assert failed["gates"]["exact_protected_write_recall"]["passed"] is True

    failed_predictions = perfect_predictions(synthetic).tolist()
    failed_predictions[:5] = ["freeze_card"] * 5
    failed_results = runner.build_results_payload(
        {"contract": synthetic.contract},
        synthetic.paths,
        synthetic.holdout_payload["examples"],
        failed_predictions,
    )
    assert failed_results["all_mandatory_safety_gates_pass"] is False
    assert failed_results["final_model_acceptance_claimed"] is False
    assert failed_results["runtime_eligible"] is False


def test_results_are_aggregate_only_text_free_and_runtime_ineligible(
    synthetic: SyntheticEvaluation,
) -> None:
    predictions = perfect_predictions(synthetic).tolist()
    inputs = {"contract": synthetic.contract}
    results = runner.build_results_payload(
        inputs,
        synthetic.paths,
        synthetic.holdout_payload["examples"],
        predictions,
    )
    serialized = runner.stable_json_bytes(results)
    assert results["prediction_count"] == 640
    assert results["all_mandatory_safety_gates_pass"] is True
    assert results["final_model_acceptance_claimed"] is True
    assert results["runtime_behavior_changed"] is False
    assert results["runtime_eligible"] is False
    assert results["threshold_tuning_performed"] is False
    assert results["model_retraining_performed"] is False
    assert all(
        record["text"].encode("utf-8") not in serialized
        for record in synthetic.holdout_payload["examples"]
    )
    assert not runner.recursive_keys(results) & {
        "text",
        "texts",
        "utterance",
        "utterances",
        "raw_text",
    }


def test_check_results_validates_lineage_without_reopening_holdout(
    synthetic: SyntheticEvaluation,
) -> None:
    initialize(synthetic)
    synthetic.classifier.prediction_values = perfect_predictions(synthetic)
    runner.evaluate(
        synthetic.paths,
        artifact_loader=synthetic.artifact_loader,
        holdout_loader=lambda path: synthetic.holdout_bytes,
        embedder=lambda texts, representation: normalized_embeddings(),
    )
    assert not synthetic.paths.final_holdout_dataset.exists()
    report = runner.check_results(synthetic.paths)
    assert report["status"] == "valid"
    assert report["files_written"] is False
    assert report["inference_performed"] is False
    assert report["runtime_behavior_changed"] is False
    assert report["runtime_eligible"] is False
    assert not synthetic.paths.final_holdout_dataset.exists()

    manifest = runner.read_json_object(synthetic.paths.results_manifest)
    manifest["results"]["sha256"] = "0" * 64
    write_json(synthetic.paths.results_manifest, manifest)
    with pytest.raises(ValueError, match="result hash"):
        runner.check_results(synthetic.paths)


def test_cli_has_no_force_reset_or_retry_backdoor() -> None:
    parser = runner.build_parser()
    assert parser.parse_args(["--preflight"]).preflight is True
    assert parser.parse_args(["--initialize-state"]).initialize_state is True
    assert parser.parse_args(["--evaluate"]).evaluate is True
    assert parser.parse_args(["--check-results"]).check_results is True
    for prohibited in ("--force", "--reset", "--retry"):
        with pytest.raises(SystemExit):
            parser.parse_args([prohibited])
