"""Fit and freeze the already-selected V2-C5 development classifier.

This Step 22C module never opens or hashes the sealed V2-C5 final-holdout
dataset. It reads only its text-free manifest for governance. The ``fit`` mode
fits exactly one classifier on all frozen development examples; it performs no
cross-validation, candidate comparison, threshold tuning, or evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import tempfile
import warnings
from collections import Counter
from collections.abc import Sequence
from importlib import import_module, metadata
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.svm import LinearSVC

model_selection = import_module(
    "scripts.run_v2c5_model_selection"
    if __package__
    else "run_v2c5_model_selection"
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ML_DIRECTORY = REPOSITORY_ROOT / "data/evals/v2/ml"
CONTRACT_PATH = ML_DIRECTORY / "v2c5_model_selection_contract.json"
RESULTS_PATH = ML_DIRECTORY / "v2c5_model_selection_results.json"
RESULTS_MANIFEST_PATH = ML_DIRECTORY / "v2c5_model_selection_results.manifest.json"
MODEL_ARTIFACT_PATH = ML_DIRECTORY / "local/v2c5_selected_classifier.joblib"
MODEL_MANIFEST_PATH = ML_DIRECTORY / "v2c5_selected_model.manifest.json"
SCRIPT_RELATIVE_PATH = "scripts/prepare_v2c5_selected_model.py"
PROHIBITED_FINAL_HOLDOUT_RELATIVE_PATH = (
    "data/evals/v2/ml/v2c5_final_holdout.json"
)

EXPECTED_SELECTED_CANDIDATE_ID = (
    "BGE_SMALL_LINEAR_SVC__C=4.0__class_weight=none"
)
MANIFEST_SCHEMA_VERSION = "v2c5-selected-model-manifest.v1"
ARTIFACT_SCHEMA_VERSION = "v2c5-selected-classifier.v1"

EXPECTED_REPRESENTATION = {
    "dimensions": 384,
    "embedding_method": "passage_embed",
    "fine_tuning": False,
    "implementation": "FastEmbed",
    "l2_normalized": True,
    "model_identifier": "BAAI/bge-small-en-v1.5",
}
EXPECTED_CLASSIFIER_PARAMETERS = {
    "C": 4.0,
    "class_weight": None,
    "dual": "auto",
    "fit_intercept": True,
    "intercept_scaling": 1.0,
    "loss": "squared_hinge",
    "max_iter": 2000,
    "multi_class": "ovr",
    "penalty": "l2",
    "random_state": 20260930,
    "tol": 0.0001,
    "verbose": 0,
}


def stable_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def repository_path(relative_path: str) -> Path:
    path = (REPOSITORY_ROOT / relative_path).resolve()
    if not path.is_relative_to(REPOSITORY_ROOT.resolve()):
        raise ValueError(f"path escapes repository: {relative_path}")
    return path


def prohibited_final_holdout_path() -> Path:
    return repository_path(PROHIBITED_FINAL_HOLDOUT_RELATIVE_PATH)


def assert_step22c_path_allowed(path: Path) -> None:
    if path.resolve() == prohibited_final_holdout_path():
        raise PermissionError(
            "Step 22C must never open or hash the sealed V2-C5 final holdout"
        )


def sha256_file(path: Path) -> str:
    assert_step22c_path_allowed(path)
    return sha256_bytes(path.read_bytes())


def read_json(path: Path) -> dict[str, Any]:
    assert_step22c_path_allowed(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return payload


def write_bytes_atomically(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary_path = Path(handle.name)
        handle.write(content)
    try:
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def validate_hashed_source(specification: dict[str, Any], label: str) -> Path:
    if not {"path", "sha256"}.issubset(specification):
        raise ValueError(f"{label} must contain path and sha256")
    path = repository_path(specification["path"])
    actual = sha256_file(path)
    if actual != specification["sha256"]:
        raise ValueError(
            f"{label} hash mismatch: expected {specification['sha256']}, "
            f"got {actual}"
        )
    return path


def expected_selected_configuration(contract: dict[str, Any]) -> dict[str, Any]:
    candidate = next(
        (
            item
            for item in model_selection.expand_candidates(contract)
            if item["candidate_id"] == EXPECTED_SELECTED_CANDIDATE_ID
        ),
        None,
    )
    if candidate is None:
        raise ValueError("expected selected candidate is absent from the frozen matrix")
    configuration = model_selection.candidate_configuration(contract, candidate)
    if configuration != {
        "candidate_id": EXPECTED_SELECTED_CANDIDATE_ID,
        "family_id": "BGE_SMALL_LINEAR_SVC",
        "representation_id": "BGE_SMALL",
        "classifier_id": "LINEAR_SVC",
        "hyperparameters": {"C": 4.0, "class_weight": None},
        "representation": EXPECTED_REPRESENTATION,
        "classifier": {
            "class": "LinearSVC",
            "parameters": EXPECTED_CLASSIFIER_PARAMETERS,
        },
    }:
        raise ValueError("expected selected candidate definition changed")
    return configuration


def validate_selected_candidate(
    results: dict[str, Any], contract: dict[str, Any]
) -> dict[str, Any]:
    selection = results.get("selection", {})
    if selection.get("model_selection_success") is not True:
        raise ValueError("successful Step 22 model selection is required")
    if selection.get("step23_permitted") is not True:
        raise ValueError("Step 22 selection does not permit Step 23")
    if selection.get("selected_candidate_id") != EXPECTED_SELECTED_CANDIDATE_ID:
        raise ValueError("frozen model-selection result chose an unexpected candidate")
    expected = expected_selected_configuration(contract)
    if selection.get("selected_candidate_configuration") != expected:
        raise ValueError("selected candidate configuration differs from the frozen contract")

    candidate_results = results.get("candidate_results")
    if not isinstance(candidate_results, list):
        raise TypeError("model-selection candidate_results must be a list")
    matches = [
        result
        for result in candidate_results
        if result.get("candidate_id") == EXPECTED_SELECTED_CANDIDATE_ID
    ]
    if len(matches) != 1:
        raise ValueError("selected candidate must occur exactly once in results")
    selected_result = matches[0]
    if selected_result.get("configuration") != expected:
        raise ValueError("selected result configuration differs from selection metadata")
    if (
        selected_result.get("safety_gate_results", {}).get("all_gates_pass")
        is not True
    ):
        raise ValueError("selected candidate did not pass every mandatory safety gate")
    gates = selected_result["safety_gate_results"].get("gates", {})
    if set(gates) != {
        "protected_write_false_positive_rate",
        "exact_protected_write_recall",
        "unsupported_or_uncertain_recall",
    } or any(gate.get("passed") is not True for gate in gates.values()):
        raise ValueError("selected candidate mandatory gate results changed")
    return selection["selected_candidate_configuration"]


def validate_model_selection_lineage(
    results: dict[str, Any],
    results_manifest: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, Any]:
    if results.get("schema_version") != model_selection.RESULTS_SCHEMA_VERSION:
        raise ValueError("unexpected V2-C5 model-selection result schema")
    if results.get("phase") != "V2-C5 Step 22B1":
        raise ValueError("unexpected V2-C5 model-selection result phase")
    if results.get("candidate_count") != 27 or results.get(
        "completed_candidate_count"
    ) != 27:
        raise ValueError("complete 27-candidate model-selection results are required")
    if results.get("development_record_count") != 8198:
        raise ValueError("model-selection development record count changed")
    if results.get("contract") != {
        "path": str(CONTRACT_PATH.relative_to(REPOSITORY_ROOT)),
        "sha256": sha256_file(CONTRACT_PATH),
    }:
        raise ValueError("model-selection contract lineage changed")
    if results.get("source_artifacts") != contract["source_artifacts"]:
        raise ValueError("model-selection source lineage changed")
    if results.get("source_development_dataset") != contract["source_artifacts"][
        "expanded_development_dataset"
    ]:
        raise ValueError("model-selection development lineage changed")
    governance = results.get("governance", {})
    required_governance = {
        "development_only": True,
        "embeddings_generated_for_development_only": True,
        "final_holdout_accessed": False,
        "final_holdout_evaluated": False,
        "final_holdout_inference_performed": False,
        "final_model_acceptance_claimed": False,
        "runtime_behavior_changed": False,
        "step23_permitted": True,
        "threshold_tuning_performed": False,
    }
    if governance != required_governance:
        raise ValueError("model-selection governance changed")

    if results_manifest.get("schema_version") != (
        model_selection.RESULTS_MANIFEST_SCHEMA_VERSION
    ):
        raise ValueError("unexpected model-selection manifest schema")
    if results_manifest.get("results") != {
        "path": str(RESULTS_PATH.relative_to(REPOSITORY_ROOT)),
        "schema_version": model_selection.RESULTS_SCHEMA_VERSION,
        "sha256": sha256_file(RESULTS_PATH),
    }:
        raise ValueError("model-selection result hash or path changed")
    if results_manifest.get("contract") != results["contract"]:
        raise ValueError("model-selection manifest contract lineage changed")
    if results_manifest.get("source_artifacts") != contract["source_artifacts"]:
        raise ValueError("model-selection manifest source lineage changed")
    if (
        results_manifest.get("candidate_count") != 27
        or results_manifest.get("completed_candidate_count") != 27
        or results_manifest.get("development_record_count") != 8198
        or results_manifest.get("model_selection_success") is not True
        or results_manifest.get("selected_candidate_id")
        != EXPECTED_SELECTED_CANDIDATE_ID
        or results_manifest.get("step23_permitted") is not True
        or results_manifest.get("threshold_tuning_performed") is not False
        or results_manifest.get("final_holdout_accessed") is not False
        or results_manifest.get("final_holdout_evaluated") is not False
        or results_manifest.get("final_holdout_inference_performed") is not False
        or results_manifest.get("runtime_behavior_changed") is not False
        or results_manifest.get("final_model_acceptance_claimed") is not False
    ):
        raise ValueError("model-selection manifest governance changed")
    if results["selection"].get("eligible_candidate_count") != sum(
        result.get("safety_gate_results", {}).get("all_gates_pass") is True
        for result in results["candidate_results"]
    ):
        raise ValueError("eligible candidate count is inconsistent")
    return validate_selected_candidate(results, contract)


def validate_holdout_manifest_governance(
    holdout_manifest: dict[str, Any], contract: dict[str, Any]
) -> None:
    status = holdout_manifest.get("execution_status", {})
    allowed = contract["holdout_isolation"]["allowed_manifest_checks"]
    for field, expected in allowed.items():
        if status.get(field) is not expected:
            raise ValueError(f"final-holdout governance changed: {field}")
    if holdout_manifest.get("dataset", {}).get("path") != (
        PROHIBITED_FINAL_HOLDOUT_RELATIVE_PATH
    ):
        raise ValueError("final-holdout manifest dataset path changed")
    if holdout_manifest["dataset"].get("sha256") != contract[
        "holdout_isolation"
    ]["final_holdout_dataset"]["declared_sha256_from_manifest"]:
        raise ValueError("final-holdout manifest dataset declaration changed")


def validate_development_population(
    examples: Sequence[dict[str, Any]], contract: dict[str, Any]
) -> dict[str, int]:
    if len(examples) != 8198:
        raise ValueError("Step 22C requires all 8,198 development records")
    labels = contract["taxonomy"]["intent_label_order"]
    counts = Counter(row["intent"] for row in examples)
    if set(counts) != set(labels) or any(counts[label] <= 0 for label in labels):
        raise ValueError("every frozen taxonomy label must be represented in training")
    return {label: counts[label] for label in labels}


def load_preconditions() -> dict[str, Any]:
    contract = read_json(CONTRACT_PATH)
    model_selection.validate_contract(contract)
    examples, taxonomy = model_selection.load_and_validate_sources(contract)
    intent_counts = validate_development_population(examples, contract)

    results = read_json(RESULTS_PATH)
    results_manifest = read_json(RESULTS_MANIFEST_PATH)
    selected_configuration = validate_model_selection_lineage(
        results, results_manifest, contract
    )
    holdout_manifest_spec = contract["source_artifacts"]["final_holdout_manifest"]
    holdout_manifest_path = validate_hashed_source(
        holdout_manifest_spec, "final-holdout manifest"
    )
    validate_holdout_manifest_governance(
        read_json(holdout_manifest_path), contract
    )
    return {
        "contract": contract,
        "examples": examples,
        "taxonomy": taxonomy,
        "intent_counts": intent_counts,
        "results": results,
        "results_manifest": results_manifest,
        "selected_configuration": selected_configuration,
    }


def preflight() -> dict[str, Any]:
    inputs = load_preconditions()
    return {
        "status": "ready",
        "selected_candidate_id": EXPECTED_SELECTED_CANDIDATE_ID,
        "selected_configuration": inputs["selected_configuration"],
        "development_record_count": len(inputs["examples"]),
        "intent_count": len(inputs["intent_counts"]),
        "model_selection_results_sha256": sha256_file(RESULTS_PATH),
        "model_selection_success": True,
        "selection_step23_permitted": True,
        "fit_performed": False,
        "embeddings_generated": False,
        "inference_performed": False,
        "files_written": False,
        "final_holdout_accessed": False,
        "final_holdout_inference_performed": False,
        "final_holdout_evaluated": False,
    }


def validate_embedding_input(
    embeddings: np.ndarray,
    examples: Sequence[dict[str, Any]],
    configuration: dict[str, Any],
) -> None:
    dimensions = configuration["representation"]["dimensions"]
    if embeddings.shape != (len(examples), dimensions):
        raise ValueError(
            f"full-development embedding shape changed: {embeddings.shape}"
        )
    if embeddings.dtype != np.float32:
        raise ValueError("full-development embeddings must use float32")
    if not np.isfinite(embeddings).all():
        raise ValueError("full-development embeddings contain non-finite values")
    if not np.allclose(
        np.linalg.norm(embeddings, axis=1), 1.0, rtol=1e-5, atol=1e-6
    ):
        raise ValueError("full-development embeddings are not L2 normalized")


def validate_fitted_classifier(
    classifier: Any,
    configuration: dict[str, Any],
    intent_labels: Sequence[str],
) -> None:
    if not isinstance(classifier, LinearSVC):
        raise TypeError("selected classifier artifact must contain LinearSVC")
    if (
        classifier.get_params(deep=False)
        != configuration["classifier"]["parameters"]
    ):
        raise ValueError("selected classifier parameters changed")
    if tuple(classifier.classes_.tolist()) != tuple(intent_labels):
        raise ValueError("selected classifier classes differ from frozen label order")
    if len(classifier.classes_) != 16:
        raise ValueError("selected classifier must contain exactly 16 classes")
    if getattr(classifier, "n_features_in_", None) != 384:
        raise ValueError("selected classifier feature dimension changed")
    if classifier.coef_.shape != (16, 384) or classifier.intercept_.shape != (16,):
        raise ValueError("selected classifier coefficient dimensions changed")


def fit_full_development_classifier(
    embeddings: np.ndarray,
    examples: Sequence[dict[str, Any]],
    configuration: dict[str, Any],
    intent_labels: Sequence[str],
) -> LinearSVC:
    validate_embedding_input(embeddings, examples, configuration)
    observed = Counter(row["intent"] for row in examples)
    if set(observed) != set(intent_labels) or any(
        observed[label] <= 0 for label in intent_labels
    ):
        raise ValueError("full-development fit requires every frozen intent")
    labels = np.asarray([row["intent"] for row in examples], dtype=object)
    classifier = LinearSVC(**configuration["classifier"]["parameters"])
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        classifier.fit(embeddings, labels)
    validate_fitted_classifier(classifier, configuration, intent_labels)
    return classifier


def ordered_values_sha256(values: Sequence[str]) -> str:
    return sha256_bytes(stable_json_bytes(list(values)))


def embedding_cache_lineage(
    cache_manifest: dict[str, Any],
    inputs: dict[str, Any],
) -> dict[str, Any]:
    examples = inputs["examples"]
    contract = inputs["contract"]
    expected_ids = model_selection.ordered_example_ids(examples)
    expected_text_hashes = model_selection.ordered_text_hashes(examples)
    if cache_manifest.get("ordered_example_ids") != expected_ids:
        raise ValueError("embedding cache ordered example IDs changed")
    if cache_manifest.get("ordered_text_sha256") != expected_text_hashes:
        raise ValueError("embedding cache ordered text hashes changed")
    if (
        cache_manifest.get("row_count") != 8198
        or cache_manifest.get("dimensions") != 384
        or cache_manifest.get("l2_normalized") is not True
        or cache_manifest.get("representation_config_sha256")
        != model_selection.bge_representation_sha256(contract)
    ):
        raise ValueError("embedding cache definition changed")
    return {
        "cache_path": str(model_selection.BGE_CACHE_PATH.relative_to(REPOSITORY_ROOT)),
        "cache_sha256": sha256_file(model_selection.BGE_CACHE_PATH),
        "cache_manifest_path": str(
            model_selection.BGE_CACHE_MANIFEST_PATH.relative_to(REPOSITORY_ROOT)
        ),
        "cache_manifest_sha256": sha256_file(
            model_selection.BGE_CACHE_MANIFEST_PATH
        ),
        "cache_schema_version": cache_manifest["schema_version"],
        "development_dataset_sha256": cache_manifest[
            "development_dataset_sha256"
        ],
        "representation_config_sha256": cache_manifest[
            "representation_config_sha256"
        ],
        "ordered_example_ids_sha256": ordered_values_sha256(expected_ids),
        "ordered_text_sha256_sha256": ordered_values_sha256(expected_text_hashes),
        "row_count": cache_manifest["row_count"],
        "dimensions": cache_manifest["dimensions"],
        "l2_normalized": cache_manifest["l2_normalized"],
    }


def artifact_payload(
    inputs: dict[str, Any],
    classifier: LinearSVC,
    cache_lineage: dict[str, Any],
) -> dict[str, Any]:
    configuration = inputs["selected_configuration"]
    return {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "selected_candidate_id": EXPECTED_SELECTED_CANDIDATE_ID,
        "representation": configuration["representation"],
        "classifier": classifier,
        "classifier_config": configuration["classifier"],
        "intent_label_order": inputs["contract"]["taxonomy"]["intent_label_order"],
        "development_record_count": len(inputs["examples"]),
        "source_model_selection_results_sha256": sha256_file(RESULTS_PATH),
        "development_dataset_sha256": inputs["contract"]["source_artifacts"][
            "expanded_development_dataset"
        ]["sha256"],
        "taxonomy_sha256": inputs["contract"]["source_artifacts"][
            "taxonomy_freeze"
        ]["sha256"],
        "embedding_cache_lineage": cache_lineage,
        "trusted_local_artifact": True,
        "runtime_authority": False,
        "fit_completed": True,
        "classifier_training_performed": True,
        "final_holdout_accessed": False,
        "final_holdout_inference_performed": False,
        "final_holdout_evaluated": False,
        "threshold_tuning_performed": False,
        "runtime_behavior_changed": False,
        "final_model_acceptance_claimed": False,
    }


def build_model_manifest(
    inputs: dict[str, Any], cache_lineage: dict[str, Any]
) -> dict[str, Any]:
    configuration = inputs["selected_configuration"]
    contract = inputs["contract"]
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "phase": "V2-C5 Step 22C",
        "selected_candidate_id": EXPECTED_SELECTED_CANDIDATE_ID,
        "source_model_selection_results": {
            "path": str(RESULTS_PATH.relative_to(REPOSITORY_ROOT)),
            "sha256": sha256_file(RESULTS_PATH),
        },
        "source_model_selection_results_manifest": {
            "path": str(RESULTS_MANIFEST_PATH.relative_to(REPOSITORY_ROOT)),
            "sha256": sha256_file(RESULTS_MANIFEST_PATH),
        },
        "source_contract": {
            "path": str(CONTRACT_PATH.relative_to(REPOSITORY_ROOT)),
            "sha256": sha256_file(CONTRACT_PATH),
        },
        "development_dataset": contract["source_artifacts"][
            "expanded_development_dataset"
        ],
        "development_dataset_manifest": contract["source_artifacts"][
            "expanded_development_manifest"
        ],
        "taxonomy": contract["source_artifacts"]["taxonomy_freeze"],
        "taxonomy_manifest": contract["source_artifacts"][
            "taxonomy_freeze_manifest"
        ],
        "final_holdout_manifest": contract["source_artifacts"][
            "final_holdout_manifest"
        ],
        "representation_config": configuration["representation"],
        "classifier_config": configuration["classifier"],
        "development_record_count": len(inputs["examples"]),
        "intent_counts": inputs["intent_counts"],
        "class_labels": contract["taxonomy"]["intent_label_order"],
        "local_classifier_artifact": {
            "path": str(MODEL_ARTIFACT_PATH.relative_to(REPOSITORY_ROOT)),
            "sha256": sha256_file(MODEL_ARTIFACT_PATH),
            "size_bytes": MODEL_ARTIFACT_PATH.stat().st_size,
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "trusted_local": True,
            "runtime_eligible": False,
        },
        "embedding_cache_lineage": cache_lineage,
        "preparation_script": {
            "path": SCRIPT_RELATIVE_PATH,
            "sha256": sha256_file(repository_path(SCRIPT_RELATIVE_PATH)),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": metadata.version("joblib"),
        },
        "fit_completed": True,
        "classifier_training_performed": True,
        "final_holdout_accessed": False,
        "final_holdout_inference_performed": False,
        "final_holdout_evaluated": False,
        "threshold_tuning_performed": False,
        "runtime_behavior_changed": False,
        "final_model_acceptance_claimed": False,
        "step23_permitted": True,
        "next_required": "v2c5_once_only_final_holdout_evaluation",
    }


def validate_text_free_manifest(
    manifest: dict[str, Any], examples: Sequence[dict[str, Any]]
) -> None:
    model_selection.validate_text_free(manifest, examples)


def load_or_create_development_embeddings(
    inputs: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    return model_selection.prepare_bge_cache(
        inputs["examples"], inputs["contract"]
    )


def fit_selected_model() -> tuple[Path, Path]:
    inputs = load_preconditions()
    if MODEL_ARTIFACT_PATH.exists() or MODEL_MANIFEST_PATH.exists():
        raise FileExistsError(
            "selected-model artifact or manifest already exists; use --check"
        )
    embeddings, cache_manifest = load_or_create_development_embeddings(inputs)
    validate_embedding_input(
        embeddings, inputs["examples"], inputs["selected_configuration"]
    )
    classifier = fit_full_development_classifier(
        embeddings,
        inputs["examples"],
        inputs["selected_configuration"],
        inputs["contract"]["taxonomy"]["intent_label_order"],
    )
    cache_lineage = embedding_cache_lineage(cache_manifest, inputs)
    payload = artifact_payload(inputs, classifier, cache_lineage)

    MODEL_ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=MODEL_ARTIFACT_PATH.parent, suffix=".joblib", delete=False
    ) as handle:
        temporary_artifact = Path(handle.name)
    try:
        joblib.dump(payload, temporary_artifact, compress=3)
        temporary_artifact.replace(MODEL_ARTIFACT_PATH)
    finally:
        temporary_artifact.unlink(missing_ok=True)
    manifest = build_model_manifest(inputs, cache_lineage)
    validate_text_free_manifest(manifest, inputs["examples"])
    write_bytes_atomically(MODEL_MANIFEST_PATH, stable_json_bytes(manifest))
    return MODEL_ARTIFACT_PATH, MODEL_MANIFEST_PATH


def validate_model_manifest(
    manifest: dict[str, Any], inputs: dict[str, Any]
) -> None:
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("unexpected selected-model manifest schema")
    if manifest.get("phase") != "V2-C5 Step 22C":
        raise ValueError("unexpected selected-model manifest phase")
    if manifest.get("selected_candidate_id") != EXPECTED_SELECTED_CANDIDATE_ID:
        raise ValueError("selected-model manifest candidate changed")
    if manifest.get("representation_config") != EXPECTED_REPRESENTATION:
        raise ValueError("selected-model manifest representation changed")
    if manifest.get("classifier_config") != inputs["selected_configuration"][
        "classifier"
    ]:
        raise ValueError("selected-model manifest classifier changed")
    if (
        manifest.get("development_record_count") != 8198
        or manifest.get("class_labels")
        != inputs["contract"]["taxonomy"]["intent_label_order"]
        or manifest.get("intent_counts") != inputs["intent_counts"]
    ):
        raise ValueError("selected-model training population changed")
    expected_sources = {
        "source_model_selection_results": {
            "path": str(RESULTS_PATH.relative_to(REPOSITORY_ROOT)),
            "sha256": sha256_file(RESULTS_PATH),
        },
        "source_model_selection_results_manifest": {
            "path": str(RESULTS_MANIFEST_PATH.relative_to(REPOSITORY_ROOT)),
            "sha256": sha256_file(RESULTS_MANIFEST_PATH),
        },
        "source_contract": {
            "path": str(CONTRACT_PATH.relative_to(REPOSITORY_ROOT)),
            "sha256": sha256_file(CONTRACT_PATH),
        },
        "development_dataset": inputs["contract"]["source_artifacts"][
            "expanded_development_dataset"
        ],
        "development_dataset_manifest": inputs["contract"]["source_artifacts"][
            "expanded_development_manifest"
        ],
        "taxonomy": inputs["contract"]["source_artifacts"]["taxonomy_freeze"],
        "taxonomy_manifest": inputs["contract"]["source_artifacts"][
            "taxonomy_freeze_manifest"
        ],
        "final_holdout_manifest": inputs["contract"]["source_artifacts"][
            "final_holdout_manifest"
        ],
    }
    for field, expected in expected_sources.items():
        if manifest.get(field) != expected:
            raise ValueError(f"selected-model manifest lineage changed: {field}")
    artifact = manifest.get("local_classifier_artifact", {})
    if artifact != {
        "path": str(MODEL_ARTIFACT_PATH.relative_to(REPOSITORY_ROOT)),
        "sha256": sha256_file(MODEL_ARTIFACT_PATH),
        "size_bytes": MODEL_ARTIFACT_PATH.stat().st_size,
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "trusted_local": True,
        "runtime_eligible": False,
    }:
        raise ValueError("selected-model local artifact lineage changed")
    required_flags = {
        "fit_completed": True,
        "classifier_training_performed": True,
        "final_holdout_accessed": False,
        "final_holdout_inference_performed": False,
        "final_holdout_evaluated": False,
        "threshold_tuning_performed": False,
        "runtime_behavior_changed": False,
        "final_model_acceptance_claimed": False,
        "step23_permitted": True,
        "next_required": "v2c5_once_only_final_holdout_evaluation",
    }
    for field, expected in required_flags.items():
        if manifest.get(field) != expected:
            raise ValueError(f"selected-model manifest governance changed: {field}")
    if manifest.get("preparation_script") != {
        "path": SCRIPT_RELATIVE_PATH,
        "sha256": sha256_file(repository_path(SCRIPT_RELATIVE_PATH)),
    }:
        raise ValueError("selected-model preparation script lineage changed")
    validate_text_free_manifest(manifest, inputs["examples"])


def validate_artifact_payload(
    payload: dict[str, Any], manifest: dict[str, Any], inputs: dict[str, Any]
) -> LinearSVC:
    if payload.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("unexpected selected-classifier artifact schema")
    if payload.get("selected_candidate_id") != EXPECTED_SELECTED_CANDIDATE_ID:
        raise ValueError("selected-classifier artifact candidate changed")
    if payload.get("representation") != EXPECTED_REPRESENTATION:
        raise ValueError("selected-classifier artifact representation changed")
    if payload.get("classifier_config") != inputs["selected_configuration"][
        "classifier"
    ]:
        raise ValueError("selected-classifier artifact configuration changed")
    if payload.get("intent_label_order") != manifest["class_labels"]:
        raise ValueError("selected-classifier artifact labels changed")
    if payload.get("embedding_cache_lineage") != manifest[
        "embedding_cache_lineage"
    ]:
        raise ValueError("selected-classifier cache lineage changed")
    if (
        payload.get("development_record_count") != 8198
        or payload.get("source_model_selection_results_sha256")
        != sha256_file(RESULTS_PATH)
        or payload.get("development_dataset_sha256")
        != inputs["contract"]["source_artifacts"]["expanded_development_dataset"][
            "sha256"
        ]
        or payload.get("taxonomy_sha256")
        != inputs["contract"]["source_artifacts"]["taxonomy_freeze"]["sha256"]
    ):
        raise ValueError("selected-classifier artifact lineage changed")
    expected_flags = {
        "trusted_local_artifact": True,
        "runtime_authority": False,
        "fit_completed": True,
        "classifier_training_performed": True,
        "final_holdout_accessed": False,
        "final_holdout_inference_performed": False,
        "final_holdout_evaluated": False,
        "threshold_tuning_performed": False,
        "runtime_behavior_changed": False,
        "final_model_acceptance_claimed": False,
    }
    for field, expected in expected_flags.items():
        if payload.get(field) != expected:
            raise ValueError(f"selected-classifier artifact boundary changed: {field}")
    classifier = payload.get("classifier")
    validate_fitted_classifier(
        classifier,
        inputs["selected_configuration"],
        inputs["contract"]["taxonomy"]["intent_label_order"],
    )
    return classifier


def check_selected_model() -> dict[str, Any]:
    inputs = load_preconditions()
    if not MODEL_ARTIFACT_PATH.is_file() or not MODEL_MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            "selected-model local artifact and tracked manifest are required"
        )
    manifest = read_json(MODEL_MANIFEST_PATH)
    validate_model_manifest(manifest, inputs)
    _, cache_manifest = model_selection.validate_bge_cache(
        inputs["examples"], inputs["contract"]
    )
    current_cache_lineage = embedding_cache_lineage(cache_manifest, inputs)
    if manifest.get("embedding_cache_lineage") != current_cache_lineage:
        raise ValueError("selected-model embedding cache lineage changed")
    payload = joblib.load(MODEL_ARTIFACT_PATH)
    if not isinstance(payload, dict):
        raise TypeError("selected-model artifact must contain a dictionary")
    validate_artifact_payload(payload, manifest, inputs)
    return {
        "status": "valid",
        "selected_candidate_id": EXPECTED_SELECTED_CANDIDATE_ID,
        "development_record_count": 8198,
        "class_count": 16,
        "local_classifier_artifact_sha256": manifest[
            "local_classifier_artifact"
        ]["sha256"],
        "fit_completed": True,
        "step23_permitted": True,
        "files_written": False,
        "inference_performed": False,
        "final_holdout_accessed": False,
        "final_holdout_inference_performed": False,
        "final_holdout_evaluated": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare the frozen V2-C5 selected development classifier"
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--fit", action="store_true")
    modes.add_argument("--check", action="store_true")
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    if arguments.preflight:
        print(json.dumps(preflight(), indent=2, sort_keys=True))
    elif arguments.fit:
        paths = fit_selected_model()
        print("\n".join(str(path.relative_to(REPOSITORY_ROOT)) for path in paths))
    elif arguments.check:
        print(json.dumps(check_selected_model(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
