"""Run the once-only V2-C5 final evaluation under durable state governance.

Preflight and state initialization never open the sealed final-holdout dataset.
Only ``--evaluate`` may open it, and only after a durable ``started`` state has
replaced the frozen ``not_started`` state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.svm import LinearSVC

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ML_DIRECTORY = REPOSITORY_ROOT / "data/evals/v2/ml"
SCRIPT_RELATIVE_PATH = "scripts/run_v2c5_final_evaluation.py"

CONTRACT_SCHEMA_VERSION = "v2c5-final-evaluation-contract.v1"
STATE_SCHEMA_VERSION = "v2c5-final-evaluation-state.v1"
RESULTS_SCHEMA_VERSION = "v2c5-final-evaluation-results.v1"
RESULTS_MANIFEST_SCHEMA_VERSION = "v2c5-final-evaluation-results-manifest.v1"
HOLDOUT_SCHEMA_VERSION = "v2c5-final-holdout.v1"
SELECTED_MODEL_SCHEMA_VERSION = "v2c5-selected-model-manifest.v1"
SELECTED_ARTIFACT_SCHEMA_VERSION = "v2c5-selected-classifier.v1"

EXPECTED_CANDIDATE_ID = "BGE_SMALL_LINEAR_SVC__C=4.0__class_weight=none"
EXPECTED_CLASS_LABELS = [
    "account_balance",
    "account_blocked",
    "cancel_transfer",
    "card_status",
    "close_account",
    "create_dispute",
    "escalation",
    "freeze_card",
    "informational_policy",
    "lost_or_stolen_phone",
    "passcode_recovery",
    "recent_transactions",
    "transaction_details",
    "transfer_failed_or_declined",
    "transfer_pending",
    "unsupported_or_uncertain",
]
EXPECTED_HISTORICAL_LABELS = [
    "account_balance",
    "card_status",
    "create_dispute",
    "escalation",
    "freeze_card",
    "informational_policy",
    "recent_transactions",
    "transaction_details",
    "unsupported_or_uncertain",
]
EXPECTED_NEW_LABELS = [
    "account_blocked",
    "cancel_transfer",
    "close_account",
    "lost_or_stolen_phone",
    "passcode_recovery",
    "transfer_failed_or_declined",
    "transfer_pending",
]
EXPECTED_REPRESENTATION = {
    "dimensions": 384,
    "embedding_method": "passage_embed",
    "fine_tuning": False,
    "implementation": "FastEmbed",
    "l2_normalized": True,
    "model_identifier": "BAAI/bge-small-en-v1.5",
}
EXPECTED_CLASSIFIER = {
    "class": "LinearSVC",
    "parameters": {
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
    },
}


@dataclass(frozen=True)
class EvaluationPaths:
    repository_root: Path
    contract: Path
    selected_model_manifest: Path
    classifier_artifact: Path
    final_holdout_manifest: Path
    final_holdout_contract: Path
    final_holdout_dataset: Path
    model_selection_results: Path
    taxonomy: Path
    state: Path
    results: Path
    results_manifest: Path
    runner: Path


DEFAULT_PATHS = EvaluationPaths(
    repository_root=REPOSITORY_ROOT,
    contract=ML_DIRECTORY / "v2c5_final_evaluation_contract.json",
    selected_model_manifest=ML_DIRECTORY / "v2c5_selected_model.manifest.json",
    classifier_artifact=ML_DIRECTORY / "local/v2c5_selected_classifier.joblib",
    final_holdout_manifest=ML_DIRECTORY / "v2c5_final_holdout.manifest.json",
    final_holdout_contract=ML_DIRECTORY / "v2c5_final_holdout_contract.json",
    final_holdout_dataset=ML_DIRECTORY / "v2c5_final_holdout.json",
    model_selection_results=ML_DIRECTORY / "v2c5_model_selection_results.json",
    taxonomy=ML_DIRECTORY / "v2c5_taxonomy_freeze.json",
    state=ML_DIRECTORY / "v2c5_final_evaluation_state.json",
    results=ML_DIRECTORY / "v2c5_final_evaluation_results.json",
    results_manifest=(
        ML_DIRECTORY / "v2c5_final_evaluation_results.manifest.json"
    ),
    runner=REPOSITORY_ROOT / SCRIPT_RELATIVE_PATH,
)


def stable_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return payload


def display_path(path: Path, paths: EvaluationPaths) -> str:
    try:
        return str(path.resolve().relative_to(paths.repository_root.resolve()))
    except ValueError:
        return str(path.resolve())


def fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def durable_write_bytes(
    path: Path,
    content: bytes,
    *,
    refuse_existing: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary_path = Path(handle.name)
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        if refuse_existing:
            try:
                os.link(temporary_path, path)
            except FileExistsError as exc:
                raise FileExistsError(f"refusing to overwrite existing file: {path}") from exc
            temporary_path.unlink()
        else:
            os.replace(temporary_path, path)
        fsync_directory(path.parent)
    finally:
        temporary_path.unlink(missing_ok=True)


def validate_file_binding(
    path: Path,
    specification: Mapping[str, Any],
    paths: EvaluationPaths,
    label: str,
) -> None:
    expected = {
        "path": display_path(path, paths),
        "sha256": sha256_file(path),
    }
    if dict(specification) != expected:
        raise ValueError(f"{label} path or hash binding changed")


def validate_contract(contract: Mapping[str, Any], paths: EvaluationPaths) -> None:
    if contract.get("schema_version") != CONTRACT_SCHEMA_VERSION:
        raise ValueError("unexpected final-evaluation contract schema")
    if contract.get("contract_version") != CONTRACT_SCHEMA_VERSION:
        raise ValueError("unexpected final-evaluation contract version")
    if contract.get("phase") != "V2-C5 Step 23A":
        raise ValueError("unexpected final-evaluation contract phase")
    status = contract.get("contract_status", {})
    if status != {
        "contract_frozen": True,
        "final_holdout_accessed": False,
        "final_holdout_evaluated": False,
        "final_holdout_inference_performed": False,
        "final_model_acceptance_claimed": False,
        "model_retraining_performed": False,
        "next_required": "v2c5_final_evaluation_runner",
        "runtime_behavior_changed": False,
        "threshold_tuning_performed": False,
    }:
        raise ValueError("final-evaluation contract status changed")
    model = contract.get("model_under_test", {})
    if (
        model.get("selected_candidate_id") != EXPECTED_CANDIDATE_ID
        or model.get("representation") != EXPECTED_REPRESENTATION
        or model.get("classifier") != EXPECTED_CLASSIFIER
        or model.get("class_labels") != EXPECTED_CLASS_LABELS
    ):
        raise ValueError("frozen model-under-test definition changed")
    if contract.get("safety_gates", {}).get("thresholds_match_step22") is not True:
        raise ValueError("Step 22 safety-gate binding changed")
    safety = contract["safety_gates"]
    if (
        safety.get("all_gates_mandatory_for_final_acceptance") is not True
        or safety.get("protected_write_intents")
        != [
            "cancel_transfer",
            "close_account",
            "create_dispute",
            "freeze_card",
        ]
    ):
        raise ValueError("protected-write intent set changed")
    gates = safety.get("gates", [])
    if gates != [
        {
            "comparison": "less_than_or_equal",
            "denominator": "count(gold_intent not in protected_write_intents)",
            "expected_denominator": 480,
            "metric": "protected_write_false_positive_rate",
            "numerator": (
                "count(gold_intent not in protected_write_intents and "
                "predicted_intent in protected_write_intents)"
            ),
            "threshold": 0.01,
        },
        {
            "comparison": "greater_than_or_equal",
            "denominator": "count(gold_intent in protected_write_intents)",
            "expected_denominator": 160,
            "metric": "exact_protected_write_recall",
            "numerator": (
                "count(gold_intent in protected_write_intents and "
                "predicted_intent == gold_intent)"
            ),
            "threshold": 0.8,
        },
        {
            "comparison": "greater_than_or_equal",
            "denominator": "count(gold_intent == unsupported_or_uncertain)",
            "expected_denominator": 40,
            "metric": "unsupported_or_uncertain_recall",
            "numerator": (
                "count(gold_intent == unsupported_or_uncertain and "
                "predicted_intent == unsupported_or_uncertain)"
            ),
            "threshold": 0.8,
        },
    ]:
        raise ValueError("final safety-gate definitions changed")
    metrics = contract.get("metrics", {})
    confusion = metrics.get("confusion_matrix", {})
    subsets = metrics.get("subset_macro_f1", {})
    if (
        metrics.get("aggregate_acceptance_threshold") is not None
        or metrics.get("aggregate_metrics")
        != ["accuracy", "balanced_accuracy", "macro_f1"]
        or metrics.get("per_intent_metrics")
        != ["precision", "recall", "f1", "support"]
        or metrics.get("macro_f1_role")
        != "reported_final_quality_metric_not_an_additional_gate"
        or confusion
        != {
            "column_axis": "predicted_intent",
            "dimensions": [16, 16],
            "label_order": EXPECTED_CLASS_LABELS,
            "row_axis": "gold_intent",
        }
        or subsets.get("historical_nine_label_macro_f1")
        != EXPECTED_HISTORICAL_LABELS
        or subsets.get("new_seven_intent_macro_f1") != EXPECTED_NEW_LABELS
    ):
        raise ValueError("final metric contract changed")
    holdout = contract.get("holdout", {})
    if (
        holdout.get("total_example_count") != 640
        or holdout.get("intent_count") != 16
        or holdout.get("examples_per_intent") != 40
        or holdout.get("intent_counts") != dict.fromkeys(EXPECTED_CLASS_LABELS, 40)
        or holdout.get("protected_write_example_count") != 160
        or holdout.get("non_protected_example_count") != 480
        or holdout.get("evaluated_before_step23") is not False
    ):
        raise ValueError("final-holdout population contract changed")
    once_only = contract.get("once_only_evaluation", {})
    if (
        once_only.get("single_final_inference_and_evaluation") is not True
        or once_only.get("evaluation_attempt_count") != 1
        or once_only.get("threshold_tuning_from_holdout_results") is not False
        or once_only.get("model_retraining_from_holdout_results") is not False
        or once_only.get("candidate_reselection_from_holdout_results") is not False
        or once_only.get("label_or_taxonomy_change_from_holdout_results") is not False
        or once_only.get("second_clean_evaluation_after_failure_inspection")
        is not False
        or once_only.get(
            "holdout_examples_may_be_added_to_development_or_training"
        )
        is not False
    ):
        raise ValueError("once-only evaluation governance changed")
    runtime = contract.get("decision_semantics", {}).get("runtime_eligibility", {})
    if (
        runtime.get("value_before_evaluation") is not False
        or runtime.get("value_after_gate_pass") is not False
        or runtime.get("runtime_behavior_changed_by_step23") is not False
    ):
        raise ValueError("runtime-eligibility boundary changed")
    isolation = contract.get("holdout_isolation", {})
    if (
        isolation.get("prohibited_step23a_path")
        != display_path(paths.final_holdout_dataset, paths)
        or isolation.get("final_holdout_dataset_opened_or_hashed_by_step23a")
        is not False
    ):
        raise ValueError("Step 23A holdout-isolation declaration changed")
    planned = contract.get("planned_outputs", {})
    expected_outputs = {
        "results": display_path(paths.results, paths),
        "results_manifest": display_path(paths.results_manifest, paths),
        "state": display_path(paths.state, paths),
    }
    if (
        any(planned.get(field) != value for field, value in expected_outputs.items())
        or planned.get("generated_by_step23a") is not False
        or planned.get("prediction_persistence") != "aggregate_only"
        or planned.get("raw_holdout_text_permitted") is not False
    ):
        raise ValueError("planned final-evaluation output paths changed")


def validate_selected_model_manifest(
    manifest: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> None:
    model = contract["model_under_test"]
    if manifest.get("schema_version") != SELECTED_MODEL_SCHEMA_VERSION:
        raise ValueError("unexpected selected-model manifest schema")
    if manifest.get("selected_candidate_id") != model["selected_candidate_id"]:
        raise ValueError("selected-model candidate changed")
    if manifest.get("representation_config") != model["representation"]:
        raise ValueError("selected-model representation changed")
    if manifest.get("classifier_config") != model["classifier"]:
        raise ValueError("selected-model classifier changed")
    if manifest.get("class_labels") != model["class_labels"]:
        raise ValueError("selected-model class labels changed")
    if manifest.get("local_classifier_artifact") != contract["source_artifacts"][
        "selected_classifier_artifact"
    ]:
        raise ValueError("selected classifier artifact binding changed")
    if manifest.get("source_model_selection_results") != contract[
        "source_artifacts"
    ]["model_selection_results"]:
        raise ValueError("selected-model result lineage changed")
    if manifest.get("development_dataset") != contract["source_artifacts"][
        "development_dataset"
    ]:
        raise ValueError("selected-model development lineage changed")
    if manifest.get("taxonomy") != contract["source_artifacts"]["taxonomy"]:
        raise ValueError("selected-model taxonomy lineage changed")
    required = {
        "classifier_training_performed": True,
        "fit_completed": True,
        "final_holdout_accessed": False,
        "final_holdout_evaluated": False,
        "final_holdout_inference_performed": False,
        "final_model_acceptance_claimed": False,
        "runtime_behavior_changed": False,
        "step23_permitted": True,
        "threshold_tuning_performed": False,
    }
    if any(manifest.get(field) != value for field, value in required.items()):
        raise ValueError("selected-model governance changed")


def validate_taxonomy(
    taxonomy: Mapping[str, Any], contract: Mapping[str, Any]
) -> None:
    labels = contract["model_under_test"]["class_labels"]
    protected = contract["safety_gates"]["protected_write_intents"]
    if (
        taxonomy.get("schema_version") != "v2c5-taxonomy-freeze.v1"
        or taxonomy.get("final_taxonomy_frozen") is not True
        or taxonomy.get("intent_label_order") != labels
        or taxonomy.get("protected_write_intents") != protected
        or set(taxonomy.get("risk_by_intent", {})) != set(labels)
    ):
        raise ValueError("frozen taxonomy changed")


def validate_holdout_manifest(
    manifest: Mapping[str, Any], contract: Mapping[str, Any]
) -> None:
    holdout = contract["holdout"]
    declaration = contract["source_artifacts"]["final_holdout_dataset_declaration"]
    if manifest.get("schema_version") != "v2c5-final-holdout-manifest.v1":
        raise ValueError("unexpected final-holdout manifest schema")
    if manifest.get("dataset", {}).get("path") != declaration["path"]:
        raise ValueError("final-holdout dataset path declaration changed")
    if manifest.get("dataset", {}).get("sha256") != declaration["sha256"]:
        raise ValueError("final-holdout dataset hash declaration changed")
    counts = manifest.get("counts", {})
    if (
        counts.get("total_count") != holdout["total_example_count"]
        or counts.get("intent_counts") != holdout["intent_counts"]
        or counts.get("protected_write_positive_count")
        != holdout["protected_write_example_count"]
        or counts.get("non_protected_count")
        != holdout["non_protected_example_count"]
    ):
        raise ValueError("final-holdout population declaration changed")
    if manifest.get("taxonomy_intent_labels") != contract["model_under_test"][
        "class_labels"
    ]:
        raise ValueError("final-holdout taxonomy declaration changed")
    status = manifest.get("execution_status", {})
    if (
        status.get("final_holdout_frozen") is not True
        or status.get("final_holdout_evaluated") is not False
        or status.get("classifier_evaluation_performed") is not False
    ):
        raise ValueError("final-holdout pre-evaluation governance changed")


def validate_classifier_payload(
    payload: Mapping[str, Any],
    selected_manifest: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> LinearSVC:
    model = contract["model_under_test"]
    if payload.get("artifact_schema_version") != SELECTED_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("unexpected selected-classifier artifact schema")
    if payload.get("selected_candidate_id") != model["selected_candidate_id"]:
        raise ValueError("selected-classifier candidate changed")
    if payload.get("representation") != model["representation"]:
        raise ValueError("selected-classifier representation changed")
    if payload.get("classifier_config") != model["classifier"]:
        raise ValueError("selected-classifier configuration changed")
    if payload.get("intent_label_order") != model["class_labels"]:
        raise ValueError("selected-classifier labels changed")
    if payload.get("development_dataset_sha256") != contract["source_artifacts"][
        "development_dataset"
    ]["sha256"]:
        raise ValueError("selected-classifier development lineage changed")
    if payload.get("taxonomy_sha256") != contract["source_artifacts"]["taxonomy"][
        "sha256"
    ]:
        raise ValueError("selected-classifier taxonomy lineage changed")
    if payload.get("source_model_selection_results_sha256") != contract[
        "source_artifacts"
    ]["model_selection_results"]["sha256"]:
        raise ValueError("selected-classifier result lineage changed")
    required_flags = {
        "classifier_training_performed": True,
        "final_holdout_accessed": False,
        "final_holdout_evaluated": False,
        "final_holdout_inference_performed": False,
        "final_model_acceptance_claimed": False,
        "fit_completed": True,
        "runtime_behavior_changed": False,
        "runtime_authority": False,
        "threshold_tuning_performed": False,
        "trusted_local_artifact": True,
    }
    if any(payload.get(field) != value for field, value in required_flags.items()):
        raise ValueError("selected-classifier governance changed")
    classifier = payload.get("classifier")
    if not isinstance(classifier, LinearSVC):
        raise TypeError("selected-classifier payload must contain LinearSVC")
    if classifier.get_params(deep=False) != model["classifier"]["parameters"]:
        raise ValueError("selected classifier parameters changed")
    labels = model["class_labels"]
    if tuple(classifier.classes_.tolist()) != tuple(labels):
        raise ValueError("selected classifier classes changed")
    if getattr(classifier, "n_features_in_", None) != 384:
        raise ValueError("selected classifier feature dimension changed")
    if classifier.coef_.shape != (16, 384) or classifier.intercept_.shape != (16,):
        raise ValueError("selected classifier coefficient dimensions changed")
    if selected_manifest.get("class_labels") != labels:
        raise ValueError("selected classifier and manifest labels differ")
    return classifier


def source_lineage(
    contract: Mapping[str, Any], paths: EvaluationPaths
) -> dict[str, str]:
    return {
        "classifier_artifact_sha256": contract["source_artifacts"][
            "selected_classifier_artifact"
        ]["sha256"],
        "declared_final_holdout_dataset_sha256": contract["source_artifacts"][
            "final_holdout_dataset_declaration"
        ]["sha256"],
        "development_dataset_sha256": contract["source_artifacts"][
            "development_dataset"
        ]["sha256"],
        "final_evaluation_contract_sha256": sha256_file(paths.contract),
        "final_holdout_contract_sha256": contract["source_artifacts"][
            "final_holdout_contract"
        ]["sha256"],
        "final_holdout_manifest_sha256": contract["source_artifacts"][
            "final_holdout_manifest"
        ]["sha256"],
        "model_selection_results_sha256": contract["source_artifacts"][
            "model_selection_results"
        ]["sha256"],
        "runner_sha256": sha256_file(paths.runner),
        "selected_model_manifest_sha256": contract["source_artifacts"][
            "selected_model_manifest"
        ]["sha256"],
        "taxonomy_sha256": contract["source_artifacts"]["taxonomy"]["sha256"],
    }


def build_state_payload(
    contract: Mapping[str, Any],
    paths: EvaluationPaths,
    state: str,
    *,
    failure: Mapping[str, Any] | None = None,
    results: Mapping[str, Any] | None = None,
    final_holdout_accessed: bool = False,
    final_holdout_inference_performed: bool = False,
    final_holdout_evaluated: bool = False,
) -> dict[str, Any]:
    return {
        "evaluation_attempt_count": 0 if state == "not_started" else 1,
        "failure": None if failure is None else dict(failure),
        "final_holdout_accessed": final_holdout_accessed,
        "final_holdout_evaluated": final_holdout_evaluated,
        "final_holdout_inference_performed": final_holdout_inference_performed,
        "phase": "V2-C5 Step 23B/C",
        "results": None if results is None else dict(results),
        "schema_version": STATE_SCHEMA_VERSION,
        "source_lineage": source_lineage(contract, paths),
        "state": state,
    }


def validate_state_payload(
    payload: Mapping[str, Any], contract: Mapping[str, Any], paths: EvaluationPaths
) -> None:
    if payload.get("schema_version") != STATE_SCHEMA_VERSION:
        raise ValueError("unexpected final-evaluation state schema")
    if payload.get("phase") != "V2-C5 Step 23B/C":
        raise ValueError("unexpected final-evaluation state phase")
    if payload.get("source_lineage") != source_lineage(contract, paths):
        raise ValueError("final-evaluation state lineage changed")
    state = payload.get("state")
    if state not in {"not_started", "started", "completed", "failed_after_access"}:
        raise ValueError("unknown final-evaluation state")
    expected_attempts = 0 if state == "not_started" else 1
    if payload.get("evaluation_attempt_count") != expected_attempts:
        raise ValueError("final-evaluation attempt count changed")
    accessed = payload.get("final_holdout_accessed")
    inferred = payload.get("final_holdout_inference_performed")
    evaluated = payload.get("final_holdout_evaluated")
    if state in {"not_started", "started"} and any((accessed, inferred, evaluated)):
        raise ValueError("pre-access final-evaluation state has consumed-work flags")
    if state == "completed" and not all((accessed, inferred, evaluated)):
        raise ValueError("completed final-evaluation state lacks completion flags")
    if state == "failed_after_access" and accessed is not True:
        raise ValueError("failed-after-access state must record holdout access")
    if state == "failed_after_access" and evaluated is not False:
        raise ValueError("failed-after-access state cannot claim completed evaluation")
    if state != "failed_after_access" and payload.get("failure") is not None:
        raise ValueError("failure metadata is allowed only after access failure")
    if state == "completed" and not isinstance(payload.get("results"), dict):
        raise ValueError("completed state must bind final results")
    if state != "completed" and payload.get("results") is not None:
        raise ValueError("only completed state may bind final results")


def ensure_result_pair_absent(paths: EvaluationPaths) -> None:
    if paths.results.exists() or paths.results_manifest.exists():
        raise FileExistsError("final-evaluation result artifacts already exist")


def load_preconditions(
    paths: EvaluationPaths = DEFAULT_PATHS,
    *,
    artifact_loader: Callable[[Path], Any] = joblib.load,
    load_classifier: bool = True,
    require_results_absent: bool = True,
) -> dict[str, Any]:
    contract = read_json_object(paths.contract)
    validate_contract(contract, paths)
    sources = contract["source_artifacts"]

    validate_file_binding(
        paths.selected_model_manifest,
        sources["selected_model_manifest"],
        paths,
        "selected-model manifest",
    )
    selected_manifest = read_json_object(paths.selected_model_manifest)
    validate_selected_model_manifest(selected_manifest, contract)

    validate_file_binding(
        paths.final_holdout_manifest,
        sources["final_holdout_manifest"],
        paths,
        "final-holdout manifest",
    )
    holdout_manifest = read_json_object(paths.final_holdout_manifest)
    validate_holdout_manifest(holdout_manifest, contract)

    validate_file_binding(
        paths.final_holdout_contract,
        sources["final_holdout_contract"],
        paths,
        "final-holdout contract",
    )
    validate_file_binding(
        paths.model_selection_results,
        sources["model_selection_results"],
        paths,
        "model-selection results",
    )
    validate_file_binding(paths.taxonomy, sources["taxonomy"], paths, "taxonomy")
    taxonomy = read_json_object(paths.taxonomy)
    validate_taxonomy(taxonomy, contract)

    artifact_specification = sources["selected_classifier_artifact"]
    if display_path(paths.classifier_artifact, paths) != artifact_specification["path"]:
        raise ValueError("selected classifier artifact path changed")
    if sha256_file(paths.classifier_artifact) != artifact_specification["sha256"]:
        raise ValueError("selected classifier artifact hash changed")
    if paths.classifier_artifact.stat().st_size != artifact_specification["size_bytes"]:
        raise ValueError("selected classifier artifact size changed")

    classifier = None
    if load_classifier:
        artifact_payload = artifact_loader(paths.classifier_artifact)
        if not isinstance(artifact_payload, dict):
            raise TypeError("selected classifier artifact must contain an object")
        classifier = validate_classifier_payload(
            artifact_payload, selected_manifest, contract
        )

    if require_results_absent:
        ensure_result_pair_absent(paths)
    state = None
    if paths.state.exists():
        state = read_json_object(paths.state)
        validate_state_payload(state, contract, paths)
    return {
        "classifier": classifier,
        "contract": contract,
        "holdout_manifest": holdout_manifest,
        "selected_model_manifest": selected_manifest,
        "state": state,
        "taxonomy": taxonomy,
    }


def preflight(
    paths: EvaluationPaths = DEFAULT_PATHS,
    *,
    artifact_loader: Callable[[Path], Any] = joblib.load,
) -> dict[str, Any]:
    inputs = load_preconditions(paths, artifact_loader=artifact_loader)
    state = inputs["state"]
    if state is not None and state["state"] != "not_started":
        raise RuntimeError(f"final evaluation cannot start from state={state['state']}")
    return {
        "classifier_artifact_validated": True,
        "embeddings_generated": False,
        "files_written": False,
        "final_holdout_accessed": False,
        "final_holdout_evaluated": False,
        "final_holdout_inference_performed": False,
        "holdout_manifest_validated": True,
        "inference_performed": False,
        "selected_candidate_id": EXPECTED_CANDIDATE_ID,
        "state": "absent" if state is None else "not_started",
        "status": "ready",
    }


def initialize_state(
    paths: EvaluationPaths = DEFAULT_PATHS,
    *,
    artifact_loader: Callable[[Path], Any] = joblib.load,
) -> Path:
    inputs = load_preconditions(paths, artifact_loader=artifact_loader)
    if paths.state.exists():
        raise FileExistsError("refusing to overwrite existing final-evaluation state")
    payload = build_state_payload(inputs["contract"], paths, "not_started")
    validate_state_payload(payload, inputs["contract"], paths)
    durable_write_bytes(
        paths.state,
        stable_json_bytes(payload),
        refuse_existing=True,
    )
    return paths.state


def load_holdout_bytes(path: Path) -> bytes:
    return path.read_bytes()


def embed_holdout_texts(
    texts: Sequence[str], representation: Mapping[str, Any]
) -> np.ndarray:
    try:
        from fastembed import TextEmbedding
    except ImportError as exc:
        raise RuntimeError("fastembed is required for final evaluation") from exc
    model = TextEmbedding(model_name=representation["model_identifier"])
    vectors = np.asarray(
        list(model.passage_embed(list(texts), batch_size=256)),
        dtype=np.float32,
    )
    if vectors.shape != (len(texts), representation["dimensions"]):
        raise ValueError(f"unexpected final-holdout embedding shape: {vectors.shape}")
    if not np.isfinite(vectors).all():
        raise ValueError("final-holdout embeddings contain non-finite values")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("final-holdout embedding contains a zero vector")
    return np.asarray(vectors / norms, dtype=np.float32)


def validate_embedding_matrix(
    embeddings: np.ndarray,
    record_count: int,
    representation: Mapping[str, Any],
) -> None:
    expected_shape = (record_count, representation["dimensions"])
    if embeddings.shape != expected_shape:
        raise ValueError(f"final-holdout embedding shape changed: {embeddings.shape}")
    if embeddings.dtype != np.float32:
        raise ValueError("final-holdout embeddings must use float32")
    if not np.isfinite(embeddings).all():
        raise ValueError("final-holdout embeddings contain non-finite values")
    if not np.allclose(
        np.linalg.norm(embeddings, axis=1),
        1.0,
        rtol=1e-5,
        atol=1e-6,
    ):
        raise ValueError("final-holdout embeddings are not L2 normalized")


def validate_holdout_payload(
    payload: Mapping[str, Any],
    contract: Mapping[str, Any],
    taxonomy: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if payload.get("schema_version") != HOLDOUT_SCHEMA_VERSION:
        raise ValueError("unexpected final-holdout dataset schema")
    examples = payload.get("examples")
    if not isinstance(examples, list):
        raise TypeError("final-holdout examples must be a list")
    holdout = contract["holdout"]
    labels = contract["model_under_test"]["class_labels"]
    if len(examples) != holdout["total_example_count"]:
        raise ValueError("final-holdout record count changed")
    if payload.get("example_count") != len(examples):
        raise ValueError("final-holdout declared record count changed")
    identifiers: set[str] = set()
    counts: Counter[str] = Counter()
    validated: list[dict[str, Any]] = []
    for index, record in enumerate(examples):
        if not isinstance(record, dict):
            raise TypeError(f"final-holdout record {index} must be an object")
        example_id = record.get("example_id")
        intent = record.get("intent")
        text = record.get("text")
        if not isinstance(example_id, str) or not example_id:
            raise ValueError(f"final-holdout record {index} has an invalid ID")
        if example_id in identifiers:
            raise ValueError(f"duplicate final-holdout example ID: {example_id}")
        if intent not in labels:
            raise ValueError(f"final-holdout record {index} has an unknown intent")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"final-holdout record {index} has empty text")
        if record.get("risk") != taxonomy["risk_by_intent"][intent]:
            raise ValueError(f"final-holdout record {index} risk changed")
        identifiers.add(example_id)
        counts[str(intent)] += 1
        validated.append(record)
    if dict(counts) != holdout["intent_counts"]:
        raise ValueError("final-holdout intent balance changed")
    protected = set(contract["safety_gates"]["protected_write_intents"])
    protected_count = sum(counts[label] for label in protected)
    if protected_count != holdout["protected_write_example_count"]:
        raise ValueError("final-holdout protected-write population changed")
    if len(validated) - protected_count != holdout["non_protected_example_count"]:
        raise ValueError("final-holdout non-protected population changed")
    return validated


def classification_metrics(
    gold: Sequence[str], predicted: Sequence[str], contract: Mapping[str, Any]
) -> dict[str, Any]:
    labels = contract["model_under_test"]["class_labels"]
    precision, recall, f1, support = precision_recall_fscore_support(
        gold,
        predicted,
        labels=labels,
        zero_division=0,
    )
    per_intent = {
        label: {
            "f1": float(f1[index]),
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(labels)
    }

    def subset_macro_f1(subset: Sequence[str]) -> float:
        indices = [index for index, label in enumerate(gold) if label in subset]
        subset_gold = [gold[index] for index in indices]
        subset_predicted = [predicted[index] for index in indices]
        return float(
            f1_score(
                subset_gold,
                subset_predicted,
                labels=list(subset),
                average="macro",
                zero_division=0,
            )
        )

    subsets = contract["metrics"]["subset_macro_f1"]
    return {
        "accuracy": float(accuracy_score(gold, predicted)),
        "balanced_accuracy": float(np.mean(recall)),
        "confusion_matrix": {
            "label_order": list(labels),
            "values": confusion_matrix(gold, predicted, labels=labels).tolist(),
        },
        "historical_nine_label_macro_f1": subset_macro_f1(
            subsets["historical_nine_label_macro_f1"]
        ),
        "macro_f1": float(
            f1_score(
                gold,
                predicted,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "new_seven_intent_macro_f1": subset_macro_f1(
            subsets["new_seven_intent_macro_f1"]
        ),
        "per_intent": per_intent,
    }


def safety_metrics(
    gold: Sequence[str], predicted: Sequence[str], contract: Mapping[str, Any]
) -> dict[str, float | int]:
    if len(gold) != len(predicted):
        raise ValueError("gold and prediction counts differ")
    protected = set(contract["safety_gates"]["protected_write_intents"])
    pairs = list(zip(gold, predicted, strict=True))
    non_protected_total = sum(expected not in protected for expected, _ in pairs)
    protected_false_positives = sum(
        expected not in protected and guess in protected
        for expected, guess in pairs
    )
    protected_total = sum(expected in protected for expected, _ in pairs)
    protected_exact = sum(
        expected in protected and guess == expected for expected, guess in pairs
    )
    unsupported_total = sum(
        expected == "unsupported_or_uncertain" for expected, _ in pairs
    )
    unsupported_exact = sum(
        expected == "unsupported_or_uncertain"
        and guess == "unsupported_or_uncertain"
        for expected, guess in pairs
    )
    if (non_protected_total, protected_total, unsupported_total) != (480, 160, 40):
        raise ValueError("final safety-metric denominators changed")
    return {
        "exact_protected_write_correct_count": protected_exact,
        "exact_protected_write_recall": protected_exact / protected_total,
        "protected_write_false_positive_count": protected_false_positives,
        "protected_write_false_positive_rate": (
            protected_false_positives / non_protected_total
        ),
        "protected_write_gold_denominator": protected_total,
        "protected_write_non_protected_denominator": non_protected_total,
        "unsupported_or_uncertain_correct_count": unsupported_exact,
        "unsupported_or_uncertain_gold_denominator": unsupported_total,
        "unsupported_or_uncertain_recall": unsupported_exact / unsupported_total,
    }


def apply_safety_gates(
    metrics: Mapping[str, float | int], contract: Mapping[str, Any]
) -> dict[str, Any]:
    results: dict[str, dict[str, Any]] = {}
    for gate in contract["safety_gates"]["gates"]:
        name = gate["metric"]
        value = metrics[name]
        if gate["comparison"] == "less_than_or_equal":
            passed = value <= gate["threshold"]
            operator = "<="
        elif gate["comparison"] == "greater_than_or_equal":
            passed = value >= gate["threshold"]
            operator = ">="
        else:
            raise ValueError(f"unsupported safety comparison: {gate['comparison']}")
        results[name] = {
            "operator": operator,
            "passed": bool(passed),
            "threshold": gate["threshold"],
            "value": value,
        }
    return {
        "all_gates_pass": all(result["passed"] for result in results.values()),
        "gates": results,
    }


def prediction_sequence_sha256(
    examples: Sequence[Mapping[str, Any]], predictions: Sequence[str]
) -> str:
    sequence = [
        {
            "example_id": example["example_id"],
            "gold_label": example["intent"],
            "predicted_label": prediction,
        }
        for example, prediction in zip(examples, predictions, strict=True)
    ]
    return sha256_bytes(stable_json_bytes(sequence))


def build_results_payload(
    inputs: Mapping[str, Any],
    paths: EvaluationPaths,
    examples: Sequence[Mapping[str, Any]],
    predictions: Sequence[str],
) -> dict[str, Any]:
    contract = inputs["contract"]
    gold = [str(example["intent"]) for example in examples]
    metrics = classification_metrics(gold, predictions, contract)
    safety = safety_metrics(gold, predictions, contract)
    gates = apply_safety_gates(safety, contract)
    accepted = gates["all_gates_pass"]
    return {
        "all_mandatory_safety_gates_pass": accepted,
        "evaluation_completed": True,
        "final_holdout_accessed": True,
        "final_holdout_evaluated": True,
        "final_holdout_inference_performed": True,
        "final_model_acceptance_claimed": accepted,
        "metrics": metrics,
        "model_retraining_performed": False,
        "phase": "V2-C5 Step 23C",
        "prediction_count": len(predictions),
        "prediction_sequence_sha256": prediction_sequence_sha256(
            examples, predictions
        ),
        "runtime_behavior_changed": False,
        "runtime_eligible": False,
        "safety_gate_results": gates,
        "safety_metrics": safety,
        "schema_version": RESULTS_SCHEMA_VERSION,
        "selected_candidate_id": contract["model_under_test"][
            "selected_candidate_id"
        ],
        "source_lineage": source_lineage(contract, paths),
        "threshold_tuning_performed": False,
    }


def build_results_manifest(
    results_bytes: bytes,
    results: Mapping[str, Any],
    contract: Mapping[str, Any],
    paths: EvaluationPaths,
) -> dict[str, Any]:
    return {
        "all_mandatory_safety_gates_pass": results[
            "all_mandatory_safety_gates_pass"
        ],
        "evaluation_completed": True,
        "final_holdout_accessed": True,
        "final_holdout_evaluated": True,
        "final_holdout_inference_performed": True,
        "final_model_acceptance_claimed": results[
            "final_model_acceptance_claimed"
        ],
        "model_retraining_performed": False,
        "phase": "V2-C5 Step 23C",
        "prediction_count": results["prediction_count"],
        "prediction_sequence_sha256": results["prediction_sequence_sha256"],
        "results": {
            "path": display_path(paths.results, paths),
            "schema_version": RESULTS_SCHEMA_VERSION,
            "sha256": sha256_bytes(results_bytes),
        },
        "runner": {
            "path": display_path(paths.runner, paths),
            "sha256": sha256_file(paths.runner),
        },
        "runtime_behavior_changed": False,
        "runtime_eligible": False,
        "schema_version": RESULTS_MANIFEST_SCHEMA_VERSION,
        "source_lineage": source_lineage(contract, paths),
        "threshold_tuning_performed": False,
    }


def failure_metadata(stage: str, error: BaseException) -> dict[str, str]:
    return {
        "exception_type": type(error).__name__,
        "message_sha256": sha256_bytes(str(error).encode("utf-8")),
        "stage": stage,
    }


def evaluate(
    paths: EvaluationPaths = DEFAULT_PATHS,
    *,
    artifact_loader: Callable[[Path], Any] = joblib.load,
    holdout_loader: Callable[[Path], bytes] = load_holdout_bytes,
    embedder: Callable[[Sequence[str], Mapping[str, Any]], np.ndarray] = (
        embed_holdout_texts
    ),
) -> tuple[Path, Path]:
    inputs = load_preconditions(
        paths,
        artifact_loader=artifact_loader,
        require_results_absent=False,
    )
    state = inputs["state"]
    if state is None:
        raise FileNotFoundError("initialized final-evaluation state is required")
    if state["state"] != "not_started":
        raise RuntimeError(f"final evaluation cannot run from state={state['state']}")
    ensure_result_pair_absent(paths)

    started = build_state_payload(inputs["contract"], paths, "started")
    validate_state_payload(started, inputs["contract"], paths)
    durable_write_bytes(paths.state, stable_json_bytes(started), refuse_existing=False)

    stage = "opening_final_holdout"
    accessed = False
    inferred = False
    try:
        accessed = True
        holdout_bytes = holdout_loader(paths.final_holdout_dataset)
        stage = "validating_final_holdout_hash"
        declared_hash = inputs["contract"]["source_artifacts"][
            "final_holdout_dataset_declaration"
        ]["sha256"]
        if sha256_bytes(holdout_bytes) != declared_hash:
            raise ValueError("final-holdout dataset hash differs from declaration")
        holdout_payload = json.loads(holdout_bytes)
        if not isinstance(holdout_payload, dict):
            raise TypeError("final-holdout dataset root must be an object")
        stage = "validating_final_holdout_population"
        examples = validate_holdout_payload(
            holdout_payload, inputs["contract"], inputs["taxonomy"]
        )

        stage = "embedding_final_holdout"
        embeddings = embedder(
            [str(example["text"]) for example in examples],
            inputs["contract"]["model_under_test"]["representation"],
        )
        validate_embedding_matrix(
            embeddings,
            len(examples),
            inputs["contract"]["model_under_test"]["representation"],
        )

        stage = "predicting_final_holdout"
        raw_predictions = inputs["classifier"].predict(embeddings)
        inferred = True
        predictions = [str(value) for value in np.asarray(raw_predictions).tolist()]
        labels = set(inputs["contract"]["model_under_test"]["class_labels"])
        if len(predictions) != len(examples) or any(
            prediction not in labels for prediction in predictions
        ):
            raise ValueError("final-holdout classifier predictions changed shape or labels")

        stage = "building_final_results"
        results = build_results_payload(inputs, paths, examples, predictions)
        results_bytes = stable_json_bytes(results)
        manifest = build_results_manifest(
            results_bytes, results, inputs["contract"], paths
        )
        manifest_bytes = stable_json_bytes(manifest)
        stage = "writing_final_results"
        durable_write_bytes(paths.results, results_bytes, refuse_existing=True)
        durable_write_bytes(
            paths.results_manifest,
            manifest_bytes,
            refuse_existing=True,
        )
        completed = build_state_payload(
            inputs["contract"],
            paths,
            "completed",
            final_holdout_accessed=True,
            final_holdout_inference_performed=True,
            final_holdout_evaluated=True,
            results={
                "results_manifest_sha256": sha256_bytes(manifest_bytes),
                "results_sha256": sha256_bytes(results_bytes),
            },
        )
        validate_state_payload(completed, inputs["contract"], paths)
        stage = "completing_evaluation_state"
        durable_write_bytes(
            paths.state,
            stable_json_bytes(completed),
            refuse_existing=False,
        )
        return paths.results, paths.results_manifest
    except BaseException as error:
        failed = build_state_payload(
            inputs["contract"],
            paths,
            "failed_after_access",
            failure=failure_metadata(stage, error),
            final_holdout_accessed=accessed,
            final_holdout_inference_performed=inferred,
            final_holdout_evaluated=False,
        )
        durable_write_bytes(
            paths.state,
            stable_json_bytes(failed),
            refuse_existing=False,
        )
        raise


def _safe_ratio(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def metrics_from_confusion(
    matrix: Sequence[Sequence[int]], contract: Mapping[str, Any]
) -> dict[str, Any]:
    values = np.asarray(matrix, dtype=np.int64)
    labels = contract["model_under_test"]["class_labels"]
    if values.shape != (16, 16) or np.any(values < 0):
        raise ValueError("final confusion matrix shape or counts changed")
    supports = values.sum(axis=1)
    predicted_counts = values.sum(axis=0)
    if values.sum() != 640 or not np.all(supports == 40):
        raise ValueError("final confusion matrix population changed")
    recalls: list[float] = []
    f1_values: list[float] = []
    per_intent: dict[str, dict[str, float | int]] = {}
    for index, label in enumerate(labels):
        true_positive = int(values[index, index])
        precision = _safe_ratio(true_positive, int(predicted_counts[index]))
        recall = _safe_ratio(true_positive, int(supports[index]))
        f1_value = _safe_ratio(2 * precision * recall, precision + recall)
        recalls.append(recall)
        f1_values.append(f1_value)
        per_intent[label] = {
            "f1": f1_value,
            "precision": precision,
            "recall": recall,
            "support": int(supports[index]),
        }

    def subset_macro_f1(subset: Sequence[str]) -> float:
        indices = [labels.index(label) for label in subset]
        subset_scores: list[float] = []
        for index in indices:
            true_positive = int(values[index, index])
            predicted_within_subset = int(values[indices, index].sum())
            precision = _safe_ratio(true_positive, predicted_within_subset)
            recall = _safe_ratio(true_positive, int(supports[index]))
            subset_scores.append(
                _safe_ratio(2 * precision * recall, precision + recall)
            )
        return float(np.mean(subset_scores))

    subsets = contract["metrics"]["subset_macro_f1"]
    return {
        "accuracy": int(np.trace(values)) / int(values.sum()),
        "balanced_accuracy": float(np.mean(recalls)),
        "confusion_matrix": {
            "label_order": list(labels),
            "values": values.tolist(),
        },
        "historical_nine_label_macro_f1": subset_macro_f1(
            subsets["historical_nine_label_macro_f1"]
        ),
        "macro_f1": float(np.mean(f1_values)),
        "new_seven_intent_macro_f1": subset_macro_f1(
            subsets["new_seven_intent_macro_f1"]
        ),
        "per_intent": per_intent,
    }


def safety_from_confusion(
    matrix: Sequence[Sequence[int]], contract: Mapping[str, Any]
) -> dict[str, float | int]:
    values = np.asarray(matrix, dtype=np.int64)
    labels = contract["model_under_test"]["class_labels"]
    protected = set(contract["safety_gates"]["protected_write_intents"])
    protected_indices = [index for index, label in enumerate(labels) if label in protected]
    non_protected_indices = [
        index for index, label in enumerate(labels) if label not in protected
    ]
    unsupported_index = labels.index("unsupported_or_uncertain")
    false_positives = int(values[np.ix_(non_protected_indices, protected_indices)].sum())
    protected_exact = int(sum(values[index, index] for index in protected_indices))
    unsupported_exact = int(values[unsupported_index, unsupported_index])
    return {
        "exact_protected_write_correct_count": protected_exact,
        "exact_protected_write_recall": protected_exact / 160,
        "protected_write_false_positive_count": false_positives,
        "protected_write_false_positive_rate": false_positives / 480,
        "protected_write_gold_denominator": 160,
        "protected_write_non_protected_denominator": 480,
        "unsupported_or_uncertain_correct_count": unsupported_exact,
        "unsupported_or_uncertain_gold_denominator": 40,
        "unsupported_or_uncertain_recall": unsupported_exact / 40,
    }


def values_close(first: Any, second: Any) -> bool:
    if isinstance(first, dict) and isinstance(second, dict):
        return set(first) == set(second) and all(
            values_close(first[key], second[key]) for key in first
        )
    if isinstance(first, list) and isinstance(second, list):
        return len(first) == len(second) and all(
            values_close(left, right)
            for left, right in zip(first, second, strict=True)
        )
    if isinstance(first, float) or isinstance(second, float):
        return math.isclose(float(first), float(second), rel_tol=0.0, abs_tol=1e-12)
    return first == second


def recursive_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            keys.add(str(key))
            keys.update(recursive_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.update(recursive_keys(child))
    return keys


def validate_results_payload(
    results: Mapping[str, Any], contract: Mapping[str, Any], paths: EvaluationPaths
) -> None:
    if results.get("schema_version") != RESULTS_SCHEMA_VERSION:
        raise ValueError("unexpected final-evaluation result schema")
    if results.get("phase") != "V2-C5 Step 23C":
        raise ValueError("unexpected final-evaluation result phase")
    if results.get("source_lineage") != source_lineage(contract, paths):
        raise ValueError("final-evaluation result lineage changed")
    if results.get("selected_candidate_id") != EXPECTED_CANDIDATE_ID:
        raise ValueError("final-evaluation result candidate changed")
    if results.get("prediction_count") != 640:
        raise ValueError("final-evaluation prediction count changed")
    digest = results.get("prediction_sequence_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("final-evaluation prediction digest is invalid")
    metrics = results.get("metrics", {})
    matrix = metrics.get("confusion_matrix", {}).get("values")
    expected_metrics = metrics_from_confusion(matrix, contract)
    if not values_close(metrics, expected_metrics):
        raise ValueError("final-evaluation aggregate metrics are inconsistent")
    expected_safety = safety_from_confusion(matrix, contract)
    if not values_close(results.get("safety_metrics"), expected_safety):
        raise ValueError("final-evaluation safety metrics are inconsistent")
    expected_gates = apply_safety_gates(expected_safety, contract)
    if not values_close(results.get("safety_gate_results"), expected_gates):
        raise ValueError("final-evaluation safety gates are inconsistent")
    accepted = expected_gates["all_gates_pass"]
    required = {
        "all_mandatory_safety_gates_pass": accepted,
        "evaluation_completed": True,
        "final_holdout_accessed": True,
        "final_holdout_evaluated": True,
        "final_holdout_inference_performed": True,
        "final_model_acceptance_claimed": accepted,
        "model_retraining_performed": False,
        "runtime_behavior_changed": False,
        "runtime_eligible": False,
        "threshold_tuning_performed": False,
    }
    if any(results.get(field) != value for field, value in required.items()):
        raise ValueError("final-evaluation result decision or governance changed")
    prohibited_keys = {"text", "texts", "utterance", "utterances", "raw_text"}
    if prohibited_keys & recursive_keys(results):
        raise ValueError("final-evaluation results contain raw-text fields")


def validate_results_manifest(
    manifest: Mapping[str, Any],
    results_bytes: bytes,
    results: Mapping[str, Any],
    contract: Mapping[str, Any],
    paths: EvaluationPaths,
) -> None:
    if manifest.get("schema_version") != RESULTS_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unexpected final-evaluation results-manifest schema")
    if manifest.get("phase") != "V2-C5 Step 23C":
        raise ValueError("unexpected final-evaluation results-manifest phase")
    if manifest.get("results") != {
        "path": display_path(paths.results, paths),
        "schema_version": RESULTS_SCHEMA_VERSION,
        "sha256": sha256_bytes(results_bytes),
    }:
        raise ValueError("final-evaluation result hash or path changed")
    if manifest.get("source_lineage") != source_lineage(contract, paths):
        raise ValueError("final-evaluation results-manifest lineage changed")
    if manifest.get("runner") != {
        "path": display_path(paths.runner, paths),
        "sha256": sha256_file(paths.runner),
    }:
        raise ValueError("final-evaluation runner lineage changed")
    mirrored = {
        "all_mandatory_safety_gates_pass",
        "evaluation_completed",
        "final_holdout_accessed",
        "final_holdout_evaluated",
        "final_holdout_inference_performed",
        "final_model_acceptance_claimed",
        "model_retraining_performed",
        "prediction_count",
        "prediction_sequence_sha256",
        "runtime_behavior_changed",
        "runtime_eligible",
        "threshold_tuning_performed",
    }
    if any(manifest.get(field) != results.get(field) for field in mirrored):
        raise ValueError("final-evaluation results-manifest summary changed")


def check_results(paths: EvaluationPaths = DEFAULT_PATHS) -> dict[str, Any]:
    inputs = load_preconditions(
        paths,
        load_classifier=False,
        require_results_absent=False,
    )
    state = inputs["state"]
    if state is None or state["state"] != "completed":
        raise RuntimeError("completed final-evaluation state is required")
    if not paths.results.is_file() or not paths.results_manifest.is_file():
        raise FileNotFoundError("complete final-evaluation result pair is required")
    results_bytes = paths.results.read_bytes()
    results = json.loads(results_bytes)
    if not isinstance(results, dict):
        raise TypeError("final-evaluation result root must be an object")
    if results_bytes != stable_json_bytes(results):
        raise ValueError("final-evaluation result serialization changed")
    validate_results_payload(results, inputs["contract"], paths)
    manifest_bytes = paths.results_manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    if not isinstance(manifest, dict):
        raise TypeError("final-evaluation results-manifest root must be an object")
    if manifest_bytes != stable_json_bytes(manifest):
        raise ValueError("final-evaluation results-manifest serialization changed")
    validate_results_manifest(
        manifest,
        results_bytes,
        results,
        inputs["contract"],
        paths,
    )
    expected_state_results = {
        "results_manifest_sha256": sha256_bytes(manifest_bytes),
        "results_sha256": sha256_bytes(results_bytes),
    }
    if state.get("results") != expected_state_results:
        raise ValueError("completed state result lineage changed")
    return {
        "all_mandatory_safety_gates_pass": results[
            "all_mandatory_safety_gates_pass"
        ],
        "files_written": False,
        "final_holdout_accessed": True,
        "final_holdout_evaluated": True,
        "final_holdout_inference_performed": True,
        "final_model_acceptance_claimed": results[
            "final_model_acceptance_claimed"
        ],
        "inference_performed": False,
        "runtime_behavior_changed": False,
        "runtime_eligible": False,
        "status": "valid",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the once-only V2-C5 final-holdout evaluation"
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--initialize-state", action="store_true")
    modes.add_argument("--evaluate", action="store_true")
    modes.add_argument("--check-results", action="store_true")
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    if arguments.preflight:
        print(json.dumps(preflight(), indent=2, sort_keys=True))
    elif arguments.initialize_state:
        path = initialize_state()
        print(path.relative_to(REPOSITORY_ROOT))
    elif arguments.evaluate:
        outputs = evaluate()
        print("\n".join(str(path.relative_to(REPOSITORY_ROOT)) for path in outputs))
    elif arguments.check_results:
        print(json.dumps(check_results(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
