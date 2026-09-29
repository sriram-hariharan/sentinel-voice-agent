"""Evaluate V2-C3, Candidate A, and Candidate B on one shared probe."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import joblib
from sklearn.svm import LinearSVC

try:
    from scripts import (
        evaluate_v2c4_candidate_a as evaluate_a,
    )
    from scripts import (
        train_v2c4_candidate_a as train_a,
    )
    from scripts import (
        train_v2c4_candidate_b as train_b,
    )
except ModuleNotFoundError:  # Direct execution from the scripts directory.
    import evaluate_v2c4_candidate_a as evaluate_a
    import train_v2c4_candidate_a as train_a
    import train_v2c4_candidate_b as train_b


EXPECTED_REPORT_KEYS = {
    "schema_version",
    "experiment_metadata",
    "input_hashes",
    "baseline_metrics",
    "candidate_a_metrics",
    "candidate_b_metrics",
    "candidate_b_deltas",
    "safety_gate_results",
    "macro_f1_regression_check",
    "selection_decision",
    "supporting_cv",
    "stage_diagnostics",
    "latency",
    "artifact_metadata",
    "integrity_checks",
    "analysis_role",
    "final_acceptance_evidence",
    "final_holdout_accessed",
    "final_improvement_claimed",
}
EXPECTED_TRAINING_REPORT_KEYS = {
    "schema_version",
    "experiment_phase",
    "analysis_role",
    "candidate_id",
    "candidate_config_sha256",
    "input_hashes",
    "training_corpus",
    "embedding_metadata",
    "stage_training",
    "supporting_cv",
    "artifact_metadata",
    "safety_boundary",
    "integrity_checks",
    "training_performed",
    "selection_probe_evaluated",
    "final_acceptance_evidence",
    "final_holdout_accessed",
    "final_improvement_claimed",
}


def validate_candidate_b_training_report(
    report: Mapping[str, Any], config: Mapping[str, Any]
) -> None:
    if set(report) != EXPECTED_TRAINING_REPORT_KEYS:
        raise ValueError("Candidate B training report structure changed")
    if report.get("schema_version") != "v2c4-candidate-b-training-report.v1":
        raise ValueError("unexpected Candidate B training report schema")
    expected_scalars = {
        "experiment_phase": "V2-C4-Step-12",
        "analysis_role": "v2c4_development_model_selection",
        "candidate_id": config["candidate_id"],
        "candidate_config_sha256": train_a.sha256_file(train_b.CONFIG_PATH),
        "training_performed": True,
        "selection_probe_evaluated": False,
        "final_acceptance_evidence": False,
        "final_holdout_accessed": False,
        "final_improvement_claimed": False,
    }
    for field, expected in expected_scalars.items():
        if report.get(field) != expected:
            raise ValueError(f"Candidate B training report changed: {field}")
    expected_hashes = {
        name: spec["sha256"] for name, spec in config["frozen_inputs"].items()
    }
    if report.get("input_hashes") != expected_hashes:
        raise ValueError("Candidate B training input hashes changed")
    training_corpus = report.get("training_corpus", {})
    if training_corpus.get("example_count") != 8558:
        raise ValueError("Candidate B training report count changed")
    embedding_metadata = report.get("embedding_metadata", {})
    if embedding_metadata.get("strategy") != config["embedding_strategy"][
        "strategy"
    ]:
        raise ValueError("Candidate B embedding strategy changed in training report")
    if embedding_metadata.get("combined_shape") != [8558, 384]:
        raise ValueError("Candidate B embedding shape changed in training report")
    if embedding_metadata.get("finite_values") is not True:
        raise ValueError("Candidate B training report has invalid embeddings")
    if embedding_metadata.get("new_embeddings_performed") is not False:
        raise ValueError("Candidate B unexpectedly created training embeddings")
    if embedding_metadata.get("development_cache_sha256") != config[
        "frozen_inputs"
    ]["v2c3_development_embedding_cache"]["sha256"]:
        raise ValueError("Candidate B development embedding cache changed")
    if embedding_metadata.get("augmentation_cache_sha256") != config[
        "frozen_inputs"
    ]["candidate_a_augmentation_embedding_cache"]["sha256"]:
        raise ValueError("Candidate B augmentation embedding cache changed")
    stages = report.get("stage_training", {})
    if set(stages) != set(train_b.STAGE_CLASSES):
        raise ValueError("Candidate B training report stage set changed")
    expected_counts = config["training_corpus"]["expected_stage_counts"]
    for stage, declared_classes in train_b.STAGE_CLASSES.items():
        metadata = stages[stage]
        if metadata.get("training_count") != expected_counts[stage]:
            raise ValueError(f"Candidate B training count changed for {stage}")
        if metadata.get("class_declaration_order") != list(declared_classes):
            raise ValueError(f"Candidate B declared classes changed for {stage}")
        if metadata.get("classes") != list(train_b.FITTED_STAGE_CLASSES[stage]):
            raise ValueError(f"Candidate B fitted classes changed for {stage}")
        if metadata.get("parameters") != config["classifier"]["parameters"]:
            raise ValueError(f"Candidate B parameters changed for {stage}")
    cv = report.get("supporting_cv", {})
    expected_cv = config["supporting_cv"]
    for field in ("method", "n_splits", "shuffle", "random_state"):
        if cv.get(field) != expected_cv[field]:
            raise ValueError(f"Candidate B supporting CV changed: {field}")
    if cv.get("record_count") != 8558:
        raise ValueError("Candidate B supporting CV record count changed")
    if cv.get("real_hierarchical_oof_inference") is not True:
        raise ValueError("Candidate B CV did not use real hierarchical inference")
    if cv.get("gold_upstream_routing_used_for_final_predictions") is not False:
        raise ValueError("Candidate B CV used gold upstream routing")
    if cv.get("every_record_received_one_final_prediction") is not True:
        raise ValueError("Candidate B CV has incomplete OOF predictions")
    if cv.get("group_leakage_detected") is not False:
        raise ValueError("Candidate B CV reports group leakage")
    if cv.get("hyperparameter_tuning_performed") is not False:
        raise ValueError("Candidate B CV reports hyperparameter tuning")
    folds = cv.get("fold_results")
    if not isinstance(folds, list) or len(folds) != 5:
        raise ValueError("Candidate B supporting CV fold count changed")
    if sum(fold.get("validation_count", 0) for fold in folds) != 8558:
        raise ValueError("Candidate B supporting CV validation coverage changed")
    if any(fold.get("group_overlap_count") != 0 for fold in folds):
        raise ValueError("Candidate B supporting CV fold reports group leakage")
    if any(
        fold.get("inference_routing") != "predicted_upstream_labels_only"
        for fold in folds
    ):
        raise ValueError("Candidate B supporting CV fold used invalid routing")
    pooled = cv.get("pooled_final_9_intent_metrics", {})
    if pooled.get("example_count") != 8558:
        raise ValueError("Candidate B pooled OOF metrics are incomplete")
    boundary = report.get("safety_boundary", {})
    if boundary.get("advisory_routing_only") is not True:
        raise ValueError("Candidate B training report crossed advisory boundary")
    if boundary.get("authorization_mechanism") is not False:
        raise ValueError("Candidate B training report claims authorization authority")
    checks = report.get("integrity_checks", {})
    required_true = (
        "all_frozen_input_hashes_verified",
        "candidate_a_trigger_verified",
        "same_candidate_a_training_corpus",
        "exact_stage_training_counts_verified",
        "candidate_a_embedding_cache_hashes_verified",
        "selection_probe_ids_text_and_groups_excluded",
        "real_hierarchical_oof_inference",
    )
    required_false = (
        "selection_probe_used_for_fitting",
        "consumed_v2c3_challenge_used",
        "consumed_v2c3_external_lockbox_used",
        "cfpb_used",
        "test_only_sources_used",
        "gold_upstream_routing_used_for_final_predictions",
        "final_holdout_accessed",
        "threshold_tuning_performed",
        "hyperparameter_search_performed",
    )
    if any(checks.get(field) is not True for field in required_true):
        raise ValueError("Candidate B training report lost a required integrity check")
    if any(checks.get(field) is not False for field in required_false):
        raise ValueError("Candidate B training report violates an integrity boundary")


def validate_candidate_b_artifact(
    artifact: Any, config: Mapping[str, Any]
) -> dict[str, LinearSVC]:
    if not isinstance(artifact, dict):
        raise TypeError("Candidate B artifact must be a dictionary")
    expected = {
        "artifact_schema_version": "v2c4-candidate-b-classifier.v1",
        "candidate_id": config["candidate_id"],
        "architecture": "hierarchical",
        "representation": config["representation"],
        "classifier_parameters": config["classifier"]["parameters"],
        "intent_label_order": list(train_a.EXPECTED_INTENTS),
        "stage_classes": {
            name: list(labels) for name, labels in train_b.STAGE_CLASSES.items()
        },
        "fitted_stage_classes": {
            name: list(labels)
            for name, labels in train_b.FITTED_STAGE_CLASSES.items()
        },
        "training_example_count": 8558,
        "training_input_sha256": {
            "v2c3_development": config["frozen_inputs"]["v2c3_development"][
                "sha256"
            ],
            "v2c4_training_augmentation": config["frozen_inputs"][
                "v2c4_training_augmentation"
            ]["sha256"],
        },
        "embedding_cache_sha256": {
            "v2c3_development": config["frozen_inputs"][
                "v2c3_development_embedding_cache"
            ]["sha256"],
            "v2c4_training_augmentation": config["frozen_inputs"][
                "candidate_a_augmentation_embedding_cache"
            ]["sha256"],
        },
        "trusted_local_artifact": True,
        "advisory_routing_only": True,
        "runtime_authority": False,
        "selection_probe_used_for_fitting": False,
        "final_holdout_accessed": False,
    }
    for field, value in expected.items():
        if artifact.get(field) != value:
            raise ValueError(f"Candidate B artifact changed: {field}")
    models = artifact.get("stage_models")
    if not isinstance(models, dict) or set(models) != set(train_b.STAGE_CLASSES):
        raise ValueError("Candidate B artifact stage set changed")
    for stage, classifier in models.items():
        if not isinstance(classifier, LinearSVC):
            raise TypeError(f"Candidate B {stage} is not LinearSVC")
        if classifier.get_params(deep=False) != config["classifier"]["parameters"]:
            raise ValueError(f"Candidate B {stage} parameters changed")
        if tuple(classifier.classes_) != train_b.FITTED_STAGE_CLASSES[stage]:
            raise ValueError(f"Candidate B {stage} classes changed")
    return models


def load_candidate_b(
    config: Mapping[str, Any],
) -> tuple[dict[str, LinearSVC], dict[str, Any], Path, Path]:
    artifact_path = train_a.repository_path(config["outputs"]["candidate_artifact"])
    report_path = train_a.repository_path(config["outputs"]["training_report"])
    if not artifact_path.is_file() or not report_path.is_file():
        raise FileNotFoundError("Candidate B artifact and training report are required")
    report = train_a.read_json(report_path)
    validate_candidate_b_training_report(report, config)
    artifact_hash = train_a.sha256_file(artifact_path)
    metadata = report.get("artifact_metadata", {})
    if metadata.get("path") != config["outputs"]["candidate_artifact"]:
        raise ValueError("Candidate B artifact path differs from training report")
    if metadata.get("sha256") != artifact_hash:
        raise ValueError("Candidate B artifact hash differs from training report")
    if metadata.get("size_bytes") != artifact_path.stat().st_size:
        raise ValueError("Candidate B artifact size differs from training report")
    artifact = joblib.load(artifact_path)
    models = validate_candidate_b_artifact(artifact, config)
    if artifact.get("stage_metadata") != report.get("stage_training"):
        raise ValueError("Candidate B artifact and report stage metadata differ")
    return models, report, artifact_path, report_path


def predict_all_on_shared_embeddings(
    models_a: Mapping[str, Any],
    models_b: Mapping[str, Any],
    embeddings: Any,
) -> dict[str, Any]:
    baseline_predictions, baseline_latency = evaluate_a.predict_with_latency(
        models_a["baseline_classifier"], embeddings
    )
    candidate_a_predictions, candidate_a_latency = evaluate_a.predict_with_latency(
        models_a["candidate_classifier"], embeddings
    )
    started = time.perf_counter()
    candidate_b_predictions, routing_counts = train_b.hierarchical_predict(
        models_b, embeddings
    )
    candidate_b_latency = time.perf_counter() - started
    return {
        "baseline_predictions": baseline_predictions,
        "candidate_a_predictions": candidate_a_predictions,
        "candidate_b_predictions": candidate_b_predictions,
        "baseline_latency": baseline_latency,
        "candidate_a_latency": candidate_a_latency,
        "candidate_b_latency": candidate_b_latency,
        "candidate_b_routing_counts": routing_counts,
    }


def validate_frozen_comparator_metrics(
    candidate_a_report: Mapping[str, Any],
    baseline_metrics: Mapping[str, Any],
    candidate_a_metrics: Mapping[str, Any],
) -> None:
    if candidate_a_report.get("baseline_metrics") != baseline_metrics:
        raise ValueError("recomputed baseline metrics differ from frozen report")
    if candidate_a_report.get("candidate_a_metrics") != candidate_a_metrics:
        raise ValueError("recomputed Candidate A metrics differ from frozen report")


def metric_deltas(
    candidate_b: Mapping[str, Any], comparator: Mapping[str, Any]
) -> dict[str, float]:
    fields = (
        "accuracy",
        "macro_f1",
        "protected_write_recall",
        "protected_write_false_positive_rate",
        "unsupported_or_uncertain_recall",
        "false_supported_rate",
        "protected_exact_intent_accuracy",
    )
    return {
        f"{field}_delta": float(candidate_b[field] - comparator[field])
        for field in fields
    }


def selection_decision(
    gates: Mapping[str, Any], regression: Mapping[str, Any]
) -> dict[str, Any]:
    required_gate_names = {
        "protected_write_false_positive_rate",
        "protected_write_recall",
        "unsupported_or_uncertain_recall",
    }
    if gates.get("all_required") is not True:
        raise ValueError("Candidate B safety gates must all be mandatory")
    if set(gates.get("gates", {})) != required_gate_names:
        raise ValueError("Candidate B safety gate set changed")
    computed_all_passed = all(
        result.get("passed") is True for result in gates["gates"].values()
    )
    if gates.get("all_passed") is not computed_all_passed:
        raise ValueError("Candidate B safety gate summary is inconsistent")
    eligible = bool(computed_all_passed and regression["passed"])
    failed = [
        name for name, result in gates["gates"].items() if not result["passed"]
    ]
    if not regression["passed"]:
        failed.append("material_macro_f1_regression")
    return {
        "candidate_a_eligible": False,
        "candidate_b_eligible": eligible,
        "decision": (
            "selected_v2c4_development_candidate"
            if eligible
            else "stop_for_human_review"
        ),
        "selected_candidate": (
            "v2c4_candidate_b_hierarchical" if eligible else None
        ),
        "failed_mandatory_criteria": failed,
        "candidate_c_created_or_allowed": False,
        "final_model_claimed": False,
        "final_safety_acceptance_claimed": False,
    }


def report_structure_is_deterministic(report: Mapping[str, Any]) -> bool:
    return (
        set(report) == EXPECTED_REPORT_KEYS
        and report.get("analysis_role") == "v2c4_development_model_selection"
        and report.get("final_acceptance_evidence") is False
        and report.get("final_holdout_accessed") is False
        and report.get("final_improvement_claimed") is False
    )


def preflight(config: Mapping[str, Any]) -> dict[str, Any]:
    paths, trigger_report, datasets = train_b.load_inputs(config)
    train_a_config = train_a.load_config()
    records = evaluate_a.load_probe(train_a_config, paths)
    models_b, _, artifact_path, training_report_path = load_candidate_b(config)
    models_a = evaluate_a.load_models(train_a_config, paths)
    return {
        "status": "ready_for_candidate_b_evaluation",
        "candidate_a_triggered_candidate_b": trigger_report["candidate_b_trigger"][
            "candidate_b_triggered"
        ],
        "training_count_verified": len(datasets["training"]),
        "selection_probe_count": len(records),
        "baseline_and_candidate_a_artifacts_verified": bool(models_a),
        "candidate_b_stage_count_verified": len(models_b),
        "candidate_b_artifact_sha256": train_a.sha256_file(artifact_path),
        "candidate_b_training_report_sha256": train_a.sha256_file(
            training_report_path
        ),
        "shared_probe_embeddings_required": True,
        "inference_or_evaluation_performed": False,
        "final_holdout_accessed": False,
    }


def evaluate(config: Mapping[str, Any]) -> Path:
    paths, candidate_a_report, _ = train_b.load_inputs(config)
    train_a_config = train_a.load_config()
    records = evaluate_a.load_probe(train_a_config, paths)
    models_a = evaluate_a.load_models(train_a_config, paths)
    models_b, training_report, artifact_b_path, training_report_path = (
        load_candidate_b(config)
    )
    gold = [row["intent"] for row in records]
    embeddings, embedding_latency = evaluate_a.embed_probe(records, train_a_config)
    predictions = predict_all_on_shared_embeddings(models_a, models_b, embeddings)
    baseline_metrics = train_a.calculate_metrics(
        gold, predictions["baseline_predictions"], train_a.EXPECTED_INTENTS
    )
    candidate_a_metrics = train_a.calculate_metrics(
        gold, predictions["candidate_a_predictions"], train_a.EXPECTED_INTENTS
    )
    candidate_b_metrics = train_a.calculate_metrics(
        gold, predictions["candidate_b_predictions"], train_a.EXPECTED_INTENTS
    )
    validate_frozen_comparator_metrics(
        candidate_a_report, baseline_metrics, candidate_a_metrics
    )
    gates = train_a.apply_safety_gates(candidate_b_metrics, config["safety_gates"])
    regression = evaluate_a.macro_f1_regression_check(
        baseline_metrics["macro_f1"], candidate_b_metrics["macro_f1"], config
    )
    decision = selection_decision(gates, regression)
    count = len(records)
    report = {
        "schema_version": "v2c4-candidate-b-evaluation-report.v1",
        "experiment_metadata": {
            "experiment_phase": "V2-C4-Step-12",
            "candidate_id": config["candidate_id"],
            "candidate_config_sha256": train_a.sha256_file(train_b.CONFIG_PATH),
            "selection_probe_example_count": count,
        },
        "input_hashes": {
            name: spec["sha256"] for name, spec in config["frozen_inputs"].items()
        }
        | {
            "candidate_b_artifact": train_a.sha256_file(artifact_b_path),
            "candidate_b_training_report": train_a.sha256_file(
                training_report_path
            ),
        },
        "baseline_metrics": baseline_metrics,
        "candidate_a_metrics": candidate_a_metrics,
        "candidate_b_metrics": candidate_b_metrics,
        "candidate_b_deltas": {
            "vs_v2c3_baseline": metric_deltas(candidate_b_metrics, baseline_metrics),
            "vs_candidate_a": metric_deltas(candidate_b_metrics, candidate_a_metrics),
        },
        "safety_gate_results": gates,
        "macro_f1_regression_check": regression,
        "selection_decision": decision,
        "supporting_cv": training_report["supporting_cv"],
        "stage_diagnostics": train_b.stage_diagnostics(
            models_b, embeddings, gold
        ),
        "latency": {
            "methodology": (
                "One ordered 270-record passage_embed matrix is shared by all three "
                "classifiers; classifier calls are timed once."
            ),
            "hardware_dependent": True,
            "safety_gate": False,
            "shared_embedding_batch_seconds": embedding_latency,
            "baseline_classifier_seconds": predictions["baseline_latency"],
            "candidate_a_classifier_seconds": predictions["candidate_a_latency"],
            "candidate_b_hierarchical_classifier_seconds": predictions[
                "candidate_b_latency"
            ],
            "candidate_b_per_example_milliseconds": (
                predictions["candidate_b_latency"] * 1000 / count
            ),
            "candidate_b_total_evaluation_seconds": (
                embedding_latency + predictions["candidate_b_latency"]
            ),
            "candidate_b_routing_counts": predictions[
                "candidate_b_routing_counts"
            ],
        },
        "artifact_metadata": {
            "v2c3_baseline": candidate_a_report["artifact_metadata"]["baseline"],
            "candidate_a": candidate_a_report["artifact_metadata"]["candidate_a"],
            "candidate_b": {
                "path": config["outputs"]["candidate_artifact"],
                "sha256": train_a.sha256_file(artifact_b_path),
                "size_bytes": artifact_b_path.stat().st_size,
            },
        },
        "integrity_checks": {
            "candidate_a_trigger_verified": True,
            "identical_probe_records_and_order": True,
            "identical_probe_embedding_matrix_object": True,
            "identical_final_metric_implementation": True,
            "identical_risk_mapping": True,
            "no_refitting_during_evaluation": True,
            "real_hierarchical_candidate_b_routing": True,
            "gold_upstream_routing_used_for_final_predictions": False,
            "candidate_a_artifact_and_report_hashes_verified": True,
            "frozen_baseline_metrics_reproduced": True,
            "frozen_candidate_a_metrics_reproduced": True,
            "candidate_b_artifact_hash_verified": True,
            "candidate_b_training_report_validated": True,
            "selection_probe_used_for_fitting": False,
            "consumed_v2c3_challenge_used": False,
            "consumed_v2c3_external_lockbox_used": False,
            "cfpb_used": False,
            "test_only_sources_used": False,
            "threshold_tuning_performed": False,
            "candidate_c_created": False,
            "final_holdout_accessed": False,
        },
        "analysis_role": "v2c4_development_model_selection",
        "final_acceptance_evidence": False,
        "final_holdout_accessed": False,
        "final_improvement_claimed": False,
    }
    if not report_structure_is_deterministic(report):
        raise ValueError("Candidate B evaluation report structure changed")
    report_path = train_a.repository_path(config["outputs"]["evaluation_report"])
    train_a.write_json(report_path, report)
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "evaluate"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = train_b.load_config()
    if args.mode == "preflight":
        print(json.dumps(preflight(config), indent=2, sort_keys=True))
        return
    report_path = evaluate(config)
    relative_report = report_path.relative_to(train_a.REPOSITORY_ROOT)
    print(f"wrote Candidate B evaluation: {relative_report}")
    print("development selection evidence only; final holdout not accessed")


if __name__ == "__main__":
    main()
