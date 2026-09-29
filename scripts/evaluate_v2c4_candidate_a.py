"""Evaluate frozen V2-C3 baseline and V2-C4 Candidate A on one shared probe."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.svm import LinearSVC

try:
    from scripts import train_v2c4_candidate_a as training
except ModuleNotFoundError:  # Direct execution from the scripts directory.
    import train_v2c4_candidate_a as training


FINAL_HOLDOUT_ROLE = "v2c4_final_safety_holdout"
EXPECTED_REPORT_KEYS = {
    "schema_version",
    "experiment_metadata",
    "input_hashes",
    "baseline_metrics",
    "candidate_a_metrics",
    "metric_deltas",
    "safety_gate_results",
    "macro_f1_regression_check",
    "candidate_b_trigger",
    "supporting_cv_metrics",
    "latency",
    "artifact_metadata",
    "integrity_checks",
    "analysis_role",
    "final_acceptance_evidence",
    "final_holdout_accessed",
    "final_improvement_claimed",
}


def validate_evaluation_records(
    records: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> None:
    expected_probe = config["selection_probe"]
    if len(records) != expected_probe["expected_count"]:
        raise ValueError("selection probe must contain exactly 270 examples")
    if any(row.get("data_role") == FINAL_HOLDOUT_ROLE for row in records):
        raise ValueError("sealed V2-C4 final holdout is forbidden in Step 11")
    if any(row.get("data_role") != expected_probe["data_role"] for row in records):
        raise ValueError("Step 11 evaluation accepts only the selection probe role")
    if Counter(row["intent"] for row in records) != dict.fromkeys(
        training.EXPECTED_INTENTS, expected_probe["examples_per_intent"]
    ):
        raise ValueError("selection probe must remain balanced at 30 per intent")
    for row in records:
        training.validate_record_risk(row)
        if row.get("training_eligible") is not False:
            raise ValueError("selection probe cannot be training eligible")
        if row.get("model_selection_eligible") is not True:
            raise ValueError("selection probe must be model-selection eligible")
        if row.get("threshold_selection_eligible") is not False:
            raise ValueError("selection probe cannot select thresholds")
        if row.get("final_acceptance_evidence") is not False:
            raise ValueError("selection probe cannot be final acceptance evidence")


def load_probe(
    config: Mapping[str, Any], paths: Mapping[str, Path]
) -> list[dict[str, Any]]:
    payload = training.read_json(paths["v2c4_selection_probe"])
    records = payload.get("examples")
    if not isinstance(records, list) or not all(
        isinstance(row, dict) for row in records
    ):
        raise TypeError("selection probe examples must be a list of objects")
    if payload.get("example_count") != len(records):
        raise ValueError("selection probe top-level count differs")
    validate_evaluation_records(records, config)
    return records


def validate_classifier_artifact(
    artifact: Any,
    *,
    expected_id_field: str,
    expected_id: str,
    expected_count: int,
    config: Mapping[str, Any],
) -> LinearSVC:
    if not isinstance(artifact, dict):
        raise TypeError("classifier artifact must contain a dictionary")
    if artifact.get(expected_id_field) != expected_id:
        raise ValueError("classifier artifact identity changed")
    if artifact.get("training_example_count") != expected_count:
        raise ValueError("classifier artifact training count changed")
    if artifact.get("classifier_parameters") != config["candidate_a"][
        "classifier"
    ]["parameters"]:
        raise ValueError("classifier artifact parameters changed")
    if artifact.get("representation") != config["candidate_a"]["representation"]:
        raise ValueError("classifier artifact representation changed")
    if tuple(artifact.get("intent_label_order", [])) != training.EXPECTED_INTENTS:
        raise ValueError("classifier artifact intent ordering changed")
    if artifact.get("trusted_local_artifact") is not True:
        raise ValueError("classifier artifact is not trusted-local")
    if artifact.get("runtime_authority") is not False:
        raise ValueError("development classifier cannot have runtime authority")
    classifier = artifact.get("classifier")
    if not isinstance(classifier, LinearSVC):
        raise TypeError("artifact classifier must be LinearSVC")
    if classifier.get_params(deep=False) != config["candidate_a"]["classifier"][
        "parameters"
    ]:
        raise ValueError("fitted LinearSVC parameters changed")
    if tuple(classifier.classes_.tolist()) != training.EXPECTED_INTENTS:
        raise ValueError("fitted LinearSVC classes changed")
    return classifier


def load_models(
    config: Mapping[str, Any], paths: Mapping[str, Path]
) -> dict[str, Any]:
    training_report_path = training.repository_path(
        config["outputs"]["training_report"]
    )
    candidate_path = training.repository_path(config["outputs"]["candidate_artifact"])
    if not training_report_path.is_file() or not candidate_path.is_file():
        raise FileNotFoundError("Candidate A artifact and training report are required")
    training_report = training.read_json(training_report_path)
    if training_report.get("schema_version") != (
        "v2c4-candidate-a-training-report.v1"
    ):
        raise ValueError("unexpected Candidate A training-report schema")
    if training_report.get("candidate_config_sha256") != training.sha256_file(
        training.CONFIG_PATH
    ):
        raise ValueError("training report config hash changed")
    artifact_metadata = training_report.get("artifact_metadata", {})
    candidate_sha256 = training.sha256_file(candidate_path)
    if artifact_metadata.get("sha256") != candidate_sha256:
        raise ValueError("Candidate A artifact SHA-256 differs from training report")
    if artifact_metadata.get("size_bytes") != candidate_path.stat().st_size:
        raise ValueError("Candidate A artifact size differs from training report")
    baseline_path = paths["v2c3_baseline_artifact"]
    baseline_artifact = joblib.load(baseline_path)
    candidate_artifact = joblib.load(candidate_path)
    baseline_classifier = validate_classifier_artifact(
        baseline_artifact,
        expected_id_field="selected_model_id",
        expected_id="final_v2c3",
        expected_count=config["training_corpus"]["v2c3_development_expected_count"],
        config=config,
    )
    candidate_classifier = validate_classifier_artifact(
        candidate_artifact,
        expected_id_field="candidate_id",
        expected_id=config["candidate_a"]["candidate_id"],
        expected_count=config["training_corpus"]["combined_expected_count"],
        config=config,
    )
    return {
        "baseline_classifier": baseline_classifier,
        "candidate_classifier": candidate_classifier,
        "training_report": training_report,
        "training_report_path": training_report_path,
        "candidate_path": candidate_path,
        "candidate_sha256": candidate_sha256,
        "baseline_path": baseline_path,
    }


def embed_probe(
    records: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> tuple[np.ndarray, float]:
    try:
        from fastembed import TextEmbedding
    except ImportError as exc:
        raise RuntimeError("fastembed is required for probe evaluation") from exc
    representation = config["candidate_a"]["representation"]
    model = TextEmbedding(model_name=representation["model_identifier"])
    texts = [row["text"] for row in records]
    started = time.perf_counter()
    vectors = list(
        model.passage_embed(
            texts, batch_size=config["embedding_strategy"]["batch_size"]
        )
    )
    elapsed = time.perf_counter() - started
    embeddings = np.asarray(vectors, dtype=np.float32)
    if embeddings.shape != (len(records), representation["dimensions"]):
        raise ValueError("selection-probe embedding shape changed")
    if not np.isfinite(embeddings).all():
        raise ValueError("selection-probe embeddings contain non-finite values")
    return embeddings, elapsed


def predict_with_latency(
    classifier: LinearSVC, embeddings: np.ndarray
) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    predictions = np.asarray(classifier.predict(embeddings), dtype=object)
    elapsed = time.perf_counter() - started
    return predictions, elapsed


def predict_both_on_shared_embeddings(
    baseline_classifier: LinearSVC,
    candidate_classifier: LinearSVC,
    embeddings: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray, float]:
    baseline_predictions, baseline_latency = predict_with_latency(
        baseline_classifier, embeddings
    )
    candidate_predictions, candidate_latency = predict_with_latency(
        candidate_classifier, embeddings
    )
    return (
        baseline_predictions,
        baseline_latency,
        candidate_predictions,
        candidate_latency,
    )


def macro_f1_regression_check(
    baseline_macro_f1: float,
    candidate_macro_f1: float,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    delta = candidate_macro_f1 - baseline_macro_f1
    threshold = config["macro_f1_regression"][
        "material_regression_if_delta_less_than"
    ]
    minimum_acceptable = baseline_macro_f1 + threshold
    material_regression = candidate_macro_f1 < minimum_acceptable
    return {
        "baseline_macro_f1": baseline_macro_f1,
        "candidate_macro_f1": candidate_macro_f1,
        "macro_f1_delta": delta,
        "material_regression_if_delta_less_than": threshold,
        "minimum_acceptable_candidate_macro_f1": minimum_acceptable,
        "material_regression": material_regression,
        "passed": not material_regression,
    }


def candidate_b_trigger(
    gate_results: Mapping[str, Any], regression_check: Mapping[str, Any]
) -> dict[str, Any]:
    failed_gates = [
        name
        for name, result in gate_results["gates"].items()
        if not result["passed"]
    ]
    material_regression = bool(regression_check["material_regression"])
    triggered = bool(failed_gates or material_regression)
    reasons = [f"failed_safety_gate:{name}" for name in failed_gates]
    if material_regression:
        reasons.append("material_macro_f1_regression")
    return {
        "candidate_b_triggered": triggered,
        "reasons": reasons,
        "candidate_a_selected_by_step_9_rule": not triggered,
        "candidate_b_built": False,
        "report_only": True,
    }


def metric_deltas(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, float]:
    fields = (
        "accuracy",
        "macro_f1",
        "false_supported_rate",
        "protected_exact_intent_accuracy",
        "protected_write_recall",
        "protected_write_false_positive_rate",
        "unsupported_or_uncertain_recall",
    )
    return {
        f"{field}_delta": float(candidate[field] - baseline[field])
        for field in fields
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
    paths = training.validate_frozen_inputs(config)
    records = load_probe(config, paths)
    return {
        "status": "ready_for_candidate_a_evaluation_after_training",
        "selection_probe_count": len(records),
        "selection_probe_sha256": config["frozen_inputs"]["v2c4_selection_probe"][
            "sha256"
        ],
        "baseline_artifact_sha256": config["frozen_inputs"][
            "v2c3_baseline_artifact"
        ]["sha256"],
        "shared_probe_embeddings_required": True,
        "inference_or_evaluation_performed": False,
        "final_holdout_accessed": False,
        "candidate_b_built": False,
    }


def evaluate(config: Mapping[str, Any]) -> Path:
    paths = training.validate_frozen_inputs(config)
    records = load_probe(config, paths)
    models = load_models(config, paths)
    gold = [row["intent"] for row in records]
    embeddings, embedding_latency = embed_probe(records, config)
    (
        baseline_predictions,
        baseline_prediction_latency,
        candidate_predictions,
        candidate_prediction_latency,
    ) = predict_both_on_shared_embeddings(
        models["baseline_classifier"],
        models["candidate_classifier"],
        embeddings,
    )
    baseline_metrics = training.calculate_metrics(
        gold, baseline_predictions, training.EXPECTED_INTENTS
    )
    candidate_metrics = training.calculate_metrics(
        gold, candidate_predictions, training.EXPECTED_INTENTS
    )
    gate_results = training.apply_safety_gates(
        candidate_metrics, config["safety_gates"]
    )
    regression = macro_f1_regression_check(
        baseline_metrics["macro_f1"], candidate_metrics["macro_f1"], config
    )
    trigger = candidate_b_trigger(gate_results, regression)
    count = len(records)
    report = {
        "schema_version": "v2c4-candidate-a-evaluation-report.v1",
        "experiment_metadata": {
            "experiment_phase": "V2-C4-Step-11",
            "candidate_id": config["candidate_a"]["candidate_id"],
            "baseline_id": "final_v2c3",
            "candidate_config_sha256": training.sha256_file(training.CONFIG_PATH),
            "selection_probe_example_count": count,
        },
        "input_hashes": {
            "selection_probe": config["frozen_inputs"]["v2c4_selection_probe"][
                "sha256"
            ],
            "baseline_artifact": config["frozen_inputs"][
                "v2c3_baseline_artifact"
            ]["sha256"],
            "candidate_artifact": models["candidate_sha256"],
            "training_report": training.sha256_file(models["training_report_path"]),
        },
        "baseline_metrics": baseline_metrics,
        "candidate_a_metrics": candidate_metrics,
        "metric_deltas": metric_deltas(baseline_metrics, candidate_metrics),
        "safety_gate_results": gate_results,
        "macro_f1_regression_check": regression,
        "candidate_b_trigger": trigger,
        "supporting_cv_metrics": models["training_report"][
            "supporting_cv_metrics"
        ],
        "latency": {
            "methodology": (
                "One 270-record FastEmbed passage_embed batch is shared by both "
                "models; each classifier receives the identical matrix once."
            ),
            "hardware_dependent": True,
            "safety_gate": False,
            "embedding_batch_seconds": embedding_latency,
            "embedding_per_example_milliseconds": embedding_latency * 1000 / count,
            "baseline_classifier_prediction_seconds": baseline_prediction_latency,
            "baseline_classifier_per_example_milliseconds": (
                baseline_prediction_latency * 1000 / count
            ),
            "candidate_classifier_prediction_seconds": candidate_prediction_latency,
            "candidate_classifier_per_example_milliseconds": (
                candidate_prediction_latency * 1000 / count
            ),
            "total_evaluation_seconds": (
                embedding_latency
                + baseline_prediction_latency
                + candidate_prediction_latency
            ),
        },
        "artifact_metadata": {
            "baseline": {
                "path": config["frozen_inputs"]["v2c3_baseline_artifact"]["path"],
                "sha256": config["frozen_inputs"]["v2c3_baseline_artifact"][
                    "sha256"
                ],
            },
            "candidate_a": {
                "path": config["outputs"]["candidate_artifact"],
                "sha256": models["candidate_sha256"],
                "size_bytes": models["candidate_path"].stat().st_size,
            },
        },
        "integrity_checks": {
            "identical_probe_records": True,
            "identical_probe_order": True,
            "identical_probe_embedding_matrix_object": True,
            "identical_metric_implementation": True,
            "identical_risk_mapping": True,
            "baseline_artifact_sha256_enforced": True,
            "candidate_artifact_sha256_enforced_from_training_report": True,
            "selection_probe_excluded_from_fitting": True,
            "threshold_tuning_performed": False,
            "hyperparameter_tuning_performed": False,
            "consumed_v2c3_challenge_used": False,
            "consumed_v2c3_external_lockbox_used": False,
            "cfpb_used": False,
            "final_holdout_accessed": False,
        },
        "analysis_role": "v2c4_development_model_selection",
        "final_acceptance_evidence": False,
        "final_holdout_accessed": False,
        "final_improvement_claimed": False,
    }
    if not report_structure_is_deterministic(report):
        raise ValueError("Candidate A evaluation report structure changed")
    report_path = training.repository_path(config["outputs"]["evaluation_report"])
    training.write_json(report_path, report)
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "evaluate"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = training.load_config()
    if args.mode == "preflight":
        print(json.dumps(preflight(config), indent=2, sort_keys=True))
        return
    report_path = evaluate(config)
    relative_report = report_path.relative_to(training.REPOSITORY_ROOT)
    print(f"wrote development evaluation: {relative_report}")
    print("final V2-C4 holdout not accessed; no final acceptance claim")


if __name__ == "__main__":
    main()
