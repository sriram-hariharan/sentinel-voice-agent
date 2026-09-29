"""Diagnose the consumed V2-C4 model-selection probe without fitting models."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import statistics
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scripts import (
        evaluate_v2c4_candidate_a as evaluate_a,
    )
    from scripts import (
        evaluate_v2c4_candidate_b as evaluate_b,
    )
    from scripts import (
        train_v2c4_candidate_a as train_a,
    )
    from scripts import (
        train_v2c4_candidate_b as train_b,
    )
except ModuleNotFoundError:  # Direct execution from the scripts directory.
    import evaluate_v2c4_candidate_a as evaluate_a
    import evaluate_v2c4_candidate_b as evaluate_b
    import train_v2c4_candidate_a as train_a
    import train_v2c4_candidate_b as train_b


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ML_ROOT = REPOSITORY_ROOT / "data/evals/v2/ml"
CONFIG_PATH = ML_ROOT / "v2c4_selection_probe_postmortem_config.json"
DEFAULT_REPORT_PATH = ML_ROOT / "v2c4_selection_probe_postmortem_report.json"
SCHEMA_VERSION = "v2c4-selection-probe-postmortem-report.v1"
ANALYSIS_ROLE = "consumed_v2c4_model_selection_diagnostic"
SOURCE_ROLE = "v2c4_model_selection_probe"
UNSUPPORTED = "unsupported_or_uncertain"
PROTECTED_INTENTS = frozenset({"create_dispute", "freeze_card"})
FORBIDDEN_INPUT_FILENAMES = frozenset(
    {"v2c4_safety_holdout.json", "v2c4_safety_holdout_seed.json"}
)
EXPECTED_HASHES = {
    "selection_probe": (
        "25c62797825b58371f1260f631e9a35bdf34c4cf1bedc895120218a5bb63e16e"
    ),
    "selection_probe_manifest": (
        "e1658f4049bf151a8b4302876034eaa948ef8dda9b61ddf7f253987aa676b945"
    ),
    "candidate_a_config": (
        "00e584f1a702a16b29537001f8fe159136193ed71224ead81f1a0d5dcd39931d"
    ),
    "candidate_a_artifact": (
        "7384c6662f11ddebdc6225f5dfdf2397d51a51a5e318cd116b585d59ed8c7c2b"
    ),
    "candidate_a_evaluation_report": (
        "3871c72cbe521bf761f5a8bd7a5f0ff95effa1411dfa414f2402df5c0b6d263f"
    ),
    "candidate_b_config": (
        "47baaadbe5efcdeb3285e1edd9939809d9a4a4b2dd219b5e75409276a320aff1"
    ),
    "candidate_b_artifact": (
        "396c21852fdd10a09d501f6567fe5f1db383e8c7ce1fe340c05afe63d568c842"
    ),
    "candidate_b_evaluation_report": (
        "3c22e16b8ee1cb78c9c19a8806be63bd75deb83b7227a99ca376bb54f478b6da"
    ),
    "candidate_b_training_report": (
        "ffe5c1f8e8a12e1bf6230bc0167b1b85274100f2f94a34da1a605ae58fabc0dd"
    ),
    "v2c3_baseline_artifact": (
        "0e03a4d8367933622a92920f14a7a4c5a2ffa74af0f53b7b0913f04849e4780c"
    ),
    "step_9_intervention_plan": (
        "4a5ed357ab8ae2fc66dd5459f59883ef20bd9b6f9996bad7f4b8f8f2b245e5ff"
    ),
}
EXPECTED_PATHS = {
    "selection_probe": "data/evals/v2/ml/v2c4_selection_probe.json",
    "selection_probe_manifest": (
        "data/evals/v2/ml/v2c4_selection_probe.manifest.json"
    ),
    "candidate_a_config": "data/evals/v2/ml/v2c4_candidate_a_config.json",
    "candidate_a_artifact": (
        "artifacts/v2/classifier/v2c4_candidate_a_classifier.joblib"
    ),
    "candidate_a_evaluation_report": (
        "data/evals/v2/ml/v2c4_candidate_a_evaluation_report.json"
    ),
    "candidate_b_config": "data/evals/v2/ml/v2c4_candidate_b_config.json",
    "candidate_b_artifact": (
        "artifacts/v2/classifier/v2c4_candidate_b_classifier.joblib"
    ),
    "candidate_b_evaluation_report": (
        "data/evals/v2/ml/v2c4_candidate_b_evaluation_report.json"
    ),
    "candidate_b_training_report": (
        "data/evals/v2/ml/v2c4_candidate_b_training_report.json"
    ),
    "v2c3_baseline_artifact": (
        "artifacts/v2/classifier/v2c3_final_classifier.joblib"
    ),
    "step_9_intervention_plan": (
        "data/evals/v2/ml/v2c4_intervention_plan.json"
    ),
}
EXPECTED_FAILURES = {
    "candidate_a": {
        "protected_total": 60,
        "protected_recognized": 56,
        "protected_misses": 4,
        "nonprotected_total": 210,
        "protected_false_positives": 7,
        "unsupported_total": 30,
        "unsupported_correct": 20,
        "unsupported_false_supported": 10,
    },
    "candidate_b": {
        "protected_total": 60,
        "protected_recognized": 48,
        "protected_misses": 12,
        "nonprotected_total": 210,
        "protected_false_positives": 9,
        "unsupported_total": 30,
        "unsupported_correct": 11,
        "unsupported_false_supported": 19,
    },
}


def repository_path(relative_path: str) -> Path:
    resolved = (REPOSITORY_ROOT / relative_path).resolve()
    if not resolved.is_relative_to(REPOSITORY_ROOT):
        raise ValueError(f"path escapes repository: {relative_path}")
    guard_path(resolved)
    return resolved


def guard_path(path: Path) -> None:
    if path.name.lower() in FORBIDDEN_INPUT_FILENAMES:
        raise ValueError(f"sealed V2-C4 final holdout input is forbidden: {path}")


def sha256_file(path: Path) -> str:
    guard_path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def load_json(path: Path) -> dict[str, Any]:
    guard_path(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def load_config() -> dict[str, Any]:
    config = load_json(CONFIG_PATH)
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != (
        "v2c4-selection-probe-postmortem-config.v1"
    ):
        raise ValueError("unexpected postmortem config schema")
    if config.get("config_version") != "2026-09-29.v1":
        raise ValueError("unexpected postmortem config version")
    if config.get("experiment_phase") != "V2-C4-Step-13":
        raise ValueError("unexpected postmortem experiment phase")
    if config.get("analysis_role") != ANALYSIS_ROLE:
        raise ValueError("postmortem analysis role changed")
    if config.get("source_data_role") != SOURCE_ROLE:
        raise ValueError("postmortem source role changed")
    if config.get("probe_consumed") is not True:
        raise ValueError("selection probe must be marked consumed")
    if config.get("eligibility") != {
        "model_selection_eligible": False,
        "training_eligible": False,
        "threshold_selection_eligible": False,
        "final_acceptance_evidence": False,
    }:
        raise ValueError("consumed probe eligibility changed")
    if config.get("candidate_c_allowed") is not False:
        raise ValueError("Candidate C is prohibited")
    if config.get("final_holdout_accessed") is not False:
        raise ValueError("final holdout must remain sealed")
    if config.get("decision_scores_are_calibrated_probabilities") is not False:
        raise ValueError("LinearSVC scores cannot be labeled probabilities")
    policy = config.get("findings_use_policy", {})
    if policy != {
        "may_inform_v2c5_taxonomy_or_discovery_hypotheses": True,
        "may_tune_or_create_another_v2c4_candidate": False,
        "candidate_c_prohibited": True,
        "start_v2c5_implementation": False,
    }:
        raise ValueError("postmortem findings-use policy changed")
    if config.get("expected_example_count") != 270:
        raise ValueError("postmortem example count changed")
    if config.get("expected_frozen_failures") != EXPECTED_FAILURES:
        raise ValueError("postmortem frozen failure counts changed")
    frozen = config.get("frozen_inputs")
    if not isinstance(frozen, Mapping) or set(frozen) != set(EXPECTED_HASHES):
        raise ValueError("postmortem frozen input set changed")
    for name, expected_hash in EXPECTED_HASHES.items():
        if frozen[name] != {
            "path": EXPECTED_PATHS[name],
            "sha256": expected_hash,
        }:
            raise ValueError(f"postmortem frozen input changed: {name}")
    if config.get("output") != {
        "path": "data/evals/v2/ml/v2c4_selection_probe_postmortem_report.json",
        "raw_text_persisted": False,
        "deterministic": True,
    }:
        raise ValueError("postmortem output policy changed")
    if config.get("execution_status") != {
        "analysis_performed": False,
        "model_fitting_performed": False,
        "model_training_performed": False,
        "model_tuning_performed": False,
        "threshold_tuning_performed": False,
        "model_selection_performed": False,
        "final_holdout_accessed": False,
    }:
        raise ValueError("postmortem config must remain pre-execution")


def validate_frozen_inputs(config: Mapping[str, Any]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for name, spec in config["frozen_inputs"].items():
        path = repository_path(spec["path"])
        actual = sha256_file(path)
        if actual != spec["sha256"]:
            raise ValueError(
                f"{name} SHA-256 mismatch: expected {spec['sha256']}, got {actual}"
            )
        paths[name] = path
    return paths


def validate_probe(
    probe: Mapping[str, Any], config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    if probe.get("schema_version") != "v2c4-development-data.v1":
        raise ValueError("unexpected selection-probe schema")
    if probe.get("data_role") != SOURCE_ROLE:
        raise ValueError("selection probe has the wrong historical source role")
    rows = probe.get("examples")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise TypeError("selection-probe examples must be objects")
    if probe.get("example_count") != len(rows):
        raise ValueError("selection-probe top-level count changed")
    if len(rows) != config["expected_example_count"]:
        raise ValueError("selection-probe record count changed")
    required = {
        "example_id",
        "intent",
        "risk",
        "design_lane",
        "family_id",
        "group_id",
        "lineage_id",
        "tags",
        "text",
        "text_sha256",
        "normalized_text_sha256",
    }
    for row in rows:
        if not required.issubset(row):
            example_id = row.get("example_id")
            raise ValueError(f"selection-probe metadata incomplete: {example_id}")
        if row.get("data_role") != SOURCE_ROLE:
            raise ValueError("selection-probe row role changed")
        if row.get("training_eligible") is not False:
            raise ValueError("selection-probe row cannot be training eligible")
        if row.get("threshold_selection_eligible") is not False:
            raise ValueError("selection-probe row cannot select thresholds")
        if row.get("final_acceptance_evidence") is not False:
            raise ValueError("selection-probe row cannot be final evidence")
    if len({row["example_id"] for row in rows}) != len(rows):
        raise ValueError("selection-probe IDs must be unique")
    return rows


def _score_matrix(classifier: Any, embeddings: np.ndarray) -> np.ndarray:
    # LinearSVC decision scores are not calibrated probabilities.
    raw = np.asarray(classifier.decision_function(embeddings), dtype=np.float64)
    classes = tuple(str(value) for value in classifier.classes_.tolist())
    if len(classes) == 2:
        values = raw.reshape(-1)
        matrix = np.column_stack((-values, values))
    else:
        matrix = raw
    if matrix.shape != (len(embeddings), len(classes)):
        raise ValueError("LinearSVC decision-score shape changed")
    if not np.isfinite(matrix).all():
        raise ValueError("LinearSVC decision scores contain non-finite values")
    return matrix


def decision_details(classifier: Any, embeddings: np.ndarray) -> list[dict[str, Any]]:
    classes = tuple(str(value) for value in classifier.classes_.tolist())
    scores = _score_matrix(classifier, embeddings)
    predictions = [str(value) for value in classifier.predict(embeddings)]
    details: list[dict[str, Any]] = []
    for prediction, row_scores in zip(predictions, scores, strict=True):
        ranked_indices = sorted(
            range(len(classes)), key=lambda index: (-row_scores[index], index)
        )
        ranked = [
            {
                "class": classes[index],
                "decision_score": float(row_scores[index]),
                "rank": rank,
            }
            for rank, index in enumerate(ranked_indices, start=1)
        ]
        if ranked[0]["class"] != prediction:
            raise ValueError("decision-score argmax differs from prediction")
        details.append(
            {
                "prediction": prediction,
                "predicted_score": ranked[0]["decision_score"],
                "decision_margin": (
                    ranked[0]["decision_score"] - ranked[1]["decision_score"]
                ),
                "class_scores": ranked,
            }
        )
    return details


def trace_hierarchy(
    models: Mapping[str, Any], embeddings: np.ndarray
) -> tuple[list[str], list[dict[str, Any]]]:
    final = np.full(len(embeddings), UNSUPPORTED, dtype=object)
    stage_1 = decision_details(models["stage_1"], embeddings)
    traces = [
        {
            "stage_1_prediction": detail["prediction"],
            "stage_1_predicted_score": detail["predicted_score"],
            "stage_1_decision_margin": detail["decision_margin"],
            "stage_1_class_scores": detail["class_scores"],
            "stage_2_executed": False,
            "stage_2_prediction": None,
            "stage_2_predicted_score": None,
            "stage_2_decision_margin": None,
            "stage_2_class_scores": None,
            "stage_3a_executed": False,
            "stage_3a_prediction": None,
            "stage_3a_predicted_score": None,
            "stage_3a_decision_margin": None,
            "stage_3a_class_scores": None,
            "stage_3b_executed": False,
            "stage_3b_prediction": None,
            "stage_3b_predicted_score": None,
            "stage_3b_decision_margin": None,
            "stage_3b_top_scores": None,
        }
        for detail in stage_1
    ]
    supported = np.asarray(
        [row["prediction"] == train_b.SUPPORTED for row in stage_1]
    )
    supported_indices = np.flatnonzero(supported)
    if len(supported_indices):
        stage_2 = decision_details(models["stage_2"], embeddings[supported_indices])
        for index, detail in zip(supported_indices, stage_2, strict=True):
            traces[index].update(
                {
                    "stage_2_executed": True,
                    "stage_2_prediction": detail["prediction"],
                    "stage_2_predicted_score": detail["predicted_score"],
                    "stage_2_decision_margin": detail["decision_margin"],
                    "stage_2_class_scores": detail["class_scores"],
                }
            )
        protected_indices = np.asarray(
            [
                index
                for index, detail in zip(supported_indices, stage_2, strict=True)
                if detail["prediction"] == train_b.PROTECTED
            ]
        )
        nonprotected_indices = np.asarray(
            [
                index
                for index, detail in zip(supported_indices, stage_2, strict=True)
                if detail["prediction"] == train_b.NONPROTECTED
            ]
        )
        if len(protected_indices):
            stage_3a = decision_details(
                models["stage_3a"], embeddings[protected_indices]
            )
            for index, detail in zip(protected_indices, stage_3a, strict=True):
                final[index] = detail["prediction"]
                traces[index].update(
                    {
                        "stage_3a_executed": True,
                        "stage_3a_prediction": detail["prediction"],
                        "stage_3a_predicted_score": detail["predicted_score"],
                        "stage_3a_decision_margin": detail["decision_margin"],
                        "stage_3a_class_scores": detail["class_scores"],
                    }
                )
        if len(nonprotected_indices):
            stage_3b = decision_details(
                models["stage_3b"], embeddings[nonprotected_indices]
            )
            for index, detail in zip(nonprotected_indices, stage_3b, strict=True):
                final[index] = detail["prediction"]
                traces[index].update(
                    {
                        "stage_3b_executed": True,
                        "stage_3b_prediction": detail["prediction"],
                        "stage_3b_predicted_score": detail["predicted_score"],
                        "stage_3b_decision_margin": detail["decision_margin"],
                        "stage_3b_top_scores": detail["class_scores"][:3],
                    }
                )
    reference, _ = train_b.hierarchical_predict(models, embeddings)
    labels = [str(value) for value in final.tolist()]
    if labels != [str(value) for value in reference.tolist()]:
        raise ValueError("hierarchy trace differs from frozen inference helper")
    return labels, traces


def ordered_predictions_sha256(predictions: Sequence[str]) -> str:
    payload = json.dumps(list(predictions), separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def verify_reproduction(
    gold: Sequence[str],
    predictions: Mapping[str, Sequence[str]],
    candidate_a_report: Mapping[str, Any],
    candidate_b_report: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    report_fields = {
        "baseline": "baseline_metrics",
        "candidate_a": "candidate_a_metrics",
        "candidate_b": "candidate_b_metrics",
    }
    metrics: dict[str, dict[str, Any]] = {}
    checks: dict[str, Any] = {}
    for model_name, field in report_fields.items():
        labels = predictions[model_name]
        if len(labels) != len(gold):
            raise ValueError(f"{model_name} prediction count changed")
        reproduced = train_a.calculate_metrics(
            gold, labels, train_a.EXPECTED_INTENTS
        )
        if reproduced != candidate_b_report.get(field):
            raise ValueError(f"{model_name} metrics differ from Candidate B report")
        if model_name != "candidate_b" and reproduced != candidate_a_report.get(
            field
        ):
            raise ValueError(f"{model_name} metrics differ from Candidate A report")
        metrics[model_name] = reproduced
        checks[model_name] = {
            "metrics_exactly_match_committed_reports": True,
            "prediction_count": len(labels),
            "ordered_predictions_sha256": ordered_predictions_sha256(labels),
            "labels_reproduced_from_pinned_artifact": True,
        }
    decision = candidate_b_report.get("selection_decision", {})
    if decision.get("decision") != "stop_for_human_review":
        raise ValueError("frozen Candidate B decision changed")
    if decision.get("selected_candidate") is not None:
        raise ValueError("V2-C4 unexpectedly selected a candidate")
    if decision.get("candidate_c_created_or_allowed") is not False:
        raise ValueError("frozen result unexpectedly permits Candidate C")
    if candidate_b_report.get("final_acceptance_evidence") is not False:
        raise ValueError("Candidate B report crossed the development boundary")
    if candidate_b_report.get("final_holdout_accessed") is not False:
        raise ValueError("Candidate B report indicates final-holdout access")
    return (
        {
            "status": "passed",
            "models": checks,
            "committed_per_example_prediction_sequence_available": False,
            "reference_limitation": (
                "Committed evaluation reports contain exact aggregate metrics and "
                "confusion matrices but no ordered per-example labels. Labels are "
                "reproduced from hash-pinned artifacts and checked against their "
                "decision outputs; ordered hashes are frozen in this report."
            ),
        },
        metrics,
    )


def failure_flags(gold: str, prediction: str) -> dict[str, bool]:
    gold_protected = gold in PROTECTED_INTENTS
    predicted_protected = prediction in PROTECTED_INTENTS
    return {
        "protected_miss": gold_protected and not predicted_protected,
        "protected_false_positive": not gold_protected and predicted_protected,
        "unsupported_false_supported": (
            gold == UNSUPPORTED and prediction != UNSUPPORTED
        ),
        "protected_wrong_subtype": (
            gold_protected and predicted_protected and gold != prediction
        ),
    }


def summarize_failures(
    examples: Sequence[Mapping[str, Any]], predictions: Sequence[str]
) -> dict[str, Any]:
    pairs = list(zip(examples, predictions, strict=True))
    protected_total = sum(row["intent"] in PROTECTED_INTENTS for row, _ in pairs)
    protected_recognized = sum(
        row["intent"] in PROTECTED_INTENTS and pred in PROTECTED_INTENTS
        for row, pred in pairs
    )
    nonprotected_total = len(pairs) - protected_total
    unsupported_total = sum(row["intent"] == UNSUPPORTED for row, _ in pairs)
    unsupported_correct = sum(
        row["intent"] == UNSUPPORTED and pred == UNSUPPORTED for row, pred in pairs
    )
    flag_rows = [failure_flags(str(row["intent"]), pred) for row, pred in pairs]
    return {
        "protected_total": protected_total,
        "protected_recognized": protected_recognized,
        "protected_misses": sum(row["protected_miss"] for row in flag_rows),
        "protected_wrong_subtype": sum(
            row["protected_wrong_subtype"] for row in flag_rows
        ),
        "nonprotected_total": nonprotected_total,
        "protected_false_positives": sum(
            row["protected_false_positive"] for row in flag_rows
        ),
        "unsupported_total": unsupported_total,
        "unsupported_correct": unsupported_correct,
        "unsupported_false_supported": sum(
            row["unsupported_false_supported"] for row in flag_rows
        ),
    }


def validate_expected_failures(
    model_name: str,
    summary: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    expected = config["expected_frozen_failures"][model_name]
    for field, value in expected.items():
        if summary.get(field) != value:
            raise ValueError(
                f"{model_name} {field} differs: expected {value}, "
                f"got {summary.get(field)}"
            )


def primary_failure_stage(gold: str, trace: Mapping[str, Any]) -> str | None:
    final = str(trace["final_prediction"])
    if final == gold:
        return None
    stage_1 = trace["stage_1_prediction"]
    if gold == UNSUPPORTED:
        return "stage_1" if stage_1 != UNSUPPORTED else None
    if stage_1 == UNSUPPORTED:
        return "stage_1"
    stage_2 = trace["stage_2_prediction"]
    if gold in PROTECTED_INTENTS:
        if stage_2 != train_b.PROTECTED:
            return "stage_2"
        return "stage_3a"
    if stage_2 != train_b.NONPROTECTED:
        return "stage_2"
    return "stage_3b"


def transition_category(a_correct: bool, b_correct: bool, same: bool) -> str:
    if a_correct and b_correct:
        return "both_correct"
    if a_correct:
        return "candidate_a_correct_candidate_b_wrong"
    if b_correct:
        return "candidate_a_wrong_candidate_b_correct"
    return "both_wrong_same_prediction" if same else "both_wrong_different_prediction"


def is_safety_failure(gold: str, prediction: str) -> bool:
    return any(failure_flags(gold, prediction).values())


def safety_transition(gold: str, a_prediction: str, b_prediction: str) -> str:
    a_unsafe = is_safety_failure(gold, a_prediction)
    b_unsafe = is_safety_failure(gold, b_prediction)
    if not a_unsafe and b_unsafe:
        return "candidate_a_safe_candidate_b_unsafe"
    if a_unsafe and not b_unsafe:
        return "candidate_a_unsafe_candidate_b_safe"
    if a_unsafe and b_unsafe:
        return "both_unsafe"
    return "both_safe"


def build_per_example(
    examples: Sequence[Mapping[str, Any]],
    baseline_details: Sequence[Mapping[str, Any]],
    candidate_a_details: Sequence[Mapping[str, Any]],
    candidate_b_predictions: Sequence[str],
    hierarchy_traces: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row, baseline, candidate_a, candidate_b, trace in zip(
        examples,
        baseline_details,
        candidate_a_details,
        candidate_b_predictions,
        hierarchy_traces,
        strict=True,
    ):
        gold = str(row["intent"])
        a_prediction = str(candidate_a["prediction"])
        gold_score_row = next(
            item for item in candidate_a["class_scores"] if item["class"] == gold
        )
        gold_rank = int(gold_score_row["rank"])
        b_trace = dict(trace)
        b_trace["final_prediction"] = candidate_b
        b_trace["primary_failure_stage"] = primary_failure_stage(gold, b_trace)
        output.append(
            {
                "example_id": row["example_id"],
                "intent": gold,
                "risk": row["risk"],
                "design_lane": row["design_lane"],
                "family_id": row["family_id"],
                "group_id": row["group_id"],
                "lineage_id": row["lineage_id"],
                "tags": list(row["tags"]),
                "text_sha256": row["text_sha256"],
                "normalized_text_sha256": row["normalized_text_sha256"],
                "baseline": {
                    "prediction": baseline["prediction"],
                    "correct": baseline["prediction"] == gold,
                },
                "candidate_a": {
                    "prediction": a_prediction,
                    "correct": a_prediction == gold,
                    "gold_score": gold_score_row["decision_score"],
                    "predicted_score": candidate_a["predicted_score"],
                    "predicted_vs_gold_margin": (
                        candidate_a["predicted_score"]
                        - gold_score_row["decision_score"]
                    ),
                    "prediction_margin": candidate_a["decision_margin"],
                    "gold_rank": gold_rank,
                    "top_3_intents": [
                        {
                            "intent": item["class"],
                            "decision_score": item["decision_score"],
                            "rank": item["rank"],
                        }
                        for item in candidate_a["class_scores"][:3]
                    ],
                    "failure_flags": failure_flags(gold, a_prediction),
                },
                "candidate_b": {
                    "final_prediction": candidate_b,
                    "correct": candidate_b == gold,
                    "failure_flags": failure_flags(gold, candidate_b),
                    "hierarchy": b_trace,
                },
                "a_vs_b_transition": transition_category(
                    a_prediction == gold,
                    candidate_b == gold,
                    a_prediction == candidate_b,
                ),
                "safety_transition": safety_transition(
                    gold, a_prediction, candidate_b
                ),
            }
        )
    return output


def _group_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    result: dict[str, Any] = {"count": count}
    for model in ("baseline", "candidate_a", "candidate_b"):
        correct_field = "correct"
        correct = sum(bool(row[model][correct_field]) for row in rows)
        result[f"{model}_correct_count"] = correct
        result[f"{model}_accuracy"] = correct / count if count else None
        result[f"{model}_error_count"] = count - correct
        result[f"{model}_error_rate"] = (count - correct) / count if count else None
    for short, model in (("a", "candidate_a"), ("b", "candidate_b")):
        flags = [row[model]["failure_flags"] for row in rows]
        result[f"candidate_{short}_protected_miss"] = sum(
            flag["protected_miss"] for flag in flags
        )
        result[f"candidate_{short}_protected_false_positive"] = sum(
            flag["protected_false_positive"] for flag in flags
        )
        result[f"candidate_{short}_unsupported_false_supported"] = sum(
            flag["unsupported_false_supported"] for flag in flags
        )
    result["a_vs_b_transitions"] = dict(
        sorted(Counter(row["a_vs_b_transition"] for row in rows).items())
    )
    result["safety_transitions"] = dict(
        sorted(Counter(row["safety_transition"] for row in rows).items())
    )
    return result


def aggregate_by(
    rows: Sequence[Mapping[str, Any]], field: str, *, multi: bool = False
) -> dict[str, Any]:
    values: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        keys = row[field] if multi else [row[field]]
        for key in keys:
            values.setdefault(str(key), []).append(row)
    return {key: _group_summary(values[key]) for key in sorted(values)}


def transition_analysis(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "overall": _group_summary(rows),
        "by_intent": aggregate_by(rows, "intent"),
        "by_risk": aggregate_by(rows, "risk"),
        "by_design_lane": aggregate_by(rows, "design_lane"),
        "by_tag": aggregate_by(rows, "tags", multi=True),
    }


def unsupported_analysis(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    unsupported = [row for row in rows if row["intent"] == UNSUPPORTED]
    lanes: dict[str, Any] = {}
    for lane in sorted({str(row["design_lane"]) for row in unsupported}):
        lane_rows = [row for row in unsupported if row["design_lane"] == lane]
        lanes[lane] = {
            "count": len(lane_rows),
            "candidate_a_correct": sum(
                row["candidate_a"]["correct"] for row in lane_rows
            ),
            "candidate_a_predictions": dict(
                sorted(
                    Counter(
                        row["candidate_a"]["prediction"] for row in lane_rows
                    ).items()
                )
            ),
            "candidate_b_correct": sum(
                row["candidate_b"]["correct"] for row in lane_rows
            ),
            "candidate_b_stage_1_misses": sum(
                row["candidate_b"]["hierarchy"]["primary_failure_stage"]
                == "stage_1"
                for row in lane_rows
            ),
            "candidate_b_final_predictions": dict(
                sorted(
                    Counter(
                        row["candidate_b"]["final_prediction"] for row in lane_rows
                    ).items()
                )
            ),
        }
    b_false_supported = [
        row
        for row in unsupported
        if row["candidate_b"]["failure_flags"]["unsupported_false_supported"]
    ]
    if any(
        row["candidate_b"]["hierarchy"]["primary_failure_stage"] != "stage_1"
        for row in b_false_supported
    ):
        raise ValueError("Candidate B unsupported failure did not begin at Stage 1")
    return {
        "count": len(unsupported),
        "by_design_lane": lanes,
        "candidate_b_false_supported_count": len(b_false_supported),
        "candidate_b_stage_1_failure_count": len(b_false_supported),
        "candidate_b_false_supported_endpoints": dict(
            sorted(
                Counter(
                    row["candidate_b"]["final_prediction"]
                    for row in b_false_supported
                ).items()
            )
        ),
        "hypothesis_assessment": {
            "hypothesis": (
                "unsupported_or_uncertain may be heterogeneous for one binary "
                "supported-vs-unsupported boundary"
            ),
            "status": "diagnostic_evidence_only_not_causal_conclusion",
            "observed_design_lane_count": len(lanes),
            "small_group_caution": True,
        },
    }


def protected_analysis(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    subtypes: dict[str, Any] = {}
    for intent in sorted(PROTECTED_INTENTS):
        selected = [row for row in rows if row["intent"] == intent]
        entry: dict[str, Any] = {"count": len(selected)}
        for short, model in (("a", "candidate_a"), ("b", "candidate_b")):
            prediction_field = (
                "prediction" if model == "candidate_a" else "final_prediction"
            )
            predictions = [row[model][prediction_field] for row in selected]
            recognized = sum(
                prediction in PROTECTED_INTENTS for prediction in predictions
            )
            exact = sum(prediction == intent for prediction in predictions)
            entry[f"candidate_{short}_protected_recognized"] = recognized
            entry[f"candidate_{short}_protected_recognition_recall"] = (
                recognized / len(selected)
            )
            entry[f"candidate_{short}_exact_correct"] = exact
            entry[f"candidate_{short}_exact_recall"] = exact / len(selected)
            entry[f"candidate_{short}_protected_misses"] = sum(
                prediction not in PROTECTED_INTENTS for prediction in predictions
            )
            entry[f"candidate_{short}_subtype_confusions"] = sum(
                prediction in PROTECTED_INTENTS and prediction != intent
                for prediction in predictions
            )
        subtypes[intent] = entry
    false_positives: dict[str, Any] = {}
    for short, model in (("a", "candidate_a"), ("b", "candidate_b")):
        prediction_field = (
            "prediction" if model == "candidate_a" else "final_prediction"
        )
        false_positives[f"candidate_{short}"] = {
            intent: sum(
                row["intent"] not in PROTECTED_INTENTS
                and row[model][prediction_field] == intent
                for row in rows
            )
            for intent in sorted(PROTECTED_INTENTS)
        }
    b_protected_errors = [
        row
        for row in rows
        if row["intent"] in PROTECTED_INTENTS
        and not row["candidate_b"]["correct"]
    ]
    return {
        "by_protected_subtype": subtypes,
        "false_positives_into_protected_subtype": false_positives,
        "candidate_b_primary_failure_stage": dict(
            sorted(
                Counter(
                    row["candidate_b"]["hierarchy"]["primary_failure_stage"]
                    for row in b_protected_errors
                ).items()
            )
        ),
    }


def numeric_summary(values: Sequence[float]) -> dict[str, Any]:
    numbers = [float(value) for value in values]
    if not numbers:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    if not all(math.isfinite(value) for value in numbers):
        raise ValueError("decision-boundary statistics require finite values")
    return {
        "count": len(numbers),
        "mean": statistics.fmean(numbers),
        "median": statistics.median(numbers),
        "min": min(numbers),
        "max": max(numbers),
    }


def decision_boundary_statistics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    a_errors = [row for row in rows if not row["candidate_a"]["correct"]]
    b_safety = [
        row
        for row in rows
        if is_safety_failure(
            str(row["intent"]), str(row["candidate_b"]["final_prediction"])
        )
    ]
    return {
        "score_semantics": (
            "LinearSVC decision scores are signed decision-function values, not "
            "calibrated probabilities. Values are comparable only within the "
            "same fitted estimator and class-score context."
        ),
        "candidate_a_error_prediction_margins": numeric_summary(
            [row["candidate_a"]["prediction_margin"] for row in a_errors]
        ),
        "candidate_a_error_predicted_vs_gold_margins": numeric_summary(
            [row["candidate_a"]["predicted_vs_gold_margin"] for row in a_errors]
        ),
        "candidate_b_safety_failure_stage_1_margins": numeric_summary(
            [
                row["candidate_b"]["hierarchy"]["stage_1_decision_margin"]
                for row in b_safety
            ]
        ),
        "candidate_b_safety_failure_stage_2_margins": numeric_summary(
            [
                row["candidate_b"]["hierarchy"]["stage_2_decision_margin"]
                for row in b_safety
                if row["candidate_b"]["hierarchy"]["stage_2_executed"]
            ]
        ),
    }


def stage_failure_decomposition(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    all_errors = [row for row in rows if not row["candidate_b"]["correct"]]
    safety_failures = [
        row
        for row in rows
        if is_safety_failure(
            str(row["intent"]), str(row["candidate_b"]["final_prediction"])
        )
    ]
    all_by_stage = Counter(
        row["candidate_b"]["hierarchy"]["primary_failure_stage"]
        for row in all_errors
    )
    safety_by_stage = Counter(
        row["candidate_b"]["hierarchy"]["primary_failure_stage"]
        for row in safety_failures
    )
    by_reason: dict[str, Counter[str]] = {}
    for row in safety_failures:
        stage = row["candidate_b"]["hierarchy"]["primary_failure_stage"]
        for reason, present in row["candidate_b"]["failure_flags"].items():
            if present:
                by_reason.setdefault(reason, Counter())[str(stage)] += 1
    return {
        "all_errors": {
            "count": len(all_errors),
            "by_primary_failure_stage": dict(sorted(all_by_stage.items())),
            "by_gold_intent_and_primary_failure_stage": {
                intent: dict(
                    sorted(
                        Counter(
                            row["candidate_b"]["hierarchy"][
                                "primary_failure_stage"
                            ]
                            for row in all_errors
                            if row["intent"] == intent
                        ).items()
                    )
                )
                for intent in train_a.EXPECTED_INTENTS
            },
        },
        "safety_failures": {
            "unique_example_count": len(safety_failures),
            "by_primary_failure_stage": dict(sorted(safety_by_stage.items())),
            "by_failure_reason_and_stage": {
                reason: dict(sorted(counts.items()))
                for reason, counts in sorted(by_reason.items())
            },
        },
    }


def v2c5_hypothesis_inputs(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    unsupported = [row for row in rows if row["intent"] == UNSUPPORTED]
    unsupported_lanes = []
    for lane in sorted({str(row["design_lane"]) for row in unsupported}):
        selected = [row for row in unsupported if row["design_lane"] == lane]
        unsupported_lanes.append(
            {
                "design_lane": lane,
                "count": len(selected),
                "candidate_a_error_count": sum(
                    not row["candidate_a"]["correct"] for row in selected
                ),
                "candidate_b_error_count": sum(
                    not row["candidate_b"]["correct"] for row in selected
                ),
                "candidate_b_stage_1_failure_count": sum(
                    row["candidate_b"]["hierarchy"]["primary_failure_stage"]
                    == "stage_1"
                    for row in selected
                ),
            }
        )
    protected = [row for row in rows if row["intent"] in PROTECTED_INTENTS]
    return {
        "role": "evidence_inputs_for_future_hypothesis_design_only",
        "taxonomy_or_intent_decisions_made": False,
        "v2c5_implementation_started": False,
        "candidate_c_justified_or_allowed": False,
        "candidate_categories": [
            {
                "category": "unsupported_semantic_subcategories",
                "evidence": {
                    "unsupported_example_count": len(unsupported),
                    "design_lane_results": unsupported_lanes,
                },
                "interpretation": (
                    "Lane-level outcome differences may motivate future human "
                    "review of whether the unsupported label contains coherent "
                    "subgroups; they do not establish new intents."
                ),
            },
            {
                "category": "protected_current_action_boundary",
                "evidence": {
                    "protected_example_count": len(protected),
                    "candidate_b_stage_1_protected_error_count": sum(
                        row["candidate_b"]["hierarchy"]["primary_failure_stage"]
                        == "stage_1"
                        for row in protected
                        if not row["candidate_b"]["correct"]
                    ),
                    "candidate_b_stage_2_protected_error_count": sum(
                        row["candidate_b"]["hierarchy"]["primary_failure_stage"]
                        == "stage_2"
                        for row in protected
                        if not row["candidate_b"]["correct"]
                    ),
                    "candidate_b_stage_3a_error_count": sum(
                        row["candidate_b"]["hierarchy"]["primary_failure_stage"]
                        == "stage_3a"
                        for row in protected
                        if not row["candidate_b"]["correct"]
                    ),
                },
                "interpretation": (
                    "The observed protected errors may inform future review of "
                    "action-state and context boundaries; they do not authorize "
                    "a new V2-C4 model or runtime behavior."
                ),
            },
        ],
        "small_group_caution": True,
        "human_adjudication_required_for_future_taxonomy_changes": True,
    }


def build_review_queue(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    queue: list[dict[str, Any]] = []
    for row in rows:
        reasons: list[str] = []
        model_labels = (
            ("candidate_a", "candidate_a"),
            ("candidate_b", "candidate_b"),
        )
        for model_key, label in model_labels:
            for reason, present in row[model_key]["failure_flags"].items():
                if present and reason != "protected_wrong_subtype":
                    reasons.append(f"{label}:{reason}")
        if not reasons:
            continue
        queue.append(
            {
                "example_id": row["example_id"],
                "gold_intent": row["intent"],
                "risk": row["risk"],
                "design_lane": row["design_lane"],
                "tags": row["tags"],
                "candidate_a_prediction": row["candidate_a"]["prediction"],
                "candidate_b_prediction": row["candidate_b"]["final_prediction"],
                "candidate_b_primary_failure_stage": row["candidate_b"][
                    "hierarchy"
                ]["primary_failure_stage"],
                "review_reasons": reasons,
                "text_sha256": row["text_sha256"],
                "normalized_text_sha256": row["normalized_text_sha256"],
            }
        )
    return queue


def validate_no_raw_text(report: Mapping[str, Any]) -> None:
    forbidden = {"text", "raw_text", "utterance"}

    def inspect(value: Any, location: str) -> None:
        if isinstance(value, Mapping):
            found = forbidden & set(value)
            if found:
                raise ValueError(f"raw text field at {location}: {sorted(found)}")
            for key, nested in value.items():
                inspect(nested, f"{location}.{key}")
        elif isinstance(value, list):
            for index, nested in enumerate(value):
                inspect(nested, f"{location}[{index}]")

    inspect(report, "report")
    if report.get("raw_text_persisted") is not False:
        raise ValueError("postmortem must declare raw_text_persisted=false")


def validate_report_governance(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected postmortem report schema")
    if report.get("analysis_role") != ANALYSIS_ROLE:
        raise ValueError("postmortem report analysis role changed")
    if report.get("selected_v2c4_candidate") is not None:
        raise ValueError("postmortem cannot select a V2-C4 candidate")
    if report.get("candidate_c_allowed") is not False:
        raise ValueError("postmortem cannot allow Candidate C")
    if report.get("final_holdout_accessed") is not False:
        raise ValueError("postmortem cannot access the final holdout")
    if report.get("final_acceptance_evidence") is not False:
        raise ValueError("postmortem cannot be final acceptance evidence")
    validate_no_raw_text(report)


def build_report() -> dict[str, Any]:
    config = load_config()
    paths = validate_frozen_inputs(config)
    probe = load_json(paths["selection_probe"])
    examples = validate_probe(probe, config)
    manifest = load_json(paths["selection_probe_manifest"])
    if manifest.get("output_sha256") != EXPECTED_HASHES["selection_probe"]:
        raise ValueError("selection-probe manifest output hash changed")
    candidate_a_report = load_json(paths["candidate_a_evaluation_report"])
    candidate_b_report = load_json(paths["candidate_b_evaluation_report"])
    candidate_b_config = train_b.load_config()
    candidate_a_config = train_a.load_config()
    model_paths = train_a.validate_frozen_inputs(candidate_b_config)
    models_a = evaluate_a.load_models(candidate_a_config, model_paths)
    models_b, _, _, _ = evaluate_b.load_candidate_b(candidate_b_config)
    os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
    embeddings, _ = evaluate_a.embed_probe(examples, candidate_a_config)
    baseline_details = decision_details(
        models_a["baseline_classifier"], embeddings
    )
    candidate_a_details = decision_details(
        models_a["candidate_classifier"], embeddings
    )
    candidate_b_predictions, hierarchy = trace_hierarchy(models_b, embeddings)
    predictions = {
        "baseline": [row["prediction"] for row in baseline_details],
        "candidate_a": [row["prediction"] for row in candidate_a_details],
        "candidate_b": candidate_b_predictions,
    }
    gold = [str(row["intent"]) for row in examples]
    reproduction, reproduced_metrics = verify_reproduction(
        gold,
        predictions,
        candidate_a_report,
        candidate_b_report,
    )
    per_example = build_per_example(
        examples,
        baseline_details,
        candidate_a_details,
        candidate_b_predictions,
        hierarchy,
    )
    candidate_a_failures = summarize_failures(examples, predictions["candidate_a"])
    candidate_b_failures = summarize_failures(examples, predictions["candidate_b"])
    validate_expected_failures("candidate_a", candidate_a_failures, config)
    validate_expected_failures("candidate_b", candidate_b_failures, config)
    review_queue = build_review_queue(per_example)
    report = {
        "schema_version": SCHEMA_VERSION,
        "experiment_phase": "V2-C4-Step-13",
        "analysis_role": ANALYSIS_ROLE,
        "provenance": {
            "source_data_role": SOURCE_ROLE,
            "source_is_consumed": True,
            "example_count": len(examples),
            "family_count": manifest["counts"]["family_count"],
            "group_count": manifest["counts"]["group_count"],
            "lineage_count": manifest["counts"]["lineage_count"],
        },
        "frozen_hashes": {
            name: {"path": spec["path"], "sha256": spec["sha256"]}
            for name, spec in config["frozen_inputs"].items()
        },
        "reproduction_checks": reproduction,
        "model_summaries": reproduced_metrics,
        "candidate_a_failure_summary": candidate_a_failures,
        "candidate_b_failure_summary": candidate_b_failures,
        "candidate_b_stage_failure_decomposition": stage_failure_decomposition(
            per_example
        ),
        "a_vs_b_transition_analysis": transition_analysis(per_example),
        "aggregations": {
            "by_intent": aggregate_by(per_example, "intent"),
            "by_risk": aggregate_by(per_example, "risk"),
            "by_design_lane": aggregate_by(per_example, "design_lane"),
            "by_tag": aggregate_by(per_example, "tags", multi=True),
        },
        "unsupported_analysis": unsupported_analysis(per_example),
        "protected_analysis": protected_analysis(per_example),
        "v2c5_hypothesis_inputs": v2c5_hypothesis_inputs(per_example),
        "decision_boundary_statistics": decision_boundary_statistics(per_example),
        "per_example_comparison": per_example,
        "review_queue": review_queue,
        "review_queue_summary": {
            "record_count": len(review_queue),
            "deduplicated_by_example_id": True,
            "raw_text_available_only_via_print_review_queue": True,
        },
        "interpretation_constraints": {
            "linear_svc_scores_are_calibrated_probabilities": False,
            "small_groups_require_caution": True,
            "causal_claims_from_metadata_only": False,
            "may_inform_v2c5_taxonomy_or_discovery_hypotheses": True,
            "may_tune_or_create_another_v2c4_candidate": False,
            "candidate_c_allowed": False,
            "v2c5_implementation_started": False,
        },
        "selected_v2c4_candidate": None,
        "candidate_c_allowed": False,
        "final_holdout_accessed": False,
        "final_acceptance_evidence": False,
        "raw_text_persisted": False,
        "execution_policy": {
            "diagnostic_inference_performed": True,
            "model_fitting_performed": False,
            "model_training_performed": False,
            "model_tuning_performed": False,
            "threshold_tuning_performed": False,
            "model_selection_performed": False,
            "final_holdout_accessed": False,
        },
    }
    validate_report_governance(report)
    return report


def write_report(path: Path, report: Mapping[str, Any]) -> None:
    guard_path(path)
    validate_report_governance(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(stable_json_bytes(report))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def check_report(path: Path, report: Mapping[str, Any]) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"postmortem report does not exist: {path}")
    if path.read_bytes() != stable_json_bytes(report):
        raise ValueError("postmortem report differs from deterministic reproduction")


def join_review_queue(
    report: Mapping[str, Any], examples: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    validate_report_governance(report)
    source = {str(row["example_id"]): row for row in examples}
    joined: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in report["review_queue"]:
        example_id = str(row["example_id"])
        if example_id in seen or example_id not in source:
            raise ValueError("review queue IDs are duplicate or absent from probe")
        seen.add(example_id)
        original = source[example_id]
        if original["intent"] != row["gold_intent"]:
            raise ValueError("review queue gold intent changed")
        if original["text_sha256"] != row["text_sha256"]:
            raise ValueError("review queue text hash changed")
        joined.append(
            {
                "example_id": example_id,
                "gold_intent": row["gold_intent"],
                "risk": row["risk"],
                "design_lane": row["design_lane"],
                "tags": row["tags"],
                "candidate_a_prediction": row["candidate_a_prediction"],
                "candidate_b_prediction": row["candidate_b_prediction"],
                "candidate_b_primary_failure_stage": row[
                    "candidate_b_primary_failure_stage"
                ],
                "review_reasons": row["review_reasons"],
                "text": original["text"],
            }
        )
    return joined


def print_review_queue(
    *,
    report_path: Path = DEFAULT_REPORT_PATH,
    report: Mapping[str, Any] | None = None,
    examples: Sequence[Mapping[str, Any]] | None = None,
    stream: io.TextIOBase | None = None,
) -> list[dict[str, Any]]:
    if (report is None) != (examples is None):
        raise ValueError("report and examples must be supplied together")
    if report is None:
        config = load_config()
        paths = validate_frozen_inputs(config)
        loaded_report = load_json(report_path)
        loaded_examples = validate_probe(load_json(paths["selection_probe"]), config)
        recorded = loaded_report.get("frozen_hashes", {}).get("selection_probe")
        if recorded != config["frozen_inputs"]["selection_probe"]:
            raise ValueError("postmortem report references the wrong probe")
    else:
        loaded_report = dict(report)
        loaded_examples = list(examples or ())
    joined = join_review_queue(loaded_report, loaded_examples)
    output = stream if stream is not None else sys.stdout
    print(
        json.dumps(
            {
                "analysis_role": ANALYSIS_ROLE,
                "record_count": len(joined),
                "records": joined,
            },
            indent=2,
            sort_keys=True,
        ),
        file=output,
    )
    return joined


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--write", action="store_true")
    actions.add_argument("--check", action="store_true")
    actions.add_argument("--print-review-queue", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.print_review_queue:
        print_review_queue()
        return
    config = load_config()
    report_path = repository_path(config["output"]["path"])
    report = build_report()
    if args.write:
        write_report(report_path, report)
        relative_report = report_path.relative_to(REPOSITORY_ROOT)
        print(f"wrote diagnostic postmortem: {relative_report}")
        return
    check_report(report_path, report)
    print("postmortem report matches deterministic diagnostic reproduction")


if __name__ == "__main__":
    main()
