"""Evaluate the frozen V2-C1 classifier on scored CLINC finance lanes."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline

from backend.app.evaluation.v2_ml import INTENT_LABELS, apply_abstention_rule

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT = REPOSITORY_ROOT / "data/evals/v2/external"
ML_ROOT = REPOSITORY_ROOT / "data/evals/v2/ml"
INPUT_PATH = (
    EXTERNAL_ROOT
    / "processed/clinc_finance/clinc_finance_external_eval.json"
)
PROCESSED_MANIFEST_PATH = EXTERNAL_ROOT / "processed/clinc_finance/manifest.json"
RAW_DATA_PATH = EXTERNAL_ROOT / "raw/clinc_oos/data_full.json"
MAPPING_PATH = EXTERNAL_ROOT / "clinc_finance_intent_mapping.json"
MODEL_SELECTION_PATH = ML_ROOT / "model_selection.json"
V2C1_REPORT_PATH = ML_ROOT / "classifier_report.json"
ARTIFACT_PATH = REPOSITORY_ROOT / "artifacts/v2/classifier/classifier.joblib"
OUTPUT_PATH = (
    EXTERNAL_ROOT / "results/clinc_finance/frozen_v2c1_report.json"
)

SCHEMA_VERSION = "clinc-finance-frozen-v2c1-report.v1"
EVALUATION_VERSION = "clinc-finance-frozen-v2c1.2026-09-27.v1"
EVALUATION_BASE_COMMIT = "97aed99021248eb0c53d7e3995deeb8cca97a533"
EXPECTED_ARTIFACT_SHA256 = (
    "e815eb629de3ae56140a09f92baf563e9aaeec1204b45c23774210452d03285c"
)
EXPECTED_EXACT_COUNT = 90
EXPECTED_UNSUPPORTED_COUNT = 630
EXPECTED_REVIEW_POOL_COUNT = 180
EXACT_EXPECTED_LABELS = (
    "account_balance",
    "recent_transactions",
    "informational_policy",
)
PROTECTED_WRITE_LABELS = ("freeze_card", "create_dispute")
PRIVATE_READ_LABELS = {
    "account_balance",
    "card_status",
    "recent_transactions",
    "transaction_details",
}


def stable_json_bytes(payload: Any) -> bytes:
    """Serialize the report without wall-clock or platform-order variation."""
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_path(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def verify_sha256(path: Path, expected: str, description: str) -> str:
    actual = sha256_path(path)
    if actual != expected:
        raise ValueError(
            f"{description} SHA-256 mismatch: expected {expected}, got {actual}"
        )
    return actual


def load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def validate_metadata_and_hashes() -> dict[str, Any]:
    processed_manifest = load_json_object(PROCESSED_MANIFEST_PATH)
    selection = load_json_object(MODEL_SELECTION_PATH)
    v2c1_report = load_json_object(V2C1_REPORT_PATH)

    if selection["model_artifact_sha256"] != EXPECTED_ARTIFACT_SHA256:
        raise ValueError("model selection does not name the expected frozen artifact")
    if v2c1_report["artifact"]["sha256"] != EXPECTED_ARTIFACT_SHA256:
        raise ValueError("V2-C1 report does not name the expected frozen artifact")
    artifact_hash = verify_sha256(
        ARTIFACT_PATH,
        EXPECTED_ARTIFACT_SHA256,
        "frozen classifier artifact",
    )
    if ARTIFACT_PATH.stat().st_size != selection["model_artifact_size_bytes"]:
        raise ValueError("frozen classifier size differs from model selection")
    if ARTIFACT_PATH.stat().st_size != v2c1_report["artifact"]["size_bytes"]:
        raise ValueError("frozen classifier size differs from V2-C1 report")

    input_metadata = processed_manifest["processed_files"][INPUT_PATH.name]
    input_hash = verify_sha256(
        INPUT_PATH,
        input_metadata["sha256"],
        "processed CLINC finance evaluation",
    )
    mapping_hash = verify_sha256(
        MAPPING_PATH,
        processed_manifest["frozen_mapping_file_sha256"],
        "frozen CLINC finance mapping",
    )
    raw_hash = verify_sha256(
        RAW_DATA_PATH,
        processed_manifest["raw_data_full_sha256"],
        "raw CLINC data",
    )

    counts = processed_manifest.get("counts", {})
    expected_counts = {
        "ambiguous_review": 60,
        "exact_match": EXPECTED_EXACT_COUNT,
        "near_match_review": 120,
        "total_finance_test_examples": 900,
        "total_review_pool": EXPECTED_REVIEW_POOL_COUNT,
        "unsupported": EXPECTED_UNSUPPORTED_COUNT,
    }
    if counts != expected_counts:
        raise ValueError("processed CLINC finance manifest counts changed")

    return {
        "artifact_sha256": artifact_hash,
        "classifier_report_sha256": sha256_path(V2C1_REPORT_PATH),
        "input_sha256": input_hash,
        "mapping_sha256": mapping_hash,
        "model_selection_sha256": sha256_path(MODEL_SELECTION_PATH),
        "processed_manifest_sha256": sha256_path(PROCESSED_MANIFEST_PATH),
        "raw_data_sha256": raw_hash,
        "selection": selection,
    }


def validate_frozen_artifact(
    artifact: Any,
    selection: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(artifact, dict):
        raise TypeError("frozen classifier artifact must contain a dictionary")
    required = {
        "abstention",
        "intent_label_order",
        "intent_model",
        "logistic_probability_intent_model",
        "risk_label_order",
        "risk_model",
        "runtime_authority",
        "trusted_local_artifact",
    }
    missing = required - artifact.keys()
    if missing:
        raise ValueError(f"frozen classifier artifact is missing: {sorted(missing)}")
    if artifact["trusted_local_artifact"] is not True:
        raise ValueError("classifier artifact is not marked trusted-local")
    if artifact["runtime_authority"] is not False:
        raise ValueError("classifier artifact must have no runtime authority")
    if artifact["intent_label_order"] != selection["intent_label_order"]:
        raise ValueError("artifact and model-selection intent label orders differ")
    if tuple(artifact["intent_label_order"]) != INTENT_LABELS:
        raise ValueError("artifact does not preserve the full nine-intent taxonomy")
    if artifact["risk_label_order"] != selection["risk_label_order"]:
        raise ValueError("artifact and model-selection risk label orders differ")
    if artifact["abstention"] != selection["abstention"]["selected"]:
        raise ValueError("artifact abstention rule differs from frozen selection")

    intent_model = artifact["intent_model"]
    logistic_model = artifact["logistic_probability_intent_model"]
    risk_model = artifact["risk_model"]
    for name, model in (
        ("intent_model", intent_model),
        ("logistic_probability_intent_model", logistic_model),
        ("risk_model", risk_model),
    ):
        if not isinstance(model, Pipeline):
            raise TypeError(f"{name} must be a fitted sklearn Pipeline")

    if intent_model.named_steps["classifier"].__class__.__name__ != "LinearSVC":
        raise ValueError("official frozen intent model must be LinearSVC")
    if (
        logistic_model.named_steps["classifier"].__class__.__name__
        != "LogisticRegression"
    ):
        raise ValueError("frozen probability model must be LogisticRegression")
    if risk_model.named_steps["classifier"].__class__.__name__ != "LinearSVC":
        raise ValueError("official frozen risk model must be LinearSVC")
    if intent_model.classes_.tolist() != selection["intent_label_order"]:
        raise ValueError("intent model classes differ from the frozen label order")
    if logistic_model.classes_.tolist() != selection["intent_label_order"]:
        raise ValueError("probability model classes differ from the frozen label order")
    if risk_model.classes_.tolist() != selection["risk_label_order"]:
        raise ValueError("risk model classes differ from the frozen label order")
    return artifact


def load_evaluation_lanes(
    dataset: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if dataset.get("source_split") != "test":
        raise ValueError("CLINC finance evaluation must use only the test split")
    lanes = dataset.get("lanes")
    if not isinstance(lanes, dict) or set(lanes) != {"exact_match", "unsupported"}:
        raise ValueError("processed evaluation must contain only two scored lanes")

    exact = lanes["exact_match"]["examples"]
    unsupported = lanes["unsupported"]["examples"]
    if len(exact) != lanes["exact_match"]["example_count"]:
        raise ValueError("exact-match lane count differs from its metadata")
    if len(exact) != EXPECTED_EXACT_COUNT:
        raise ValueError("exact-match lane must contain 90 examples")
    if len(unsupported) != lanes["unsupported"]["example_count"]:
        raise ValueError("unsupported lane count differs from its metadata")
    if len(unsupported) != EXPECTED_UNSUPPORTED_COUNT:
        raise ValueError("unsupported lane must contain 630 examples")

    if any(row["mapping_status"] != "EXACT_MATCH" for row in exact):
        raise ValueError("exact-match lane contains a non-EXACT_MATCH row")
    if any(row["mapping_status"] != "UNSUPPORTED" for row in unsupported):
        raise ValueError("unsupported lane contains a non-UNSUPPORTED row")
    if {row["expected_sentinelvoice_intent"] for row in exact} != set(
        EXACT_EXPECTED_LABELS
    ):
        raise ValueError("exact-match lane has unexpected target classes")
    if any(
        row["expected_sentinelvoice_intent"] != "unsupported_or_uncertain"
        for row in unsupported
    ):
        raise ValueError("unsupported lane has an unexpected target class")
    if any(row["source_split"] != "test" for row in [*exact, *unsupported]):
        raise ValueError("a scored row does not originate from the CLINC test split")

    example_ids = [row["example_id"] for row in [*exact, *unsupported]]
    if len(example_ids) != len(set(example_ids)):
        raise ValueError("duplicate examples exist across scored lanes")
    return exact, unsupported


def distribution(values: list[str]) -> dict[str, dict[str, float | int]]:
    counts = Counter(values)
    total = len(values)
    return {
        label: {"count": counts[label], "rate": counts[label] / total}
        for label in INTENT_LABELS
    }


def numeric_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "maximum": float(np.max(values)),
        "mean": float(np.mean(values)),
        "minimum": float(np.min(values)),
        "p25": float(np.percentile(values, 25)),
        "p50": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
    }


def decision_margin_summary(model: Pipeline, texts: list[str]) -> dict[str, Any]:
    scores = np.asarray(model.decision_function(texts), dtype=float)
    if scores.ndim != 2 or scores.shape[1] != len(INTENT_LABELS):
        raise ValueError("frozen SVM must expose one decision score per intent")
    ordered = np.sort(scores, axis=1)
    top_one = ordered[:, -1]
    top_two = ordered[:, -2]
    return {
        "example_count": len(texts),
        "interpretation": "Raw LinearSVC scores and margins are not probabilities.",
        "top_1_score": numeric_summary(top_one),
        "top_2_score": numeric_summary(top_two),
        "top_1_minus_top_2_margin": numeric_summary(top_one - top_two),
    }


def protected_write_false_positives(predictions: list[str]) -> dict[str, Any]:
    total = len(predictions)
    by_intent = {
        label: {
            "count": predictions.count(label),
            "rate": predictions.count(label) / total,
        }
        for label in PROTECTED_WRITE_LABELS
    }
    combined_count = sum(row["count"] for row in by_intent.values())
    return {
        "by_intent": by_intent,
        "combined_count": combined_count,
        "combined_rate": combined_count / total,
    }


def exact_lane_metrics(
    examples: list[dict[str, Any]], predictions: list[str]
) -> dict[str, Any]:
    expected = [row["expected_sentinelvoice_intent"] for row in examples]
    precision, recall, f1, support = precision_recall_fscore_support(
        expected,
        predictions,
        labels=list(EXACT_EXPECTED_LABELS),
        zero_division=0,
    )
    confusion = {
        expected_label: {
            predicted_label: sum(
                expected_value == expected_label
                and predicted_value == predicted_label
                for expected_value, predicted_value in zip(
                    expected, predictions, strict=True
                )
            )
            for predicted_label in INTENT_LABELS
        }
        for expected_label in EXACT_EXPECTED_LABELS
    }
    major_confusions = sorted(
        (
            {
                "count": count,
                "expected": expected_label,
                "predicted": predicted_label,
                "rate_within_expected_class": count
                / expected.count(expected_label),
            }
            for expected_label, row in confusion.items()
            for predicted_label, count in row.items()
            if predicted_label != expected_label and count
        ),
        key=lambda row: (-row["count"], row["expected"], row["predicted"]),
    )
    return {
        "accuracy": float(accuracy_score(expected, predictions)),
        "confusion_by_expected_and_predicted_intent": confusion,
        "example_count": len(examples),
        "false_protected_write": protected_write_false_positives(predictions),
        "macro_f1": float(
            f1_score(
                expected,
                predictions,
                labels=list(EXACT_EXPECTED_LABELS),
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1_scope": list(EXACT_EXPECTED_LABELS),
        "major_confusions": major_confusions,
        "per_expected_class": {
            label: {
                "f1": float(f1[index]),
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(EXACT_EXPECTED_LABELS)
        },
        "prediction_distribution_full_nine_intents": distribution(predictions),
    }


def unsupported_lane_metrics(predictions: list[str]) -> dict[str, Any]:
    total = len(predictions)
    unsupported_count = predictions.count("unsupported_or_uncertain")
    private_read_count = sum(
        prediction in PRIVATE_READ_LABELS for prediction in predictions
    )
    return {
        "example_count": total,
        "false_private_read_prediction": {
            "count": private_read_count,
            "rate": private_read_count / total,
        },
        "false_protected_write": protected_write_false_positives(predictions),
        "false_supported_intent_rate": (total - unsupported_count) / total,
        "prediction_distribution_full_nine_intents": distribution(predictions),
        "unsupported_or_uncertain_recall": unsupported_count / total,
    }


def frozen_advisory_abstention_analysis(
    *,
    exact: list[dict[str, Any]],
    unsupported: list[dict[str, Any]],
    logistic_model: Pipeline,
    risk_model: Pipeline,
    selected_rule: dict[str, Any],
) -> dict[str, Any]:
    def evaluate_lane(
        examples: list[dict[str, Any]], *, exact_lane: bool
    ) -> dict[str, Any]:
        texts = [row["text"] for row in examples]
        expected = np.asarray(
            [row["expected_sentinelvoice_intent"] for row in examples]
        )
        probabilities = logistic_model.predict_proba(texts)
        top_indices = np.argmax(probabilities, axis=1)
        predictions = np.asarray(logistic_model.classes_)[top_indices]
        accepted = apply_abstention_rule(
            rule={"selected": selected_rule},
            intent_probabilities=probabilities,
            intent_classes=logistic_model.classes_.tolist(),
            independent_risk_predictions=risk_model.predict(texts).tolist(),
        )
        accepted_count = int(np.sum(accepted))
        common: dict[str, Any] = {
            "abstained_count": len(examples) - accepted_count,
            "abstention_rate": 1.0 - (accepted_count / len(examples)),
            "accepted_count": accepted_count,
            "coverage": accepted_count / len(examples),
            "example_count": len(examples),
            "unabstained_prediction_distribution": distribution(
                predictions.tolist()
            ),
        }
        if exact_lane:
            common.update(
                {
                    "selective_accuracy": (
                        float(np.mean(predictions[accepted] == expected[accepted]))
                        if accepted_count
                        else None
                    ),
                    "unabstained_accuracy": float(np.mean(predictions == expected)),
                }
            )
        else:
            accepted_unsupported = int(
                np.sum(predictions[accepted] == "unsupported_or_uncertain")
            )
            total_unsupported = int(
                np.sum(predictions == "unsupported_or_uncertain")
            )
            common.update(
                {
                    "accepted_as_unsupported_count": accepted_unsupported,
                    "accepted_as_unsupported_rate": (
                        accepted_unsupported / accepted_count
                        if accepted_count
                        else None
                    ),
                    "accepted_false_supported_count": (
                        accepted_count - accepted_unsupported
                    ),
                    "accepted_false_supported_rate": (
                        (accepted_count - accepted_unsupported) / accepted_count
                        if accepted_count
                        else None
                    ),
                    "unabstained_unsupported_recall": (
                        total_unsupported / len(examples)
                    ),
                }
            )
        return common

    return {
        "analysis_role": (
            "Frozen Logistic Regression probability path; separate from primary "
            "SVM metrics and not validation of the SVM."
        ),
        "exact_match_lane": evaluate_lane(exact, exact_lane=True),
        "frozen_rule": selected_rule,
        "threshold_selection_source": "V2-C1 validation only",
        "unsupported_lane": evaluate_lane(unsupported, exact_lane=False),
    }


def build_report() -> dict[str, Any]:
    metadata = validate_metadata_and_hashes()
    dataset = load_json_object(INPUT_PATH)
    exact, unsupported = load_evaluation_lanes(dataset)
    artifact = validate_frozen_artifact(
        joblib.load(ARTIFACT_PATH), metadata["selection"]
    )

    exact_texts = [row["text"] for row in exact]
    unsupported_texts = [row["text"] for row in unsupported]
    intent_model = artifact["intent_model"]
    exact_predictions = intent_model.predict(exact_texts).tolist()
    unsupported_predictions = intent_model.predict(unsupported_texts).tolist()

    return {
        "advisory_only": True,
        "evaluation_base_commit": EVALUATION_BASE_COMMIT,
        "evaluation_version": EVALUATION_VERSION,
        "external_metrics_boundary": {
            "directly_comparable_to_v2c1_nine_intent_locked_test_macro_f1": False,
            "exact_match_macro_f1_expected_class_count": 3,
            "reported_separately_from_v2c1": True,
        },
        "frozen_advisory_abstention_analysis": frozen_advisory_abstention_analysis(
            exact=exact,
            unsupported=unsupported,
            logistic_model=artifact["logistic_probability_intent_model"],
            risk_model=artifact["risk_model"],
            selected_rule=artifact["abstention"],
        ),
        "frozen_classifier": {
            "artifact_path": str(ARTIFACT_PATH.relative_to(REPOSITORY_ROOT)),
            "artifact_sha256": metadata["artifact_sha256"],
            "components": {
                "advisory_abstention_rule": artifact["abstention"],
                "independent_risk_model": "LinearSVC",
                "logistic_probability_intent_model": "LogisticRegression",
                "selected_intent_model": "LinearSVC",
            },
            "full_intent_label_order": artifact["intent_label_order"],
            "modified_or_retrained": False,
            "runtime_authority": False,
        },
        "integrity": {
            "classifier_report_sha256": metadata["classifier_report_sha256"],
            "frozen_mapping_sha256": metadata["mapping_sha256"],
            "model_selection_sha256": metadata["model_selection_sha256"],
            "processed_evaluation_sha256": metadata["input_sha256"],
            "processed_manifest_sha256": metadata["processed_manifest_sha256"],
            "raw_clinc_data_sha256": metadata["raw_data_sha256"],
        },
        "primary_classifier": "frozen V2-C1 LinearSVC intent model",
        "primary_svm": {
            "exact_match_lane": {
                **exact_lane_metrics(exact, exact_predictions),
                "decision_margin": decision_margin_summary(
                    intent_model, exact_texts
                ),
            },
            "unsupported_lane": {
                **unsupported_lane_metrics(unsupported_predictions),
                "decision_margin": decision_margin_summary(
                    intent_model, unsupported_texts
                ),
            },
        },
        "schema_version": SCHEMA_VERSION,
        "scored_source": {
            "dataset_path": str(INPUT_PATH.relative_to(REPOSITORY_ROOT)),
            "dataset_sha256": metadata["input_sha256"],
            "dataset_version": dataset["dataset_version"],
            "exact_match_examples": len(exact),
            "oos_examples_scored": False,
            "review_pool_examples_excluded": EXPECTED_REVIEW_POOL_COUNT,
            "review_pool_scored": False,
            "source_id": dataset["source_id"],
            "source_revision": dataset["source_revision"],
            "source_split": dataset["source_split"],
            "unsupported_examples": len(unsupported),
        },
        "training_or_tuning_performed": False,
    }


def build_report_bytes() -> bytes:
    return stable_json_bytes(build_report())


def main() -> None:
    report_bytes = build_report_bytes()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_bytes(report_bytes)
    report = json.loads(report_bytes)
    exact = report["primary_svm"]["exact_match_lane"]
    unsupported = report["primary_svm"]["unsupported_lane"]
    print(
        "Evaluated frozen V2-C1 model on CLINC finance: "
        f"exact_accuracy={exact['accuracy']:.6f}, "
        f"exact_macro_f1={exact['macro_f1']:.6f}, "
        "unsupported_recall="
        f"{unsupported['unsupported_or_uncertain_recall']:.6f}"
    )


if __name__ == "__main__":
    main()
