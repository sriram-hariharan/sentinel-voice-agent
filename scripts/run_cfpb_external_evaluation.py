"""Evaluate the frozen V2-C1 classifier on frozen CFPB semantic labels."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from backend.app.evaluation.v2_ml import INTENT_LABELS, apply_abstention_rule

try:
    from scripts import export_cfpb_semantic_final_labels as label_contract
except ModuleNotFoundError:  # Direct execution adds scripts/, not the repo root.
    import export_cfpb_semantic_final_labels as label_contract


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT = REPOSITORY_ROOT / "data/evals/v2/external"
ML_ROOT = REPOSITORY_ROOT / "data/evals/v2/ml"
LABEL_PATH = (
    EXTERNAL_ROOT / "processed/cfpb/cfpb_semantic_final_labels.jsonl"
)
REVIEW_POOL_MANIFEST_PATH = (
    EXTERNAL_ROOT / "processed/cfpb/review_pool_manifest.json"
)
SOURCE_POOL_PATH = (
    EXTERNAL_ROOT / "processed/cfpb/local/cfpb_review_pool.jsonl"
)
MODEL_SELECTION_PATH = ML_ROOT / "model_selection.json"
V2C1_REPORT_PATH = ML_ROOT / "classifier_report.json"
ARTIFACT_PATH = REPOSITORY_ROOT / "artifacts/v2/classifier/classifier.joblib"
OUTPUT_PATH = EXTERNAL_ROOT / "results/cfpb/frozen_v2c1_report.json"

SCHEMA_VERSION = "cfpb-frozen-v2c1-report.v1"
EVALUATION_VERSION = "cfpb-frozen-v2c1.2026-09-28.v1"
EVALUATION_BASE_COMMIT = "b7f52f29fd81e17a604604af45d0cca551e15162"
EXPECTED_ARTIFACT_SHA256 = (
    "e815eb629de3ae56140a09f92baf563e9aaeec1204b45c23774210452d03285c"
)
EXPECTED_MODEL_SELECTION_SHA256 = (
    "700af92377110b585967a7300ff1b1682f45f84ac8d0aa09e18c8b85d540dab3"
)
EXPECTED_V2C1_REPORT_SHA256 = (
    "dcf940890b2602e040ed3f0a30b0f45f0c34e352707896ea64079b20c929f843"
)
EXPECTED_LABEL_SHA256 = (
    "40da78787aa5eb7d2a0542d51a9af007b4c836b899559d9185052d24f24f490e"
)
EXPECTED_REVIEW_POOL_MANIFEST_SHA256 = (
    "d078cc46dcf705261f13a713e778168d681bc5fb8c8397979383f85bbe273576"
)
EXPECTED_SOURCE_COUNT = 3_800
EXPECTED_HOLDOUT_COUNT = 1_800
EXPECTED_PRIMARY_COUNT = 1_776
EXPECTED_MULTI_COUNT = 24
EXPECTED_SOURCE_MAPPING_COUNTS = {
    "AMBIGUOUS": 1_200,
    "NEAR_MATCH": 600,
    "UNSUPPORTED": 2_000,
}
EXPECTED_CATEGORY_COUNTS = {
    "MULTI_SUPPORTED_INTENT": 24,
    "NO_CURRENT_REQUEST": 616,
    "SINGLE_SUPPORTED_INTENT": 303,
    "UNCLEAR_OR_INSUFFICIENT": 37,
    "UNSUPPORTED": 820,
}
EXPECTED_PROTECTED_WRITE_COUNTS = {"create_dispute": 28, "freeze_card": 1}
PROTECTED_WRITE_LABELS = ("freeze_card", "create_dispute")
SOURCE_FIELDS = frozenset(
    {
        "candidate_sentinelvoice_intents",
        "complaint_id",
        "date_received",
        "issue",
        "length_bin",
        "mapping_status",
        "narrative",
        "narrative_length",
        "narrative_sha256",
        "product",
        "source_archive",
        "sub_issue",
        "sub_product",
    }
)


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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"{path} line {line_number} must be a JSON object")
        records.append(value)
    return records


def validate_metadata_and_hashes() -> dict[str, Any]:
    """Pin every tracked contract and the manifest-pinned local source bytes."""
    selection_hash = verify_sha256(
        MODEL_SELECTION_PATH,
        EXPECTED_MODEL_SELECTION_SHA256,
        "frozen V2-C1 model selection",
    )
    classifier_report_hash = verify_sha256(
        V2C1_REPORT_PATH,
        EXPECTED_V2C1_REPORT_SHA256,
        "frozen V2-C1 classifier report",
    )
    label_hash = verify_sha256(
        LABEL_PATH,
        EXPECTED_LABEL_SHA256,
        "frozen CFPB semantic labels",
    )
    manifest_hash = verify_sha256(
        REVIEW_POOL_MANIFEST_PATH,
        EXPECTED_REVIEW_POOL_MANIFEST_SHA256,
        "frozen CFPB review-pool manifest",
    )
    artifact_hash = verify_sha256(
        ARTIFACT_PATH,
        EXPECTED_ARTIFACT_SHA256,
        "frozen classifier artifact",
    )

    selection = load_json_object(MODEL_SELECTION_PATH)
    v2c1_report = load_json_object(V2C1_REPORT_PATH)
    manifest = load_json_object(REVIEW_POOL_MANIFEST_PATH)
    if selection.get("model_artifact_sha256") != EXPECTED_ARTIFACT_SHA256:
        raise ValueError("model selection does not name the frozen artifact")
    if v2c1_report.get("artifact", {}).get("sha256") != EXPECTED_ARTIFACT_SHA256:
        raise ValueError("V2-C1 report does not name the frozen artifact")
    if ARTIFACT_PATH.stat().st_size != selection.get("model_artifact_size_bytes"):
        raise ValueError("frozen classifier size differs from model selection")
    if ARTIFACT_PATH.stat().st_size != v2c1_report.get("artifact", {}).get(
        "size_bytes"
    ):
        raise ValueError("frozen classifier size differs from V2-C1 report")
    if selection.get("locked_test_evaluated") is not False:
        raise ValueError("model selection must predate locked-test evaluation")
    if selection.get("selection_splits") != ["train", "validation"]:
        raise ValueError("model selection must use only train and validation")
    if selection.get("selected_intent_model", {}).get("model") != "linear_svm":
        raise ValueError("frozen selected intent model must be LinearSVC")
    if selection["selected_intent_model"].get("c") != 0.5:
        raise ValueError("frozen selected intent model must use C=0.5")
    if selection.get("probability_intent_model", {}).get("model") != (
        "logistic_regression"
    ):
        raise ValueError("frozen probability intent model must be LogisticRegression")
    if selection["probability_intent_model"].get("c") != 2.0:
        raise ValueError("frozen probability intent model must use C=2.0")

    local_material = manifest.get("local_review_material", {})
    expected_source_path = str(SOURCE_POOL_PATH.relative_to(REPOSITORY_ROOT))
    if local_material.get("path") != expected_source_path:
        raise ValueError("review-pool manifest names an unexpected local source")
    if local_material.get("row_count") != EXPECTED_SOURCE_COUNT:
        raise ValueError("review-pool manifest source count changed")
    if local_material.get("contains_narrative_text") is not True:
        raise ValueError("local CFPB source must be declared narrative-bearing")
    source_hash = verify_sha256(
        SOURCE_POOL_PATH,
        local_material.get("sha256"),
        "manifest-pinned local CFPB review pool",
    )
    return {
        "artifact_sha256": artifact_hash,
        "classifier_report_sha256": classifier_report_hash,
        "label_sha256": label_hash,
        "manifest": manifest,
        "model_selection_sha256": selection_hash,
        "review_pool_manifest_sha256": manifest_hash,
        "selection": selection,
        "source_pool_sha256": source_hash,
    }


def _validate_tfidf_features(model: Pipeline, name: str) -> None:
    if tuple(model.named_steps) != ("features", "classifier"):
        raise ValueError(f"{name} must preserve the frozen Pipeline steps")
    features = model.named_steps["features"]
    if not isinstance(features, FeatureUnion):
        raise TypeError(f"{name} features must be an sklearn FeatureUnion")
    transformers = dict(features.transformer_list)
    if set(transformers) != {"word", "character"}:
        raise ValueError(f"{name} must preserve word and character TF-IDF")
    word = transformers["word"]
    character = transformers["character"]
    if not isinstance(word, TfidfVectorizer) or not isinstance(
        character, TfidfVectorizer
    ):
        raise TypeError(f"{name} feature branches must be TfidfVectorizer")
    if word.analyzer != "word" or word.ngram_range != (1, 2):
        raise ValueError(f"{name} word TF-IDF configuration changed")
    if character.analyzer != "char_wb" or character.ngram_range != (3, 5):
        raise ValueError(f"{name} character TF-IDF configuration changed")
    if word.sublinear_tf is not True or character.sublinear_tf is not True:
        raise ValueError(f"{name} TF-IDF sublinear scaling changed")


def validate_frozen_artifact(
    artifact: Any,
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate fitted model types, feature contracts, labels, and thresholds."""
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
        raise ValueError("artifact does not preserve the frozen nine intents")
    if artifact["risk_label_order"] != selection["risk_label_order"]:
        raise ValueError("artifact and model-selection risk label orders differ")
    selected_rule = selection["abstention"]["selected"]
    if artifact["abstention"] != selected_rule:
        raise ValueError("artifact abstention rule differs from frozen selection")
    if (
        selected_rule.get("mode") != "threshold"
        or selected_rule.get("confidence_threshold") != 0.0
        or selected_rule.get("margin_threshold") != 0.2
        or selected_rule.get("require_intent_risk_agreement") is not False
    ):
        raise ValueError("frozen abstention rule changed")

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
        _validate_tfidf_features(model, name)

    intent_classifier = intent_model.named_steps["classifier"]
    logistic_classifier = logistic_model.named_steps["classifier"]
    risk_classifier = risk_model.named_steps["classifier"]
    if not isinstance(intent_classifier, LinearSVC) or intent_classifier.C != 0.5:
        raise ValueError("official frozen intent model must be LinearSVC C=0.5")
    if not isinstance(logistic_classifier, LogisticRegression) or (
        logistic_classifier.C != 2.0
    ):
        raise ValueError("frozen probability model must be LogisticRegression C=2")
    if not isinstance(risk_classifier, LinearSVC):
        raise TypeError("official frozen risk model must be LinearSVC")
    if intent_model.classes_.tolist() != selection["intent_label_order"]:
        raise ValueError("intent model classes differ from the frozen label order")
    if logistic_model.classes_.tolist() != selection["intent_label_order"]:
        raise ValueError("probability model classes differ from the frozen order")
    if risk_model.classes_.tolist() != selection["risk_label_order"]:
        raise ValueError("risk model classes differ from the frozen label order")
    return artifact


def _validate_review_pool_manifest(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    if manifest.get("schema_version") != "cfpb-review-pool-manifest.v1":
        raise ValueError("unexpected CFPB review-pool manifest schema")
    summary = manifest.get("summary", {})
    if summary.get("total_selected") != EXPECTED_SOURCE_COUNT:
        raise ValueError("CFPB review-pool manifest total changed")
    if summary.get("selected_by_mapping_status") != EXPECTED_SOURCE_MAPPING_COUNTS:
        raise ValueError("CFPB review-pool lane counts changed")
    records = manifest.get("selected_records")
    if not isinstance(records, list) or not all(
        isinstance(record, dict) for record in records
    ):
        raise TypeError("review-pool manifest selected_records must be objects")
    if len(records) != EXPECTED_SOURCE_COUNT:
        raise ValueError("review-pool manifest selected-record count changed")
    hashes = [record.get("narrative_sha256") for record in records]
    if len(hashes) != len(set(hashes)):
        raise ValueError("review-pool manifest contains duplicate source hashes")
    return records


def _validate_source_pool(
    source_records: Sequence[Mapping[str, Any]],
    manifest_records: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    if len(source_records) != EXPECTED_SOURCE_COUNT:
        raise ValueError(f"CFPB source pool must contain {EXPECTED_SOURCE_COUNT} rows")
    source_hashes = [record.get("narrative_sha256") for record in source_records]
    if len(source_hashes) != len(set(source_hashes)):
        raise ValueError("CFPB source pool contains duplicate narrative hashes")
    complaint_ids = [record.get("complaint_id") for record in source_records]
    if len(complaint_ids) != len(set(complaint_ids)):
        raise ValueError("CFPB source pool contains duplicate complaint IDs")
    mapping_counts = Counter(record.get("mapping_status") for record in source_records)
    if dict(mapping_counts) != EXPECTED_SOURCE_MAPPING_COUNTS:
        raise ValueError("CFPB source-pool lane counts changed")

    manifest_by_hash = {
        record["narrative_sha256"]: record for record in manifest_records
    }
    source_by_hash: dict[str, Mapping[str, Any]] = {}
    for record in source_records:
        if set(record) != SOURCE_FIELDS:
            raise ValueError("CFPB local source fields differ from the frozen schema")
        narrative = record["narrative"]
        digest = record["narrative_sha256"]
        if not isinstance(narrative, str) or not narrative.strip():
            raise ValueError("CFPB source row requires narrative text")
        if sha256_bytes(narrative.encode("utf-8")) != digest:
            raise ValueError("CFPB source narrative differs from its SHA-256")
        if record["narrative_length"] != len(narrative):
            raise ValueError("CFPB source narrative length metadata changed")
        manifest_record = manifest_by_hash.get(digest)
        if manifest_record is None:
            raise ValueError("CFPB source hash is absent from the frozen manifest")
        for field in SOURCE_FIELDS - {"narrative"}:
            if record[field] != manifest_record[field]:
                raise ValueError(f"CFPB source {field} differs from the manifest")
        source_by_hash[digest] = record
    if source_hashes != [row["narrative_sha256"] for row in manifest_records]:
        raise ValueError("CFPB source order differs from the frozen manifest")
    return source_by_hash


def _validate_final_labels(
    labels: Sequence[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    label_contract.validate_final_export_records(labels)
    if Counter(row["final_review_category"] for row in labels) != Counter(
        EXPECTED_CATEGORY_COUNTS
    ):
        raise ValueError("frozen CFPB final-category counts changed")
    primary = [row for row in labels if row["primary_single_label_evaluable"]]
    multi = [
        row for row in labels
        if row["final_review_category"] == "MULTI_SUPPORTED_INTENT"
    ]
    if len(primary) != EXPECTED_PRIMARY_COUNT or len(multi) != EXPECTED_MULTI_COUNT:
        raise ValueError("frozen CFPB primary/MULTI scoring split changed")
    protected_counts = {
        intent: sum(intent in row["final_supported_intents"] for row in labels)
        for intent in PROTECTED_WRITE_LABELS
    }
    if protected_counts != EXPECTED_PROTECTED_WRITE_COUNTS:
        raise ValueError("frozen CFPB protected-write label counts changed")
    return primary, multi


def load_and_validate_cfpb_inputs(
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Join exactly 1,800 frozen labels to the 3,800-row local source by hash."""
    if metadata is None:
        metadata = validate_metadata_and_hashes()
    labels = load_jsonl(LABEL_PATH)
    primary_labels, multi_labels = _validate_final_labels(labels)
    primary_hashes = {row["narrative_sha256"] for row in primary_labels}
    multi_hashes = {row["narrative_sha256"] for row in multi_labels}
    manifest_records = _validate_review_pool_manifest(metadata["manifest"])
    source_by_hash = _validate_source_pool(
        load_jsonl(SOURCE_POOL_PATH), manifest_records
    )

    selected: list[dict[str, Any]] = []
    for label in labels:
        source = source_by_hash.get(label["narrative_sha256"])
        if source is None:
            raise ValueError("frozen CFPB label hash is absent from the source pool")
        if source["complaint_id"] != label["complaint_id"]:
            raise ValueError("CFPB complaint ID differs after hash linkage")
        if source["mapping_status"] != label["original_mapping_status"]:
            raise ValueError("CFPB mapping status differs after hash linkage")
        selected.append({**label, "text": source["narrative"]})
    if len(selected) != EXPECTED_HOLDOUT_COUNT:
        raise ValueError("CFPB label/source join did not select exactly 1,800 rows")
    if len({row["narrative_sha256"] for row in selected}) != EXPECTED_HOLDOUT_COUNT:
        raise ValueError("CFPB label/source join contains duplicate hashes")
    return {
        "all": selected,
        "multi": [
            row for row in selected if row["narrative_sha256"] in multi_hashes
        ],
        "primary": [
            row for row in selected if row["narrative_sha256"] in primary_hashes
        ],
        "source_pool_count": len(source_by_hash),
    }


def distribution(values: Sequence[str]) -> dict[str, dict[str, float | int]]:
    counts = Counter(values)
    total = len(values)
    return {
        label: {
            "count": counts[label],
            "rate": counts[label] / total if total else 0.0,
        }
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


def decision_margin_summary(model: Pipeline, texts: Sequence[str]) -> dict[str, Any]:
    scores = np.asarray(model.decision_function(list(texts)), dtype=float)
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


def primary_single_label_metrics(
    records: Sequence[Mapping[str, Any]],
    predictions: Sequence[str],
) -> dict[str, Any]:
    if len(records) != len(predictions):
        raise ValueError("primary records and predictions differ in length")
    if any(prediction not in INTENT_LABELS for prediction in predictions):
        raise ValueError("primary prediction is outside the frozen taxonomy")
    expected = [str(row["primary_single_label_target"]) for row in records]
    precision, recall, f1, support = precision_recall_fscore_support(
        expected,
        predictions,
        labels=list(INTENT_LABELS),
        zero_division=0,
    )
    confusion = {
        expected_label: {
            predicted_label: sum(
                gold == expected_label and predicted == predicted_label
                for gold, predicted in zip(expected, predictions, strict=True)
            )
            for predicted_label in INTENT_LABELS
        }
        for expected_label in INTENT_LABELS
    }
    major_confusions = sorted(
        (
            {
                "count": count,
                "expected": expected_label,
                "predicted": predicted_label,
                "rate_within_expected_class": count / expected.count(expected_label),
            }
            for expected_label, row in confusion.items()
            for predicted_label, count in row.items()
            if predicted_label != expected_label and count
        ),
        key=lambda row: (-row["count"], row["expected"], row["predicted"]),
    )
    unsupported = "unsupported_or_uncertain"
    unsupported_total = expected.count(unsupported)
    false_supported_predictions = [
        predicted
        for gold, predicted in zip(expected, predictions, strict=True)
        if gold == unsupported and predicted != unsupported
    ]
    return {
        "accuracy": float(accuracy_score(expected, predictions)),
        "confusion_by_expected_and_predicted_intent": confusion,
        "example_count": len(records),
        "false_supported_behavior": {
            "false_supported_prediction_count": len(false_supported_predictions),
            "false_supported_prediction_distribution": distribution(
                false_supported_predictions
            ),
            "false_supported_rate_within_gold_unsupported": (
                len(false_supported_predictions) / unsupported_total
            ),
            "gold_unsupported_count": unsupported_total,
        },
        "gold_distribution_full_nine_intents": distribution(expected),
        "macro_f1": float(
            f1_score(
                expected,
                predictions,
                labels=list(INTENT_LABELS),
                average="macro",
                zero_division=0,
            )
        ),
        "major_confusions": major_confusions,
        "per_class": {
            label: {
                "f1": float(f1[index]),
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(INTENT_LABELS)
        },
        "prediction_distribution_full_nine_intents": distribution(predictions),
        "unsupported_or_uncertain_recall": float(
            recall[INTENT_LABELS.index(unsupported)]
        ),
        "weighted_f1": float(
            f1_score(
                expected,
                predictions,
                labels=list(INTENT_LABELS),
                average="weighted",
                zero_division=0,
            )
        ),
    }


def multi_intent_metrics(
    records: Sequence[Mapping[str, Any]],
    predictions: Sequence[str],
) -> dict[str, Any]:
    if len(records) != len(predictions):
        raise ValueError("multi-intent records and predictions differ in length")
    hits = sum(
        label_contract.multi_intent_prediction_is_hit(
            prediction, row["final_supported_intents"]
        )
        for row, prediction in zip(records, predictions, strict=True)
    )
    return {
        "example_count": len(records),
        "interpretation": (
            "Secondary set-membership metric only; this does not make the frozen "
            "classifier multi-label."
        ),
        "prediction_distribution_full_nine_intents": distribution(predictions),
        "prediction_is_supported_intent_hit_count": hits,
        "prediction_is_supported_intent_accuracy": hits / len(records),
        "scoring_rule": "prediction in final_supported_intents",
    }


def protected_write_metrics(
    records: Sequence[Mapping[str, Any]],
    predictions: Sequence[str],
    *,
    included: Sequence[bool] | None = None,
) -> dict[str, Any]:
    if len(records) != len(predictions):
        raise ValueError("protected-write records and predictions differ in length")
    mask = list(included) if included is not None else [True] * len(records)
    if len(mask) != len(records):
        raise ValueError("protected-write inclusion mask differs in length")
    by_intent: dict[str, Any] = {}
    included_count = sum(mask)
    for intent in PROTECTED_WRITE_LABELS:
        gold = [intent in row["final_supported_intents"] for row in records]
        predicted = [value == intent for value in predictions]
        gold_positive = sum(
            use and is_gold for use, is_gold in zip(mask, gold, strict=True)
        )
        predicted_positive = sum(
            use and is_predicted
            for use, is_predicted in zip(mask, predicted, strict=True)
        )
        true_positive = sum(
            use and is_gold and is_predicted
            for use, is_gold, is_predicted in zip(
                mask, gold, predicted, strict=True
            )
        )
        false_positive = predicted_positive - true_positive
        negative_count = included_count - gold_positive
        by_intent[intent] = {
            "false_discovery_rate": (
                false_positive / predicted_positive if predicted_positive else 0.0
            ),
            "false_positive_count": false_positive,
            "false_positive_rate": (
                false_positive / negative_count if negative_count else 0.0
            ),
            "gold_negative_count": negative_count,
            "gold_supported_count": gold_positive,
            "predicted_positive_count": predicted_positive,
            "true_positive_count": true_positive,
        }
    return {
        "by_intent": by_intent,
        "evaluated_example_count": included_count,
        "gold_rule": "intent membership in final_supported_intents",
    }


def frozen_logistic_abstention_analysis(
    *,
    records: Sequence[Mapping[str, Any]],
    logistic_model: Pipeline,
    risk_model: Pipeline,
    selected_rule: Mapping[str, Any],
) -> dict[str, Any]:
    texts = [str(row["text"]) for row in records]
    expected = np.asarray(
        [str(row["primary_single_label_target"]) for row in records]
    )
    probabilities = np.asarray(logistic_model.predict_proba(texts), dtype=float)
    top_indices = np.argmax(probabilities, axis=1)
    predictions = np.asarray(logistic_model.classes_)[top_indices]
    risk_predictions = risk_model.predict(texts).tolist()
    accepted = apply_abstention_rule(
        rule={"selected": dict(selected_rule)},
        intent_probabilities=probabilities,
        intent_classes=logistic_model.classes_.tolist(),
        independent_risk_predictions=risk_predictions,
    )
    accepted_count = int(np.sum(accepted))
    correct = predictions == expected
    errors = ~correct
    selective_macro_f1 = (
        float(
            f1_score(
                expected[accepted],
                predictions[accepted],
                labels=list(INTENT_LABELS),
                average="macro",
                zero_division=0,
            )
        )
        if accepted_count
        else None
    )
    return {
        "analysis_role": (
            "Frozen Logistic Regression probability path; separate from the "
            "primary SVM and not validation or replacement of it."
        ),
        "abstained_count": len(records) - accepted_count,
        "abstention_rate": 1.0 - (accepted_count / len(records)),
        "accepted_count": accepted_count,
        "accepted_protected_write_predictions": protected_write_metrics(
            records,
            predictions.tolist(),
            included=accepted.tolist(),
        ),
        "correct_predictions_removed": int(np.sum(correct & ~accepted)),
        "coverage": accepted_count / len(records),
        "errors_remaining": int(np.sum(errors & accepted)),
        "errors_removed_by_abstention": int(np.sum(errors & ~accepted)),
        "example_count": len(records),
        "frozen_rule": dict(selected_rule),
        "selective_accuracy": (
            float(np.mean(correct[accepted])) if accepted_count else None
        ),
        "selective_macro_f1_full_nine_intents": selective_macro_f1,
        "threshold_selection_source": "V2-C1 validation only",
        "unabstained_accuracy": float(np.mean(correct)),
        "unabstained_prediction_distribution": distribution(
            predictions.tolist()
        ),
    }


def build_report() -> dict[str, Any]:
    metadata = validate_metadata_and_hashes()
    inputs = load_and_validate_cfpb_inputs(metadata)
    artifact = validate_frozen_artifact(
        joblib.load(ARTIFACT_PATH), metadata["selection"]
    )

    records = inputs["all"]
    texts = [row["text"] for row in records]
    intent_model = artifact["intent_model"]
    all_predictions = intent_model.predict(texts).tolist()
    primary_records = [
        row for row in records if row["primary_single_label_evaluable"]
    ]
    primary_predictions = [
        prediction
        for row, prediction in zip(records, all_predictions, strict=True)
        if row["primary_single_label_evaluable"]
    ]
    multi_records = [
        row
        for row in records
        if row["final_review_category"] == "MULTI_SUPPORTED_INTENT"
    ]
    multi_predictions = [
        prediction
        for row, prediction in zip(records, all_predictions, strict=True)
        if row["final_review_category"] == "MULTI_SUPPORTED_INTENT"
    ]
    category_counts = Counter(row["final_review_category"] for row in records)
    protected_write_counts = {
        intent: sum(intent in row["final_supported_intents"] for row in records)
        for intent in PROTECTED_WRITE_LABELS
    }

    return {
        "advisory_only": True,
        "consumer_narrative_text_emitted": False,
        "evaluation_base_commit": EVALUATION_BASE_COMMIT,
        "evaluation_version": EVALUATION_VERSION,
        "external_metrics_boundary": {
            "directly_comparable_to_v2c1_balanced_nine_intent_test": False,
            "external_results_used_to_select_v2c1": False,
            "label_and_scoring_contract_frozen_before_inference": True,
            "reported_separately_from_v2c1": True,
        },
        "frozen_advisory_abstention_analysis": (
            frozen_logistic_abstention_analysis(
                records=primary_records,
                logistic_model=artifact["logistic_probability_intent_model"],
                risk_model=artifact["risk_model"],
                selected_rule=artifact["abstention"],
            )
        ),
        "frozen_classifier": {
            "artifact_path": str(ARTIFACT_PATH.relative_to(REPOSITORY_ROOT)),
            "artifact_sha256": metadata["artifact_sha256"],
            "components": {
                "advisory_abstention_rule": artifact["abstention"],
                "independent_risk_model": "LinearSVC",
                "logistic_probability_intent_model": "LogisticRegression C=2.0",
                "selected_intent_model": "LinearSVC C=0.5",
                "text_features": "word 1-2 + char_wb 3-5 TF-IDF",
            },
            "full_intent_label_order": artifact["intent_label_order"],
            "modified_or_retrained": False,
            "runtime_authority": False,
        },
        "integrity": {
            "classifier_report_sha256": metadata["classifier_report_sha256"],
            "final_labels_sha256": metadata["label_sha256"],
            "model_selection_sha256": metadata["model_selection_sha256"],
            "review_pool_manifest_sha256": metadata[
                "review_pool_manifest_sha256"
            ],
            "source_pool_sha256": metadata["source_pool_sha256"],
        },
        "label_contract": {
            "category_counts": dict(sorted(category_counts.items())),
            "excluded_multi_intent_count": len(multi_records),
            "multi_intent_scoring_rule": (
                "prediction in final_supported_intents"
            ),
            "primary_evaluation_coverage": len(primary_records) / len(records),
            "primary_scoring_mapping": {
                "MULTI_SUPPORTED_INTENT": None,
                "NO_CURRENT_REQUEST": "unsupported_or_uncertain",
                "SINGLE_SUPPORTED_INTENT": "sole final_supported_intent",
                "UNCLEAR_OR_INSUFFICIENT": "unsupported_or_uncertain",
                "UNSUPPORTED": "unsupported_or_uncertain",
            },
            "primary_single_label_evaluable_count": len(primary_records),
            "protected_write_supported_counts": protected_write_counts,
            "scoring_contract_version": records[0]["scoring_contract_version"],
            "total_holdout_count": len(records),
        },
        "multi_intent_secondary": multi_intent_metrics(
            multi_records, multi_predictions
        ),
        "primary_classifier": "frozen V2-C1 LinearSVC intent model",
        "primary_svm": {
            **primary_single_label_metrics(
                primary_records, primary_predictions
            ),
            "decision_margin": decision_margin_summary(
                intent_model,
                [row["text"] for row in primary_records],
            ),
            "protected_write_behavior_all_1800": protected_write_metrics(
                records, all_predictions
            ),
        },
        "report_content_boundary": {
            "contains_classifier_aggregate_metrics": True,
            "contains_consumer_narrative_text": False,
            "contains_per_record_predictions": False,
        },
        "schema_version": SCHEMA_VERSION,
        "scored_source": {
            "extra_review_pool_records_not_scored": (
                inputs["source_pool_count"] - len(records)
            ),
            "final_label_path": str(LABEL_PATH.relative_to(REPOSITORY_ROOT)),
            "final_label_schema_version": records[0]["schema_version"],
            "multi_intent_examples": len(multi_records),
            "primary_single_label_examples": len(primary_records),
            "selected_holdout_examples": len(records),
            "source_pool_examples": inputs["source_pool_count"],
            "source_pool_path": str(SOURCE_POOL_PATH.relative_to(REPOSITORY_ROOT)),
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
    primary = report["primary_svm"]
    multi = report["multi_intent_secondary"]
    print(
        "Evaluated frozen V2-C1 model on CFPB: "
        f"primary_accuracy={primary['accuracy']:.6f}, "
        f"primary_macro_f1={primary['macro_f1']:.6f}, "
        "multi_membership_accuracy="
        f"{multi['prediction_is_supported_intent_accuracy']:.6f}"
    )


if __name__ == "__main__":
    main()
