"""Diagnose frozen V2-C3 challenge errors for the V2-C4 recovery phase.

This workflow performs inference only with the already-frozen V2-C3 model. It
does not train, tune, select a model, alter thresholds, or read V2-C4 final
holdout data.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import statistics
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

try:
    from scripts import run_v2c3_final_evaluation as final_evaluation
except ModuleNotFoundError:  # Direct execution from the scripts directory.
    import run_v2c3_final_evaluation as final_evaluation


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ML_ROOT = REPOSITORY_ROOT / "data/evals/v2/ml"
CONTRACT_PATH = ML_ROOT / "v2c4_experiment_contract.json"
ERROR_CONFIG_PATH = ML_ROOT / "v2c4_error_analysis_config.json"
FINAL_CONFIG_PATH = ML_ROOT / "v2c3_final_evaluation_config.json"
FINAL_REPORT_PATH = ML_ROOT / "v2c3_final_evaluation_report.json"
DEFAULT_OUTPUT_PATH = ML_ROOT / "v2c4_error_analysis_report.json"

SCHEMA_VERSION = "v2c4-v2c3-challenge-error-analysis.v1"
ANALYSIS_ROLE = "consumed_v2c3_diagnostic"
CONSUMED_CHALLENGE_ROLE = "consumed_v2c3_safety_challenge"
SEALED_V2C4_ROLE = "v2c4_final_safety_holdout"
EXPECTED_ARTIFACT_SHA256 = (
    "0e03a4d8367933622a92920f14a7a4c5a2ffa74af0f53b7b0913f04849e4780c"
)
EXPECTED_CHALLENGE_SHA256 = (
    "4493b9baa8363408c917fe29de069dec82f925c8c3a605878a08eba06b051e5c"
)
EXPECTED_CHALLENGE_MANIFEST_SHA256 = (
    "51389b7e693156433edd6e7f0c2fe5b2c3a1ab0f34a261b47de7a7cda6e48b38"
)
EXPECTED_EXAMPLE_COUNT = 270
EXPECTED_EXAMPLES_PER_INTENT = 30
EXPECTED_INTENTS = tuple(final_evaluation.EXPECTED_INTENTS)
PROTECTED_WRITE_INTENTS = frozenset({"create_dispute", "freeze_card"})
UNSUPPORTED_INTENT = "unsupported_or_uncertain"
FORBIDDEN_INPUT_FILENAMES = frozenset(
    {
        "v2c4_safety_holdout.json",
        "v2c4_safety_holdout_seed.json",
    }
)

FROZEN_CATEGORY_IDS = frozenset(
    {
        "unsupported_to_supported",
        "supported_to_unsupported",
        "protected_to_wrong_protected_intent",
        "protected_to_non_protected",
        "non_protected_to_protected",
        "freeze_card_create_dispute_confusion",
        "informational_mention_vs_action_request",
        "ambiguity_or_insufficient_context",
        "lexical_trigger_over_reliance",
    }
)

MECHANICAL_BUCKET_DEFINITIONS = {
    "freeze_dispute_cross_confusion": (
        "Gold and prediction are the two different protected-write intents."
    ),
    "nonprotected_to_protected": (
        "Gold is non-protected and prediction is a protected-write intent."
    ),
    "protected_to_nonprotected": (
        "Gold is protected-write and prediction is non-protected."
    ),
    "protected_to_wrong_protected": (
        "Gold is protected-write and prediction is the other protected intent."
    ),
    "supported_to_unsupported": (
        "Gold is supported and prediction is unsupported_or_uncertain."
    ),
    "unsupported_to_supported": (
        "Gold is unsupported_or_uncertain and prediction is supported."
    ),
}

MECHANICAL_TO_FROZEN_CATEGORY = {
    "freeze_dispute_cross_confusion": "freeze_card_create_dispute_confusion",
    "nonprotected_to_protected": "non_protected_to_protected",
    "protected_to_nonprotected": "protected_to_non_protected",
    "protected_to_wrong_protected": "protected_to_wrong_protected_intent",
    "supported_to_unsupported": "supported_to_unsupported",
    "unsupported_to_supported": "unsupported_to_supported",
}

SEMANTIC_CATEGORY_TAG_EVIDENCE = {
    "ambiguity_or_insufficient_context": frozenset(
        {"fragment", "unclear_no_current_request", "vague_complaint"}
    ),
    "informational_mention_vs_action_request": frozenset(
        {"protected_action_information"}
    ),
}


def sha256_file(path: Path) -> str:
    return final_evaluation.tournament.sha256_file(path)


def stable_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def repository_path(relative_path: str) -> Path:
    return final_evaluation.repository_path(relative_path)


def guard_input_path(path: Path) -> None:
    if path.name.lower() in FORBIDDEN_INPUT_FILENAMES:
        raise ValueError(f"sealed V2-C4 final holdout input is forbidden: {path}")


def guard_data_role(payload: Mapping[str, Any], *, label: str) -> None:
    if payload.get("data_role") == SEALED_V2C4_ROLE:
        raise ValueError(f"sealed V2-C4 final data role is forbidden: {label}")


def load_json_object(path: Path, *, dataset: bool = False) -> dict[str, Any]:
    guard_input_path(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    if dataset:
        guard_data_role(value, label=str(path))
    return value


def validate_hash(path: Path, expected: str, label: str) -> str:
    guard_input_path(path)
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: expected {expected}, got {actual}")
    return actual


def load_frozen_category_ids(config: Mapping[str, Any]) -> tuple[str, ...]:
    if config.get("schema_version") != "v2c4-error-analysis-config.v1":
        raise ValueError("unexpected V2-C4 error-analysis config schema")
    categories = config.get("categories")
    if not isinstance(categories, list):
        raise TypeError("frozen error-analysis categories must be a list")
    category_ids = tuple(row.get("id") for row in categories)
    if len(category_ids) != len(set(category_ids)):
        raise ValueError("frozen error-analysis categories contain duplicates")
    if set(category_ids) != FROZEN_CATEGORY_IDS:
        raise ValueError("frozen V2-C4 error-analysis categories changed")
    return category_ids


def validate_governance(
    contract: Mapping[str, Any], error_config: Mapping[str, Any]
) -> tuple[tuple[str, ...], dict[str, str], tuple[str, ...]]:
    if contract.get("schema_version") != "v2c4-experiment-contract.v1":
        raise ValueError("unexpected V2-C4 contract schema")
    challenge_policy = contract.get("consumed_data_policy", {}).get(
        "v2c3_challenge_set", {}
    )
    if challenge_policy.get("diagnostic_role") != CONSUMED_CHALLENGE_ROLE:
        raise ValueError("V2-C3 challenge is not frozen as consumed diagnostic data")
    for field in (
        "model_selection_eligible",
        "threshold_selection_eligible",
        "v2c4_final_acceptance_eligible",
        "may_be_described_as_untouched",
    ):
        if challenge_policy.get(field) is not False:
            raise ValueError(f"invalid consumed challenge policy: {field}")
    if challenge_policy.get("example_count") != EXPECTED_EXAMPLE_COUNT:
        raise ValueError("consumed challenge count changed")

    intents = tuple(contract.get("intent_taxonomy", ()))
    risk_by_intent = contract.get("risk_by_intent")
    if intents != EXPECTED_INTENTS:
        raise ValueError("V2-C4 intent taxonomy differs from frozen V2-C3 order")
    if not isinstance(risk_by_intent, dict) or set(risk_by_intent) != set(intents):
        raise ValueError("V2-C4 risk mapping is incomplete")

    category_ids = load_frozen_category_ids(error_config)
    if error_config.get("frozen_before_error_analysis") is not True:
        raise ValueError("error-analysis categories were not frozen first")
    if error_config.get("error_analysis_performed") is not False:
        raise ValueError("frozen config already marks error analysis performed")
    if error_config.get("output_policy", {}).get("raw_text_persisted") is not False:
        raise ValueError("frozen error-analysis config must prohibit raw text")
    if error_config.get("source_roles", {}).get("v2c3_challenge") != (
        CONSUMED_CHALLENGE_ROLE
    ):
        raise ValueError("error-analysis config has the wrong challenge role")
    return intents, dict(risk_by_intent), category_ids


def extract_record_level_predictions(
    final_report: Mapping[str, Any], example_ids: Sequence[str]
) -> list[str] | None:
    final_results = final_report.get("challenge_set_results", {}).get(
        "final_v2c3", {}
    )
    candidates = (
        final_report.get("challenge_record_predictions"),
        final_results.get("record_predictions")
        if isinstance(final_results, Mapping)
        else None,
    )
    rows = next((value for value in candidates if isinstance(value, list)), None)
    if rows is None:
        return None
    by_id: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("record-level prediction must be an object")
        example_id = row.get("example_id")
        prediction = row.get("predicted_intent", row.get("prediction"))
        if not isinstance(example_id, str) or not isinstance(prediction, str):
            raise TypeError("record-level prediction lacks ID or predicted intent")
        if example_id in by_id:
            raise ValueError(f"duplicate frozen prediction: {example_id}")
        by_id[example_id] = prediction
    if set(by_id) != set(example_ids):
        raise ValueError("frozen record predictions do not match challenge IDs")
    return [by_id[example_id] for example_id in example_ids]


def mechanical_buckets(gold: str, predicted: str) -> list[str]:
    buckets: list[str] = []
    gold_protected = gold in PROTECTED_WRITE_INTENTS
    predicted_protected = predicted in PROTECTED_WRITE_INTENTS
    if gold == UNSUPPORTED_INTENT and predicted != UNSUPPORTED_INTENT:
        buckets.append("unsupported_to_supported")
    if gold != UNSUPPORTED_INTENT and predicted == UNSUPPORTED_INTENT:
        buckets.append("supported_to_unsupported")
    if gold_protected and not predicted_protected:
        buckets.append("protected_to_nonprotected")
    if gold_protected and predicted_protected and gold != predicted:
        buckets.append("protected_to_wrong_protected")
        buckets.append("freeze_dispute_cross_confusion")
    if not gold_protected and predicted_protected:
        buckets.append("nonprotected_to_protected")
    return buckets


def semantic_review_metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    tags = set(row.get("tags", ()))
    candidates = []
    for category_id, evidence_tags in SEMANTIC_CATEGORY_TAG_EVIDENCE.items():
        matched = sorted(tags & evidence_tags)
        if matched:
            candidates.append(
                {
                    "category_id": category_id,
                    "evidence_tags": matched,
                    "status": "metadata_supported_review_candidate",
                }
            )
    return {
        "automatic_causal_claims": [],
        "metadata_supported_candidates": candidates,
        "semantic_review_required": True,
    }


def rank_class_scores(
    classifier_classes: Sequence[str], decision_scores: Sequence[float]
) -> list[dict[str, Any]]:
    if len(classifier_classes) != len(decision_scores):
        raise ValueError("classifier classes and decision scores differ in length")
    numeric_scores = [float(score) for score in decision_scores]
    if not all(math.isfinite(score) for score in numeric_scores):
        raise ValueError("decision scores must all be finite")
    ranked_indices = sorted(
        range(len(classifier_classes)),
        key=lambda index: (-numeric_scores[index], index),
    )
    return [
        {
            "intent": str(classifier_classes[index]),
            "decision_score": numeric_scores[index],
            "rank": rank,
        }
        for rank, index in enumerate(ranked_indices, start=1)
    ]


def numeric_summary(values: Sequence[float]) -> dict[str, Any]:
    numeric = [float(value) for value in values]
    if not numeric:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
        }
    return {
        "count": len(numeric),
        "mean": statistics.fmean(numeric),
        "median": statistics.median(numeric),
        "min": min(numeric),
        "max": max(numeric),
    }


def _rank_distribution(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter(str(row["gold_class_rank"]) for row in rows)
    return dict(sorted(counts.items(), key=lambda item: int(item[0])))


def analyze_decision_boundaries(
    examples: Sequence[Mapping[str, Any]],
    predictions: Sequence[str],
    decision_scores: Sequence[Sequence[float]],
    classifier_classes: Sequence[str],
    baseline_analysis: Mapping[str, Any],
) -> dict[str, Any]:
    if not (
        len(examples) == len(predictions) == len(decision_scores)
    ):
        raise ValueError("examples, predictions, and decision scores differ in length")
    classes = tuple(str(intent) for intent in classifier_classes)
    if (
        len(classes) != len(EXPECTED_INTENTS)
        or len(set(classes)) != len(classes)
        or set(classes) != set(EXPECTED_INTENTS)
    ):
        raise ValueError("classifier.classes_ must contain each frozen intent once")
    class_indices = {intent: index for index, intent in enumerate(classes)}

    per_error: list[dict[str, Any]] = []
    protected_examples: list[dict[str, Any]] = []
    tag_counts: dict[str, Counter[str]] = {}
    all_score_rows: list[dict[str, Any]] = []

    for row, predicted, raw_scores in zip(
        examples, predictions, decision_scores, strict=True
    ):
        if len(raw_scores) != len(classes):
            raise ValueError(
                "LinearSVC decision_function must return one score per class"
            )
        ranked = rank_class_scores(classes, raw_scores)
        if ranked[0]["intent"] != predicted:
            raise ValueError(
                f"decision-score argmax differs from model prediction: "
                f"{row.get('example_id')}"
            )
        score_by_intent = {
            intent: float(raw_scores[index])
            for intent, index in class_indices.items()
        }
        gold = str(row["intent"])
        gold_rank = next(
            item["rank"] for item in ranked if item["intent"] == gold
        )
        gold_score = score_by_intent[gold]
        predicted_score = score_by_intent[predicted]
        decision_margin = predicted_score - gold_score
        best_protected = min(
            PROTECTED_WRITE_INTENTS,
            key=lambda intent: (-score_by_intent[intent], class_indices[intent]),
        )
        nonprotected_intents = [
            intent for intent in classes if intent not in PROTECTED_WRITE_INTENTS
        ]
        best_nonprotected = min(
            nonprotected_intents,
            key=lambda intent: (-score_by_intent[intent], class_indices[intent]),
        )
        max_protected_score = score_by_intent[best_protected]
        max_nonprotected_score = score_by_intent[best_nonprotected]
        protected_gap = max_protected_score - max_nonprotected_score
        detail = {
            "example_id": row["example_id"],
            "gold_intent": gold,
            "predicted_intent": predicted,
            "gold_class_decision_score": gold_score,
            "predicted_class_decision_score": predicted_score,
            "decision_margin": decision_margin,
            "gold_class_rank": gold_rank,
            "top_3_predictions": ranked[:3],
            "best_protected_intent": best_protected,
            "max_protected_score": max_protected_score,
            "best_nonprotected_intent": best_nonprotected,
            "max_nonprotected_score": max_nonprotected_score,
            "protected_vs_nonprotected_gap": protected_gap,
        }
        for field in ("text_sha256", "normalized_text_sha256"):
            if field in row:
                detail[field] = row[field]
        all_score_rows.append(detail)

        is_correct = gold == predicted
        is_protected = gold in PROTECTED_WRITE_INTENTS
        is_protected_miss = is_protected and not is_correct
        is_protected_false_positive = (
            not is_protected and predicted in PROTECTED_WRITE_INTENTS
        )
        is_unsupported = gold == UNSUPPORTED_INTENT
        is_unsupported_false_supported = is_unsupported and not is_correct
        for tag in sorted(set(row.get("tags", ()))):
            if not isinstance(tag, str) or not tag:
                raise ValueError(f"invalid challenge tag: {row.get('example_id')}")
            counts = tag_counts.setdefault(tag, Counter())
            counts["example_count"] += 1
            counts["correct_count" if is_correct else "error_count"] += 1
            counts["protected_example_count"] += int(is_protected)
            counts["protected_miss_count"] += int(is_protected_miss)
            counts["protected_false_positive_count"] += int(
                is_protected_false_positive
            )
            counts["unsupported_example_count"] += int(is_unsupported)
            counts["unsupported_false_supported_count"] += int(
                is_unsupported_false_supported
            )

        if not is_correct:
            per_error.append(
                {
                    key: detail[key]
                    for key in (
                        "example_id",
                        "gold_intent",
                        "predicted_intent",
                        "gold_class_decision_score",
                        "predicted_class_decision_score",
                        "decision_margin",
                        "gold_class_rank",
                        "top_3_predictions",
                        "text_sha256",
                        "normalized_text_sha256",
                    )
                    if key in detail
                }
            )
        if is_protected:
            protected_examples.append(
                {
                    key: detail[key]
                    for key in (
                        "example_id",
                        "gold_intent",
                        "predicted_intent",
                        "best_protected_intent",
                        "max_protected_score",
                        "best_nonprotected_intent",
                        "max_nonprotected_score",
                        "protected_vs_nonprotected_gap",
                        "text_sha256",
                        "normalized_text_sha256",
                    )
                    if key in detail
                }
            )

    baseline_error_predictions = {
        row["example_id"]: row["predicted_intent"]
        for row in baseline_analysis["error_records"]
    }
    decision_error_predictions = {
        row["example_id"]: row["predicted_intent"] for row in per_error
    }
    if decision_error_predictions != baseline_error_predictions:
        raise ValueError(
            "decision-boundary predictions differ from reproduced predictions"
        )
    if len(protected_examples) != baseline_analysis["protected_safety"][
        "protected_total"
    ]:
        raise ValueError("decision-boundary protected count changed")

    tag_analysis: dict[str, dict[str, Any]] = {}
    count_fields = (
        "example_count",
        "correct_count",
        "error_count",
        "protected_example_count",
        "protected_miss_count",
        "protected_false_positive_count",
        "unsupported_example_count",
        "unsupported_false_supported_count",
    )
    for tag in sorted(tag_counts):
        counts = tag_counts[tag]
        example_count = counts["example_count"]
        tag_analysis[tag] = {
            field: counts[field] for field in count_fields
        }
        tag_analysis[tag]["error_rate"] = counts["error_count"] / example_count

    protected_misses = [
        row
        for row in all_score_rows
        if row["gold_intent"] in PROTECTED_WRITE_INTENTS
        and row["gold_intent"] != row["predicted_intent"]
    ]
    protected_false_positives = [
        row
        for row in all_score_rows
        if row["gold_intent"] not in PROTECTED_WRITE_INTENTS
        and row["predicted_intent"] in PROTECTED_WRITE_INTENTS
    ]
    unsupported_false_supported = [
        row
        for row in all_score_rows
        if row["gold_intent"] == UNSUPPORTED_INTENT
        and row["predicted_intent"] != UNSUPPORTED_INTENT
    ]
    expected_reason_counts = baseline_analysis["review_queue_summary"][
        "reason_counts"
    ]
    actual_reason_counts = {
        "protected_miss": len(protected_misses),
        "protected_false_positive": len(protected_false_positives),
        "unsupported_false_supported": len(unsupported_false_supported),
    }
    if actual_reason_counts != expected_reason_counts:
        raise ValueError(
            f"decision-boundary safety counts changed: {actual_reason_counts}"
        )

    summary = {
        "protected_misses": {
            "count": len(protected_misses),
            "gold_class_rank_distribution": _rank_distribution(protected_misses),
            "decision_margin": numeric_summary(
                [row["decision_margin"] for row in protected_misses]
            ),
            "protected_vs_nonprotected_gap": numeric_summary(
                [
                    row["protected_vs_nonprotected_gap"]
                    for row in protected_misses
                ]
            ),
        },
        "protected_false_positives": {
            "count": len(protected_false_positives),
            "predicted_protected_score": numeric_summary(
                [
                    row["predicted_class_decision_score"]
                    for row in protected_false_positives
                ]
            ),
            "best_nonprotected_score": numeric_summary(
                [row["max_nonprotected_score"] for row in protected_false_positives]
            ),
            "protected_vs_nonprotected_gap": numeric_summary(
                [
                    row["predicted_class_decision_score"]
                    - row["max_nonprotected_score"]
                    for row in protected_false_positives
                ]
            ),
        },
        "unsupported_false_supported": {
            "count": len(unsupported_false_supported),
            "gold_unsupported_score": numeric_summary(
                [
                    row["gold_class_decision_score"]
                    for row in unsupported_false_supported
                ]
            ),
            "predicted_supported_score": numeric_summary(
                [
                    row["predicted_class_decision_score"]
                    for row in unsupported_false_supported
                ]
            ),
            "gold_class_rank_distribution": _rank_distribution(
                unsupported_false_supported
            ),
            "decision_margin": numeric_summary(
                [row["decision_margin"] for row in unsupported_false_supported]
            ),
        },
    }
    return {
        "score_definitions": {
            "decision_margin": (
                "predicted_class_decision_score - gold_class_decision_score"
            ),
            "protected_vs_nonprotected_gap": (
                "max_protected_score - max_nonprotected_score"
            ),
            "ranking": (
                "Descending LinearSVC decision_function score; ties retain "
                "classifier.classes_ order."
            ),
        },
        "classifier_class_order": list(classes),
        "per_error": per_error,
        "protected_analysis": {
            "example_count": len(protected_examples),
            "per_example": protected_examples,
            "gap_summary": numeric_summary(
                [row["protected_vs_nonprotected_gap"] for row in protected_examples]
            ),
        },
        "tag_analysis": tag_analysis,
        "summary": summary,
        "integrity": {
            "class_count": len(classes),
            "decision_score_count_per_example": len(classes),
            "prediction_argmax_matches": True,
            "persisted_predictions_match_reproduced_predictions": True,
            "frozen_error_count": len(per_error),
            "frozen_safety_counts": actual_reason_counts,
        },
    }


def _error_record(
    row: Mapping[str, Any],
    predicted: str,
    risk_by_intent: Mapping[str, str],
    frozen_category_ids: set[str],
) -> dict[str, Any]:
    buckets = mechanical_buckets(str(row["intent"]), predicted)
    semantic_review = semantic_review_metadata(row)
    frozen_matches = sorted(
        {
            MECHANICAL_TO_FROZEN_CATEGORY[bucket]
            for bucket in buckets
            if MECHANICAL_TO_FROZEN_CATEGORY[bucket] in frozen_category_ids
        }
    )
    result: dict[str, Any] = {
        "example_id": row["example_id"],
        "gold_intent": row["intent"],
        "predicted_intent": predicted,
        "gold_risk": row["risk"],
        "predicted_risk": risk_by_intent[predicted],
        "mechanical_buckets": buckets,
        "frozen_category_matches": frozen_matches,
        "automatic_causal_claims": semantic_review["automatic_causal_claims"],
        "semantic_review_candidates": semantic_review[
            "metadata_supported_candidates"
        ],
        "semantic_review_required": semantic_review["semantic_review_required"],
        "tags": list(row.get("tags", ())),
    }
    if "group_id" in row:
        result["group_id"] = row["group_id"]
    if "safety_pattern" in row:
        result["safety_pattern"] = row["safety_pattern"]
    for field in ("text_sha256", "normalized_text_sha256"):
        if field in row:
            result[field] = row[field]
    return result


def analyze_predictions(
    examples: Sequence[Mapping[str, Any]],
    predictions: Sequence[str],
    intents: Sequence[str],
    risk_by_intent: Mapping[str, str],
    frozen_category_ids: Sequence[str],
) -> dict[str, Any]:
    if len(examples) != len(predictions):
        raise ValueError("challenge examples and predictions differ in length")
    if len({row.get("example_id") for row in examples}) != len(examples):
        raise ValueError("challenge example IDs must be unique")
    allowed_intents = set(intents)
    frozen_categories = set(frozen_category_ids)
    confusion = {
        gold: {predicted: 0 for predicted in intents} for gold in intents
    }
    error_records: list[dict[str, Any]] = []
    review_queue: list[dict[str, Any]] = []
    correct_count = 0

    for row, predicted in zip(examples, predictions, strict=True):
        guard_data_role(row, label=str(row.get("example_id", "challenge row")))
        gold = row.get("intent")
        if gold not in allowed_intents or predicted not in allowed_intents:
            raise ValueError("gold or predicted intent is outside frozen taxonomy")
        if row.get("risk") != risk_by_intent[gold]:
            raise ValueError(f"gold risk mismatch: {row.get('example_id')}")
        confusion[gold][predicted] += 1
        if gold == predicted:
            correct_count += 1
            continue

        error = _error_record(
            row,
            predicted,
            risk_by_intent,
            frozen_categories,
        )
        error_records.append(error)
        review_reasons: list[str] = []
        if gold in PROTECTED_WRITE_INTENTS:
            review_reasons.append("protected_miss")
        if gold not in PROTECTED_WRITE_INTENTS and (
            predicted in PROTECTED_WRITE_INTENTS
        ):
            review_reasons.append("protected_false_positive")
        if gold == UNSUPPORTED_INTENT:
            review_reasons.append("unsupported_false_supported")
        if review_reasons:
            queue_row: dict[str, Any] = {
                "example_id": row["example_id"],
                "gold_intent": gold,
                "predicted_intent": predicted,
                "gold_risk": row["risk"],
                "predicted_risk": risk_by_intent[predicted],
                "mechanical_buckets": error["mechanical_buckets"],
                "review_reasons": review_reasons,
                "semantic_review_required": True,
                "tags": list(row.get("tags", ())),
            }
            if "group_id" in row:
                queue_row["group_id"] = row["group_id"]
            if "safety_pattern" in row:
                queue_row["safety_pattern"] = row["safety_pattern"]
            for field in ("text_sha256", "normalized_text_sha256"):
                if field in row:
                    queue_row[field] = row[field]
            review_queue.append(queue_row)

    example_count = len(examples)
    error_count = example_count - correct_count
    per_intent: dict[str, Any] = {}
    for gold in intents:
        row_counts = confusion[gold]
        support = sum(row_counts.values())
        correct = row_counts[gold]
        wrong = Counter(
            {
                predicted: count
                for predicted, count in row_counts.items()
                if predicted != gold and count
            }
        )
        if wrong:
            highest = max(wrong.values())
            winner = next(
                intent
                for intent in intents
                if intent in wrong and wrong[intent] == highest
            )
            most_common_wrong: dict[str, Any] | None = {
                "intent": winner,
                "count": highest,
            }
        else:
            most_common_wrong = None
        per_intent[gold] = {
            "support": support,
            "correct": correct,
            "incorrect": support - correct,
            "recall": correct / support if support else 0.0,
            "most_common_wrong_prediction": most_common_wrong,
        }

    protected_rows = [
        (row, predicted)
        for row, predicted in zip(examples, predictions, strict=True)
        if row["intent"] in PROTECTED_WRITE_INTENTS
    ]
    nonprotected_rows = [
        (row, predicted)
        for row, predicted in zip(examples, predictions, strict=True)
        if row["intent"] not in PROTECTED_WRITE_INTENTS
    ]
    protected_exact = sum(
        row["intent"] == predicted for row, predicted in protected_rows
    )
    protected_recognized = sum(
        predicted in PROTECTED_WRITE_INTENTS for _, predicted in protected_rows
    )
    protected_wrong = sum(
        predicted in PROTECTED_WRITE_INTENTS and row["intent"] != predicted
        for row, predicted in protected_rows
    )
    protected_to_nonprotected = sum(
        predicted not in PROTECTED_WRITE_INTENTS for _, predicted in protected_rows
    )
    nonprotected_to_freeze = sum(
        predicted == "freeze_card" for _, predicted in nonprotected_rows
    )
    nonprotected_to_dispute = sum(
        predicted == "create_dispute" for _, predicted in nonprotected_rows
    )
    false_positive_count = nonprotected_to_freeze + nonprotected_to_dispute

    unsupported_rows = [
        predicted
        for row, predicted in zip(examples, predictions, strict=True)
        if row["intent"] == UNSUPPORTED_INTENT
    ]
    unsupported_distribution = Counter(unsupported_rows)
    unsupported_correct = unsupported_distribution[UNSUPPORTED_INTENT]

    confusion_values = [
        [confusion[gold][predicted] for predicted in intents] for gold in intents
    ]
    review_reason_counts = Counter(
        reason for row in review_queue for reason in row["review_reasons"]
    )
    if len({row["example_id"] for row in review_queue}) != len(review_queue):
        raise ValueError("safety-critical review queue contains duplicate examples")

    return {
        "overall": {
            "example_count": example_count,
            "correct_count": correct_count,
            "error_count": error_count,
            "accuracy": correct_count / example_count if example_count else 0.0,
        },
        "confusion_matrix": {
            "label_order": list(intents),
            "values": confusion_values,
            "gold_to_predicted": confusion,
        },
        "per_intent": per_intent,
        "protected_safety": {
            "protected_total": len(protected_rows),
            "protected_exact_intent_correct": protected_exact,
            "protected_correctly_recognized_as_protected": protected_recognized,
            "protected_miss_count": len(protected_rows) - protected_exact,
            "protected_to_nonprotected_count": protected_to_nonprotected,
            "protected_to_wrong_protected_intent_count": protected_wrong,
            "protected_write_recall": (
                protected_exact / len(protected_rows) if protected_rows else 0.0
            ),
            "nonprotected_total": len(nonprotected_rows),
            "nonprotected_to_freeze_card": nonprotected_to_freeze,
            "nonprotected_to_create_dispute": nonprotected_to_dispute,
            "protected_false_positive_count": false_positive_count,
            "protected_write_false_positive_rate": (
                false_positive_count / len(nonprotected_rows)
                if nonprotected_rows
                else 0.0
            ),
        },
        "unsupported_analysis": {
            "total": len(unsupported_rows),
            "correctly_unsupported": unsupported_correct,
            "predicted_supported": len(unsupported_rows) - unsupported_correct,
            "predicted_intent_distribution": {
                intent: unsupported_distribution[intent] for intent in intents
            },
            "recall": (
                unsupported_correct / len(unsupported_rows)
                if unsupported_rows
                else 0.0
            ),
        },
        "error_records": error_records,
        "safety_critical_review_queue": review_queue,
        "review_queue_summary": {
            "unique_example_count": len(review_queue),
            "reason_counts": dict(sorted(review_reason_counts.items())),
        },
    }


def validate_frozen_baseline(
    analysis: Mapping[str, Any], final_report: Mapping[str, Any]
) -> None:
    frozen = final_report["challenge_set_results"]["final_v2c3"]
    overall = analysis["overall"]
    protected = analysis["protected_safety"]
    unsupported = analysis["unsupported_analysis"]
    expected_exact = {
        "example_count": EXPECTED_EXAMPLE_COUNT,
        "protected_total": 60,
        "protected_exact_intent_correct": 45,
        "protected_miss_count": 15,
        "nonprotected_total": 210,
        "protected_false_positive_count": 4,
        "unsupported_total": 30,
        "unsupported_correct": 21,
        "unsupported_false_supported": 9,
    }
    actual_exact = {
        "example_count": overall["example_count"],
        "protected_total": protected["protected_total"],
        "protected_exact_intent_correct": protected[
            "protected_exact_intent_correct"
        ],
        "protected_miss_count": protected["protected_miss_count"],
        "nonprotected_total": protected["nonprotected_total"],
        "protected_false_positive_count": protected[
            "protected_false_positive_count"
        ],
        "unsupported_total": unsupported["total"],
        "unsupported_correct": unsupported["correctly_unsupported"],
        "unsupported_false_supported": unsupported["predicted_supported"],
    }
    if actual_exact != expected_exact:
        raise ValueError(
            f"reproduced V2-C3 safety counts changed: {actual_exact}"
        )
    if analysis["confusion_matrix"]["label_order"] != frozen[
        "confusion_matrix"
    ]["label_order"] or analysis["confusion_matrix"]["values"] != frozen[
        "confusion_matrix"
    ]["values"]:
        raise ValueError("reproduced challenge confusion matrix changed")
    expected_metrics = {
        "accuracy": frozen["accuracy"],
        "protected_write_recall": frozen["protected_write_recall"],
        "protected_write_false_positive_rate": frozen[
            "protected_write_false_positive_rate"
        ],
        "unsupported_or_uncertain_recall": frozen[
            "unsupported_or_uncertain_recall"
        ],
    }
    actual_metrics = {
        "accuracy": overall["accuracy"],
        "protected_write_recall": protected["protected_write_recall"],
        "protected_write_false_positive_rate": protected[
            "protected_write_false_positive_rate"
        ],
        "unsupported_or_uncertain_recall": unsupported["recall"],
    }
    if actual_metrics != expected_metrics:
        raise ValueError(
            f"reproduced V2-C3 diagnostic metrics changed: {actual_metrics}"
        )


def build_report_payload(
    *,
    analysis: Mapping[str, Any],
    decision_boundary_analysis: Mapping[str, Any],
    prediction_source: str,
    frozen_category_ids: Sequence[str],
    input_hashes: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis_role": ANALYSIS_ROLE,
        "source_data_role": CONSUMED_CHALLENGE_ROLE,
        "final_holdout": False,
        "model_selection_evidence": False,
        "v2c4_final_acceptance_evidence": False,
        "prediction_source": prediction_source,
        "selected_v2c3_variant": "bge_svc_c_4_0",
        "representation": {
            "model_identifier": "BAAI/bge-small-en-v1.5",
            "method": "FastEmbed passage_embed",
            "dimensions": 384,
        },
        "input_hashes": dict(input_hashes),
        "frozen_error_category_ids": list(frozen_category_ids),
        "mechanical_bucket_definitions": MECHANICAL_BUCKET_DEFINITIONS,
        "semantic_review_policy": {
            "automatic_semantic_causality_allowed": False,
            "metadata_may_identify_review_candidates": True,
            "human_review_required_before_semantic_category_assignment": True,
        },
        "raw_text_persisted": False,
        "temporary_human_review": {
            "mode": "--print-review-queue",
            "report_modified": False,
            "stdout_only": True,
        },
        "diagnostic_regression_metrics": {
            "overall": analysis["overall"],
            "confusion_matrix": analysis["confusion_matrix"],
            "per_intent": analysis["per_intent"],
            "protected_safety": analysis["protected_safety"],
            "unsupported_analysis": analysis["unsupported_analysis"],
        },
        "decision_boundary_analysis": dict(decision_boundary_analysis),
        "error_records": analysis["error_records"],
        "safety_critical_review_queue": analysis[
            "safety_critical_review_queue"
        ],
        "review_queue_summary": analysis["review_queue_summary"],
        "execution_policy": {
            "diagnostic_inference_performed": prediction_source
            in {
                "committed_v2c3_record_predictions",
                "frozen_v2c3_model_diagnostic_reproduction",
            },
            "model_fitting_performed": False,
            "model_training_performed": False,
            "model_tuning_performed": False,
            "threshold_tuning_performed": False,
            "model_selection_performed": False,
            "runtime_integration_performed": False,
            "v2c4_final_holdout_accessed": False,
        },
    }


def validate_report_contains_no_raw_text(payload: Mapping[str, Any]) -> None:
    forbidden_fields = {"text", "raw_text", "utterance"}
    for section in ("error_records", "safety_critical_review_queue"):
        if not isinstance(payload.get(section), list):
            raise TypeError(f"diagnostic report {section} must be a list")

    def inspect(value: Any, location: str) -> None:
        if isinstance(value, Mapping):
            present = forbidden_fields & set(value)
            if present:
                raise ValueError(
                    "diagnostic report cannot persist raw-text fields at "
                    f"{location}: {sorted(present)}"
                )
            for key, nested in value.items():
                inspect(nested, f"{location}.{key}")
        elif isinstance(value, list):
            for index, nested in enumerate(value):
                inspect(nested, f"{location}[{index}]")

    inspect(payload, "report")
    if payload.get("raw_text_persisted") is not False:
        raise ValueError("diagnostic report must declare raw_text_persisted=false")


def write_report(path: Path, payload: Mapping[str, Any]) -> None:
    guard_input_path(path)
    validate_report_contains_no_raw_text(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(stable_json_bytes(payload))


def join_review_queue_with_challenge(
    report: Mapping[str, Any],
    challenge_examples: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    validate_report_contains_no_raw_text(report)
    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected diagnostic report schema")
    if report.get("analysis_role") != ANALYSIS_ROLE:
        raise ValueError("diagnostic report has the wrong analysis role")
    if report.get("final_holdout") is not False:
        raise ValueError("review mode cannot consume a final holdout report")

    challenge_by_id: dict[str, Mapping[str, Any]] = {}
    for row in challenge_examples:
        guard_data_role(row, label=str(row.get("example_id", "challenge row")))
        example_id = row.get("example_id")
        if not isinstance(example_id, str) or example_id in challenge_by_id:
            raise ValueError("consumed challenge example IDs must be unique strings")
        challenge_by_id[example_id] = row

    joined: list[dict[str, Any]] = []
    seen: set[str] = set()
    for queue_row in report["safety_critical_review_queue"]:
        example_id = queue_row.get("example_id")
        if not isinstance(example_id, str) or example_id in seen:
            raise ValueError("diagnostic review queue IDs must be unique strings")
        seen.add(example_id)
        source = challenge_by_id.get(example_id)
        if source is None:
            raise ValueError(f"review queue ID is absent from challenge: {example_id}")
        if source.get("intent") != queue_row.get("gold_intent"):
            raise ValueError(f"review queue gold intent changed: {example_id}")
        for field in ("text_sha256", "normalized_text_sha256"):
            if field in queue_row and queue_row[field] != source.get(field):
                raise ValueError(f"review queue {field} changed: {example_id}")

        display: dict[str, Any] = {
            "example_id": example_id,
            "text": source["text"],
            "gold_intent": queue_row["gold_intent"],
            "predicted_intent": queue_row["predicted_intent"],
            "mechanical_buckets": queue_row["mechanical_buckets"],
            "semantic_review_required": queue_row[
                "semantic_review_required"
            ],
            "tags": list(source.get("tags", ())),
        }
        if "group_id" in source:
            display["group_id"] = source["group_id"]
        if "safety_pattern" in source:
            display["safety_pattern"] = source["safety_pattern"]
        joined.append(display)
    return joined


def load_review_inputs(
    report_path: Path = DEFAULT_OUTPUT_PATH,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    guard_input_path(report_path)
    if not report_path.is_file():
        raise FileNotFoundError(
            f"diagnostic report is required before review mode: {report_path}"
        )
    report = load_json_object(report_path)
    validate_report_contains_no_raw_text(report)

    final_config = load_json_object(FINAL_CONFIG_PATH)
    final_evaluation.validate_config(final_config)
    challenge_spec = final_config["challenge_set"]
    challenge_path = repository_path(challenge_spec["path"])
    challenge_manifest_path = repository_path(challenge_spec["manifest_path"])
    validate_hash(challenge_path, EXPECTED_CHALLENGE_SHA256, "V2-C3 challenge")
    validate_hash(
        challenge_manifest_path,
        EXPECTED_CHALLENGE_MANIFEST_SHA256,
        "V2-C3 challenge manifest",
    )
    recorded_challenge = report.get("input_hashes", {}).get(
        "consumed_v2c3_challenge", {}
    )
    if recorded_challenge.get("sha256") != EXPECTED_CHALLENGE_SHA256:
        raise ValueError("diagnostic report references the wrong challenge hash")
    challenge = final_evaluation.load_challenge_examples(final_config)
    return report, challenge


def print_review_queue(
    *,
    report_path: Path = DEFAULT_OUTPUT_PATH,
    report_payload: Mapping[str, Any] | None = None,
    challenge_examples: Sequence[Mapping[str, Any]] | None = None,
    stream: io.TextIOBase | None = None,
) -> list[dict[str, Any]]:
    if (report_payload is None) != (challenge_examples is None):
        raise ValueError(
            "report_payload and challenge_examples must be supplied together"
        )
    if report_payload is None:
        loaded_report, loaded_challenge = load_review_inputs(report_path)
    else:
        loaded_report = dict(report_payload)
        loaded_challenge = list(challenge_examples or ())
    joined = join_review_queue_with_challenge(loaded_report, loaded_challenge)
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


def run_diagnostic(output_path: Path = DEFAULT_OUTPUT_PATH) -> dict[str, Any]:
    guard_input_path(output_path)
    contract = load_json_object(CONTRACT_PATH)
    error_config = load_json_object(ERROR_CONFIG_PATH)
    contract_spec = error_config.get("v2c4_contract", {})
    if contract_spec.get("path") != CONTRACT_PATH.relative_to(
        REPOSITORY_ROOT
    ).as_posix():
        raise ValueError("error-analysis config points to the wrong V2-C4 contract")
    validate_hash(
        CONTRACT_PATH,
        str(contract_spec.get("sha256")),
        "frozen V2-C4 contract",
    )
    intents, risk_by_intent, frozen_category_ids = validate_governance(
        contract, error_config
    )

    parent_hashes = contract["parent_v2c3_hashes"]
    validate_hash(
        FINAL_CONFIG_PATH,
        parent_hashes["final_evaluation_config"]["sha256"],
        "frozen V2-C3 final-evaluation config",
    )
    validate_hash(
        FINAL_REPORT_PATH,
        parent_hashes["final_evaluation_report"]["sha256"],
        "frozen V2-C3 final-evaluation report",
    )

    final_config = load_json_object(FINAL_CONFIG_PATH)
    final_report = load_json_object(FINAL_REPORT_PATH)
    final_evaluation.validate_config(final_config)
    if final_report.get("raw_text_persisted") is not False:
        raise ValueError("V2-C3 final report unexpectedly contains persisted text")

    challenge_spec = final_config["challenge_set"]
    challenge_path = repository_path(challenge_spec["path"])
    challenge_manifest_path = repository_path(challenge_spec["manifest_path"])
    guard_input_path(challenge_path)
    guard_input_path(challenge_manifest_path)
    validate_hash(challenge_path, EXPECTED_CHALLENGE_SHA256, "V2-C3 challenge")
    validate_hash(
        challenge_manifest_path,
        EXPECTED_CHALLENGE_MANIFEST_SHA256,
        "V2-C3 challenge manifest",
    )
    challenge = final_evaluation.load_challenge_examples(final_config)
    if len(challenge) != EXPECTED_EXAMPLE_COUNT:
        raise ValueError("V2-C3 challenge count changed")
    counts = Counter(row["intent"] for row in challenge)
    if set(counts.values()) != {EXPECTED_EXAMPLES_PER_INTENT}:
        raise ValueError("V2-C3 challenge is no longer balanced at 30 per intent")
    for row in challenge:
        guard_data_role(row, label=str(row["example_id"]))

    example_ids = [row["example_id"] for row in challenge]
    persisted_predictions = extract_record_level_predictions(
        final_report, example_ids
    )
    artifact_path = repository_path(
        final_config["selected_model"]["artifact"]["path"]
    )
    artifact_sha256 = validate_hash(
        artifact_path,
        EXPECTED_ARTIFACT_SHA256,
        "frozen V2-C3 classifier",
    )
    frozen_model = final_evaluation.validate_final_model_artifact(final_config)
    embeddings = final_evaluation.embed_final_texts(
        [row["text"] for row in challenge], final_config
    )
    classifier = frozen_model["classifier"]
    predictions = classifier.predict(embeddings).tolist()
    decision_scores = classifier.decision_function(embeddings)
    classifier_classes = [str(intent) for intent in classifier.classes_.tolist()]
    if persisted_predictions is None:
        prediction_source = "frozen_v2c3_model_diagnostic_reproduction"
    else:
        if predictions != persisted_predictions:
            raise ValueError(
                "frozen model predictions differ from persisted predictions"
            )
        prediction_source = "committed_v2c3_record_predictions"

    analysis = analyze_predictions(
        challenge,
        predictions,
        intents,
        risk_by_intent,
        frozen_category_ids,
    )
    validate_frozen_baseline(analysis, final_report)
    decision_boundary = analyze_decision_boundaries(
        challenge,
        predictions,
        decision_scores.tolist(),
        classifier_classes,
        analysis,
    )
    input_hashes = {
        "consumed_v2c3_challenge": {
            "path": challenge_spec["path"],
            "sha256": EXPECTED_CHALLENGE_SHA256,
        },
        "consumed_v2c3_challenge_manifest": {
            "path": challenge_spec["manifest_path"],
            "sha256": EXPECTED_CHALLENGE_MANIFEST_SHA256,
        },
        "frozen_v2c3_classifier": {
            "path": final_config["selected_model"]["artifact"]["path"],
            "sha256": artifact_sha256,
        },
        "frozen_v2c3_final_config": {
            "path": FINAL_CONFIG_PATH.relative_to(REPOSITORY_ROOT).as_posix(),
            "sha256": parent_hashes["final_evaluation_config"]["sha256"],
        },
        "frozen_v2c3_final_report": {
            "path": FINAL_REPORT_PATH.relative_to(REPOSITORY_ROOT).as_posix(),
            "sha256": parent_hashes["final_evaluation_report"]["sha256"],
        },
        "frozen_v2c4_error_analysis_config": {
            "path": ERROR_CONFIG_PATH.relative_to(REPOSITORY_ROOT).as_posix(),
            "sha256": sha256_file(ERROR_CONFIG_PATH),
        },
        "frozen_v2c4_experiment_contract": {
            "path": CONTRACT_PATH.relative_to(REPOSITORY_ROOT).as_posix(),
            "sha256": contract_spec["sha256"],
        },
    }
    report = build_report_payload(
        analysis=analysis,
        decision_boundary_analysis=decision_boundary,
        prediction_source=prediction_source,
        frozen_category_ids=frozen_category_ids,
        input_hashes=input_hashes,
    )
    write_report(output_path, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce and diagnose consumed V2-C3 challenge errors."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Diagnostic JSON output path or existing report path for review.",
    )
    parser.add_argument(
        "--print-review-queue",
        action="store_true",
        help=(
            "Print review records with source text to stdout without inference "
            "or writes."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.print_review_queue:
        print_review_queue(report_path=args.output)
        return
    report = run_diagnostic(args.output)
    overall = report["diagnostic_regression_metrics"]["overall"]
    safety = report["diagnostic_regression_metrics"]["protected_safety"]
    unsupported = report["diagnostic_regression_metrics"][
        "unsupported_analysis"
    ]
    print(
        "Wrote consumed V2-C3 diagnostic report: "
        f"examples={overall['example_count']}, errors={overall['error_count']}, "
        f"protected_misses={safety['protected_miss_count']}, "
        f"protected_false_positives={safety['protected_false_positive_count']}, "
        f"unsupported_false_supported={unsupported['predicted_supported']}"
    )


if __name__ == "__main__":
    main()
