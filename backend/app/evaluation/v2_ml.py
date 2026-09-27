"""Offline-only conventional ML helpers for the SentinelVoice V2-C experiment."""

from __future__ import annotations

import re
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from backend.app.evaluation.v2_contracts import IntentLabel, RiskLabel

INTENT_LABELS = tuple(sorted(label.value for label in IntentLabel))
RISK_LABELS = tuple(sorted(label.value for label in RiskLabel))
RISK_BY_INTENT = {
    IntentLabel.INFORMATIONAL_POLICY.value: RiskLabel.PUBLIC.value,
    IntentLabel.ACCOUNT_BALANCE.value: RiskLabel.PRIVATE_READ.value,
    IntentLabel.RECENT_TRANSACTIONS.value: RiskLabel.PRIVATE_READ.value,
    IntentLabel.TRANSACTION_DETAILS.value: RiskLabel.PRIVATE_READ.value,
    IntentLabel.CARD_STATUS.value: RiskLabel.PRIVATE_READ.value,
    IntentLabel.FREEZE_CARD.value: RiskLabel.PROTECTED_WRITE.value,
    IntentLabel.CREATE_DISPUTE.value: RiskLabel.PROTECTED_WRITE.value,
    IntentLabel.ESCALATION.value: RiskLabel.ESCALATION_OR_UNCERTAIN.value,
    IntentLabel.UNSUPPORTED_OR_UNCERTAIN.value: (
        RiskLabel.ESCALATION_OR_UNCERTAIN.value
    ),
}


@dataclass(frozen=True)
class MajorityBaseline:
    intent: str
    risk: str

    @classmethod
    def fit(cls, examples: Sequence[dict[str, Any]]) -> MajorityBaseline:
        return cls(
            intent=_deterministic_mode(example["intent"] for example in examples),
            risk=_deterministic_mode(example["risk"] for example in examples),
        )

    def predict_intent(self, texts: Sequence[str]) -> list[str]:
        return [self.intent] * len(texts)

    def predict_risk(self, texts: Sequence[str]) -> list[str]:
        return [self.risk] * len(texts)


class RuleBaseline:
    """Small transparent baseline; it intentionally is not a routing engine."""

    def predict_intent(self, texts: Sequence[str]) -> list[str]:
        return [self._predict_one(text) for text in texts]

    def predict_risk(self, texts: Sequence[str]) -> list[str]:
        return [RISK_BY_INTENT[intent] for intent in self.predict_intent(texts)]

    @staticmethod
    def _predict_one(text: str) -> str:
        normalized = re.sub(r"\s+", " ", text.lower()).strip()
        if _contains_any(
            normalized,
            "human",
            "representative",
            "live agent",
            "support team",
            "support person",
            "take over",
            "transfer me",
            "escalate",
        ):
            return IntentLabel.ESCALATION.value

        policy_signal = _contains_any(
            normalized,
            "policy",
            "rules",
            "requirements",
            "what happens",
            "what would happen",
            "how does the process",
            "how long does",
            "qualifies",
            "qualify for",
            "explain the difference",
            "before i act",
            "before i take any action",
        )
        if policy_signal and _contains_any(
            normalized,
            "freeze",
            "frozen",
            "dispute",
            "pending",
            "posted",
            "unauthorized",
            "replacement",
        ):
            return IntentLabel.INFORMATIONAL_POLICY.value

        status_only = _contains_any(
            normalized,
            "status",
            "currently frozen",
            "frozen already",
            "already frozen",
            "is frozen",
            "been locked",
            "still use",
            "still be used",
            "current state",
            "card state",
            "card works",
            "card work",
            "purchases are enabled",
        )
        no_card_action = _contains_any(
            normalized,
            "don't freeze",
            "dont freeze",
            "don't disable",
            "dont disable",
            "without locking",
            "without changing",
            "take no action",
            "only want to know",
            "only check",
        )
        if status_only or no_card_action:
            return IntentLabel.CARD_STATUS.value

        freeze_signal = _contains_any(
            normalized,
            "freeze",
            "lock the card",
            "lock card",
            "disable",
            "block the card",
            "block card",
            "stop purchases",
            "make the card unusable",
            "make my card unusable",
        )
        if freeze_signal and not policy_signal:
            return IntentLabel.FREEZE_CARD.value

        dispute_signal = _contains_any(
            normalized,
            "open a dispute",
            "file a dispute",
            "start a dispute",
            "start a charge dispute",
            "contest the",
            "challenge the",
            "file a claim",
            "start a claim",
            "begin a claim",
            "report the charge",
            "report the payment",
            "dispute the",
            "dispute for",
        )
        if dispute_signal and not policy_signal:
            return IntentLabel.CREATE_DISPUTE.value

        if _contains_any(
            normalized,
            "balance",
            "available funds",
            "funds in",
            "money is left",
            "money i can spend",
            "how much i have",
            "how much is available",
            "account total",
            "enough in",
        ):
            return IntentLabel.ACCOUNT_BALANCE.value

        if _contains_any(
            normalized,
            "recent transactions",
            "latest transactions",
            "newest transactions",
            "transaction list",
            "transaction history",
            "recent activity",
            "latest account activity",
            "recent charges",
            "last few purchases",
            "last several",
            "new charges",
            "new payments",
            "posted lately",
            "since monday",
            "past week",
            "this week's",
        ):
            return IntentLabel.RECENT_TRANSACTIONS.value

        if _contains_any(
            normalized,
            "transaction",
            "charge",
            "purchase",
            "payment",
            "merchant",
        ) and _contains_any(
            normalized,
            "tell me about",
            "details",
            "identify",
            "inspect",
            "what happened",
            "where",
            "when",
            "which merchant",
            "posted or reversed",
            "card paid",
        ):
            return IntentLabel.TRANSACTION_DETAILS.value

        return IntentLabel.UNSUPPORTED_OR_UNCERTAIN.value


def _contains_any(text: str, *needles: str) -> bool:
    return any(needle in text for needle in needles)


def _deterministic_mode(values: Iterable[str]) -> str:
    counts = Counter(values)
    if not counts:
        raise ValueError("cannot learn a majority label from no examples")
    return min(counts, key=lambda value: (-counts[value], value))


def make_pipeline(
    *,
    model_name: str,
    c_value: float,
    random_state: int,
    max_iter: int,
) -> Pipeline:
    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                ),
            ),
            (
                "character",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    sublinear_tf=True,
                ),
            ),
        ]
    )
    if model_name == "logistic_regression":
        estimator = LogisticRegression(
            C=c_value,
            max_iter=max_iter,
            random_state=random_state,
        )
    elif model_name == "linear_svm":
        estimator = LinearSVC(C=c_value, random_state=random_state)
    else:
        raise ValueError(f"unsupported model: {model_name}")
    return Pipeline([("features", features), ("classifier", estimator)])


def classification_metrics(
    expected: Sequence[str],
    predicted: Sequence[str],
    *,
    labels: Sequence[str],
) -> dict[str, Any]:
    report = classification_report(
        expected,
        predicted,
        labels=list(labels),
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(expected, predicted)),
        "macro_f1": float(f1_score(expected, predicted, labels=labels, average="macro")),
        "per_class": {
            label: {
                metric: float(report[label][metric])
                for metric in ("precision", "recall", "f1-score")
            }
            for label in labels
        },
        "confusion_matrix": {
            "label_order": list(labels),
            "values": confusion_matrix(expected, predicted, labels=labels).tolist(),
        },
    }


def probability_metrics(
    expected: Sequence[str],
    probabilities: np.ndarray,
    *,
    classes: Sequence[str],
) -> dict[str, Any]:
    class_to_index = {label: index for index, label in enumerate(classes)}
    one_hot = np.zeros_like(probabilities)
    for row, label in enumerate(expected):
        one_hot[row, class_to_index[label]] = 1.0
    confidence = probabilities.max(axis=1)
    return {
        "log_loss": float(log_loss(expected, probabilities, labels=list(classes))),
        "multiclass_brier_score": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "confidence_distribution": {
            "minimum": float(np.min(confidence)),
            "p25": float(np.percentile(confidence, 25)),
            "p50": float(np.percentile(confidence, 50)),
            "p75": float(np.percentile(confidence, 75)),
            "p90": float(np.percentile(confidence, 90)),
            "p95": float(np.percentile(confidence, 95)),
            "maximum": float(np.max(confidence)),
            "mean": float(np.mean(confidence)),
        },
    }


def protected_write_recall_from_intent(
    expected_intents: Sequence[str], predicted_intents: Sequence[str]
) -> float:
    expected = [RISK_BY_INTENT[value] for value in expected_intents]
    predicted = [RISK_BY_INTENT[value] for value in predicted_intents]
    metrics = classification_metrics(expected, predicted, labels=RISK_LABELS)
    return metrics["per_class"][RiskLabel.PROTECTED_WRITE.value]["recall"]


def evaluate_candidate_grid(
    *,
    train_texts: Sequence[str],
    train_labels: Sequence[str],
    validation_texts: Sequence[str],
    validation_labels: Sequence[str],
    labels: Sequence[str],
    config: dict[str, Any],
    target: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], Pipeline]:
    results: list[dict[str, Any]] = []
    fitted: dict[tuple[str, float], Pipeline] = {}
    for model_name in ("logistic_regression", "linear_svm"):
        model_config = config["candidate_models"][model_name]
        for c_value in model_config["c_values"]:
            pipeline = make_pipeline(
                model_name=model_name,
                c_value=float(c_value),
                random_state=config["candidate_models"]["logistic_regression"][
                    "random_state"
                ],
                max_iter=config["candidate_models"]["logistic_regression"]["max_iter"],
            )
            pipeline.fit(train_texts, train_labels)
            predictions = pipeline.predict(validation_texts).tolist()
            metrics = classification_metrics(validation_labels, predictions, labels=labels)
            if target == "intent":
                protected_recall = protected_write_recall_from_intent(
                    validation_labels, predictions
                )
            else:
                protected_recall = metrics["per_class"][
                    RiskLabel.PROTECTED_WRITE.value
                ]["recall"]
            probability = None
            if model_name == "logistic_regression":
                probability = probability_metrics(
                    validation_labels,
                    pipeline.predict_proba(validation_texts),
                    classes=pipeline.classes_.tolist(),
                )
            result = {
                "model": model_name,
                "c": float(c_value),
                "validation": metrics,
                "protected_write_recall": protected_recall,
                "probability_metrics": probability,
            }
            results.append(result)
            fitted[(model_name, float(c_value))] = pipeline

    def selection_key(result: dict[str, Any]) -> tuple[float, float, int, float, float]:
        probability = result["probability_metrics"]
        return (
            round(result["validation"]["macro_f1"], 12),
            round(result["protected_write_recall"], 12),
            int(probability is not None),
            -(probability["log_loss"] if probability is not None else float("inf")),
            -result["c"],
        )

    selected = max(results, key=selection_key)
    selected_model = fitted[(selected["model"], selected["c"])]
    return results, selected, selected_model


def select_abstention_rule(
    *,
    expected_intents: Sequence[str],
    intent_probabilities: np.ndarray,
    intent_classes: Sequence[str],
    independent_risk_predictions: Sequence[str],
    config: dict[str, Any],
) -> dict[str, Any]:
    class_array = np.asarray(intent_classes)
    order = np.argsort(intent_probabilities, axis=1)
    top_indices = order[:, -1]
    second_indices = order[:, -2]
    predicted_intents = class_array[top_indices]
    confidence = intent_probabilities[np.arange(len(top_indices)), top_indices]
    margin = confidence - intent_probabilities[
        np.arange(len(second_indices)), second_indices
    ]
    derived_risks = np.asarray([RISK_BY_INTENT[value] for value in predicted_intents])
    independent_risks = np.asarray(independent_risk_predictions)
    expected = np.asarray(expected_intents)
    expected_risks = np.asarray([RISK_BY_INTENT[value] for value in expected])
    base_accuracy = float(np.mean(predicted_intents == expected))
    rows: list[dict[str, Any]] = []
    for require_agreement in (False, True):
        for confidence_threshold in config["abstention"]["confidence_thresholds"]:
            for margin_threshold in config["abstention"]["margin_thresholds"]:
                evaluated = (confidence >= confidence_threshold) & (
                    margin >= margin_threshold
                )
                if require_agreement:
                    evaluated &= derived_risks == independent_risks
                count = int(np.sum(evaluated))
                protected = evaluated & (
                    expected_risks == RiskLabel.PROTECTED_WRITE.value
                )
                protected_count = int(np.sum(protected))
                if count == 0 or protected_count == 0:
                    continue
                selective_accuracy = float(
                    np.mean(predicted_intents[evaluated] == expected[evaluated])
                )
                protected_recall = float(
                    np.mean(predicted_intents[protected] == expected[protected])
                )
                rows.append(
                    {
                        "confidence_threshold": float(confidence_threshold),
                        "margin_threshold": float(margin_threshold),
                        "require_intent_risk_agreement": require_agreement,
                        "coverage": count / len(expected),
                        "evaluated_count": count,
                        "selective_accuracy": selective_accuracy,
                        "evaluated_protected_write_count": protected_count,
                        "evaluated_protected_write_recall": protected_recall,
                        "eligible": (
                            selective_accuracy >= base_accuracy
                            and protected_recall == 1.0
                        ),
                    }
                )
    eligible = [row for row in rows if row["eligible"]]
    if not eligible:
        selected = {
            "mode": "no_abstention",
            "reason": "no validation threshold satisfied the frozen criterion",
            "coverage": 1.0,
            "selective_accuracy": base_accuracy,
        }
    else:
        best = max(
            eligible,
            key=lambda row: (
                row["coverage"],
                row["selective_accuracy"],
                -row["confidence_threshold"],
                -row["margin_threshold"],
                -int(row["require_intent_risk_agreement"]),
            ),
        )
        if (
            best["coverage"] == 1.0
            and best["confidence_threshold"] == 0.0
            and best["margin_threshold"] == 0.0
            and not best["require_intent_risk_agreement"]
        ):
            selected = {
                **best,
                "mode": "no_abstention",
                "reason": "highest eligible coverage is the un-abstained model",
            }
        else:
            selected = {**best, "mode": "threshold"}
    return {
        "selection_split": "validation",
        "unabstained_accuracy": base_accuracy,
        "selected": selected,
        "coverage_curve": rows,
    }


def apply_abstention_rule(
    *,
    rule: dict[str, Any],
    intent_probabilities: np.ndarray,
    intent_classes: Sequence[str],
    independent_risk_predictions: Sequence[str],
) -> np.ndarray:
    selected = rule["selected"]
    if selected["mode"] == "no_abstention":
        return np.ones(len(intent_probabilities), dtype=bool)
    order = np.argsort(intent_probabilities, axis=1)
    top = order[:, -1]
    second = order[:, -2]
    confidence = intent_probabilities[np.arange(len(top)), top]
    margin = confidence - intent_probabilities[np.arange(len(second)), second]
    evaluated = (confidence >= selected["confidence_threshold"]) & (
        margin >= selected["margin_threshold"]
    )
    if selected["require_intent_risk_agreement"]:
        predicted = np.asarray(intent_classes)[top]
        derived = np.asarray([RISK_BY_INTENT[value] for value in predicted])
        evaluated &= derived == np.asarray(independent_risk_predictions)
    return evaluated


def grouped_learning_curve_subsets(
    train_examples: Sequence[dict[str, Any]], fractions: Sequence[float]
) -> list[tuple[float, list[dict[str, Any]]]]:
    groups_by_intent: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for example in train_examples:
        groups_by_intent[example["intent"]][example["group_id"]].append(example)
    subsets: list[tuple[float, list[dict[str, Any]]]] = []
    for fraction in fractions:
        selected: list[dict[str, Any]] = []
        for intent in sorted(groups_by_intent):
            groups = groups_by_intent[intent]
            group_ids = sorted(groups)
            count = max(1, int(np.ceil(len(group_ids) * fraction)))
            for group_id in group_ids[:count]:
                selected.extend(groups[group_id])
        subsets.append(
            (fraction, sorted(selected, key=lambda example: example["example_id"]))
        )
    return subsets


def measure_local_inference_latency(
    model: Pipeline,
    texts: Sequence[str],
    *,
    warmup_calls: int,
    sample_count: int,
) -> dict[str, Any]:
    for index in range(warmup_calls):
        model.predict([texts[index % len(texts)]])
    samples_ms: list[float] = []
    for index in range(sample_count):
        started = time.perf_counter_ns()
        model.predict([texts[index % len(texts)]])
        samples_ms.append((time.perf_counter_ns() - started) / 1_000_000)
    return {
        "measurement_source": "local_ml",
        "sample_count": sample_count,
        "p50_ms": float(np.percentile(samples_ms, 50)),
        "p90_ms": float(np.percentile(samples_ms, 90)),
        "p95_ms": float(np.percentile(samples_ms, 95)),
    }
