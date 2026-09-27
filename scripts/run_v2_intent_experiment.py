"""Run the offline-only SentinelVoice V2-C intent/risk ML experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn

from backend.app.evaluation.v2_ml import (
    INTENT_LABELS,
    RISK_BY_INTENT,
    RISK_LABELS,
    MajorityBaseline,
    RuleBaseline,
    apply_abstention_rule,
    classification_metrics,
    evaluate_candidate_grid,
    grouped_learning_curve_subsets,
    make_pipeline,
    measure_local_inference_latency,
    probability_metrics,
    select_abstention_rule,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ML_DIR = REPOSITORY_ROOT / "data/evals/v2/ml"
DEFAULT_DATASET_PATH = DEFAULT_ML_DIR / "intent_risk_dataset.json"
DEFAULT_MANIFEST_PATH = DEFAULT_ML_DIR / "intent_risk_dataset.manifest.json"
DEFAULT_CONFIG_PATH = DEFAULT_ML_DIR / "classifier_config.json"
DEFAULT_SELECTION_PATH = DEFAULT_ML_DIR / "model_selection.json"
DEFAULT_REPORT_PATH = DEFAULT_ML_DIR / "classifier_report.json"
DEFAULT_ARTIFACT_PATH = REPOSITORY_ROOT / "artifacts/v2/classifier/classifier.joblib"
REPORT_VERSION = "v2-classifier-report.v1"
SELECTION_VERSION = "v2-model-selection.v1"
ARTIFACT_VERSION = "v2-classifier-artifact.v1"


def stable_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _examples_for_split(
    dataset: dict[str, Any], split: str
) -> list[dict[str, Any]]:
    return [example for example in dataset["examples"] if example["split"] == split]


def _texts(examples: Sequence[dict[str, Any]]) -> list[str]:
    return [example["text"] for example in examples]


def _labels(examples: Sequence[dict[str, Any]], field: str) -> list[str]:
    return [example[field] for example in examples]


def _baseline_metrics(
    baseline: MajorityBaseline | RuleBaseline,
    examples: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    texts = _texts(examples)
    expected_intents = _labels(examples, "intent")
    expected_risks = _labels(examples, "risk")
    return {
        "intent": classification_metrics(
            expected_intents,
            baseline.predict_intent(texts),
            labels=INTENT_LABELS,
        ),
        "risk": classification_metrics(
            expected_risks,
            baseline.predict_risk(texts),
            labels=RISK_LABELS,
        ),
    }


def _model_metrics(
    model: Any,
    examples: Sequence[dict[str, Any]],
    *,
    target: str,
    labels: Sequence[str],
) -> dict[str, Any]:
    texts = _texts(examples)
    expected = _labels(examples, target)
    predicted = model.predict(texts).tolist()
    result = classification_metrics(expected, predicted, labels=labels)
    if hasattr(model, "predict_proba"):
        result["probability"] = probability_metrics(
            expected,
            model.predict_proba(texts),
            classes=model.classes_.tolist(),
        )
    return result


def _selected_reason(selected: dict[str, Any]) -> str:
    return (
        "Selected using validation only: highest intent macro-F1, then "
        "intent-derived PROTECTED_WRITE recall, then probability/calibration "
        "availability and log loss, then lower C as a deterministic complexity "
        "tie-breaker. Locked-test results were not available to selection."
    )


def _selection_summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": result["model"],
        "c": result["c"],
        "validation_accuracy": result["validation"]["accuracy"],
        "validation_macro_f1": result["validation"]["macro_f1"],
        "protected_write_recall": result["protected_write_recall"],
        "probability_metrics": result["probability_metrics"],
    }


def _evaluate_abstention(
    *,
    examples: Sequence[dict[str, Any]],
    intent_model: Any,
    risk_model: Any,
    abstention: dict[str, Any],
) -> dict[str, Any]:
    texts = _texts(examples)
    expected = np.asarray(_labels(examples, "intent"))
    probabilities = intent_model.predict_proba(texts)
    predicted = intent_model.classes_[np.argmax(probabilities, axis=1)]
    risk_predictions = risk_model.predict(texts).tolist()
    evaluated = apply_abstention_rule(
        rule=abstention,
        intent_probabilities=probabilities,
        intent_classes=intent_model.classes_.tolist(),
        independent_risk_predictions=risk_predictions,
    )
    evaluated_count = int(np.sum(evaluated))
    return {
        "coverage": evaluated_count / len(examples),
        "evaluated_count": evaluated_count,
        "abstained_count": len(examples) - evaluated_count,
        "selective_accuracy": (
            float(np.mean(predicted[evaluated] == expected[evaluated]))
            if evaluated_count
            else None
        ),
    }


def _risk_disagreement(
    *,
    examples: Sequence[dict[str, Any]],
    intent_model: Any,
    risk_model: Any,
) -> dict[str, Any]:
    texts = _texts(examples)
    intent_predictions = intent_model.predict(texts).tolist()
    derived = [RISK_BY_INTENT[value] for value in intent_predictions]
    independent = risk_model.predict(texts).tolist()
    disagreements = [
        {
            "example_id": example["example_id"],
            "expected_intent": example["intent"],
            "predicted_intent": intent,
            "intent_derived_risk": derived_risk,
            "independent_risk": independent_risk,
            "source_tags": example["tags"],
        }
        for example, intent, derived_risk, independent_risk in zip(
            examples, intent_predictions, derived, independent, strict=True
        )
        if derived_risk != independent_risk
    ]
    return {
        "count": len(disagreements),
        "rate": len(disagreements) / len(examples),
        "by_expected_intent": dict(
            sorted(Counter(row["expected_intent"] for row in disagreements).items())
        ),
        "examples": disagreements,
        "interpretation": (
            "Independent risk classification is experimental and redundant because "
            "the dataset risk label is deterministic from intent. It has no authority."
        ),
    }


def _learning_curve(
    *,
    train_examples: Sequence[dict[str, Any]],
    validation_examples: Sequence[dict[str, Any]],
    logistic_candidate: dict[str, Any],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    validation_texts = _texts(validation_examples)
    validation_labels = _labels(validation_examples, "intent")
    for fraction, subset in grouped_learning_curve_subsets(
        train_examples, config["learning_curve_train_group_fractions"]
    ):
        model = make_pipeline(
            model_name="logistic_regression",
            c_value=logistic_candidate["c"],
            random_state=config["candidate_models"]["logistic_regression"][
                "random_state"
            ],
            max_iter=config["candidate_models"]["logistic_regression"]["max_iter"],
        )
        model.fit(_texts(subset), _labels(subset, "intent"))
        predicted = model.predict(validation_texts).tolist()
        rows.append(
            {
                "train_group_fraction": fraction,
                "train_examples": len(subset),
                "train_groups": len({example["group_id"] for example in subset}),
                "validation_macro_f1": classification_metrics(
                    validation_labels, predicted, labels=INTENT_LABELS
                )["macro_f1"],
            }
        )
    return rows


def _best_logistic_candidate(
    candidates: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    logistic = [
        candidate
        for candidate in candidates
        if candidate["model"] == "logistic_regression"
    ]
    return max(
        logistic,
        key=lambda candidate: (
            candidate["validation"]["macro_f1"],
            candidate["protected_write_recall"],
            -candidate["probability_metrics"]["log_loss"],
            -candidate["c"],
        ),
    )


def run_experiment(
    *,
    dataset_path: Path = DEFAULT_DATASET_PATH,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    config_path: Path = DEFAULT_CONFIG_PATH,
    selection_path: Path = DEFAULT_SELECTION_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
    artifact_path: Path = DEFAULT_ARTIFACT_PATH,
) -> dict[str, Any]:
    dataset_bytes = dataset_path.read_bytes()
    dataset = json.loads(dataset_bytes)
    manifest = _load(manifest_path)
    config = _load(config_path)
    if sha256_bytes(dataset_bytes) != manifest["hashes"]["dataset_sha256"]:
        raise ValueError("dataset hash does not match the frozen manifest")
    if config["excluded_from_selection"] != ["locked_test"]:
        raise ValueError("locked_test must remain excluded from selection")

    train = _examples_for_split(dataset, "train")
    validation = _examples_for_split(dataset, "validation")
    training_started = time.perf_counter()

    majority = MajorityBaseline.fit(train)
    rules = RuleBaseline()
    intent_candidates, selected_intent, intent_model = evaluate_candidate_grid(
        train_texts=_texts(train),
        train_labels=_labels(train, "intent"),
        validation_texts=_texts(validation),
        validation_labels=_labels(validation, "intent"),
        labels=INTENT_LABELS,
        config=config,
        target="intent",
    )
    risk_candidates, selected_risk, risk_model = evaluate_candidate_grid(
        train_texts=_texts(train),
        train_labels=_labels(train, "risk"),
        validation_texts=_texts(validation),
        validation_labels=_labels(validation, "risk"),
        labels=RISK_LABELS,
        config=config,
        target="risk",
    )
    logistic_candidate = _best_logistic_candidate(intent_candidates)
    logistic_model = make_pipeline(
        model_name="logistic_regression",
        c_value=logistic_candidate["c"],
        random_state=config["candidate_models"]["logistic_regression"][
            "random_state"
        ],
        max_iter=config["candidate_models"]["logistic_regression"]["max_iter"],
    )
    logistic_model.fit(_texts(train), _labels(train, "intent"))
    validation_probabilities = logistic_model.predict_proba(_texts(validation))
    validation_risk_predictions = risk_model.predict(_texts(validation)).tolist()
    abstention = select_abstention_rule(
        expected_intents=_labels(validation, "intent"),
        intent_probabilities=validation_probabilities,
        intent_classes=logistic_model.classes_.tolist(),
        independent_risk_predictions=validation_risk_predictions,
        config=config,
    )
    learning_curve = _learning_curve(
        train_examples=train,
        validation_examples=validation,
        logistic_candidate=logistic_candidate,
        config=config,
    )

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_payload = {
        "artifact_version": ARTIFACT_VERSION,
        "classifier_version": config["classifier_version"],
        "dataset_version": dataset["dataset_version"],
        "dataset_sha256": manifest["hashes"]["dataset_sha256"],
        "intent_label_order": list(INTENT_LABELS),
        "risk_label_order": list(RISK_LABELS),
        "intent_model": intent_model,
        "logistic_probability_intent_model": logistic_model,
        "risk_model": risk_model,
        "abstention": abstention["selected"],
        "trusted_local_artifact": True,
        "runtime_authority": False,
    }
    joblib.dump(artifact_payload, artifact_path, compress=3)
    artifact_bytes = artifact_path.read_bytes()
    training_time_seconds = time.perf_counter() - training_started

    selection = {
        "schema_version": SELECTION_VERSION,
        "experiment_id": config["experiment_id"],
        "classifier_version": config["classifier_version"],
        "dataset_version": dataset["dataset_version"],
        "dataset_sha256": manifest["hashes"]["dataset_sha256"],
        "v2_a_seed_sha256": manifest["hashes"]["v2_a_seed_sha256"],
        "selection_splits": ["train", "validation"],
        "excluded_split": "locked_test",
        "locked_test_evaluated": False,
        "selection_criteria": config["selection_criteria"],
        "feature_config": config["feature_config"],
        "intent_candidates": [_selection_summary(row) for row in intent_candidates],
        "risk_candidates": [_selection_summary(row) for row in risk_candidates],
        "selected_intent_model": _selection_summary(selected_intent),
        "probability_intent_model": _selection_summary(logistic_candidate),
        "selected_risk_model": _selection_summary(selected_risk),
        "selection_reason": _selected_reason(selected_intent),
        "abstention": abstention,
        "training_group_count": len({example["group_id"] for example in train}),
        "validation_group_count": len(
            {example["group_id"] for example in validation}
        ),
        "locked_test_group_count": manifest["counts"]["locked_test_groups"],
        "intent_label_order": list(INTENT_LABELS),
        "risk_label_order": list(RISK_LABELS),
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
        "model_artifact_path": str(artifact_path.relative_to(REPOSITORY_ROOT)),
        "model_artifact_sha256": sha256_bytes(artifact_bytes),
        "model_artifact_size_bytes": len(artifact_bytes),
        "experiment_timestamp": None,
    }
    selection_path.write_bytes(stable_json_bytes(selection))

    # Stage D begins only after selection metadata and the model artifact exist.
    locked_test = _examples_for_split(dataset, "locked_test")
    locked_intent_metrics = _model_metrics(
        intent_model, locked_test, target="intent", labels=INTENT_LABELS
    )
    locked_risk_metrics = _model_metrics(
        risk_model, locked_test, target="risk", labels=RISK_LABELS
    )
    report = {
        "schema_version": REPORT_VERSION,
        "experiment_id": config["experiment_id"],
        "classifier_version": config["classifier_version"],
        "dataset_version": dataset["dataset_version"],
        "dataset_sha256": manifest["hashes"]["dataset_sha256"],
        "data_scope": {
            "sentinelvoice_controlled_synthetic": True,
            "public_external_datasets": [],
            "production_transcripts": [],
        },
        "advisory_only": True,
        "runtime_integrated": False,
        "measurement_source": "local_ml",
        "majority_baseline": {
            "learned_intent": majority.intent,
            "learned_risk": majority.risk,
            "validation": _baseline_metrics(majority, validation),
            "locked_test": _baseline_metrics(majority, locked_test),
        },
        "rule_baseline": {
            "validation": _baseline_metrics(rules, validation),
            "locked_test": _baseline_metrics(rules, locked_test),
        },
        "logistic_regression_validation_candidates": [
            _selection_summary(row)
            for row in intent_candidates
            if row["model"] == "logistic_regression"
        ],
        "linear_svm_validation_candidates": [
            _selection_summary(row)
            for row in intent_candidates
            if row["model"] == "linear_svm"
        ],
        "selected": {
            "intent": _selection_summary(selected_intent),
            "risk": _selection_summary(selected_risk),
            "probability_intent_candidate": _selection_summary(
                logistic_candidate
            ),
            "reason": _selected_reason(selected_intent),
        },
        "validation": {
            "intent": _model_metrics(
                intent_model, validation, target="intent", labels=INTENT_LABELS
            ),
            "logistic_probability_intent": _model_metrics(
                logistic_model,
                validation,
                target="intent",
                labels=INTENT_LABELS,
            ),
            "risk": _model_metrics(
                risk_model, validation, target="risk", labels=RISK_LABELS
            ),
            "abstention": _evaluate_abstention(
                examples=validation,
                intent_model=logistic_model,
                risk_model=risk_model,
                abstention=abstention,
            ),
            "risk_disagreement": _risk_disagreement(
                examples=validation,
                intent_model=intent_model,
                risk_model=risk_model,
            ),
        },
        "locked_test": {
            "evaluated_after_selection_metadata_written": True,
            "intent": locked_intent_metrics,
            "logistic_probability_intent": _model_metrics(
                logistic_model,
                locked_test,
                target="intent",
                labels=INTENT_LABELS,
            ),
            "risk": locked_risk_metrics,
            "abstention": _evaluate_abstention(
                examples=locked_test,
                intent_model=logistic_model,
                risk_model=risk_model,
                abstention=abstention,
            ),
            "risk_disagreement": _risk_disagreement(
                examples=locked_test,
                intent_model=intent_model,
                risk_model=risk_model,
            ),
        },
        "learning_curve": learning_curve,
        "training": {
            "training_time_seconds": training_time_seconds,
            "python_version": platform.python_version(),
            "scikit_learn_version": sklearn.__version__,
            "train_examples": len(train),
            "train_groups": len({example["group_id"] for example in train}),
        },
        "local_inference_latency": measure_local_inference_latency(
            intent_model,
            _texts(validation),
            warmup_calls=config["latency"]["warmup_calls"],
            sample_count=config["latency"]["sample_count"],
        ),
        "artifact": {
            "path": str(artifact_path.relative_to(REPOSITORY_ROOT)),
            "size_bytes": len(artifact_bytes),
            "sha256": sha256_bytes(artifact_bytes),
            "trusted_local_only": True,
        },
        "safety_invariants": {
            "classifier_runtime_authority": False,
            "authorization_remains_deterministic": True,
            "confirmation_remains_deterministic": True,
            "resource_ownership_remains_deterministic": True,
        },
    }
    report_path.write_bytes(stable_json_bytes(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT_PATH)
    args = parser.parse_args()
    report = run_experiment(
        dataset_path=args.dataset,
        manifest_path=args.manifest,
        config_path=args.config,
        selection_path=args.selection,
        report_path=args.report,
        artifact_path=args.artifact,
    )
    print(
        json.dumps(
            {
                "selected": report["selected"],
                "locked_test_intent": report["locked_test"]["intent"],
                "artifact": report["artifact"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
