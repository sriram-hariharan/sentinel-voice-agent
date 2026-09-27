import json
from pathlib import Path

import joblib

from backend.app.evaluation.v2_ml import (
    MajorityBaseline,
    RuleBaseline,
    make_pipeline,
)

ML_DIR = Path("data/evals/v2/ml")
DATASET_PATH = ML_DIR / "intent_risk_dataset.json"
MANIFEST_PATH = ML_DIR / "intent_risk_dataset.manifest.json"
CONFIG_PATH = ML_DIR / "classifier_config.json"
SELECTION_PATH = ML_DIR / "model_selection.json"
REPORT_PATH = ML_DIR / "classifier_report.json"
ARTIFACT_PATH = Path("artifacts/v2/classifier/classifier.joblib")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _split_examples(split: str) -> list[dict]:
    return [
        example
        for example in _load(DATASET_PATH)["examples"]
        if example["split"] == split
    ]


def test_majority_and_rule_baselines_are_deterministic() -> None:
    train = _split_examples("train")
    validation_texts = [example["text"] for example in _split_examples("validation")]

    first_majority = MajorityBaseline.fit(train)
    second_majority = MajorityBaseline.fit(train)
    first_rules = RuleBaseline().predict_intent(validation_texts)
    second_rules = RuleBaseline().predict_intent(validation_texts)

    assert first_majority == second_majority
    assert first_majority.predict_intent(validation_texts) == (
        second_majority.predict_intent(validation_texts)
    )
    assert first_rules == second_rules


def test_model_predictions_are_reproducible() -> None:
    config = _load(CONFIG_PATH)
    train = _split_examples("train")
    validation = _split_examples("validation")
    train_texts = [example["text"] for example in train]
    train_labels = [example["intent"] for example in train]
    validation_texts = [example["text"] for example in validation]
    kwargs = {
        "model_name": "logistic_regression",
        "c_value": 2.0,
        "random_state": config["candidate_models"]["logistic_regression"][
            "random_state"
        ],
        "max_iter": config["candidate_models"]["logistic_regression"]["max_iter"],
    }
    first = make_pipeline(**kwargs).fit(train_texts, train_labels)
    second = make_pipeline(**kwargs).fit(train_texts, train_labels)

    assert first.predict(validation_texts).tolist() == second.predict(
        validation_texts
    ).tolist()


def test_serialized_artifact_reload_predicts_identically() -> None:
    first = joblib.load(ARTIFACT_PATH)
    second = joblib.load(ARTIFACT_PATH)
    texts = [example["text"] for example in _split_examples("validation")]

    assert first["trusted_local_artifact"] is True
    assert first["runtime_authority"] is False
    assert first["intent_model"].predict(texts).tolist() == second[
        "intent_model"
    ].predict(texts).tolist()
    assert first["risk_model"].predict(texts).tolist() == second[
        "risk_model"
    ].predict(texts).tolist()


def test_selection_excludes_locked_test_and_precedes_its_evaluation() -> None:
    manifest = _load(MANIFEST_PATH)
    selection = _load(SELECTION_PATH)
    report = _load(REPORT_PATH)

    assert selection["selection_splits"] == ["train", "validation"]
    assert selection["excluded_split"] == "locked_test"
    assert selection["locked_test_evaluated"] is False
    assert selection["dataset_sha256"] == manifest["hashes"]["dataset_sha256"]
    assert selection["locked_test_group_count"] == 81
    assert report["locked_test"]["evaluated_after_selection_metadata_written"]


def test_report_metadata_and_measurement_boundary_are_valid() -> None:
    report = _load(REPORT_PATH)

    assert report["schema_version"] == "v2-classifier-report.v1"
    assert report["advisory_only"] is True
    assert report["runtime_integrated"] is False
    assert report["measurement_source"] == "local_ml"
    assert report["local_inference_latency"]["measurement_source"] == "local_ml"
    assert report["data_scope"] == {
        "sentinelvoice_controlled_synthetic": True,
        "public_external_datasets": [],
        "production_transcripts": [],
    }
    assert report["artifact"]["size_bytes"] == ARTIFACT_PATH.stat().st_size


def test_selected_model_beats_both_locked_test_baselines() -> None:
    report = _load(REPORT_PATH)
    selected = report["locked_test"]["intent"]["macro_f1"]

    assert selected > report["majority_baseline"]["locked_test"]["intent"][
        "macro_f1"
    ]
    assert selected > report["rule_baseline"]["locked_test"]["intent"][
        "macro_f1"
    ]


def test_no_production_agent_runtime_imports_classifier() -> None:
    production_files = [
        *Path("backend/app/agent").rglob("*.py"),
        *Path("backend/app/voice").rglob("*.py"),
    ]
    contents = "\n".join(path.read_text(encoding="utf-8") for path in production_files)

    assert "v2_ml" not in contents
    assert "classifier.joblib" not in contents
