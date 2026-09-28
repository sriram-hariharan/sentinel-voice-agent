from __future__ import annotations

import inspect
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from sklearn.pipeline import FeatureUnion, Pipeline

from scripts import run_v2c3_model_tournament as tournament

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "data/evals/v2/ml/v2c3_tournament_config.json"
V2C1_CONFIG_PATH = ROOT / "data/evals/v2/ml/classifier_config.json"
V2C1_SELECTION_PATH = ROOT / "data/evals/v2/ml/model_selection.json"
RUNNER_PATH = ROOT / "scripts/run_v2c3_model_tournament.py"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def config() -> dict[str, Any]:
    return load_json(CONFIG_PATH)


def synthetic_examples(config: dict[str, Any]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for label in config["intent_taxonomy"]:
        for group_number in range(5):
            for variant in range(2):
                examples.append(
                    {
                        "example_id": f"{label}-{group_number}-{variant}",
                        "group_id": f"{label}-group-{group_number}",
                        "intent": label,
                        "source_id": "sentinelvoice_v2c1_internal",
                        "text": f"synthetic {label} {group_number} {variant}",
                        "data_role": "development",
                    }
                )
    return examples


def test_tournament_config_schema_and_fixed_candidate_count(config: dict[str, Any]) -> None:
    tournament.validate_config(config)
    assert config["schema_version"] == "v2c3-model-tournament-config.v1"
    assert config["training_or_evaluation_performed"] is False
    assert len(config["candidates"]) == 14
    assert len({candidate["id"] for candidate in config["candidates"]}) == 14


def test_v2c1_anchor_matches_existing_frozen_configuration(
    config: dict[str, Any],
) -> None:
    old_config = load_json(V2C1_CONFIG_PATH)
    old_selection = load_json(V2C1_SELECTION_PATH)
    representation = config["representations"]["v2c1_word_char_tfidf"]
    classifier = config["classifiers"]["v2c1_linear_svc_anchor"]
    assert representation["word_tfidf"]["ngram_range"] == old_config["feature_config"][
        "word_tfidf"
    ]["ngram_range"]
    assert representation["character_tfidf"]["ngram_range"] == old_config[
        "feature_config"
    ]["character_tfidf"]["ngram_range"]
    assert representation["word_tfidf"]["sublinear_tf"] is True
    assert representation["character_tfidf"]["sublinear_tf"] is True
    assert classifier["class"] == "LinearSVC"
    assert classifier["parameters"]["C"] == old_selection["selected_intent_model"]["c"]
    assert classifier["parameters"]["random_state"] == old_config["candidate_models"][
        "logistic_regression"
    ]["random_state"]
    assert classifier["parameters"]["class_weight"] is None


def test_required_representations_classifiers_and_architectures_are_present(
    config: dict[str, Any],
) -> None:
    candidate_representations = {
        candidate["representation"] for candidate in config["candidates"]
    }
    assert candidate_representations == {
        "v2c1_word_char_tfidf",
        "word_char_tfidf_alt",
        "word_tfidf",
        "char_tfidf",
        "tfidf_lsa",
        "bge_small_en_v1_5",
    }
    candidate_classifier_classes = {
        config["classifiers"][candidate["classifier"]]["class"]
        for candidate in config["candidates"]
    }
    assert candidate_classifier_classes == {
        "LinearSVC",
        "LogisticRegression",
        "SGDClassifier",
        "RidgeClassifier",
    }
    assert {candidate["architecture"] for candidate in config["candidates"]} == {
        "direct_9_way",
        "hierarchical_supported_then_intent",
    }


def test_exact_candidate_matrix_is_frozen(config: dict[str, Any]) -> None:
    observed = [
        (
            candidate["architecture"],
            candidate["representation"],
            candidate["classifier"],
        )
        for candidate in config["candidates"]
    ]
    assert observed == [
        ("direct_9_way", "v2c1_word_char_tfidf", "v2c1_linear_svc_anchor"),
        ("direct_9_way", "v2c1_word_char_tfidf", "linear_svc_balanced"),
        ("direct_9_way", "v2c1_word_char_tfidf", "logistic_regression_balanced"),
        ("direct_9_way", "v2c1_word_char_tfidf", "sgd_classifier_balanced"),
        ("direct_9_way", "v2c1_word_char_tfidf", "ridge_classifier_balanced"),
        ("direct_9_way", "word_tfidf", "linear_svc_balanced"),
        ("direct_9_way", "char_tfidf", "linear_svc_balanced"),
        ("direct_9_way", "word_char_tfidf_alt", "linear_svc_balanced"),
        ("direct_9_way", "tfidf_lsa", "linear_svc_balanced"),
        ("direct_9_way", "tfidf_lsa", "logistic_regression_balanced"),
        ("direct_9_way", "bge_small_en_v1_5", "linear_svc_balanced"),
        ("direct_9_way", "bge_small_en_v1_5", "logistic_regression_balanced"),
        (
            "hierarchical_supported_then_intent",
            "v2c1_word_char_tfidf",
            "linear_svc_balanced",
        ),
        (
            "hierarchical_supported_then_intent",
            "bge_small_en_v1_5",
            "logistic_regression_balanced",
        ),
    ]


def test_five_fold_decision_is_frozen(config: dict[str, Any]) -> None:
    assert config["cv"] == {
        "artifact_path": "data/evals/v2/ml/v2c3_cv_folds.json",
        "method": "StratifiedGroupKFold",
        "n_splits": 5,
        "shuffle": True,
        "random_state": 20260928,
        "fallback_allowed": False,
        "feasibility_decided_before_model_scores": True,
        "folds_reused_for_every_candidate": True,
        "group_field": "group_id",
        "target_field": "intent",
    }


def test_fold_preparation_is_deterministic_group_safe_and_complete(
    config: dict[str, Any],
) -> None:
    examples = synthetic_examples(config)
    first = tournament.fold_assignments_payload(examples, config)
    second = tournament.fold_assignments_payload(examples, config)
    assert first == second
    assert first["n_splits"] == 5
    assert first["model_scores_available_when_prepared"] is False
    assert len(first["assignments"]) == len(examples)
    assert len({row["example_id"] for row in first["assignments"]}) == len(examples)
    folds = tournament.validate_fold_artifact(first, examples, config)
    group_folds: dict[str, set[int]] = {}
    for example, fold in zip(examples, folds, strict=True):
        group_folds.setdefault(example["group_id"], set()).add(int(fold))
    assert all(len(values) == 1 for values in group_folds.values())
    for summary in first["fold_summaries"]:
        assert all(summary["counts_by_intent"][label] > 0 for label in config["intent_taxonomy"])


def test_only_frozen_development_dataset_is_an_allowed_input(config: dict[str, Any]) -> None:
    assert config["development_dataset"]["path"] == (
        "data/evals/v2/ml/v2c3_development_dataset.json"
    )
    assert config["development_dataset"]["sha256"] == (
        "3783042b3656e3170c2bf001a0fe059d65cad415ed259365d095a1e19fe689ca"
    )
    assert config["development_dataset"]["allowed_data_role"] == "development"
    assert config["report"]["final_lockbox_used"] is False
    assert config["report"]["challenge_set_used"] is False
    assert config["report"]["cfpb_used"] is False
    assert not any(
        field in inspect.signature(tournament.load_development_data).parameters
        for field in ("dataset_path", "test_path", "lockbox_path", "challenge_path", "cfpb_path")
    )


def test_runner_has_no_prohibited_search_or_evaluation_implementation() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert "GridSearchCV" not in source
    assert "RandomizedSearchCV" not in source
    assert "backend.app.main" not in source
    assert "backend.app.agent" not in source
    assert "locked_test" not in source
    assert "banking77/test" not in source
    assert "cfpb_narrative" not in source
    assert "v2c3_challenge_set.json" not in source
    assert "v2c3_fresh_lockbox_manifest.json" not in source


def test_tfidf_and_lsa_are_inside_fresh_candidate_pipelines(
    config: dict[str, Any],
) -> None:
    candidate = next(
        row for row in config["candidates"] if row["id"] == "direct_lsa_balanced_svc"
    )
    representation = tournament.build_representation(candidate["representation"], config)
    assert isinstance(representation, Pipeline)
    assert list(representation.named_steps) == ["tfidf", "truncated_svd", "normalize"]
    assert isinstance(representation.named_steps["tfidf"], FeatureUnion)
    source = inspect.getsource(tournament.run_candidate_cv)
    assert "fit_candidate" in source
    assert "train_indices" in source
    assert "validation_indices" in source
    assert "model.predict(validation_features)" in source


class StaticEstimator:
    def __init__(self, predictions: list[str]) -> None:
        self.predictions = predictions
        self.received: list[Any] = []

    def predict(self, features: Any) -> np.ndarray:
        self.received.append(features)
        return np.asarray(self.predictions[: len(features)])


def test_hierarchical_prediction_routes_only_supported_examples() -> None:
    stage_1 = StaticEstimator(["supported", tournament.UNSUPPORTED, "supported"])
    stage_2 = StaticEstimator(["freeze_card", "account_balance"])
    model = tournament.HierarchicalClassifier(stage_1, stage_2)
    predicted = model.predict(["a", "b", "c"])
    assert predicted.tolist() == ["freeze_card", tournament.UNSUPPORTED, "account_balance"]
    assert stage_2.received == [["a", "c"]]


def test_hierarchical_fit_uses_supported_only_for_stage_2(
    config: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    fitted_labels: list[list[str]] = []

    class RecordingEstimator:
        def fit(self, features: Any, labels: np.ndarray) -> RecordingEstimator:
            fitted_labels.append(labels.tolist())
            return self

        def predict(self, features: Any) -> np.ndarray:
            return np.asarray(["supported"] * len(features))

    monkeypatch.setattr(tournament, "build_classifier", lambda *args: RecordingEstimator())
    candidate = {
        "id": "synthetic-hierarchy",
        "architecture": "hierarchical_supported_then_intent",
        "representation": "bge_small_en_v1_5",
        "classifier": "logistic_regression_balanced",
    }
    labels = np.asarray(["account_balance", tournament.UNSUPPORTED, "freeze_card"])
    tournament.fit_candidate(candidate, np.ones((3, 2)), labels, config)
    assert fitted_labels[0] == ["supported", tournament.UNSUPPORTED, "supported"]
    assert fitted_labels[1] == ["account_balance", "freeze_card"]


def test_metric_formulas_use_exact_protected_intent_match(config: dict[str, Any]) -> None:
    gold = [
        tournament.UNSUPPORTED,
        tournament.UNSUPPORTED,
        "freeze_card",
        "create_dispute",
        "account_balance",
        "informational_policy",
    ]
    predicted = [
        tournament.UNSUPPORTED,
        "account_balance",
        "create_dispute",
        "create_dispute",
        "freeze_card",
        "informational_policy",
    ]
    metrics = tournament.safety_metrics(gold, predicted, config)
    assert metrics["unsupported_or_uncertain_recall"] == pytest.approx(0.5)
    assert metrics["false_supported_count"] == 1
    assert metrics["false_supported_rate"] == pytest.approx(0.5)
    assert metrics["protected_write_recall"] == pytest.approx(0.5)
    assert metrics["protected_write_false_positive_count"] == 1
    assert metrics["protected_write_false_positive_rate"] == pytest.approx(0.25)
    assert metrics["supported_intent_accuracy"] == pytest.approx(0.5)


def test_safety_gate_logic_requires_all_three_gates(config: dict[str, Any]) -> None:
    metrics = {
        "protected_write_false_positive_rate": 0.01,
        "protected_write_recall": 0.8,
        "unsupported_or_uncertain_recall": 0.8,
    }
    assert tournament.apply_safety_gates(metrics, config)["eligible"] is True
    for field, failed_value in (
        ("protected_write_false_positive_rate", 0.01001),
        ("protected_write_recall", 0.79999),
        ("unsupported_or_uncertain_recall", 0.79999),
    ):
        changed = dict(metrics)
        changed[field] = failed_value
        assert tournament.apply_safety_gates(changed, config)["eligible"] is False


def ranking_result(
    candidate_id: str,
    macro_f1: float,
    *,
    false_supported: float = 0.1,
    unsupported_recall: float = 0.9,
    protected_fpr: float = 0.005,
    standard_deviation: float = 0.02,
    latency: float = 1.0,
    size: int = 100,
    architecture: str = "direct_9_way",
    representation: str = "v2c1_word_char_tfidf",
    eligible: bool = True,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "macro_f1_mean": macro_f1,
        "macro_f1_standard_deviation": standard_deviation,
        "definition": {"architecture": architecture, "representation": representation},
        "pooled_oof_metrics": {
            "false_supported_rate": false_supported,
            "unsupported_or_uncertain_recall": unsupported_recall,
            "protected_write_false_positive_rate": protected_fpr,
        },
        "latency": {
            "primary_scope": "representation_and_classifier",
            "representation_and_classifier": {"p95_ms": latency},
        },
        "model_size": {"serialized_fitted_candidate_bytes": size},
        "safety_gate_result": {"eligible": eligible},
    }


def test_ranking_is_deterministic_and_uses_frozen_tie_breaks() -> None:
    high_macro = ranking_result("high", 0.91, false_supported=0.2)
    within_tie = ranking_result("tie-winner", 0.906, false_supported=0.05)
    outside_tie = ranking_result("outside", 0.9049, false_supported=0.0)
    ineligible = ranking_result("ineligible", 0.99, eligible=False)
    ranked = tournament.rank_eligible_candidates(
        [outside_tie, ineligible, high_macro, within_tie]
    )
    assert [row["candidate_id"] for row in ranked] == ["tie-winner", "high", "outside"]


def test_finalists_are_limited_to_three_and_prefer_distinct_families(
    config: dict[str, Any],
) -> None:
    ranked = [
        ranking_result("lexical-1", 0.9),
        ranking_result("lexical-2", 0.89, representation="word_tfidf"),
        ranking_result("lsa", 0.88, representation="tfidf_lsa"),
        ranking_result("bge", 0.87, representation="bge_small_en_v1_5"),
    ]
    assert tournament.select_finalists(ranked, config) == ["lexical-1", "lsa", "bge"]


def test_embedding_cache_validation_is_keyed_and_label_free(
    tmp_path: Path, config: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    local_config = deepcopy(config)
    local_config["development_dataset"]["sha256"] = "d" * 64
    spec = local_config["representations"]["bge_small_en_v1_5"]
    spec["dimensions"] = 3
    spec["cache_path"] = "local/cache.npz"
    spec["cache_metadata_path"] = "local/cache.json"
    examples = synthetic_examples(local_config)[:4]
    monkeypatch.setattr(tournament, "REPOSITORY_ROOT", tmp_path)
    cache_path = tmp_path / spec["cache_path"]
    cache_path.parent.mkdir(parents=True)
    np.savez_compressed(
        cache_path,
        embeddings=np.ones((4, 3), dtype=np.float32),
        example_ids=np.asarray([row["example_id"] for row in examples]),
    )
    metadata = {
        "schema_version": tournament.CACHE_SCHEMA_VERSION,
        "model_identifier": "BAAI/bge-small-en-v1.5",
        "development_dataset_sha256": "d" * 64,
        "ordered_example_ids_sha256": tournament.ordered_ids_sha256(examples),
        "example_count": 4,
        "dimensions": 3,
        "cache_file_sha256": tournament.sha256_file(cache_path),
        "label_data_used": False,
        "fine_tuning_performed": False,
    }
    (tmp_path / spec["cache_metadata_path"]).write_bytes(
        tournament.stable_json_bytes(metadata)
    )
    assert tournament.validate_embedding_cache(local_config, examples).shape == (4, 3)
    metadata["ordered_example_ids_sha256"] = "bad"
    (tmp_path / spec["cache_metadata_path"]).write_bytes(
        tournament.stable_json_bytes(metadata)
    )
    with pytest.raises(ValueError, match="ordered_example_ids_sha256"):
        tournament.validate_embedding_cache(local_config, examples)


def test_embedding_preparation_does_not_use_labels() -> None:
    source = inspect.getsource(tournament.prepare_embeddings)
    assert '["intent"]' not in source
    assert '["risk"]' not in source
    assert "passage_embed" in source
    assert "BAAI/bge-small-en-v1.5" not in source  # read only from frozen config


def test_report_contract_has_no_final_evaluation_or_runtime_integration(
    config: dict[str, Any],
) -> None:
    assert config["report"] == {
        "path": "data/evals/v2/ml/v2c3_tournament_report.json",
        "persist_raw_text": False,
        "persist_oof_predictions": False,
        "final_lockbox_used": False,
        "challenge_set_used": False,
        "cfpb_used": False,
        "hyperparameter_search_performed": False,
    }
    report_source = inspect.getsource(tournament.run_tournament)
    for expected in (
        '"final_lockbox_used": False',
        '"challenge_set_used": False',
        '"cfpb_used": False',
        '"hyperparameter_search_performed": False',
        '"final_test_metric_reported": False',
        '"runtime_integration_performed": False',
    ):
        assert expected in report_source
