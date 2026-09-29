from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import pytest
from sklearn.pipeline import FeatureUnion, Pipeline

from scripts import run_v2c3_bounded_tuning as tuning
from scripts import run_v2c3_model_tournament as tournament

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "data/evals/v2/ml/v2c3_tuning_config.json"
RUNNER_PATH = ROOT / "scripts/run_v2c3_bounded_tuning.py"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def config() -> dict[str, Any]:
    return load_json(CONFIG_PATH)


@pytest.fixture
def step_4_config() -> dict[str, Any]:
    return load_json(ROOT / "data/evals/v2/ml/v2c3_tournament_config.json")


def test_config_freezes_exact_three_finalists(config: dict[str, Any]) -> None:
    tuning.validate_config(config)
    assert config["frozen_finalist_ids"] == [
        "direct_v2c1_tfidf_balanced_svc",
        "direct_bge_balanced_svc",
        "direct_lsa_balanced_svc",
    ]
    assert config["step_4_winner"] == "direct_v2c1_tfidf_balanced_svc"


def test_config_freezes_exact_15_variants(config: dict[str, Any]) -> None:
    assert len(config["variants"]) == 15
    assert [row["id"] for row in config["variants"]] == tuning.EXPECTED_VARIANT_IDS
    assert len({row["id"] for row in config["variants"]}) == 15
    assert config["search_scope"]["cartesian_grid_generation_allowed"] is False
    assert config["hyperparameter_search_type"] == "bounded_predeclared_manual_search"


def test_exact_tfidf_and_bge_c_values(config: dict[str, Any]) -> None:
    by_family = {
        family: [row for row in config["variants"] if row["family"] == family]
        for family in ("tfidf_svc", "bge_svc")
    }
    assert [row["parameters"]["C"] for row in by_family["tfidf_svc"]] == [
        0.5,
        0.75,
        1.0,
        1.5,
        2.0,
    ]
    assert [row["parameters"]["C"] for row in by_family["bge_svc"]] == [
        0.25,
        0.5,
        1.0,
        2.0,
        4.0,
    ]


def test_exact_lsa_component_and_c_pairs(config: dict[str, Any]) -> None:
    variants = [row for row in config["variants"] if row["family"] == "lsa_svc"]
    assert [
        (row["parameters"]["n_components"], row["parameters"]["C"])
        for row in variants
    ] == [(128, 1.0), (256, 0.5), (256, 1.0), (256, 2.0), (384, 1.0)]


def test_no_extra_models_representations_or_architectures(config: dict[str, Any]) -> None:
    assert {row["architecture"] for row in config["variants"]} == {"direct_9_way"}
    assert {row["representation"] for row in config["variants"]} == {
        "v2c1_word_char_tfidf",
        "bge_small_en_v1_5",
        "tfidf_lsa",
    }
    assert config["search_scope"]["classifier_family"] == "LinearSVC"
    assert config["search_scope"]["class_weight"] == "balanced"
    assert config["search_scope"]["new_classifier_families_allowed"] is False
    assert config["search_scope"]["new_representation_families_allowed"] is False
    assert config["search_scope"]["architecture_changes_allowed"] is False


def test_all_frozen_input_hashes_match_tracked_artifacts(config: dict[str, Any]) -> None:
    for spec in config["frozen_inputs"].values():
        assert tournament.sha256_file(ROOT / spec["path"]) == spec["sha256"]


def test_same_frozen_fold_assignment_is_required(config: dict[str, Any]) -> None:
    fold_spec = config["frozen_inputs"]["cv_folds"]
    fold_artifact = load_json(ROOT / fold_spec["path"])
    assert fold_spec["assignment_sha256"] == (
        "71bb40c78ffbff5d33d8527c77725f784c5d81b37d6348eb1fac30de96e532f8"
    )
    assert fold_artifact["assignment_sha256"] == fold_spec["assignment_sha256"]
    assert fold_spec["n_splits"] == 5
    assert fold_spec["example_count"] == 8198
    assert fold_spec["new_fold_generation_allowed"] is False


def test_runner_cannot_recompute_or_write_folds() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert "from sklearn.model_selection import StratifiedGroupKFold" not in source
    assert "StratifiedGroupKFold(" not in source
    assert "prepare_folds" not in source
    assert "fold_assignments_payload" not in source
    assert 'choices=("preflight", "run")' in source
    load_source = inspect.getsource(tuning.load_and_validate_inputs)
    assert "validate_fold_artifact" in load_source
    assert "write_json" not in load_source


def test_preflight_performs_no_fitting_or_inference() -> None:
    source = inspect.getsource(tuning.preflight)
    assert ".fit(" not in source
    assert ".predict(" not in source
    assert "fit_candidate" not in source
    assert "run_candidate_cv" not in source
    assert "measure_full_development_candidate" not in source


def test_development_is_the_only_dataset_input(config: dict[str, Any]) -> None:
    assert config["frozen_inputs"]["development_dataset"] == {
        "path": "data/evals/v2/ml/v2c3_development_dataset.json",
        "sha256": "3783042b3656e3170c2bf001a0fe059d65cad415ed259365d095a1e19fe689ca",
        "example_count": 8198,
    }
    allowed = config["allowed_read_paths"]
    assert sum("development_dataset.json" in path for path in allowed) == 1
    assert all("test.csv" not in path for path in allowed)
    assert all("challenge_set" not in path for path in allowed)
    assert all("fresh_lockbox" not in path for path in allowed)
    assert all("cfpb" not in path for path in allowed)


def test_runner_has_no_prohibited_dataset_reads() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    for prohibited in (
        "v2c3_fresh_lockbox_manifest.json",
        "v2c3_challenge_set.json",
        "cfpb_narratives",
        "banking77/test.csv",
        "data_full.json",
        "intent_risk_dataset.json",
        "locked_test",
    ):
        assert prohibited not in source


def test_bge_cache_is_required_and_never_regenerated(config: dict[str, Any]) -> None:
    policy = config["embedding_cache_policy"]
    assert policy["existing_step_4_cache_required"] is True
    assert policy["automatic_regeneration_allowed"] is False
    assert policy["fine_tuning_allowed"] is False
    assert len(policy["cache_file_sha256"]) == 64
    assert len(policy["cache_metadata_sha256"]) == 64
    source = inspect.getsource(tuning.load_and_validate_inputs)
    assert "validate_embedding_cache" in source
    assert "cache_file_sha256" in source
    assert "cache_metadata_sha256" in source
    runner_source = RUNNER_PATH.read_text(encoding="utf-8")
    assert "prepare_embeddings" not in runner_source
    assert "passage_embed" not in runner_source


def test_runtime_definitions_change_only_predeclared_parameters(
    config: dict[str, Any], step_4_config: dict[str, Any]
) -> None:
    base_classifier = step_4_config["classifiers"]["linear_svc_balanced"]["parameters"]
    for variant in config["variants"]:
        candidate, runtime_config = tuning.runtime_definition(variant, step_4_config)
        parameters = runtime_config["classifiers"]["step5_balanced_linear_svc"][
            "parameters"
        ]
        assert parameters["C"] == variant["parameters"]["C"]
        assert {key: value for key, value in parameters.items() if key != "C"} == {
            key: value for key, value in base_classifier.items() if key != "C"
        }
        assert candidate["architecture"] == "direct_9_way"
        if variant["family"] == "lsa_svc":
            assert candidate["representation"] == "tfidf_lsa"
            lsa = runtime_config["representations"]["tfidf_lsa"]
            assert lsa["truncated_svd"]["n_components"] == variant["parameters"][
                "n_components"
            ]
            assert lsa["truncated_svd"]["algorithm"] == "randomized"
            assert lsa["truncated_svd"]["n_iter"] == 5
            assert lsa["truncated_svd"]["random_state"] == 20260928


def test_tfidf_and_lsa_remain_fold_local(
    config: dict[str, Any], step_4_config: dict[str, Any]
) -> None:
    tfidf_variant = config["variants"][0]
    tfidf_candidate, tfidf_runtime = tuning.runtime_definition(
        tfidf_variant, step_4_config
    )
    tfidf_representation = tournament.build_representation(
        tfidf_candidate["representation"], tfidf_runtime
    )
    assert isinstance(tfidf_representation, FeatureUnion)

    lsa_variant = config["variants"][10]
    lsa_candidate, lsa_runtime = tuning.runtime_definition(lsa_variant, step_4_config)
    lsa_representation = tournament.build_representation(
        lsa_candidate["representation"], lsa_runtime
    )
    assert isinstance(lsa_representation, Pipeline)
    assert list(lsa_representation.named_steps) == ["tfidf", "truncated_svd", "normalize"]
    assert isinstance(lsa_representation.named_steps["tfidf"], FeatureUnion)
    evaluate_source = inspect.getsource(tuning.evaluate_variant)
    assert "run_candidate_cv" in evaluate_source


def test_metric_and_safety_implementations_are_reused_unchanged(
    config: dict[str, Any], step_4_config: dict[str, Any]
) -> None:
    assert config["metrics"]["definitions_unchanged_from_step_4"] is True
    assert config["metrics"]["implementation_source"] == (
        "scripts/run_v2c3_model_tournament.py"
    )
    gold = [
        tournament.UNSUPPORTED,
        tournament.UNSUPPORTED,
        "freeze_card",
        "create_dispute",
        "account_balance",
    ]
    predicted = [
        tournament.UNSUPPORTED,
        "account_balance",
        "create_dispute",
        "create_dispute",
        "freeze_card",
    ]
    metrics = tournament.safety_metrics(gold, predicted, step_4_config)
    assert metrics["false_supported_rate"] == pytest.approx(0.5)
    assert metrics["protected_write_recall"] == pytest.approx(0.5)
    assert metrics["protected_write_false_positive_rate"] == pytest.approx(1 / 3)


def test_safety_gates_and_ranking_match_step_4(
    config: dict[str, Any], step_4_config: dict[str, Any]
) -> None:
    tuning_gates = {
        key: config["safety_gates"][key]
        for key in (
            "protected_write_false_positive_rate_maximum",
            "protected_write_recall_minimum",
            "unsupported_or_uncertain_recall_minimum",
            "all_gates_required",
            "relaxation_after_results_allowed",
        )
    }
    assert tuning_gates == step_4_config["safety_gates"]
    assert config["ranking"]["practical_tie_absolute_difference_below"] == 0.005
    assert config["ranking"]["tie_break_order"] == step_4_config["ranking"][
        "tie_break_order"
    ]


def ranking_result(
    variant_id: str,
    family: str,
    macro_f1: float,
    *,
    false_supported: float = 0.1,
    eligible: bool = True,
) -> dict[str, Any]:
    return {
        "candidate_id": variant_id,
        "tuning_variant": {"family": family},
        "macro_f1_mean": macro_f1,
        "macro_f1_standard_deviation": 0.02,
        "definition": {"architecture": "direct_9_way"},
        "pooled_oof_metrics": {
            "false_supported_rate": false_supported,
            "unsupported_or_uncertain_recall": 0.9,
            "protected_write_false_positive_rate": 0.005,
        },
        "latency": {
            "primary_scope": "representation_and_classifier",
            "representation_and_classifier": {"p95_ms": 1.0},
        },
        "model_size": {"serialized_fitted_candidate_bytes": 100},
        "safety_gate_result": {"eligible": eligible},
    }


def test_deterministic_ranking_and_family_winners() -> None:
    results = [
        ranking_result("tfidf-high", "tfidf_svc", 0.91, false_supported=0.2),
        ranking_result("tfidf-tie", "tfidf_svc", 0.906, false_supported=0.05),
        ranking_result("bge", "bge_svc", 0.89),
        ranking_result("lsa", "lsa_svc", 0.88),
        ranking_result("ineligible", "bge_svc", 0.99, eligible=False),
    ]
    assert [row["candidate_id"] for row in tuning.rank_results(results)] == [
        "tfidf-tie",
        "tfidf-high",
        "bge",
        "lsa",
    ]
    assert tuning.best_by_family(results) == {
        "tfidf_svc": "tfidf-tie",
        "bge_svc": "bge",
        "lsa_svc": "lsa",
    }


def test_no_broad_search_runtime_or_final_evaluation(config: dict[str, Any]) -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert "GridSearchCV" not in source
    assert "RandomizedSearchCV" not in source
    assert "ParameterGrid" not in source
    assert "backend.app.main" not in source
    assert "runtime routing" not in source.lower()
    assert config["tuning_performed"] is False
    assert config["final_evaluation_performed"] is False


def test_report_flags_are_explicit(config: dict[str, Any]) -> None:
    assert config["report"] == {
        "path": "data/evals/v2/ml/v2c3_tuning_report.json",
        "persist_raw_text": False,
        "persist_oof_predictions": False,
        "final_lockbox_used": False,
        "challenge_set_used": False,
        "cfpb_used": False,
        "new_fold_generation_performed": False,
        "embedding_fine_tuning_performed": False,
        "broad_search_performed": False,
        "bounded_hyperparameter_search_performed": True,
    }
    report_source = inspect.getsource(tuning.run_tuning)
    for expected in (
        '"final_lockbox_used": False',
        '"challenge_set_used": False',
        '"cfpb_used": False',
        '"new_fold_generation_performed": False',
        '"embedding_fine_tuning_performed": False',
        '"broad_search_performed": False',
        '"bounded_hyperparameter_search_performed": True',
        '"final_evaluation_performed": False',
        '"runtime_integration_performed": False',
        '"final_model_artifact_frozen": False',
    ):
        assert expected in report_source
