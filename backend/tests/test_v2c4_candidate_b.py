from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scripts import (
    evaluate_v2c4_candidate_b as evaluator,
)
from scripts import (
    train_v2c4_candidate_a as candidate_a,
)
from scripts import (
    train_v2c4_candidate_b as candidate_b,
)

ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = ROOT / "scripts/train_v2c4_candidate_b.py"
EVALUATION_SCRIPT = ROOT / "scripts/evaluate_v2c4_candidate_b.py"


@pytest.fixture(scope="module")
def config() -> dict[str, Any]:
    return candidate_b.load_config()


def test_config_matches_frozen_hierarchical_recipe(config: dict[str, Any]) -> None:
    assert config["candidate_id"] == "v2c4_candidate_b_hierarchical"
    assert config["architecture"] == "hierarchical"
    assert config["representation"] == {
        "family": "frozen_local_sentence_embedding",
        "model_identifier": "BAAI/bge-small-en-v1.5",
        "fastembed_method": "passage_embed",
        "dimensions": 384,
        "fine_tuning": False,
        "external_api": False,
    }
    assert config["classifier"]["parameters"] == (
        candidate_a.EXPECTED_CLASSIFIER_PARAMETERS
    )
    assert config["classifier"]["parameters"]["C"] == 4.0
    assert config["classifier"]["parameters"]["class_weight"] == "balanced"
    assert all(value is False for value in config["prohibited_options"].values())
    assert config["embedding_strategy"]["new_training_embeddings_required"] is False
    assert config["execution_status"] == {
        "training_performed": False,
        "selection_probe_evaluated": False,
        "final_holdout_accessed": False,
        "threshold_tuning_performed": False,
    }


def test_config_validator_rejects_recipe_and_hash_changes(
    config: dict[str, Any],
) -> None:
    changed_recipe = deepcopy(config)
    changed_recipe["classifier"]["parameters"]["C"] = 1.0
    with pytest.raises(ValueError, match="classifier changed"):
        candidate_b.validate_config(changed_recipe)

    changed_hash = deepcopy(config)
    changed_hash["frozen_inputs"]["v2c4_selection_probe"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="frozen input paths or hashes changed"):
        candidate_b.validate_config(changed_hash)


def test_hierarchy_mappings_cover_every_intent_once(config: dict[str, Any]) -> None:
    hierarchy = config["hierarchy"]
    assert tuple(hierarchy["stage_1"]["classes"]) == (
        "supported_current_request",
        "unsupported_or_uncertain",
    )
    assert tuple(hierarchy["stage_2"]["classes"]) == (
        "protected_write",
        "nonprotected",
    )
    assert tuple(hierarchy["stage_3a"]["classes"]) == (
        "create_dispute",
        "freeze_card",
    )
    assert tuple(hierarchy["stage_3b"]["classes"]) == (
        "account_balance",
        "card_status",
        "escalation",
        "informational_policy",
        "recent_transactions",
        "transaction_details",
    )
    terminal_groups = [
        hierarchy["stage_1"]["unsupported_intents"],
        hierarchy["stage_2"]["protected_intents"],
        hierarchy["stage_2"]["nonprotected_intents"],
    ]
    flattened = [intent for group in terminal_groups for intent in group]
    assert len(flattened) == 9
    assert len(set(flattened)) == 9
    assert set(flattened) == set(candidate_a.EXPECTED_INTENTS)


def test_stage_training_subsets_use_gold_mappings() -> None:
    targets = candidate_b.stage_targets(candidate_a.EXPECTED_INTENTS)
    assert len(targets["stage_1"][0]) == 9
    assert len(targets["stage_2"][0]) == 8
    assert len(targets["stage_3a"][0]) == 2
    assert len(targets["stage_3b"][0]) == 6
    assert set(targets["stage_1"][1]) == {
        "supported_current_request",
        "unsupported_or_uncertain",
    }
    assert set(targets["stage_2"][1]) == {"protected_write", "nonprotected"}
    assert set(targets["stage_3a"][1]) == set(candidate_b.PROTECTED_INTENTS)
    assert set(targets["stage_3b"][1]) == set(candidate_b.NONPROTECTED_INTENTS)


def test_training_corpus_and_stage_counts_are_frozen(config: dict[str, Any]) -> None:
    corpus = config["training_corpus"]
    assert corpus["v2c3_development_expected_count"] == 8198
    assert corpus["augmentation_expected_count"] == 360
    assert corpus["combined_expected_count"] == 8558
    assert corpus["expected_stage_counts"] == {
        "stage_1": 8558,
        "stage_2": 2047,
        "stage_3a": 234,
        "stage_3b": 1813,
    }
    assert corpus["same_as_candidate_a"] is True
    assert corpus["additional_data_allowed"] is False


@pytest.mark.parametrize(
    "role",
    [
        "v2c4_model_selection_probe",
        "consumed_v2c3_safety_challenge",
        "consumed_v2c3_external_lockbox",
        "consumed_v2c3_external_regression",
        "v2c4_final_safety_holdout",
    ],
)
def test_candidate_b_reuses_candidate_a_forbidden_role_guards(role: str) -> None:
    config_a = candidate_a.load_config()
    row = {
        "data_role": role,
        "source_id": "synthetic",
        "source_split": "train",
        "original_split": "train",
    }
    with pytest.raises(ValueError, match="forbidden training data role"):
        candidate_a.validate_training_role(row, config_a)


@pytest.mark.parametrize(
    ("source_id", "source_split"),
    [
        ("cfpb-complaint", "train"),
        ("banking77", "test"),
        ("clinc150", "test"),
        ("v2c3_safety_challenge", "challenge"),
        ("v2c3_external_lockbox", "locked"),
    ],
)
def test_candidate_b_reuses_candidate_a_source_guards(
    source_id: str, source_split: str
) -> None:
    config_a = candidate_a.load_config()
    row = {
        "data_role": "development",
        "source_id": source_id,
        "source_domain": source_id,
        "source_split": source_split,
        "original_split": source_split,
    }
    with pytest.raises(ValueError, match="forbidden"):
        candidate_a.validate_training_role(row, config_a)


class FixedModel:
    def __init__(self, stage: str) -> None:
        self.stage = stage
        self.calls: list[np.ndarray] = []

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        self.calls.append(embeddings)
        intent_indices = embeddings[:, 0].astype(int)
        if self.stage == "stage_1":
            return np.where(
                intent_indices == 8,
                candidate_b.UNSUPPORTED,
                candidate_b.SUPPORTED,
            )
        if self.stage == "stage_2":
            return np.where(
                np.isin(intent_indices, (2, 4)),
                candidate_b.PROTECTED,
                candidate_b.NONPROTECTED,
            )
        return np.asarray(
            [candidate_a.EXPECTED_INTENTS[index] for index in intent_indices]
        )


def fixed_models() -> dict[str, FixedModel]:
    return {stage: FixedModel(stage) for stage in candidate_b.STAGE_CLASSES}


def test_hierarchical_inference_uses_predicted_routes_and_short_circuits() -> None:
    models = fixed_models()
    embeddings = np.asarray([[8.0], [2.0], [4.0], [0.0], [7.0]])
    predictions, counts = candidate_b.hierarchical_predict(models, embeddings)

    assert predictions.tolist() == [
        "unsupported_or_uncertain",
        "create_dispute",
        "freeze_card",
        "account_balance",
        "transaction_details",
    ]
    assert counts == {
        "stage_1_executions": 5,
        "stage_2_executions": 4,
        "stage_3a_executions": 2,
        "stage_3b_executions": 2,
        "stage_1_unsupported_short_circuits": 1,
    }
    assert len(models["stage_2"].calls[0]) == 4
    assert len(models["stage_3a"].calls[0]) == 2
    assert len(models["stage_3b"].calls[0]) == 2


def test_hierarchical_inference_rejects_unknown_predicted_routes() -> None:
    class UnknownStageOne:
        def predict(self, embeddings: np.ndarray) -> np.ndarray:
            return np.asarray(["not_a_frozen_route"] * len(embeddings))

    models = fixed_models()
    models["stage_1"] = UnknownStageOne()  # type: ignore[assignment]
    with pytest.raises(ValueError, match="Stage 1 produced an unknown route label"):
        candidate_b.hierarchical_predict(models, np.asarray([[0.0]]))


def test_group_cv_uses_real_hierarchical_oof_inference(
    config: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    records: list[dict[str, str]] = []
    rows: list[list[float]] = []
    for intent_index, intent in enumerate(candidate_a.EXPECTED_INTENTS):
        for group_index in range(10):
            records.append(
                {
                    "intent": intent,
                    "group_id": f"{intent}:group:{group_index}",
                }
            )
            rows.append([float(intent_index)])
    embeddings = np.asarray(rows, dtype=np.float32)
    monkeypatch.setattr(
        candidate_b,
        "fit_hierarchy",
        lambda config, embeddings, intents: (fixed_models(), {}),
    )
    result = candidate_b.supporting_cv(config, records, embeddings)

    assert result["n_splits"] == 5
    assert result["random_state"] == 20260928
    assert result["real_hierarchical_oof_inference"] is True
    assert result["gold_upstream_routing_used_for_final_predictions"] is False
    assert result["every_record_received_one_final_prediction"] is True
    assert result["group_leakage_detected"] is False
    assert sum(row["validation_count"] for row in result["fold_results"]) == 90
    assert all(
        row["inference_routing"] == "predicted_upstream_labels_only"
        for row in result["fold_results"]
    )
    assert all(row["group_overlap_count"] == 0 for row in result["fold_results"])
    assert result["pooled_final_9_intent_metrics"]["example_count"] == 90


def passing_gates() -> dict[str, Any]:
    return {
        "all_required": True,
        "all_passed": True,
        "gates": {
            "protected_write_false_positive_rate": {"passed": True},
            "protected_write_recall": {"passed": True},
            "unsupported_or_uncertain_recall": {"passed": True},
        },
    }


def test_selection_requires_every_mandatory_criterion() -> None:
    selected = evaluator.selection_decision(passing_gates(), {"passed": True})
    assert selected["candidate_b_eligible"] is True
    assert selected["decision"] == "selected_v2c4_development_candidate"
    assert selected["selected_candidate"] == "v2c4_candidate_b_hierarchical"
    assert selected["candidate_c_created_or_allowed"] is False

    for gate_name in passing_gates()["gates"]:
        failed_gates = passing_gates()
        failed_gates["all_passed"] = False
        failed_gates["gates"][gate_name]["passed"] = False
        stopped = evaluator.selection_decision(failed_gates, {"passed": True})
        assert stopped["candidate_b_eligible"] is False
        assert stopped["decision"] == "stop_for_human_review"
        assert stopped["selected_candidate"] is None
        assert gate_name in stopped["failed_mandatory_criteria"]

    regression = evaluator.selection_decision(passing_gates(), {"passed": False})
    assert regression["candidate_b_eligible"] is False
    assert "material_macro_f1_regression" in regression["failed_mandatory_criteria"]

    inconsistent = passing_gates()
    inconsistent["gates"]["protected_write_recall"]["passed"] = False
    with pytest.raises(ValueError, match="summary is inconsistent"):
        evaluator.selection_decision(inconsistent, {"passed": True})


def test_fairness_and_metric_implementation_are_shared() -> None:
    assert evaluator.train_a.calculate_metrics is candidate_a.calculate_metrics
    embeddings = np.zeros((2, 384), dtype=np.float32)
    models = fixed_models()
    models["stage_1"].predict(embeddings)
    assert models["stage_1"].calls[0] is embeddings
    evaluation_source = EVALUATION_SCRIPT.read_text(encoding="utf-8")
    assert ".fit(" not in evaluation_source


def test_baseline_candidate_a_and_b_receive_same_embedding_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embeddings = np.zeros((2, 384), dtype=np.float32)
    seen: list[np.ndarray] = []

    def fake_direct_predict(
        model: object, matrix: np.ndarray
    ) -> tuple[np.ndarray, float]:
        seen.append(matrix)
        return np.asarray(["account_balance", "card_status"]), 0.01

    def fake_hierarchical_predict(
        models: object, matrix: np.ndarray
    ) -> tuple[np.ndarray, dict[str, int]]:
        seen.append(matrix)
        return np.asarray(["account_balance", "card_status"]), {}

    monkeypatch.setattr(
        evaluator.evaluate_a, "predict_with_latency", fake_direct_predict
    )
    monkeypatch.setattr(
        evaluator.train_b, "hierarchical_predict", fake_hierarchical_predict
    )
    evaluator.predict_all_on_shared_embeddings(
        {"baseline_classifier": object(), "candidate_classifier": object()},
        {},
        embeddings,
    )
    assert len(seen) == 3
    assert all(matrix is embeddings for matrix in seen)


def test_frozen_comparator_metrics_must_reproduce() -> None:
    report = {
        "baseline_metrics": {"macro_f1": 0.69},
        "candidate_a_metrics": {"macro_f1": 0.81},
    }
    evaluator.validate_frozen_comparator_metrics(
        report,
        {"macro_f1": 0.69},
        {"macro_f1": 0.81},
    )
    with pytest.raises(ValueError, match="Candidate A metrics differ"):
        evaluator.validate_frozen_comparator_metrics(
            report,
            {"macro_f1": 0.69},
            {"macro_f1": 0.80},
        )


def test_candidate_a_inputs_are_hash_pinned(config: dict[str, Any]) -> None:
    frozen = config["frozen_inputs"]
    assert frozen["parent_intervention_plan"]["sha256"] == (
        "4a5ed357ab8ae2fc66dd5459f59883ef20bd9b6f9996bad7f4b8f8f2b245e5ff"
    )
    assert frozen["candidate_a_artifact"]["sha256"] == (
        "7384c6662f11ddebdc6225f5dfdf2397d51a51a5e318cd116b585d59ed8c7c2b"
    )
    assert frozen["candidate_a_training_report"]["sha256"] == (
        "77eab72edb3bb86fd57d19af26df9d5a33e6577669e9f17d6e8e77a930b2eb6d"
    )
    assert frozen["candidate_a_evaluation_report"]["sha256"] == (
        "3871c72cbe521bf761f5a8bd7a5f0ff95effa1411dfa414f2402df5c0b6d263f"
    )
    assert frozen["v2c3_development"]["sha256"] == (
        "3783042b3656e3170c2bf001a0fe059d65cad415ed259365d095a1e19fe689ca"
    )
    assert frozen["v2c4_training_augmentation"]["sha256"] == (
        "90da31583b0a57d52a55721e223499a080660235837545d517ce9e65c390bffc"
    )
    assert frozen["v2c4_selection_probe"]["sha256"] == (
        "25c62797825b58371f1260f631e9a35bdf34c4cf1bedc895120218a5bb63e16e"
    )


def test_stage_two_keeps_semantic_and_fitted_class_orders_distinct() -> None:
    assert candidate_b.STAGE_CLASSES["stage_2"] == (
        "protected_write",
        "nonprotected",
    )
    assert candidate_b.FITTED_STAGE_CLASSES["stage_2"] == (
        "nonprotected",
        "protected_write",
    )


def test_sealed_holdout_paths_are_absent_and_role_is_rejected() -> None:
    for script_path in (TRAIN_SCRIPT, EVALUATION_SCRIPT):
        source = script_path.read_text(encoding="utf-8")
        assert "v2c4_safety_holdout.json" not in source
        assert "v2c4_safety_holdout_seed.json" not in source
    config_a = candidate_a.load_config()
    with pytest.raises(ValueError, match="forbidden training data role"):
        candidate_a.validate_training_role(
            {
                "data_role": "v2c4_final_safety_holdout",
                "source_id": "sealed",
            },
            config_a,
        )


def test_report_structure_preserves_development_boundary() -> None:
    report = dict.fromkeys(evaluator.EXPECTED_REPORT_KEYS)
    report.update(
        {
            "analysis_role": "v2c4_development_model_selection",
            "final_acceptance_evidence": False,
            "final_holdout_accessed": False,
            "final_improvement_claimed": False,
        }
    )
    assert evaluator.report_structure_is_deterministic(report) is True
    report["final_acceptance_evidence"] = True
    assert evaluator.report_structure_is_deterministic(report) is False
