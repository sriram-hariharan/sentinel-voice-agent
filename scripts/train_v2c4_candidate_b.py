"""Train the frozen V2-C4 Candidate B hierarchy with supporting CV."""

from __future__ import annotations

import argparse
import json
import os
import platform
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from importlib import metadata
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import LinearSVC

try:
    from scripts import train_v2c4_candidate_a as candidate_a
except ModuleNotFoundError:  # Direct execution from the scripts directory.
    import train_v2c4_candidate_a as candidate_a


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY_ROOT / "data/evals/v2/ml/v2c4_candidate_b_config.json"
SUPPORTED = "supported_current_request"
UNSUPPORTED = "unsupported_or_uncertain"
PROTECTED = "protected_write"
NONPROTECTED = "nonprotected"
PROTECTED_INTENTS = ("create_dispute", "freeze_card")
NONPROTECTED_INTENTS = (
    "account_balance",
    "card_status",
    "escalation",
    "informational_policy",
    "recent_transactions",
    "transaction_details",
)
STAGE_CLASSES = {
    "stage_1": (SUPPORTED, UNSUPPORTED),
    "stage_2": (PROTECTED, NONPROTECTED),
    "stage_3a": PROTECTED_INTENTS,
    "stage_3b": NONPROTECTED_INTENTS,
}
FITTED_STAGE_CLASSES = {
    stage: tuple(sorted(labels)) for stage, labels in STAGE_CLASSES.items()
}
EXPECTED_FROZEN_INPUTS = {
    "parent_intervention_plan": {
        "path": "data/evals/v2/ml/v2c4_intervention_plan.json",
        "sha256": "4a5ed357ab8ae2fc66dd5459f59883ef20bd9b6f9996bad7f4b8f8f2b245e5ff",
    },
    "candidate_a_config": {
        "path": "data/evals/v2/ml/v2c4_candidate_a_config.json",
        "sha256": "00e584f1a702a16b29537001f8fe159136193ed71224ead81f1a0d5dcd39931d",
    },
    "candidate_a_training_report": {
        "path": "data/evals/v2/ml/v2c4_candidate_a_training_report.json",
        "sha256": "77eab72edb3bb86fd57d19af26df9d5a33e6577669e9f17d6e8e77a930b2eb6d",
    },
    "candidate_a_evaluation_report": {
        "path": "data/evals/v2/ml/v2c4_candidate_a_evaluation_report.json",
        "sha256": "3871c72cbe521bf761f5a8bd7a5f0ff95effa1411dfa414f2402df5c0b6d263f",
    },
    "candidate_a_artifact": {
        "path": "artifacts/v2/classifier/v2c4_candidate_a_classifier.joblib",
        "sha256": "7384c6662f11ddebdc6225f5dfdf2397d51a51a5e318cd116b585d59ed8c7c2b",
    },
    "v2c3_development": {
        "path": "data/evals/v2/ml/v2c3_development_dataset.json",
        "sha256": "3783042b3656e3170c2bf001a0fe059d65cad415ed259365d095a1e19fe689ca",
    },
    "v2c4_training_augmentation": {
        "path": "data/evals/v2/ml/v2c4_training_augmentation.json",
        "sha256": "90da31583b0a57d52a55721e223499a080660235837545d517ce9e65c390bffc",
    },
    "v2c4_selection_probe": {
        "path": "data/evals/v2/ml/v2c4_selection_probe.json",
        "sha256": "25c62797825b58371f1260f631e9a35bdf34c4cf1bedc895120218a5bb63e16e",
    },
    "v2c3_development_embedding_cache": {
        "path": "data/evals/v2/ml/local/v2c3_bge_small_en_v1_5.npz",
        "sha256": "eddaa00ddec7335dd63a49d1650e274d993603f8d52e6b48016089f79514ba89",
    },
    "v2c3_development_embedding_cache_metadata": {
        "path": "data/evals/v2/ml/local/v2c3_bge_small_en_v1_5.metadata.json",
        "sha256": "a8477d65c231faebb1edc29c8ded60384d800df991c655a7411e6d6e619eaea3",
    },
    "candidate_a_augmentation_embedding_cache": {
        "path": "data/evals/v2/ml/local/v2c4_candidate_a_augmentation_bge.npz",
        "sha256": "53ac1fc7b6349995fd4fd04e35d43f378d8c2dcecc60201832386908b695bbbb",
    },
    "candidate_a_augmentation_embedding_cache_metadata": {
        "path": (
            "data/evals/v2/ml/local/"
            "v2c4_candidate_a_augmentation_bge.metadata.json"
        ),
        "sha256": "362e09aa618d1d3e260c9bc757ccfc233bbda816d893618168f699c49b114aed",
    },
    "v2c3_baseline_artifact": {
        "path": "artifacts/v2/classifier/v2c3_final_classifier.joblib",
        "sha256": "0e03a4d8367933622a92920f14a7a4c5a2ffa74af0f53b7b0913f04849e4780c",
    },
}
EXPECTED_OUTPUTS = {
    "candidate_artifact": (
        "artifacts/v2/classifier/v2c4_candidate_b_classifier.joblib"
    ),
    "training_report": "data/evals/v2/ml/v2c4_candidate_b_training_report.json",
    "evaluation_report": (
        "data/evals/v2/ml/v2c4_candidate_b_evaluation_report.json"
    ),
}


def load_config() -> dict[str, Any]:
    config = candidate_a.read_json(CONFIG_PATH)
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != "v2c4-candidate-b-config.v1":
        raise ValueError("unexpected Candidate B config schema")
    if config.get("config_version") != "2026-09-29.v1":
        raise ValueError("unexpected Candidate B config version")
    if config.get("candidate_id") != "v2c4_candidate_b_hierarchical":
        raise ValueError("Candidate B identity changed")
    if config.get("architecture") != "hierarchical":
        raise ValueError("Candidate B architecture changed")
    if config.get("experiment_phase") != "V2-C4-Step-12":
        raise ValueError("Candidate B experiment phase changed")
    if config.get("analysis_role") != "v2c4_development_model_selection":
        raise ValueError("Candidate B analysis role changed")
    if config.get("final_acceptance_evidence") is not False:
        raise ValueError("Candidate B cannot provide final acceptance evidence")
    if config.get("advisory_routing_only") is not True:
        raise ValueError("Candidate B must remain advisory routing only")
    if config.get("runtime_authority") is not False:
        raise ValueError("Candidate B cannot have runtime authority")
    if config.get("random_seed") != 20260928:
        raise ValueError("Candidate B random seed changed")
    if tuple(config.get("intent_order", [])) != candidate_a.EXPECTED_INTENTS:
        raise ValueError("Candidate B intent order changed")
    if config.get("risk_by_intent") != candidate_a.EXPECTED_RISK_BY_INTENT:
        raise ValueError("Candidate B risk mapping changed")
    if config.get("representation") != {
        "family": "frozen_local_sentence_embedding",
        "model_identifier": "BAAI/bge-small-en-v1.5",
        "fastembed_method": "passage_embed",
        "dimensions": 384,
        "fine_tuning": False,
        "external_api": False,
    }:
        raise ValueError("Candidate B representation changed")
    if config.get("classifier") != {
        "class": "LinearSVC",
        "parameters": candidate_a.EXPECTED_CLASSIFIER_PARAMETERS,
    }:
        raise ValueError("Candidate B classifier changed")
    hierarchy = config["hierarchy"]
    if tuple(hierarchy["stage_1"]["classes"]) != STAGE_CLASSES["stage_1"]:
        raise ValueError("Stage 1 classes changed")
    if tuple(hierarchy["stage_2"]["classes"]) != STAGE_CLASSES["stage_2"]:
        raise ValueError("Stage 2 classes changed")
    if tuple(hierarchy["stage_3a"]["classes"]) != STAGE_CLASSES["stage_3a"]:
        raise ValueError("Stage 3A classes changed")
    if tuple(hierarchy["stage_3b"]["classes"]) != STAGE_CLASSES["stage_3b"]:
        raise ValueError("Stage 3B classes changed")
    if hierarchy["stage_1"]["input"] != "all_examples":
        raise ValueError("Stage 1 input changed")
    if hierarchy["stage_2"]["input"] != (
        "gold_supported_rows_for_training; stage_1_supported_rows_for_inference"
    ):
        raise ValueError("Stage 2 input or routing changed")
    if hierarchy["stage_3a"]["input"] != (
        "gold_protected_rows_for_training; stage_2_protected_rows_for_inference"
    ):
        raise ValueError("Stage 3A input or routing changed")
    if hierarchy["stage_3b"]["input"] != (
        "gold_supported_nonprotected_rows_for_training; "
        "stage_2_nonprotected_rows_for_inference"
    ):
        raise ValueError("Stage 3B input or routing changed")
    if tuple(hierarchy["stage_1"]["unsupported_intents"]) != (UNSUPPORTED,):
        raise ValueError("Stage 1 unsupported mapping changed")
    if tuple(hierarchy["stage_1"]["supported_intents"]) != tuple(
        intent for intent in candidate_a.EXPECTED_INTENTS if intent != UNSUPPORTED
    ):
        raise ValueError("Stage 1 supported mapping changed")
    if tuple(hierarchy["stage_2"]["protected_intents"]) != PROTECTED_INTENTS:
        raise ValueError("Stage 2 protected mapping changed")
    if tuple(hierarchy["stage_2"]["nonprotected_intents"]) != NONPROTECTED_INTENTS:
        raise ValueError("Stage 2 nonprotected mapping changed")
    mapped = (
        set(hierarchy["stage_1"]["unsupported_intents"])
        | set(hierarchy["stage_2"]["protected_intents"])
        | set(hierarchy["stage_2"]["nonprotected_intents"])
    )
    mapped_sequence = (
        list(hierarchy["stage_1"]["unsupported_intents"])
        + list(hierarchy["stage_2"]["protected_intents"])
        + list(hierarchy["stage_2"]["nonprotected_intents"])
    )
    if (
        mapped != set(candidate_a.EXPECTED_INTENTS)
        or len(mapped_sequence) != len(set(mapped_sequence))
    ):
        raise ValueError("hierarchy does not map every intent exactly once")
    if config["training_corpus"] != {
        "v2c3_development_expected_count": 8198,
        "augmentation_expected_count": 360,
        "combined_expected_count": 8558,
        "expected_stage_counts": {
            "stage_1": 8558,
            "stage_2": 2047,
            "stage_3a": 234,
            "stage_3b": 1813,
        },
        "same_as_candidate_a": True,
        "additional_data_allowed": False,
    }:
        raise ValueError("Candidate B training corpus changed")
    if config["embedding_strategy"] != {
        "strategy": "reuse_verified_candidate_a_component_caches",
        "v2c3_development_cache_reused": True,
        "candidate_a_augmentation_cache_reused": True,
        "new_training_embeddings_required": False,
        "shared_probe_embedding_for_all_models": True,
        "query_embed_allowed": False,
    }:
        raise ValueError("Candidate B embedding strategy changed")
    if config["safety_gates"] != {
        "protected_write_false_positive_rate_maximum": 0.01,
        "protected_write_recall_minimum": 0.8,
        "unsupported_or_uncertain_recall_minimum": 0.8,
        "all_gates_required": True,
    }:
        raise ValueError("Candidate B safety gates changed")
    if config["supporting_cv"] != {
        "role": "supporting_development_evidence",
        "primary_selection_evidence": False,
        "method": "StratifiedGroupKFold",
        "n_splits": 5,
        "shuffle": True,
        "random_state": 20260928,
        "group_field": "group_id",
        "real_hierarchical_inference_required": True,
        "hyperparameter_tuning_allowed": False,
    }:
        raise ValueError("Candidate B supporting CV changed")
    if config["selection_rule"] != {
        "candidate_a_eligible": False,
        "candidate_b_selected_only_if_all_safety_gates_pass": True,
        "candidate_b_selected_only_if_no_material_macro_f1_regression": True,
        "eligible_wording": "selected_v2c4_development_candidate",
        "failure_action": "stop_for_human_review",
        "candidate_c_allowed": False,
    }:
        raise ValueError("Candidate B selection rule changed")
    if config["macro_f1_regression"][
        "material_regression_if_delta_less_than"
    ] != -0.01:
        raise ValueError("Candidate B macro-F1 regression threshold changed")
    if config["macro_f1_regression"].get("baseline") != "frozen_v2c3":
        raise ValueError("Candidate B macro-F1 baseline changed")
    if config["selection_probe"] != {
        "data_role": "v2c4_model_selection_probe",
        "expected_count": 270,
        "examples_per_intent": 30,
        "primary_model_selection_evidence": True,
        "training_eligible": False,
        "model_selection_eligible": True,
        "threshold_selection_eligible": False,
        "final_acceptance_evidence": False,
    }:
        raise ValueError("Candidate B selection probe changed")
    if config["prohibited_options"] != {
        "hyperparameter_search": False,
        "threshold_tuning": False,
        "probability_calibration": False,
        "abstention_change": False,
        "embedding_change": False,
        "transformer_fine_tuning": False,
        "llm_classifier": False,
    }:
        raise ValueError("Candidate B prohibited options changed")
    if config.get("frozen_inputs") != EXPECTED_FROZEN_INPUTS:
        raise ValueError("Candidate B frozen input paths or hashes changed")
    if config.get("outputs") != EXPECTED_OUTPUTS:
        raise ValueError("Candidate B output paths changed")
    if config["execution_status"] != {
        "training_performed": False,
        "selection_probe_evaluated": False,
        "final_holdout_accessed": False,
        "threshold_tuning_performed": False,
    }:
        raise ValueError("Candidate B config must remain pre-execution")


def validate_candidate_a_trigger(paths: Mapping[str, Path]) -> dict[str, Any]:
    report = candidate_a.read_json(paths["candidate_a_evaluation_report"])
    trigger = report.get("candidate_b_trigger", {})
    gates = report.get("safety_gate_results", {})
    if trigger.get("candidate_b_triggered") is not True:
        raise ValueError("Candidate A did not trigger Candidate B")
    if gates.get("all_required") is not True or gates.get("all_passed") is not False:
        raise ValueError("Candidate A gate outcome does not permit Candidate B")
    failed = {
        name for name, result in gates.get("gates", {}).items() if not result["passed"]
    }
    if failed != {
        "protected_write_false_positive_rate",
        "unsupported_or_uncertain_recall",
    }:
        raise ValueError("Candidate A failure set changed")
    if report.get("final_acceptance_evidence") is not False:
        raise ValueError("Candidate A report crossed the development boundary")
    if report.get("final_holdout_accessed") is not False:
        raise ValueError("Candidate A report indicates final-holdout access")
    return report


def load_inputs(
    config: Mapping[str, Any],
) -> tuple[dict[str, Path], dict[str, Any], dict[str, list[dict[str, Any]]]]:
    paths = candidate_a.validate_frozen_inputs(config)
    trigger_report = validate_candidate_a_trigger(paths)
    candidate_a_config = candidate_a.load_config()
    datasets = candidate_a.load_and_validate_datasets(
        candidate_a_config,
        {
            "v2c3_development": paths["v2c3_development"],
            "v2c4_training_augmentation": paths["v2c4_training_augmentation"],
            "v2c4_selection_probe": paths["v2c4_selection_probe"],
        },
    )
    return paths, trigger_report, datasets


def load_combined_embeddings(
    config: Mapping[str, Any],
    paths: Mapping[str, Path],
    datasets: Mapping[str, Sequence[Mapping[str, Any]]],
) -> np.ndarray:
    candidate_a_config = candidate_a.load_config()
    development = candidate_a.load_v2c3_development_embeddings(
        candidate_a_config, paths, datasets["development"]
    )
    augmentation = candidate_a._validate_new_embedding_cache(
        candidate_a_config,
        datasets["augmentation"],
        paths["candidate_a_augmentation_embedding_cache"],
        paths["candidate_a_augmentation_embedding_cache_metadata"],
    )
    combined = np.concatenate([development, augmentation], axis=0)
    if combined.shape != (config["training_corpus"]["combined_expected_count"], 384):
        raise ValueError("Candidate B combined embedding shape changed")
    if not np.isfinite(combined).all():
        raise ValueError("Candidate B embeddings contain non-finite values")
    return combined


def stage_targets(intents: Sequence[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    values = np.asarray(intents)
    all_indices = np.arange(len(values))
    supported_mask = values != UNSUPPORTED
    protected_mask = np.isin(values, PROTECTED_INTENTS)
    nonprotected_mask = np.isin(values, NONPROTECTED_INTENTS)
    return {
        "stage_1": (
            all_indices,
            np.where(supported_mask, SUPPORTED, UNSUPPORTED),
        ),
        "stage_2": (
            np.flatnonzero(supported_mask),
            np.where(protected_mask[supported_mask], PROTECTED, NONPROTECTED),
        ),
        "stage_3a": (np.flatnonzero(protected_mask), values[protected_mask]),
        "stage_3b": (np.flatnonzero(nonprotected_mask), values[nonprotected_mask]),
    }


def fit_hierarchy(
    config: Mapping[str, Any],
    embeddings: np.ndarray,
    intents: Sequence[str],
    *,
    enforce_expected_counts: bool = False,
) -> tuple[dict[str, LinearSVC], dict[str, Any]]:
    targets = stage_targets(intents)
    models: dict[str, LinearSVC] = {}
    stage_metadata: dict[str, Any] = {}
    for stage, (indices, labels) in targets.items():
        classifier = LinearSVC(**config["classifier"]["parameters"])
        classifier.fit(embeddings[indices], labels)
        fitted_classes = tuple(classifier.classes_.tolist())
        if fitted_classes != FITTED_STAGE_CLASSES[stage]:
            raise ValueError(f"fitted Candidate B classes changed for {stage}")
        models[stage] = classifier
        iterations = int(np.max(np.atleast_1d(classifier.n_iter_)))
        stage_metadata[stage] = {
            "training_count": len(indices),
            "class_distribution": dict(sorted(Counter(labels.tolist()).items())),
            "class_declaration_order": list(STAGE_CLASSES[stage]),
            "classes": list(fitted_classes),
            "parameters": config["classifier"]["parameters"],
            "n_iter": iterations,
            "max_iter": config["classifier"]["parameters"]["max_iter"],
            "converged_within_max_iter": iterations
            < config["classifier"]["parameters"]["max_iter"],
        }
    if enforce_expected_counts:
        expected = config["training_corpus"]["expected_stage_counts"]
        actual = {
            name: row["training_count"] for name, row in stage_metadata.items()
        }
        if actual != expected:
            raise ValueError("derived Candidate B stage counts differ from config")
    return models, stage_metadata


def hierarchical_predict(
    models: Mapping[str, Any], embeddings: np.ndarray
) -> tuple[np.ndarray, dict[str, int]]:
    final = np.full(len(embeddings), UNSUPPORTED, dtype=object)
    stage_1 = np.asarray(models["stage_1"].predict(embeddings), dtype=object)
    if not set(stage_1.tolist()).issubset(STAGE_CLASSES["stage_1"]):
        raise ValueError("Stage 1 produced an unknown route label")
    supported_indices = np.flatnonzero(stage_1 == SUPPORTED)
    protected_count = 0
    nonprotected_count = 0
    if len(supported_indices):
        stage_2 = np.asarray(
            models["stage_2"].predict(embeddings[supported_indices]), dtype=object
        )
        if not set(stage_2.tolist()).issubset(STAGE_CLASSES["stage_2"]):
            raise ValueError("Stage 2 produced an unknown route label")
        protected_relative = np.flatnonzero(stage_2 == PROTECTED)
        nonprotected_relative = np.flatnonzero(stage_2 == NONPROTECTED)
        protected_indices = supported_indices[protected_relative]
        nonprotected_indices = supported_indices[nonprotected_relative]
        protected_count = len(protected_indices)
        nonprotected_count = len(nonprotected_indices)
        if protected_count:
            protected_predictions = np.asarray(
                models["stage_3a"].predict(embeddings[protected_indices]),
                dtype=object,
            )
            if not set(protected_predictions.tolist()).issubset(
                STAGE_CLASSES["stage_3a"]
            ):
                raise ValueError("Stage 3A produced an unknown intent")
            final[protected_indices] = protected_predictions
        if nonprotected_count:
            nonprotected_predictions = np.asarray(
                models["stage_3b"].predict(embeddings[nonprotected_indices]),
                dtype=object,
            )
            if not set(nonprotected_predictions.tolist()).issubset(
                STAGE_CLASSES["stage_3b"]
            ):
                raise ValueError("Stage 3B produced an unknown intent")
            final[nonprotected_indices] = nonprotected_predictions
    return final, {
        "stage_1_executions": len(embeddings),
        "stage_2_executions": len(supported_indices),
        "stage_3a_executions": protected_count,
        "stage_3b_executions": nonprotected_count,
        "stage_1_unsupported_short_circuits": len(embeddings) - len(supported_indices),
    }


def stage_metric(
    gold: Sequence[str], predicted: Sequence[str], labels: Sequence[str]
) -> dict[str, Any]:
    report = classification_report(
        gold, predicted, labels=list(labels), output_dict=True, zero_division=0
    )
    return {
        "example_count": len(gold),
        "accuracy": float(accuracy_score(gold, predicted)),
        "per_class": {
            label: {
                "precision": float(report[label]["precision"]),
                "recall": float(report[label]["recall"]),
                "f1": float(report[label]["f1-score"]),
                "support": int(report[label]["support"]),
            }
            for label in labels
        },
        "confusion_matrix": {
            "label_order": list(labels),
            "values": confusion_matrix(gold, predicted, labels=list(labels)).tolist(),
        },
    }


def stage_diagnostics(
    models: Mapping[str, Any], embeddings: np.ndarray, intents: Sequence[str]
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    for stage, (indices, labels) in stage_targets(intents).items():
        predictions = models[stage].predict(embeddings[indices])
        diagnostics[stage] = stage_metric(labels, predictions, STAGE_CLASSES[stage])
    return {
        "role": "supporting_diagnostics",
        "routing_mode": "gold_eligible_subsets_for_stage_diagnostics_only",
        "not_used_for_final_predictions": True,
        "stages": diagnostics,
    }


def supporting_cv(
    config: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    embeddings: np.ndarray,
) -> dict[str, Any]:
    if len(records) != len(embeddings):
        raise ValueError("Candidate B CV records and embeddings differ in length")
    intents = np.asarray([row["intent"] for row in records])
    groups = np.asarray([row["group_id"] for row in records])
    cv_config = config["supporting_cv"]
    splitter = StratifiedGroupKFold(
        n_splits=cv_config["n_splits"],
        shuffle=cv_config["shuffle"],
        random_state=cv_config["random_state"],
    )
    oof = np.full(len(records), "", dtype=object)
    assignments = np.zeros(len(records), dtype=np.int8)
    diagnostic_oof = {
        stage: np.full(len(records), "", dtype=object) for stage in STAGE_CLASSES
    }
    diagnostic_assignments = {
        stage: np.zeros(len(records), dtype=np.int8) for stage in STAGE_CLASSES
    }
    folds: list[dict[str, Any]] = []
    for fold, (train_indices, validation_indices) in enumerate(
        splitter.split(embeddings, intents, groups)
    ):
        training_groups = set(groups[train_indices].tolist())
        validation_groups = set(groups[validation_indices].tolist())
        if training_groups & validation_groups:
            raise ValueError(f"group leakage in Candidate B fold {fold}")
        if set(intents[train_indices]) != set(candidate_a.EXPECTED_INTENTS):
            raise ValueError(f"Candidate B fold {fold} training lacks an intent")
        if set(intents[validation_indices]) != set(candidate_a.EXPECTED_INTENTS):
            raise ValueError(f"Candidate B fold {fold} lacks an intent")
        models, _ = fit_hierarchy(
            config, embeddings[train_indices], intents[train_indices]
        )
        predictions, routing = hierarchical_predict(
            models, embeddings[validation_indices]
        )
        oof[validation_indices] = predictions
        assignments[validation_indices] += 1
        fold_stage_metrics: dict[str, Any] = {}
        for stage, (relative_indices, labels) in stage_targets(
            intents[validation_indices]
        ).items():
            stage_predictions = models[stage].predict(
                embeddings[validation_indices[relative_indices]]
            )
            global_indices = validation_indices[relative_indices]
            diagnostic_oof[stage][global_indices] = stage_predictions
            diagnostic_assignments[stage][global_indices] += 1
            fold_stage_metrics[stage] = stage_metric(
                labels, stage_predictions, STAGE_CLASSES[stage]
            )
        folds.append(
            {
                "validation_fold": fold,
                "training_count": len(train_indices),
                "validation_count": len(validation_indices),
                "training_group_count": len(training_groups),
                "validation_group_count": len(validation_groups),
                "group_overlap_count": 0,
                "inference_routing": "predicted_upstream_labels_only",
                "routing_counts": routing,
                "final_9_intent_metrics": candidate_a.calculate_metrics(
                    intents[validation_indices],
                    predictions,
                    candidate_a.EXPECTED_INTENTS,
                ),
                "stage_diagnostics": {
                    "role": "supporting_diagnostics",
                    "routing_mode": (
                        "gold_eligible_subsets_for_stage_diagnostics_only"
                    ),
                    "not_used_for_final_predictions": True,
                    "stages": fold_stage_metrics,
                },
            }
        )
    if not np.all(assignments == 1) or np.any(oof == ""):
        raise ValueError("each record requires one Candidate B OOF prediction")
    macro_values = np.asarray(
        [row["final_9_intent_metrics"]["macro_f1"] for row in folds]
    )
    pooled_stage_metrics: dict[str, Any] = {}
    for stage, (indices, labels) in stage_targets(intents).items():
        if not np.all(diagnostic_assignments[stage][indices] == 1):
            raise ValueError(f"each eligible record requires one {stage} diagnostic")
        pooled_stage_metrics[stage] = stage_metric(
            labels, diagnostic_oof[stage][indices], STAGE_CLASSES[stage]
        )
    return {
        "role": "supporting_development_evidence",
        "primary_selection_evidence": False,
        "method": "StratifiedGroupKFold",
        "n_splits": 5,
        "shuffle": True,
        "random_state": 20260928,
        "record_count": len(records),
        "real_hierarchical_oof_inference": True,
        "gold_upstream_routing_used_for_final_predictions": False,
        "every_record_received_one_final_prediction": True,
        "group_leakage_detected": False,
        "fold_results": folds,
        "pooled_final_9_intent_metrics": candidate_a.calculate_metrics(
            intents, oof, candidate_a.EXPECTED_INTENTS
        ),
        "pooled_stage_diagnostics": {
            "role": "supporting_diagnostics",
            "routing_mode": "gold_eligible_subsets_for_stage_diagnostics_only",
            "not_used_for_final_predictions": True,
            "stages": pooled_stage_metrics,
        },
        "macro_f1_mean": float(np.mean(macro_values)),
        "macro_f1_standard_deviation": float(np.std(macro_values, ddof=0)),
        "hyperparameter_tuning_performed": False,
    }


def preflight(config: Mapping[str, Any]) -> dict[str, Any]:
    _, trigger_report, datasets = load_inputs(config)
    derived = stage_targets([row["intent"] for row in datasets["training"]])
    counts = {stage: len(rows[0]) for stage, rows in derived.items()}
    if counts != config["training_corpus"]["expected_stage_counts"]:
        raise ValueError("preflight stage counts differ")
    return {
        "status": "ready",
        "candidate_a_triggered_candidate_b": trigger_report["candidate_b_trigger"][
            "candidate_b_triggered"
        ],
        "training_count": len(datasets["training"]),
        "stage_training_counts": counts,
        "existing_embedding_caches_required": True,
        "integrity_checks": {
            "all_frozen_input_hashes_verified": True,
            "candidate_a_failure_trigger_verified": True,
            "same_candidate_a_training_corpus_verified": True,
            "selection_probe_excluded_from_fitting": True,
            "consumed_challenge_excluded": True,
            "consumed_external_lockbox_excluded": True,
            "cfpb_and_test_only_sources_excluded": True,
            "final_holdout_role_and_path_excluded": True,
        },
        "training_performed": False,
        "inference_or_evaluation_performed": False,
        "final_holdout_accessed": False,
    }


def train(config: Mapping[str, Any]) -> tuple[Path, Path]:
    paths, _, datasets = load_inputs(config)
    embeddings = load_combined_embeddings(config, paths, datasets)
    cv = supporting_cv(config, datasets["training"], embeddings)
    intents = [row["intent"] for row in datasets["training"]]
    models, stages = fit_hierarchy(
        config, embeddings, intents, enforce_expected_counts=True
    )
    artifact_path = candidate_a.repository_path(config["outputs"]["candidate_artifact"])
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "artifact_schema_version": "v2c4-candidate-b-classifier.v1",
        "candidate_id": config["candidate_id"],
        "architecture": "hierarchical",
        "representation": config["representation"],
        "classifier_parameters": config["classifier"]["parameters"],
        "intent_label_order": list(candidate_a.EXPECTED_INTENTS),
        "stage_classes": {name: list(labels) for name, labels in STAGE_CLASSES.items()},
        "fitted_stage_classes": {
            name: list(labels) for name, labels in FITTED_STAGE_CLASSES.items()
        },
        "stage_models": models,
        "stage_metadata": stages,
        "training_example_count": len(intents),
        "training_input_sha256": {
            "v2c3_development": config["frozen_inputs"]["v2c3_development"][
                "sha256"
            ],
            "v2c4_training_augmentation": config["frozen_inputs"][
                "v2c4_training_augmentation"
            ]["sha256"],
        },
        "embedding_cache_sha256": {
            "v2c3_development": config["frozen_inputs"][
                "v2c3_development_embedding_cache"
            ]["sha256"],
            "v2c4_training_augmentation": config["frozen_inputs"][
                "candidate_a_augmentation_embedding_cache"
            ]["sha256"],
        },
        "trusted_local_artifact": True,
        "advisory_routing_only": True,
        "runtime_authority": False,
        "selection_probe_used_for_fitting": False,
        "final_holdout_accessed": False,
    }
    with tempfile.NamedTemporaryFile(
        dir=artifact_path.parent, suffix=".joblib", delete=False
    ) as handle:
        temporary_path = Path(handle.name)
    try:
        joblib.dump(artifact, temporary_path, compress=3)
        os.replace(temporary_path, artifact_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    report = {
        "schema_version": "v2c4-candidate-b-training-report.v1",
        "experiment_phase": "V2-C4-Step-12",
        "analysis_role": "v2c4_development_model_selection",
        "candidate_id": config["candidate_id"],
        "candidate_config_sha256": candidate_a.sha256_file(CONFIG_PATH),
        "input_hashes": {
            name: spec["sha256"] for name, spec in config["frozen_inputs"].items()
        },
        "training_corpus": {
            "example_count": len(intents),
            "counts_by_intent": dict(sorted(Counter(intents).items())),
            "group_count": len({row["group_id"] for row in datasets["training"]}),
            "ordered_example_ids_sha256": candidate_a.ordered_ids_sha256(
                datasets["training"]
            ),
        },
        "embedding_metadata": {
            "strategy": config["embedding_strategy"]["strategy"],
            "combined_shape": list(embeddings.shape),
            "finite_values": True,
            "new_embeddings_performed": False,
            "development_cache_sha256": config["frozen_inputs"][
                "v2c3_development_embedding_cache"
            ]["sha256"],
            "augmentation_cache_sha256": config["frozen_inputs"][
                "candidate_a_augmentation_embedding_cache"
            ]["sha256"],
        },
        "stage_training": stages,
        "supporting_cv": cv,
        "artifact_metadata": {
            "path": config["outputs"]["candidate_artifact"],
            "sha256": candidate_a.sha256_file(artifact_path),
            "size_bytes": artifact_path.stat().st_size,
            "scikit_learn_version": sklearn.__version__,
            "joblib_version": metadata.version("joblib"),
            "python_version": platform.python_version(),
        },
        "safety_boundary": {
            "advisory_routing_only": True,
            "authorization_mechanism": False,
            "runtime_deterministic_controls_remain_responsible_for": [
                "authentication",
                "authorization",
                "ownership",
                "explicit_confirmation",
                "protected_tool_execution",
            ],
        },
        "integrity_checks": {
            "all_frozen_input_hashes_verified": True,
            "candidate_a_trigger_verified": True,
            "same_candidate_a_training_corpus": True,
            "exact_stage_training_counts_verified": True,
            "candidate_a_embedding_cache_hashes_verified": True,
            "selection_probe_used_for_fitting": False,
            "selection_probe_ids_text_and_groups_excluded": True,
            "consumed_v2c3_challenge_used": False,
            "consumed_v2c3_external_lockbox_used": False,
            "cfpb_used": False,
            "test_only_sources_used": False,
            "real_hierarchical_oof_inference": True,
            "gold_upstream_routing_used_for_final_predictions": False,
            "final_holdout_accessed": False,
            "threshold_tuning_performed": False,
            "hyperparameter_search_performed": False,
        },
        "training_performed": True,
        "selection_probe_evaluated": False,
        "final_acceptance_evidence": False,
        "final_holdout_accessed": False,
        "final_improvement_claimed": False,
    }
    report_path = candidate_a.repository_path(config["outputs"]["training_report"])
    candidate_a.write_json(report_path, report)
    return artifact_path, report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "train"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config()
    if args.mode == "preflight":
        print(json.dumps(preflight(config), indent=2, sort_keys=True))
        return
    artifact_path, report_path = train(config)
    print(f"wrote Candidate B artifact: {artifact_path.relative_to(REPOSITORY_ROOT)}")
    print(f"wrote Candidate B report: {report_path.relative_to(REPOSITORY_ROOT)}")
    print("advisory development routing only; final holdout not accessed")


if __name__ == "__main__":
    main()
