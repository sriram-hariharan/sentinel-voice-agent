"""Train frozen V2-C4 Candidate A and produce supporting development CV."""

from __future__ import annotations

import argparse
import hashlib
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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import LinearSVC

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY_ROOT / "data/evals/v2/ml/v2c4_candidate_a_config.json"
EXPECTED_INTENTS = (
    "account_balance",
    "card_status",
    "create_dispute",
    "escalation",
    "freeze_card",
    "informational_policy",
    "recent_transactions",
    "transaction_details",
    "unsupported_or_uncertain",
)
EXPECTED_RISK_BY_INTENT = {
    "account_balance": "PRIVATE_READ",
    "card_status": "PRIVATE_READ",
    "create_dispute": "PROTECTED_WRITE",
    "escalation": "ESCALATION_OR_UNCERTAIN",
    "freeze_card": "PROTECTED_WRITE",
    "informational_policy": "PUBLIC",
    "recent_transactions": "PRIVATE_READ",
    "transaction_details": "PRIVATE_READ",
    "unsupported_or_uncertain": "ESCALATION_OR_UNCERTAIN",
}
EXPECTED_CLASSIFIER_PARAMETERS = {
    "C": 4.0,
    "class_weight": "balanced",
    "dual": "auto",
    "fit_intercept": True,
    "intercept_scaling": 1.0,
    "loss": "squared_hinge",
    "max_iter": 2000,
    "multi_class": "ovr",
    "penalty": "l2",
    "random_state": 20260928,
    "tol": 0.0001,
    "verbose": 0,
}
PROTECTED_INTENTS = {"create_dispute", "freeze_card"}
UNSUPPORTED = "unsupported_or_uncertain"
EXPECTED_BASELINE_SHA256 = (
    "0e03a4d8367933622a92920f14a7a4c5a2ffa74af0f53b7b0913f04849e4780c"
)


def repository_path(relative_path: str) -> Path:
    resolved = (REPOSITORY_ROOT / relative_path).resolve()
    if not resolved.is_relative_to(REPOSITORY_ROOT):
        raise ValueError(f"path escapes repository: {relative_path}")
    return resolved


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def stable_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_ids_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    identifiers = [row["example_id"] for row in records]
    return hashlib.sha256(stable_json_bytes(identifiers)).hexdigest()


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def write_json(path: Path, value: Any) -> None:
    atomic_write_bytes(path, stable_json_bytes(value))


def load_config() -> dict[str, Any]:
    config = read_json(CONFIG_PATH)
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != "v2c4-candidate-a-config.v1":
        raise ValueError("unexpected Candidate A config schema")
    if tuple(config.get("intent_order", [])) != EXPECTED_INTENTS:
        raise ValueError("Candidate A intent ordering changed")
    if config.get("risk_by_intent") != EXPECTED_RISK_BY_INTENT:
        raise ValueError("Candidate A risk mapping changed")
    candidate = config["candidate_a"]
    if (
        candidate["candidate_id"] != "v2c4_candidate_a_targeted_data"
        or candidate["architecture"] != "direct_9_way_classifier"
    ):
        raise ValueError("Candidate A identity or architecture changed")
    representation = candidate["representation"]
    expected_representation = {
        "family": "frozen_local_sentence_embedding",
        "model_identifier": "BAAI/bge-small-en-v1.5",
        "fastembed_method": "passage_embed",
        "dimensions": 384,
        "fine_tuning": False,
        "external_api": False,
    }
    if representation != expected_representation:
        raise ValueError("Candidate A representation changed")
    classifier = candidate["classifier"]
    if (
        classifier["class"] != "LinearSVC"
        or classifier["parameters"] != EXPECTED_CLASSIFIER_PARAMETERS
    ):
        raise ValueError("Candidate A classifier changed")
    for field in (
        "threshold_tuning",
        "probability_calibration",
        "abstention_change",
        "embedding_fine_tuning",
        "hyperparameter_search",
    ):
        if candidate[field] is not False:
            raise ValueError(f"Candidate A prohibited option enabled: {field}")
    cv = config["supporting_cv"]
    if {
        "method": cv["method"],
        "n_splits": cv["n_splits"],
        "shuffle": cv["shuffle"],
        "random_state": cv["random_state"],
    } != {
        "method": "StratifiedGroupKFold",
        "n_splits": 5,
        "shuffle": True,
        "random_state": 20260928,
    }:
        raise ValueError("supporting CV definition changed")
    if config["safety_gates"] != {
        "protected_write_false_positive_rate_maximum": 0.01,
        "protected_write_recall_minimum": 0.8,
        "unsupported_or_uncertain_recall_minimum": 0.8,
        "all_gates_required": True,
    }:
        raise ValueError("Candidate A safety gates changed")
    if config["macro_f1_regression"][
        "material_regression_if_delta_less_than"
    ] != -0.01:
        raise ValueError("material macro-F1 regression threshold changed")
    if config["training_corpus"] != {
        "allowed_data_roles": [
            "development",
            "v2c4_targeted_training_augmentation",
        ],
        "v2c3_development_expected_count": 8198,
        "augmentation_expected_count": 360,
        "combined_expected_count": 8558,
        "selection_probe_excluded": True,
    }:
        raise ValueError("Candidate A training corpus definition changed")
    if config["embedding_strategy"] != {
        "strategy": (
            "reuse_verified_v2c3_development_cache_and_embed_augmentation_only"
        ),
        "shared_probe_embedding_for_both_models": True,
        "deterministic_text_order": True,
        "batch_size": 256,
        "query_embed_allowed": False,
        "fine_tuning_allowed": False,
        "external_api_allowed": False,
    }:
        raise ValueError("Candidate A embedding strategy changed")
    if config["selection_probe"] != {
        "data_role": "v2c4_model_selection_probe",
        "expected_count": 270,
        "examples_per_intent": 30,
        "training_eligible": False,
        "model_selection_eligible": True,
        "threshold_selection_eligible": False,
        "final_acceptance_evidence": False,
        "primary_model_selection_evidence": True,
    }:
        raise ValueError("Candidate A selection-probe definition changed")
    if config["candidate_b_trigger"] != {
        "report_only": True,
        "trigger_if_any_safety_gate_fails": True,
        "trigger_if_material_macro_f1_regression": True,
        "candidate_b_construction_allowed": False,
    }:
        raise ValueError("Candidate B trigger definition changed")
    baseline_spec = config["frozen_inputs"]["v2c3_baseline_artifact"]
    if baseline_spec != {
        "path": "artifacts/v2/classifier/v2c3_final_classifier.joblib",
        "sha256": EXPECTED_BASELINE_SHA256,
    }:
        raise ValueError("frozen V2-C3 baseline artifact identity changed")
    if config["execution_status"] != {
        "training_performed": False,
        "selection_probe_evaluated": False,
        "final_holdout_accessed": False,
        "candidate_b_built": False,
    }:
        raise ValueError("execution config must remain pre-execution")


def validate_frozen_inputs(config: Mapping[str, Any]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for name, spec in config["frozen_inputs"].items():
        path = repository_path(spec["path"])
        actual = sha256_file(path)
        if actual != spec["sha256"]:
            raise ValueError(
                f"{name} SHA-256 mismatch: expected {spec['sha256']}, got {actual}"
            )
        paths[name] = path
    return paths


def _dataset_examples(payload: Mapping[str, Any], label: str) -> list[dict[str, Any]]:
    examples = payload.get("examples")
    if not isinstance(examples, list) or not all(
        isinstance(row, dict) for row in examples
    ):
        raise TypeError(f"{label} examples must be a list of objects")
    if payload.get("example_count") != len(examples):
        raise ValueError(f"{label} example count differs from records")
    return examples


def validate_record_risk(row: Mapping[str, Any]) -> None:
    intent = row.get("intent")
    if intent not in EXPECTED_RISK_BY_INTENT:
        raise ValueError(f"unknown training intent: {intent}")
    if row.get("risk") != EXPECTED_RISK_BY_INTENT[intent]:
        raise ValueError(f"risk mapping differs for {row.get('example_id')}")


def validate_training_role(row: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    role = row.get("data_role")
    if role in config["forbidden_training_roles"]:
        raise ValueError(f"forbidden training data role: {role}")
    if role not in config["training_corpus"]["allowed_data_roles"]:
        raise ValueError(f"unapproved training data role: {role}")
    searchable = " ".join(
        str(row.get(field, ""))
        for field in (
            "data_role",
            "source_id",
            "source_domain",
            "source_split",
            "original_split",
        )
    ).lower()
    for fragment in config["forbidden_source_fragments"]:
        if fragment in searchable:
            raise ValueError(f"forbidden training source fragment: {fragment}")
    source_identity = " ".join(
        str(row.get(field, "")) for field in ("source_id", "source_domain")
    ).lower()
    source_splits = {
        str(row.get(field, "")).lower()
        for field in ("source_split", "original_split")
    }
    if {"test", "oos_test", "locked_test"} & source_splits and any(
        source in source_identity for source in ("banking77", "clinc")
    ):
        raise ValueError("forbidden test-only training source")


def load_and_validate_datasets(
    config: Mapping[str, Any], paths: Mapping[str, Path]
) -> dict[str, list[dict[str, Any]]]:
    development_payload = read_json(paths["v2c3_development"])
    augmentation_payload = read_json(paths["v2c4_training_augmentation"])
    probe_payload = read_json(paths["v2c4_selection_probe"])
    development = _dataset_examples(development_payload, "V2-C3 development")
    augmentation = _dataset_examples(augmentation_payload, "V2-C4 augmentation")
    probe = _dataset_examples(probe_payload, "V2-C4 selection probe")
    expected = config["training_corpus"]
    if len(development) != expected["v2c3_development_expected_count"]:
        raise ValueError("derived V2-C3 development count differs from config")
    if len(augmentation) != expected["augmentation_expected_count"]:
        raise ValueError("derived augmentation count differs from config")
    training = [*development, *augmentation]
    if len(training) != expected["combined_expected_count"]:
        raise ValueError("combined Candidate A training count differs")
    if len(probe) != config["selection_probe"]["expected_count"]:
        raise ValueError("selection probe count differs")
    for row in training:
        validate_training_role(row, config)
        validate_record_risk(row)
        if not isinstance(row.get("group_id"), str) or not row["group_id"]:
            raise ValueError("every training record requires a group_id")
    if {row["intent"] for row in training} != set(EXPECTED_INTENTS):
        raise ValueError("training corpus does not contain all nine intents")
    ids = [row["example_id"] for row in training]
    if len(ids) != len(set(ids)):
        raise ValueError("training example IDs are not unique")
    if any(row.get("data_role") != "v2c4_model_selection_probe" for row in probe):
        raise ValueError("selection probe role changed")
    if any(row.get("training_eligible") is not False for row in probe):
        raise ValueError("selection probe cannot be training eligible")
    training_ids = set(ids)
    probe_ids = {row["example_id"] for row in probe}
    if training_ids & probe_ids:
        raise ValueError("selection-probe IDs appear in Candidate A training")
    training_hashes = {row["normalized_text_sha256"] for row in training}
    probe_hashes = {row["normalized_text_sha256"] for row in probe}
    if training_hashes & probe_hashes:
        raise ValueError("selection-probe text appears in Candidate A training")
    training_groups = {row["group_id"] for row in training}
    probe_groups = {row["group_id"] for row in probe}
    if training_groups & probe_groups:
        raise ValueError("selection-probe group appears in Candidate A training")
    return {
        "development": development,
        "augmentation": augmentation,
        "training": training,
        "probe": probe,
    }


def load_v2c3_development_embeddings(
    config: Mapping[str, Any],
    paths: Mapping[str, Path],
    development: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    metadata_payload = read_json(paths["v2c3_development_embedding_cache_metadata"])
    representation = config["candidate_a"]["representation"]
    expected_metadata = {
        "model_identifier": representation["model_identifier"],
        "dimensions": representation["dimensions"],
        "example_count": len(development),
        "development_dataset_sha256": config["frozen_inputs"][
            "v2c3_development"
        ]["sha256"],
        "ordered_example_ids_sha256": ordered_ids_sha256(development),
        "cache_file_sha256": config["frozen_inputs"][
            "v2c3_development_embedding_cache"
        ]["sha256"],
        "fine_tuning_performed": False,
        "label_data_used": False,
    }
    for field, expected in expected_metadata.items():
        if metadata_payload.get(field) != expected:
            raise ValueError(f"V2-C3 embedding metadata changed: {field}")
    cache_path = paths["v2c3_development_embedding_cache"]
    with np.load(cache_path, allow_pickle=False) as cache:
        embeddings = np.asarray(cache["embeddings"], dtype=np.float32)
        example_ids = cache["example_ids"].astype(str).tolist()
    if example_ids != [row["example_id"] for row in development]:
        raise ValueError("V2-C3 embedding cache ordering changed")
    expected_shape = (len(development), representation["dimensions"])
    if embeddings.shape != expected_shape or not np.isfinite(embeddings).all():
        raise ValueError("V2-C3 embedding cache shape or values changed")
    return embeddings


def _validate_new_embedding_cache(
    config: Mapping[str, Any],
    augmentation: Sequence[Mapping[str, Any]],
    cache_path: Path,
    metadata_path: Path,
) -> np.ndarray:
    cache_metadata = read_json(metadata_path)
    expected = {
        "schema_version": "v2c4-candidate-a-augmentation-bge-cache.v1",
        "model_identifier": config["candidate_a"]["representation"][
            "model_identifier"
        ],
        "fastembed_method": "passage_embed",
        "dimensions": 384,
        "example_count": len(augmentation),
        "augmentation_sha256": config["frozen_inputs"][
            "v2c4_training_augmentation"
        ]["sha256"],
        "ordered_example_ids_sha256": ordered_ids_sha256(augmentation),
        "fine_tuning_performed": False,
        "label_data_used": False,
    }
    for field, expected_value in expected.items():
        if cache_metadata.get(field) != expected_value:
            raise ValueError(f"augmentation embedding metadata changed: {field}")
    if cache_metadata.get("cache_file_sha256") != sha256_file(cache_path):
        raise ValueError("augmentation embedding cache hash differs from metadata")
    with np.load(cache_path, allow_pickle=False) as cache:
        embeddings = np.asarray(cache["embeddings"], dtype=np.float32)
        example_ids = cache["example_ids"].astype(str).tolist()
    if example_ids != [row["example_id"] for row in augmentation]:
        raise ValueError("augmentation embedding cache ordering changed")
    if embeddings.shape != (len(augmentation), 384):
        raise ValueError("augmentation embedding cache shape changed")
    if not np.isfinite(embeddings).all():
        raise ValueError("augmentation embedding cache contains non-finite values")
    return embeddings


def load_or_create_augmentation_embeddings(
    config: Mapping[str, Any], augmentation: Sequence[Mapping[str, Any]]
) -> tuple[np.ndarray, dict[str, Any]]:
    cache_path = repository_path(config["outputs"]["augmentation_embedding_cache"])
    metadata_path = repository_path(
        config["outputs"]["augmentation_embedding_cache_metadata"]
    )
    if cache_path.exists() != metadata_path.exists():
        raise FileNotFoundError("augmentation cache and metadata must exist together")
    if cache_path.exists():
        embeddings = _validate_new_embedding_cache(
            config, augmentation, cache_path, metadata_path
        )
        return embeddings, read_json(metadata_path)
    try:
        from fastembed import TextEmbedding
    except ImportError as exc:
        raise RuntimeError("fastembed is required to embed the augmentation") from exc
    model_identifier = config["candidate_a"]["representation"]["model_identifier"]
    model = TextEmbedding(model_name=model_identifier)
    texts = [row["text"] for row in augmentation]
    vectors = list(
        model.passage_embed(
            texts, batch_size=config["embedding_strategy"]["batch_size"]
        )
    )
    embeddings = np.asarray(vectors, dtype=np.float32)
    if embeddings.shape != (len(augmentation), 384):
        raise ValueError("unexpected augmentation embedding shape")
    if not np.isfinite(embeddings).all():
        raise ValueError("augmentation embeddings contain non-finite values")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=cache_path.parent, prefix=f".{cache_path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(
                handle,
                embeddings=embeddings,
                example_ids=np.asarray([row["example_id"] for row in augmentation]),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, cache_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    cache_metadata = {
        "schema_version": "v2c4-candidate-a-augmentation-bge-cache.v1",
        "model_identifier": model_identifier,
        "fastembed_method": "passage_embed",
        "dimensions": 384,
        "dtype": "float32",
        "example_count": len(augmentation),
        "augmentation_sha256": config["frozen_inputs"][
            "v2c4_training_augmentation"
        ]["sha256"],
        "ordered_example_ids_sha256": ordered_ids_sha256(augmentation),
        "cache_file_sha256": sha256_file(cache_path),
        "fastembed_version": metadata.version("fastembed"),
        "fine_tuning_performed": False,
        "label_data_used": False,
        "source_scope": "v2c4_training_augmentation_only",
        "training_or_evaluation_performed": False,
    }
    write_json(metadata_path, cache_metadata)
    _validate_new_embedding_cache(config, augmentation, cache_path, metadata_path)
    return embeddings, cache_metadata


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def calculate_metrics(
    gold: Sequence[str], predicted: Sequence[str], intent_order: Sequence[str]
) -> dict[str, Any]:
    if len(gold) != len(predicted):
        raise ValueError("gold and prediction counts differ")
    report = classification_report(
        gold,
        predicted,
        labels=list(intent_order),
        output_dict=True,
        zero_division=0,
    )
    pairs = list(zip(gold, predicted, strict=True))
    protected_total = sum(expected in PROTECTED_INTENTS for expected, _ in pairs)
    protected_true_positive = sum(
        expected in PROTECTED_INTENTS and guess in PROTECTED_INTENTS
        for expected, guess in pairs
    )
    protected_exact = sum(
        expected in PROTECTED_INTENTS and guess == expected
        for expected, guess in pairs
    )
    non_protected_total = sum(
        expected not in PROTECTED_INTENTS for expected, _ in pairs
    )
    protected_false_positive = sum(
        expected not in PROTECTED_INTENTS and guess in PROTECTED_INTENTS
        for expected, guess in pairs
    )
    unsupported_total = sum(expected == UNSUPPORTED for expected, _ in pairs)
    unsupported_correct = sum(
        expected == UNSUPPORTED and guess == UNSUPPORTED for expected, guess in pairs
    )
    unsupported_false_supported = sum(
        expected == UNSUPPORTED and guess != UNSUPPORTED for expected, guess in pairs
    )
    return {
        "example_count": len(gold),
        "accuracy": float(accuracy_score(gold, predicted)),
        "macro_f1": float(
            f1_score(
                gold,
                predicted,
                labels=list(intent_order),
                average="macro",
                zero_division=0,
            )
        ),
        "per_intent": {
            intent: {
                "precision": float(report[intent]["precision"]),
                "recall": float(report[intent]["recall"]),
                "f1": float(report[intent]["f1-score"]),
                "support": int(report[intent]["support"]),
            }
            for intent in intent_order
        },
        "confusion_matrix": {
            "label_order": list(intent_order),
            "values": confusion_matrix(
                gold, predicted, labels=list(intent_order)
            ).tolist(),
        },
        "protected_total": protected_total,
        "protected_true_positive": protected_true_positive,
        "protected_misses": protected_total - protected_true_positive,
        "protected_write_recall": ratio(
            protected_true_positive, protected_total
        ),
        "protected_exact_intent_correct": protected_exact,
        "protected_exact_intent_accuracy": ratio(protected_exact, protected_total),
        "non_protected_total": non_protected_total,
        "protected_false_positive": protected_false_positive,
        "protected_write_false_positive_rate": ratio(
            protected_false_positive, non_protected_total
        ),
        "unsupported_total": unsupported_total,
        "unsupported_correctly_rejected": unsupported_correct,
        "unsupported_false_supported": unsupported_false_supported,
        "unsupported_or_uncertain_recall": ratio(
            unsupported_correct, unsupported_total
        ),
        "false_supported_rate": ratio(
            unsupported_false_supported, unsupported_total
        ),
    }


def apply_safety_gates(
    metrics: Mapping[str, Any], gates: Mapping[str, Any]
) -> dict[str, Any]:
    results = {
        "protected_write_false_positive_rate": {
            "value": metrics["protected_write_false_positive_rate"],
            "operator": "<=",
            "threshold": gates["protected_write_false_positive_rate_maximum"],
            "passed": metrics["protected_write_false_positive_rate"]
            <= gates["protected_write_false_positive_rate_maximum"],
        },
        "protected_write_recall": {
            "value": metrics["protected_write_recall"],
            "operator": ">=",
            "threshold": gates["protected_write_recall_minimum"],
            "passed": metrics["protected_write_recall"]
            >= gates["protected_write_recall_minimum"],
        },
        "unsupported_or_uncertain_recall": {
            "value": metrics["unsupported_or_uncertain_recall"],
            "operator": ">=",
            "threshold": gates["unsupported_or_uncertain_recall_minimum"],
            "passed": metrics["unsupported_or_uncertain_recall"]
            >= gates["unsupported_or_uncertain_recall_minimum"],
        },
    }
    return {
        "all_required": gates["all_gates_required"],
        "all_passed": all(result["passed"] for result in results.values()),
        "gates": results,
    }


def supporting_cv(
    config: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    embeddings: np.ndarray,
) -> dict[str, Any]:
    labels = np.asarray([row["intent"] for row in records])
    groups = np.asarray([row["group_id"] for row in records])
    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=config["supporting_cv"]["random_state"],
    )
    oof = np.full(len(records), "", dtype=object)
    assignment_count = np.zeros(len(records), dtype=np.int8)
    fold_results: list[dict[str, Any]] = []
    for fold, (train_indices, validation_indices) in enumerate(
        splitter.split(embeddings, labels, groups)
    ):
        train_groups = set(groups[train_indices].tolist())
        validation_groups = set(groups[validation_indices].tolist())
        if train_groups & validation_groups:
            raise ValueError(f"group leakage in supporting fold {fold}")
        validation_intents = set(labels[validation_indices].tolist())
        if validation_intents != set(EXPECTED_INTENTS):
            raise ValueError(f"supporting fold {fold} does not contain all intents")
        classifier = LinearSVC(
            **config["candidate_a"]["classifier"]["parameters"]
        )
        classifier.fit(embeddings[train_indices], labels[train_indices])
        predictions = classifier.predict(embeddings[validation_indices])
        oof[validation_indices] = predictions
        assignment_count[validation_indices] += 1
        fold_metrics = calculate_metrics(
            labels[validation_indices], predictions, EXPECTED_INTENTS
        )
        fold_results.append(
            {
                "validation_fold": fold,
                "training_count": len(train_indices),
                "validation_count": len(validation_indices),
                "training_group_count": len(train_groups),
                "validation_group_count": len(validation_groups),
                "group_overlap_count": 0,
                "metrics": fold_metrics,
            }
        )
    if not np.all(assignment_count == 1) or np.any(oof == ""):
        raise ValueError("each training record must receive one validation prediction")
    pooled = calculate_metrics(labels, oof, EXPECTED_INTENTS)
    macro_values = np.asarray(
        [row["metrics"]["macro_f1"] for row in fold_results], dtype=np.float64
    )
    return {
        "role": "supporting_development_evidence",
        "primary_selection_evidence": False,
        "method": "StratifiedGroupKFold",
        "n_splits": 5,
        "shuffle": True,
        "random_state": 20260928,
        "record_count": len(records),
        "every_record_received_one_validation_prediction": True,
        "group_leakage_detected": False,
        "fold_results": fold_results,
        "pooled_metrics": pooled,
        "macro_f1_mean": float(np.mean(macro_values)),
        "macro_f1_standard_deviation": float(np.std(macro_values, ddof=0)),
        "hyperparameter_tuning_performed": False,
    }


def preflight(config: Mapping[str, Any]) -> dict[str, Any]:
    paths = validate_frozen_inputs(config)
    datasets = load_and_validate_datasets(config, paths)
    return {
        "status": "ready",
        "candidate_id": config["candidate_a"]["candidate_id"],
        "v2c3_development_count": len(datasets["development"]),
        "augmentation_count": len(datasets["augmentation"]),
        "combined_training_count": len(datasets["training"]),
        "selection_probe_count_verified_but_not_used_for_fitting": len(
            datasets["probe"]
        ),
        "embedding_strategy": config["embedding_strategy"]["strategy"],
        "training_performed": False,
        "embedding_performed": False,
        "inference_or_evaluation_performed": False,
        "final_holdout_accessed": False,
        "candidate_b_built": False,
    }


def train_candidate(config: Mapping[str, Any]) -> tuple[Path, Path]:
    paths = validate_frozen_inputs(config)
    datasets = load_and_validate_datasets(config, paths)
    development_embeddings = load_v2c3_development_embeddings(
        config, paths, datasets["development"]
    )
    augmentation_embeddings, augmentation_cache_metadata = (
        load_or_create_augmentation_embeddings(config, datasets["augmentation"])
    )
    embeddings = np.concatenate(
        [development_embeddings, augmentation_embeddings], axis=0
    )
    if embeddings.shape != (len(datasets["training"]), 384):
        raise ValueError("combined Candidate A embedding shape differs")
    if not np.isfinite(embeddings).all():
        raise ValueError("combined Candidate A embeddings contain non-finite values")
    cv_metrics = supporting_cv(config, datasets["training"], embeddings)
    labels = np.asarray([row["intent"] for row in datasets["training"]])
    classifier = LinearSVC(**config["candidate_a"]["classifier"]["parameters"])
    classifier.fit(embeddings, labels)
    artifact_path = repository_path(config["outputs"]["candidate_artifact"])
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_payload = {
        "artifact_schema_version": "v2c4-candidate-a-classifier.v1",
        "candidate_id": config["candidate_a"]["candidate_id"],
        "architecture": config["candidate_a"]["architecture"],
        "representation": config["candidate_a"]["representation"],
        "classifier_parameters": config["candidate_a"]["classifier"]["parameters"],
        "intent_label_order": list(EXPECTED_INTENTS),
        "training_example_count": len(datasets["training"]),
        "training_input_sha256": {
            "v2c3_development": config["frozen_inputs"]["v2c3_development"][
                "sha256"
            ],
            "v2c4_training_augmentation": config["frozen_inputs"][
                "v2c4_training_augmentation"
            ]["sha256"],
        },
        "classifier": classifier,
        "trusted_local_artifact": True,
        "runtime_authority": False,
        "selection_probe_used_for_fitting": False,
        "final_holdout_accessed": False,
    }
    with tempfile.NamedTemporaryFile(
        dir=artifact_path.parent, suffix=".joblib", delete=False
    ) as handle:
        temporary_artifact = Path(handle.name)
    try:
        joblib.dump(artifact_payload, temporary_artifact, compress=3)
        os.replace(temporary_artifact, artifact_path)
    finally:
        temporary_artifact.unlink(missing_ok=True)
    artifact_sha256 = sha256_file(artifact_path)
    training = datasets["training"]
    report = {
        "schema_version": "v2c4-candidate-a-training-report.v1",
        "experiment_phase": "V2-C4-Step-11",
        "analysis_role": "v2c4_development_model_selection",
        "candidate_id": config["candidate_a"]["candidate_id"],
        "candidate_config_sha256": sha256_file(CONFIG_PATH),
        "input_hashes": {
            name: spec["sha256"] for name, spec in config["frozen_inputs"].items()
        },
        "training_corpus": {
            "example_count": len(training),
            "counts_by_intent": dict(
                sorted(Counter(row["intent"] for row in training).items())
            ),
            "counts_by_risk": dict(
                sorted(Counter(row["risk"] for row in training).items())
            ),
            "counts_by_data_role": dict(
                sorted(Counter(row["data_role"] for row in training).items())
            ),
            "counts_by_source_id": dict(
                sorted(Counter(row["source_id"] for row in training).items())
            ),
            "group_count": len({row["group_id"] for row in training}),
            "ordered_example_ids_sha256": ordered_ids_sha256(training),
        },
        "embedding_metadata": {
            "strategy": config["embedding_strategy"]["strategy"],
            "model_identifier": config["candidate_a"]["representation"][
                "model_identifier"
            ],
            "method": "passage_embed",
            "dimensions": 384,
            "combined_shape": list(embeddings.shape),
            "finite_values": True,
            "v2c3_cache_reused": True,
            "v2c3_cache_sha256": config["frozen_inputs"][
                "v2c3_development_embedding_cache"
            ]["sha256"],
            "augmentation_cache_sha256": augmentation_cache_metadata[
                "cache_file_sha256"
            ],
            "fine_tuning_performed": False,
            "external_api_used": False,
        },
        "classifier": config["candidate_a"]["classifier"],
        "supporting_cv_metrics": cv_metrics,
        "artifact_metadata": {
            "path": config["outputs"]["candidate_artifact"],
            "sha256": artifact_sha256,
            "size_bytes": artifact_path.stat().st_size,
            "scikit_learn_version": sklearn.__version__,
            "joblib_version": metadata.version("joblib"),
            "python_version": platform.python_version(),
        },
        "integrity_checks": {
            "all_frozen_input_hashes_verified": True,
            "all_nine_intents_present": True,
            "risk_mapping_valid": True,
            "group_ids_present": True,
            "selection_probe_ids_text_and_groups_excluded": True,
            "forbidden_data_roles_absent": True,
            "selection_probe_used_for_fitting": False,
            "final_holdout_accessed": False,
            "consumed_v2c3_challenge_used": False,
            "consumed_v2c3_external_lockbox_used": False,
            "cfpb_used": False,
        },
        "training_performed": True,
        "selection_probe_evaluated": False,
        "final_acceptance_evidence": False,
        "final_holdout_accessed": False,
        "candidate_b_built": False,
        "hyperparameter_search_performed": False,
        "threshold_tuning_performed": False,
        "final_improvement_claimed": False,
    }
    report_path = repository_path(config["outputs"]["training_report"])
    write_json(report_path, report)
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
    artifact_path, report_path = train_candidate(config)
    print(f"wrote Candidate A artifact: {artifact_path.relative_to(REPOSITORY_ROOT)}")
    print(f"wrote training report: {report_path.relative_to(REPOSITORY_ROOT)}")
    print("final V2-C4 holdout not accessed; no final acceptance claim")


if __name__ == "__main__":
    main()
