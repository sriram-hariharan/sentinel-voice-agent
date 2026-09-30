"""Run the frozen SentinelVoice V2-C5 Step 22 development experiment.

The sealed V2-C5 final-holdout dataset is deliberately inaccessible here. Step
22 reads only its text-free manifest for governance and uses only the frozen
expanded development dataset for folds, features, fitting, and selection.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import pickle
import platform
import tempfile
import time
import warnings
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from sklearn.base import BaseEstimator
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ML_DIRECTORY = REPOSITORY_ROOT / "data/evals/v2/ml"
CONTRACT_PATH = ML_DIRECTORY / "v2c5_model_selection_contract.json"
FOLDS_PATH = ML_DIRECTORY / "v2c5_model_selection_folds.json"
FOLDS_MANIFEST_PATH = ML_DIRECTORY / "v2c5_model_selection_folds.manifest.json"
RESULTS_PATH = ML_DIRECTORY / "v2c5_model_selection_results.json"
RESULTS_MANIFEST_PATH = ML_DIRECTORY / "v2c5_model_selection_results.manifest.json"
BGE_CACHE_PATH = ML_DIRECTORY / "local/v2c5_model_selection_bge_cache.npz"
BGE_CACHE_MANIFEST_PATH = (
    ML_DIRECTORY / "local/v2c5_model_selection_bge_cache.manifest.json"
)
PROHIBITED_FINAL_HOLDOUT_RELATIVE_PATH = (
    "data/evals/v2/ml/v2c5_final_holdout.json"
)
RUNNER_RELATIVE_PATH = "scripts/run_v2c5_model_selection.py"

CONTRACT_SCHEMA_VERSION = "v2c5-model-selection-contract.v1"
FOLDS_SCHEMA_VERSION = "v2c5-model-selection-folds.v1"
FOLDS_MANIFEST_SCHEMA_VERSION = "v2c5-model-selection-folds-manifest.v1"
RESULTS_SCHEMA_VERSION = "v2c5-model-selection-results.v1"
RESULTS_MANIFEST_SCHEMA_VERSION = "v2c5-model-selection-results-manifest.v1"
BGE_CACHE_SCHEMA_VERSION = "v2c5-model-selection-bge-cache.v1"
UNSUPPORTED_INTENT = "unsupported_or_uncertain"


def stable_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def repository_path(relative_path: str) -> Path:
    path = (REPOSITORY_ROOT / relative_path).resolve()
    if not path.is_relative_to(REPOSITORY_ROOT.resolve()):
        raise ValueError(f"path escapes repository: {relative_path}")
    return path


def prohibited_final_holdout_path() -> Path:
    return repository_path(PROHIBITED_FINAL_HOLDOUT_RELATIVE_PATH)


def assert_step22_path_allowed(path: Path) -> None:
    if path.resolve() == prohibited_final_holdout_path():
        raise PermissionError(
            "Step 22 must never open or hash the sealed V2-C5 final holdout"
        )


def sha256_file(path: Path) -> str:
    assert_step22_path_allowed(path)
    return sha256_bytes(path.read_bytes())


def read_json(path: Path) -> dict[str, Any]:
    assert_step22_path_allowed(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def write_bytes_atomically(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary_path = Path(handle.name)
        handle.write(content)
    try:
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def validate_frozen_source(spec: dict[str, Any]) -> Path:
    if not {"path", "sha256"}.issubset(spec):
        raise ValueError("frozen source spec must contain path and sha256")
    path = repository_path(spec["path"])
    actual = sha256_file(path)
    if actual != spec["sha256"]:
        raise ValueError(f"frozen source hash mismatch for {spec['path']}: {actual}")
    return path


def load_contract() -> dict[str, Any]:
    contract = read_json(CONTRACT_PATH)
    validate_contract(contract)
    return contract


def validate_contract(contract: dict[str, Any]) -> None:
    if contract.get("schema_version") != CONTRACT_SCHEMA_VERSION:
        raise ValueError("unexpected V2-C5 model-selection contract schema")
    if contract.get("contract_version") != CONTRACT_SCHEMA_VERSION:
        raise ValueError("unexpected V2-C5 model-selection contract version")
    if contract.get("phase") != "V2-C5 Step 22A":
        raise ValueError("unexpected model-selection contract phase")

    status = contract["contract_status"]
    expected_false = {
        "candidate_evaluation_performed",
        "cross_validation_performed",
        "embeddings_generated",
        "model_selection_performed",
        "model_training_performed",
        "runtime_behavior_changed",
        "sealed_final_holdout_accessed",
        "step23_permitted",
    }
    if any(status.get(field) is not False for field in expected_false):
        raise ValueError("Step 22A contract records unexpected prior execution")
    if status.get("model_selection_contract_frozen") is not True:
        raise ValueError("Step 22A contract is not frozen")

    dataset = contract["dataset_contract"]
    if (
        dataset["only_eligible_dataset"]
        != "data/evals/v2/ml/v2c5_expanded_development_dataset.json"
        or dataset["classifier_input_fields"] != ["text"]
        or dataset["eligible_data_roles"] != ["development"]
        or dataset["example_count"] != 8198
        or dataset["intent_count"] != 16
        or dataset["group_field"] != "group_id"
        or dataset["model_selection_probe_created"] is not False
    ):
        raise ValueError("frozen development-only dataset contract changed")

    taxonomy = contract["taxonomy"]
    labels = taxonomy["intent_label_order"]
    if len(labels) != 16 or len(set(labels)) != 16:
        raise ValueError("frozen taxonomy must contain 16 unique labels")
    if taxonomy["intent_count"] != len(labels):
        raise ValueError("frozen taxonomy count changed")
    protected = taxonomy["protected_write_intents"]
    if protected != [
        "cancel_transfer",
        "close_account",
        "create_dispute",
        "freeze_card",
    ]:
        raise ValueError("exact protected-write intent set changed")
    if contract["safety_gates"]["protected_write_intents"] != protected:
        raise ValueError("safety protected-write set does not match taxonomy")

    cv = contract["cross_validation"]
    if (
        cv["method"] != "StratifiedGroupKFold"
        or cv["n_splits"] != 5
        or cv["shuffle"] is not True
        or cv["random_state"] != 20260930
        or cv["group_field"] != "group_id"
        or cv["label_field"] != "intent"
        or cv["non_group_aware_fallback_allowed"] is not False
        or cv["fail_if_group_aware_cv_unavailable"] is not True
    ):
        raise ValueError("frozen group-aware CV settings changed")

    representations = contract["representations"]
    if set(representations) != {
        "WORD_TFIDF",
        "CHAR_TFIDF",
        "WORD_CHAR_TFIDF",
        "BGE_SMALL",
    }:
        raise ValueError("frozen representation set changed")
    if representations["WORD_TFIDF"] != {
        "class": "TfidfVectorizer",
        "parameters": {
            "analyzer": "word",
            "lowercase": True,
            "min_df": 2,
            "ngram_range": [1, 2],
            "sublinear_tf": True,
        },
    }:
        raise ValueError("frozen word TF-IDF representation changed")
    if representations["CHAR_TFIDF"] != {
        "class": "TfidfVectorizer",
        "parameters": {
            "analyzer": "char_wb",
            "lowercase": True,
            "min_df": 2,
            "ngram_range": [3, 5],
            "sublinear_tf": True,
        },
    }:
        raise ValueError("frozen character TF-IDF representation changed")
    if representations["WORD_CHAR_TFIDF"] != {
        "class": "FeatureUnion",
        "exact_member_definitions_reused": True,
        "members": ["WORD_TFIDF", "CHAR_TFIDF"],
    }:
        raise ValueError("frozen word/character FeatureUnion changed")
    bge = representations["BGE_SMALL"]
    if (
        bge["implementation"] != "FastEmbed"
        or bge["embedding_method"] != "passage_embed"
        or bge["model_identifier"] != "BAAI/bge-small-en-v1.5"
        or bge["dimensions"] != 384
        or bge["l2_normalized"] is not True
        or bge["fine_tuning"] is not False
    ):
        raise ValueError("frozen BGE representation changed")

    classifiers = contract["classifiers"]
    if set(classifiers) != {"LINEAR_SVC", "LOGISTIC_REGRESSION"}:
        raise ValueError("frozen classifier set changed")
    if classifiers["LINEAR_SVC"]["class"] != "LinearSVC":
        raise ValueError("LINEAR_SVC class changed")
    if classifiers["LINEAR_SVC"]["fixed_parameters"] != {
        "dual": "auto",
        "fit_intercept": True,
        "intercept_scaling": 1.0,
        "loss": "squared_hinge",
        "max_iter": 2000,
        "multi_class": "ovr",
        "penalty": "l2",
        "random_state": 20260930,
        "tol": 0.0001,
        "verbose": 0,
    }:
        raise ValueError("deterministic LinearSVC settings changed")
    logistic = classifiers["LOGISTIC_REGRESSION"]
    if (
        logistic["class"] != "LogisticRegression"
        or logistic["fixed_parameters"]
        != {
            "dual": False,
            "fit_intercept": True,
            "intercept_scaling": 1.0,
            "max_iter": 2000,
            "penalty": "l2",
            "random_state": 20260930,
            "solver": "lbfgs",
            "tol": 0.0001,
            "verbose": 0,
            "warm_start": False,
        }
    ):
        raise ValueError("deterministic LogisticRegression settings changed")

    candidates = expand_candidates(contract)
    matrix = contract["candidate_matrix"]
    if (
        matrix["candidate_family_count"] != 5
        or len(matrix["families"]) != 5
        or matrix["candidate_count"] != 27
        or len(candidates) != 27
        or len({row["candidate_id"] for row in candidates}) != 27
    ):
        raise ValueError("frozen 27-candidate matrix changed")
    expected_families = [
        ("WORD_TFIDF_LINEAR_SVC", "WORD_TFIDF", "LINEAR_SVC", 6),
        ("CHAR_TFIDF_LINEAR_SVC", "CHAR_TFIDF", "LINEAR_SVC", 6),
        ("WORD_CHAR_TFIDF_LINEAR_SVC", "WORD_CHAR_TFIDF", "LINEAR_SVC", 6),
        ("BGE_SMALL_LINEAR_SVC", "BGE_SMALL", "LINEAR_SVC", 6),
        (
            "BGE_SMALL_LOGISTIC_REGRESSION",
            "BGE_SMALL",
            "LOGISTIC_REGRESSION",
            3,
        ),
    ]
    actual_families = [
        (
            family["family_id"],
            family["representation_id"],
            family["classifier_id"],
            family["configuration_count"],
        )
        for family in matrix["families"]
    ]
    if actual_families != expected_families:
        raise ValueError("frozen candidate families or ordering changed")
    for family in matrix["families"]:
        expected_weights = (
            ["balanced"]
            if family["classifier_id"] == "LOGISTIC_REGRESSION"
            else [None, "balanced"]
        )
        if family["hyperparameter_grid"] != {
            "C": [0.25, 1.0, 4.0],
            "class_weight": expected_weights,
        }:
            raise ValueError(f"frozen grid changed for {family['family_id']}")
    historical_id = matrix["historical_v2c3_recipe"]["candidate_id"]
    if sum(row["candidate_id"] == historical_id for row in candidates) != 1:
        raise ValueError("historical V2-C3 recipe must occur exactly once")

    gates = contract["safety_gates"]
    expected_gates = [
        ("protected_write_false_positive_rate", "less_than_or_equal", 0.01),
        ("exact_protected_write_recall", "greater_than_or_equal", 0.8),
        ("unsupported_or_uncertain_recall", "greater_than_or_equal", 0.8),
    ]
    actual_gates = [
        (gate["metric"], gate["comparison"], gate["threshold"])
        for gate in gates["gates"]
    ]
    if (
        actual_gates != expected_gates
        or gates["all_gates_mandatory_for_selection"] is not True
        or gates["threshold_tuning_permitted"] is not False
        or gates["threshold_tuning_performed"] is not False
    ):
        raise ValueError("frozen mandatory safety gates changed")

    selection = contract["selection_rule"]
    if selection["ordered_tie_breaks"] != [
        {"direction": "highest", "metric": "mean_fold_macro_f1"},
        {"direction": "highest", "metric": "worst_fold_macro_f1"},
        {
            "direction": "lowest",
            "metric": "protected_write_false_positive_rate",
        },
        {
            "direction": "highest",
            "metric": "unsupported_or_uncertain_recall",
        },
        {"direction": "lowest", "metric": "local_prediction_latency"},
        {"direction": "ascending_lexical", "metric": "candidate_id"},
    ]:
        raise ValueError("frozen candidate selection order changed")
    tolerance = selection["numerical_tie_tolerance"]
    if tolerance["absolute"] != 1e-12 or tolerance["relative"] != 0.0:
        raise ValueError("frozen numerical tie tolerance changed")

    latency = contract["metrics"]["latency_measurement"]
    if latency != {
        "aggregation": (
            "median_of_five_timed_full_validation_fold_predictions_divided_by_"
            "example_count"
        ),
        "clock": "time.perf_counter",
        "scope": "representation_transform_or_embedding_plus_classifier_prediction",
        "unit": "milliseconds_per_example",
        "warmup_full_validation_prediction_runs": 1,
    }:
        raise ValueError("frozen local prediction latency protocol changed")

    isolation = contract["holdout_isolation"]
    holdout = isolation["final_holdout_dataset"]
    if (
        holdout["path"] != PROHIBITED_FINAL_HOLDOUT_RELATIVE_PATH
        or holdout["accessed"] is not False
        or isolation["step22_may_open_holdout_dataset"] is not False
        or isolation["step22_may_perform_holdout_inference"] is not False
    ):
        raise ValueError("sealed final-holdout isolation changed")


def expand_candidates(contract: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand only the candidate matrix encoded in the frozen contract."""
    candidates: list[dict[str, Any]] = []
    matrix = contract["candidate_matrix"]
    for family in matrix["families"]:
        grid = family["hyperparameter_grid"]
        before = len(candidates)
        for c_value in grid["C"]:
            for class_weight in grid["class_weight"]:
                weight_id = "none" if class_weight is None else class_weight
                candidate_id = matrix["candidate_id_format"].format(
                    family_id=family["family_id"],
                    C=c_value,
                    class_weight_or_none=weight_id,
                )
                candidates.append(
                    {
                        "candidate_id": candidate_id,
                        "family_id": family["family_id"],
                        "representation_id": family["representation_id"],
                        "classifier_id": family["classifier_id"],
                        "hyperparameters": {
                            "C": c_value,
                            "class_weight": class_weight,
                        },
                    }
                )
        if len(candidates) - before != family["configuration_count"]:
            raise ValueError(
                f"configuration count changed for family {family['family_id']}"
            )
    return candidates


def candidate_configuration(
    contract: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    classifier = contract["classifiers"][candidate["classifier_id"]]
    return {
        **candidate,
        "representation": contract["representations"][candidate["representation_id"]],
        "classifier": {
            "class": classifier["class"],
            "parameters": {
                **classifier["fixed_parameters"],
                **candidate["hyperparameters"],
            },
        },
    }


def candidate_configuration_sha256(
    contract: dict[str, Any], candidate: dict[str, Any]
) -> str:
    return sha256_bytes(stable_json_bytes(candidate_configuration(contract, candidate)))


def validate_development_dataset(
    dataset: dict[str, Any], contract: dict[str, Any]
) -> list[dict[str, Any]]:
    specification = contract["dataset_contract"]
    if dataset.get("schema_version") != "v2c5-expanded-development-dataset.v1":
        raise ValueError("expanded development dataset schema changed")
    examples = dataset.get("examples")
    if not isinstance(examples, list):
        raise TypeError("expanded development examples must be a list")
    if (
        dataset.get("example_count") != len(examples)
        or len(examples) != specification["example_count"]
        or dataset.get("classifier_input_fields") != ["text"]
    ):
        raise ValueError("expanded development population changed")
    labels = contract["taxonomy"]["intent_label_order"]
    example_ids: set[str] = set()
    observed_labels: set[str] = set()
    for row in examples:
        example_id = row.get("example_id")
        group_id = row.get("group_id")
        text = row.get("text")
        intent = row.get("intent")
        if not isinstance(example_id, str) or not example_id:
            raise ValueError("development example_id must be a non-empty string")
        if example_id in example_ids:
            raise ValueError(f"duplicate development example_id: {example_id}")
        example_ids.add(example_id)
        if not isinstance(group_id, str) or not group_id:
            raise ValueError(f"development group_id is invalid: {example_id}")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"development text is invalid: {example_id}")
        if row.get("text_sha256") != sha256_bytes(text.encode("utf-8")):
            raise ValueError(f"development text hash mismatch: {example_id}")
        if intent not in labels:
            raise ValueError(f"development intent is invalid: {example_id}")
        observed_labels.add(intent)
        if row.get("data_role") != "development":
            raise ValueError(f"non-development row encountered: {example_id}")
    if observed_labels != set(labels):
        raise ValueError("development data does not contain the exact frozen taxonomy")
    return examples


def validate_taxonomy(taxonomy: dict[str, Any], contract: dict[str, Any]) -> None:
    expected = contract["taxonomy"]
    if taxonomy.get("schema_version") != "v2c5-taxonomy-freeze.v1":
        raise ValueError("taxonomy freeze schema changed")
    if taxonomy.get("taxonomy_version") != expected["taxonomy_version"]:
        raise ValueError("taxonomy version changed")
    if taxonomy.get("intent_label_order") != expected["intent_label_order"]:
        raise ValueError("taxonomy label order changed")
    if taxonomy.get("protected_write_intents") != expected["protected_write_intents"]:
        raise ValueError("taxonomy protected-write intents changed")
    if taxonomy.get("final_taxonomy_frozen") is not True:
        raise ValueError("expanded taxonomy is not frozen")


def validate_source_manifests(
    development_manifest: dict[str, Any],
    taxonomy_manifest: dict[str, Any],
    final_holdout_manifest: dict[str, Any],
    contract: dict[str, Any],
) -> None:
    dataset_spec = contract["source_artifacts"]["expanded_development_dataset"]
    manifest_dataset = development_manifest.get("dataset", {})
    if (
        development_manifest.get("schema_version")
        != "v2c5-expanded-development-dataset-manifest.v1"
        or manifest_dataset.get("path") != dataset_spec["path"]
        or manifest_dataset.get("sha256") != dataset_spec["sha256"]
        or development_manifest.get("counts", {}).get("output_occurrences") != 8198
    ):
        raise ValueError("expanded development manifest changed")
    if (
        taxonomy_manifest.get("schema_version")
        != "v2c5-taxonomy-freeze-manifest.v1"
        or taxonomy_manifest.get("final_taxonomy_frozen") is not True
        or taxonomy_manifest.get("final_intent_count") != 16
    ):
        raise ValueError("taxonomy freeze manifest changed")

    status = final_holdout_manifest.get("execution_status", {})
    allowed = contract["holdout_isolation"]["allowed_manifest_checks"]
    for field, expected in allowed.items():
        if status.get(field) is not expected:
            raise ValueError(f"final-holdout manifest governance changed: {field}")
    declared = contract["holdout_isolation"]["final_holdout_dataset"][
        "declared_sha256_from_manifest"
    ]
    if final_holdout_manifest.get("dataset", {}).get("sha256") != declared:
        raise ValueError("final-holdout manifest dataset declaration changed")


def load_and_validate_sources(
    contract: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sources = contract["source_artifacts"]
    paths = {name: validate_frozen_source(spec) for name, spec in sources.items()}
    dataset = read_json(paths["expanded_development_dataset"])
    examples = validate_development_dataset(dataset, contract)
    taxonomy = read_json(paths["taxonomy_freeze"])
    validate_taxonomy(taxonomy, contract)
    validate_source_manifests(
        read_json(paths["expanded_development_manifest"]),
        read_json(paths["taxonomy_freeze_manifest"]),
        read_json(paths["final_holdout_manifest"]),
        contract,
    )
    return examples, taxonomy


def validate_group_feasibility(
    examples: Sequence[dict[str, Any]], contract: dict[str, Any]
) -> dict[str, int]:
    groups_by_label: dict[str, set[str]] = defaultdict(set)
    for row in examples:
        groups_by_label[row["intent"]].add(row["group_id"])
    counts = {
        label: len(groups_by_label[label])
        for label in contract["taxonomy"]["intent_label_order"]
    }
    insufficient = {
        label: count
        for label, count in counts.items()
        if count < contract["cross_validation"]["n_splits"]
    }
    if insufficient:
        raise ValueError(f"five group-aware folds are infeasible: {insufficient}")
    return counts


def fold_summaries(
    examples: Sequence[dict[str, Any]], folds: np.ndarray, contract: dict[str, Any]
) -> list[dict[str, Any]]:
    labels = contract["taxonomy"]["intent_label_order"]
    summaries: list[dict[str, Any]] = []
    for fold in range(contract["cross_validation"]["n_splits"]):
        indices = np.flatnonzero(folds == fold)
        rows = [examples[int(index)] for index in indices]
        counts = Counter(row["intent"] for row in rows)
        if set(counts) != set(labels):
            raise ValueError(f"validation fold {fold} does not contain all frozen intents")
        summaries.append(
            {
                "validation_fold": fold,
                "example_count": len(rows),
                "group_count": len({row["group_id"] for row in rows}),
                "intent_counts": {label: counts[label] for label in labels},
            }
        )
    return summaries


def build_fold_payload(
    examples: Sequence[dict[str, Any]], contract: dict[str, Any]
) -> dict[str, Any]:
    """Build deterministic text-free folds without fitting any representation."""
    group_counts = validate_group_feasibility(examples, contract)
    cv = contract["cross_validation"]
    labels = np.asarray([row["intent"] for row in examples], dtype=object)
    groups = np.asarray([row["group_id"] for row in examples], dtype=object)
    splitter = StratifiedGroupKFold(
        n_splits=cv["n_splits"],
        shuffle=cv["shuffle"],
        random_state=cv["random_state"],
    )
    folds = np.full(len(examples), -1, dtype=np.int8)
    placeholder = np.zeros(len(examples), dtype=np.int8)
    for fold, (_, validation_indices) in enumerate(
        splitter.split(placeholder, labels, groups)
    ):
        if np.any(folds[validation_indices] != -1):
            raise ValueError("a development record received multiple validation folds")
        folds[validation_indices] = fold
    if np.any(folds < 0):
        raise ValueError("a development record received no validation fold")

    assignments = [
        {
            "example_id": row["example_id"],
            "group_id": row["group_id"],
            "intent": row["intent"],
            "validation_fold": int(folds[index]),
        }
        for index, row in enumerate(examples)
    ]
    payload = {
        "schema_version": FOLDS_SCHEMA_VERSION,
        "phase": "V2-C5 Step 22B1",
        "contract": {
            "path": str(CONTRACT_PATH.relative_to(REPOSITORY_ROOT)),
            "sha256": sha256_file(CONTRACT_PATH),
        },
        "source_development_dataset": contract["source_artifacts"][
            "expanded_development_dataset"
        ],
        "cross_validation": dict(cv),
        "record_count": len(examples),
        "unique_group_count": len(set(groups.tolist())),
        "independent_group_counts_by_intent": group_counts,
        "assignment_count": len(assignments),
        "assignment_sha256": sha256_bytes(stable_json_bytes(assignments)),
        "assignments": assignments,
        "fold_summaries": fold_summaries(examples, folds, contract),
        "integrity": {
            "each_record_validation_assignment_count": 1,
            "group_leakage_count": 0,
            "raw_text_persisted": False,
        },
        "governance": {
            "candidate_scoring_performed": False,
            "classifier_fitting_performed": False,
            "final_holdout_accessed": False,
            "final_holdout_inference_performed": False,
        },
    }
    validate_fold_payload(payload, examples, contract)
    return payload


def validate_fold_payload(
    payload: dict[str, Any],
    examples: Sequence[dict[str, Any]],
    contract: dict[str, Any],
) -> np.ndarray:
    if payload.get("schema_version") != FOLDS_SCHEMA_VERSION:
        raise ValueError("fold artifact schema changed")
    if payload.get("contract", {}).get("sha256") != sha256_file(CONTRACT_PATH):
        raise ValueError("fold artifact contract lineage changed")
    if payload.get("source_development_dataset") != contract["source_artifacts"][
        "expanded_development_dataset"
    ]:
        raise ValueError("fold artifact development lineage changed")
    if payload.get("cross_validation") != contract["cross_validation"]:
        raise ValueError("fold artifact CV settings changed")
    assignments = payload.get("assignments")
    if not isinstance(assignments, list):
        raise TypeError("fold assignments must be a list")
    if sha256_bytes(stable_json_bytes(assignments)) != payload.get("assignment_sha256"):
        raise ValueError("fold assignment digest mismatch")
    if len(assignments) != len(examples) or payload.get("assignment_count") != len(examples):
        raise ValueError("fold assignments do not cover the development population")

    by_id: dict[str, dict[str, Any]] = {}
    for assignment in assignments:
        example_id = assignment.get("example_id")
        if example_id in by_id:
            raise ValueError(f"duplicate validation assignment: {example_id}")
        by_id[example_id] = assignment
    expected_ids = [row["example_id"] for row in examples]
    if set(by_id) != set(expected_ids):
        raise ValueError("fold assignments do not match development example IDs")

    folds = np.empty(len(examples), dtype=np.int8)
    group_folds: dict[str, set[int]] = defaultdict(set)
    for index, row in enumerate(examples):
        assignment = by_id[row["example_id"]]
        if assignment.get("group_id") != row["group_id"]:
            raise ValueError("fold assignment group lineage changed")
        if assignment.get("intent") != row["intent"]:
            raise ValueError("fold assignment intent lineage changed")
        fold = assignment.get("validation_fold")
        if not isinstance(fold, int) or isinstance(fold, bool) or fold not in range(5):
            raise ValueError("fold assignment must be an integer from 0 through 4")
        folds[index] = fold
        group_folds[row["group_id"]].add(fold)
    leakage_count = sum(len(values) > 1 for values in group_folds.values())
    if leakage_count:
        raise ValueError("a group crosses validation folds")
    if set(folds.tolist()) != set(range(5)):
        raise ValueError("fold assignments must use exactly folds 0 through 4")
    if payload.get("record_count") != len(examples):
        raise ValueError("fold record count changed")
    if payload.get("unique_group_count") != len(group_folds):
        raise ValueError("fold unique-group count changed")
    if payload.get("independent_group_counts_by_intent") != validate_group_feasibility(
        examples, contract
    ):
        raise ValueError("fold group counts changed")
    if payload.get("fold_summaries") != fold_summaries(examples, folds, contract):
        raise ValueError("fold summaries do not match assignments")
    if payload.get("integrity") != {
        "each_record_validation_assignment_count": 1,
        "group_leakage_count": 0,
        "raw_text_persisted": False,
    }:
        raise ValueError("fold integrity declaration changed")
    governance = payload.get("governance", {})
    if any(governance.get(field) is not False for field in governance):
        raise ValueError("fold artifact claims prohibited execution")
    return folds


def build_fold_manifest(fold_bytes: bytes, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": FOLDS_MANIFEST_SCHEMA_VERSION,
        "phase": "V2-C5 Step 22B1",
        "fold_artifact": {
            "path": str(FOLDS_PATH.relative_to(REPOSITORY_ROOT)),
            "schema_version": FOLDS_SCHEMA_VERSION,
            "sha256": sha256_bytes(fold_bytes),
        },
        "runner": {
            "path": RUNNER_RELATIVE_PATH,
            "sha256": sha256_file(repository_path(RUNNER_RELATIVE_PATH)),
        },
        "contract": payload["contract"],
        "source_development_dataset": payload["source_development_dataset"],
        "record_count": payload["record_count"],
        "unique_group_count": payload["unique_group_count"],
        "fold_summaries": payload["fold_summaries"],
        "cross_validation": payload["cross_validation"],
        "integrity": payload["integrity"],
        "final_holdout_accessed": False,
    }


def prepare_folds(
    examples: Sequence[dict[str, Any]], contract: dict[str, Any]
) -> tuple[Path, Path]:
    payload = build_fold_payload(examples, contract)
    fold_bytes = stable_json_bytes(payload)
    manifest_bytes = stable_json_bytes(build_fold_manifest(fold_bytes, payload))
    existing = (FOLDS_PATH.exists(), FOLDS_MANIFEST_PATH.exists())
    if any(existing):
        if not all(existing):
            raise FileExistsError("incomplete existing fold artifact pair; refusing overwrite")
        if (
            FOLDS_PATH.read_bytes() != fold_bytes
            or FOLDS_MANIFEST_PATH.read_bytes() != manifest_bytes
        ):
            raise FileExistsError("incompatible fold artifacts exist; refusing overwrite")
        return FOLDS_PATH, FOLDS_MANIFEST_PATH
    write_bytes_atomically(FOLDS_PATH, fold_bytes)
    write_bytes_atomically(FOLDS_MANIFEST_PATH, manifest_bytes)
    return FOLDS_PATH, FOLDS_MANIFEST_PATH


def check_folds(
    examples: Sequence[dict[str, Any]], contract: dict[str, Any]
) -> tuple[dict[str, Any], np.ndarray]:
    if not FOLDS_PATH.is_file() or not FOLDS_MANIFEST_PATH.is_file():
        raise FileNotFoundError("frozen fold artifact pair is required")
    payload = read_json(FOLDS_PATH)
    folds = validate_fold_payload(payload, examples, contract)
    expected_bytes = stable_json_bytes(build_fold_payload(examples, contract))
    if FOLDS_PATH.read_bytes() != expected_bytes:
        raise ValueError("fold artifact is not the deterministic expected serialization")
    manifest = read_json(FOLDS_MANIFEST_PATH)
    expected_manifest = build_fold_manifest(FOLDS_PATH.read_bytes(), payload)
    if manifest != expected_manifest or FOLDS_MANIFEST_PATH.read_bytes() != stable_json_bytes(
        expected_manifest
    ):
        raise ValueError("fold manifest content or serialization changed")
    return payload, folds


def ordered_example_ids(examples: Sequence[dict[str, Any]]) -> list[str]:
    return [row["example_id"] for row in examples]


def ordered_text_hashes(examples: Sequence[dict[str, Any]]) -> list[str]:
    return [sha256_bytes(row["text"].encode("utf-8")) for row in examples]


def bge_representation_sha256(contract: dict[str, Any]) -> str:
    return sha256_bytes(stable_json_bytes(contract["representations"]["BGE_SMALL"]))


def expected_bge_cache_manifest(
    examples: Sequence[dict[str, Any]],
    contract: dict[str, Any],
    *,
    cache_sha256: str,
    generation_elapsed_seconds: float,
    fastembed_version: str,
) -> dict[str, Any]:
    representation = contract["representations"]["BGE_SMALL"]
    return {
        "schema_version": BGE_CACHE_SCHEMA_VERSION,
        "source_scope": "v2c5_expanded_development_dataset_only",
        "development_dataset_sha256": contract["source_artifacts"][
            "expanded_development_dataset"
        ]["sha256"],
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "representation_config_sha256": bge_representation_sha256(contract),
        "model_identifier": representation["model_identifier"],
        "embedding_method": representation["embedding_method"],
        "dimensions": representation["dimensions"],
        "l2_normalized": representation["l2_normalized"],
        "row_count": len(examples),
        "ordered_example_ids": ordered_example_ids(examples),
        "ordered_text_sha256": ordered_text_hashes(examples),
        "cache_file_sha256": cache_sha256,
        "dtype": "float32",
        "generation_elapsed_seconds": generation_elapsed_seconds,
        "fastembed_version": fastembed_version,
        "raw_text_persisted": False,
        "labels_used": False,
        "fine_tuning_performed": False,
        "final_holdout_accessed": False,
    }


def validate_embedding_matrix(
    embeddings: np.ndarray, examples: Sequence[dict[str, Any]], contract: dict[str, Any]
) -> None:
    dimensions = contract["representations"]["BGE_SMALL"]["dimensions"]
    if embeddings.shape != (len(examples), dimensions):
        raise ValueError(f"BGE cache shape mismatch: {embeddings.shape}")
    if embeddings.dtype != np.float32:
        raise ValueError("BGE cache dtype must be float32")
    if not np.isfinite(embeddings).all():
        raise ValueError("BGE cache contains non-finite values")
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, rtol=1e-5, atol=1e-6):
        raise ValueError("BGE cache rows are not L2 normalized")


def validate_bge_cache(
    examples: Sequence[dict[str, Any]], contract: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    if not BGE_CACHE_PATH.is_file() or not BGE_CACHE_MANIFEST_PATH.is_file():
        raise FileNotFoundError("complete local V2-C5 BGE cache is required")
    manifest = read_json(BGE_CACHE_MANIFEST_PATH)
    invariant = expected_bge_cache_manifest(
        examples,
        contract,
        cache_sha256=manifest.get("cache_file_sha256", ""),
        generation_elapsed_seconds=manifest.get("generation_elapsed_seconds", -1.0),
        fastembed_version=manifest.get("fastembed_version", ""),
    )
    if manifest != invariant:
        raise ValueError("stale or mismatched BGE cache manifest")
    if not isinstance(manifest["generation_elapsed_seconds"], (int, float)) or (
        isinstance(manifest["generation_elapsed_seconds"], bool)
        or not math.isfinite(float(manifest["generation_elapsed_seconds"]))
        or manifest["generation_elapsed_seconds"] < 0
    ):
        raise ValueError("BGE cache generation time is invalid")
    if (
        not isinstance(manifest["fastembed_version"], str)
        or not manifest["fastembed_version"]
        or not isinstance(manifest["cache_file_sha256"], str)
        or len(manifest["cache_file_sha256"]) != 64
    ):
        raise ValueError("BGE cache version or digest metadata is invalid")
    if sha256_file(BGE_CACHE_PATH) != manifest["cache_file_sha256"]:
        raise ValueError("BGE cache file hash mismatch")
    with np.load(BGE_CACHE_PATH, allow_pickle=False) as cached:
        if set(cached.files) != {"embeddings", "example_ids", "text_sha256"}:
            raise ValueError("unexpected arrays in BGE cache")
        embeddings = np.asarray(cached["embeddings"])
        example_ids = cached["example_ids"].astype(str).tolist()
        text_hashes = cached["text_sha256"].astype(str).tolist()
    if example_ids != ordered_example_ids(examples):
        raise ValueError("BGE cache example row order changed")
    if text_hashes != ordered_text_hashes(examples):
        raise ValueError("BGE cache text-hash row order changed")
    validate_embedding_matrix(embeddings, examples, contract)
    return embeddings, manifest


def load_fastembed_model(contract: dict[str, Any]) -> Any:
    try:
        from fastembed import TextEmbedding
    except ImportError as exc:
        raise RuntimeError("fastembed is required for the frozen BGE representation") from exc
    model_name = contract["representations"]["BGE_SMALL"]["model_identifier"]
    try:
        return TextEmbedding(model_name=model_name)
    except Exception as exc:
        raise RuntimeError(f"FastEmbed could not load frozen model {model_name}") from exc


def embed_texts(model: Any, texts: Sequence[str], contract: dict[str, Any]) -> np.ndarray:
    representation = contract["representations"]["BGE_SMALL"]
    vectors = np.asarray(list(model.passage_embed(list(texts), batch_size=256)), dtype=np.float32)
    expected_shape = (len(texts), representation["dimensions"])
    if vectors.shape != expected_shape or not np.isfinite(vectors).all():
        raise ValueError(f"unexpected BGE embedding matrix: {vectors.shape}")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("BGE returned a zero vector")
    normalized = np.asarray(vectors / norms, dtype=np.float32)
    if not np.allclose(np.linalg.norm(normalized, axis=1), 1.0, rtol=1e-5, atol=1e-6):
        raise ValueError("BGE L2 normalization failed")
    return normalized


def prepare_bge_cache(
    examples: Sequence[dict[str, Any]], contract: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    existing = (BGE_CACHE_PATH.exists(), BGE_CACHE_MANIFEST_PATH.exists())
    if any(existing):
        if not all(existing):
            raise FileExistsError("incomplete local BGE cache pair; refusing overwrite")
        return validate_bge_cache(examples, contract)
    model = load_fastembed_model(contract)
    started = time.perf_counter()
    embeddings = embed_texts(model, [row["text"] for row in examples], contract)
    elapsed = time.perf_counter() - started
    BGE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=BGE_CACHE_PATH.parent, suffix=".npz", delete=False
    ) as handle:
        temporary_path = Path(handle.name)
    try:
        np.savez_compressed(
            temporary_path,
            embeddings=embeddings,
            example_ids=np.asarray(ordered_example_ids(examples)),
            text_sha256=np.asarray(ordered_text_hashes(examples)),
        )
        temporary_path.replace(BGE_CACHE_PATH)
    finally:
        temporary_path.unlink(missing_ok=True)
    manifest = expected_bge_cache_manifest(
        examples,
        contract,
        cache_sha256=sha256_file(BGE_CACHE_PATH),
        generation_elapsed_seconds=elapsed,
        fastembed_version=importlib.metadata.version("fastembed"),
    )
    write_bytes_atomically(BGE_CACHE_MANIFEST_PATH, stable_json_bytes(manifest))
    return validate_bge_cache(examples, contract)


def build_representation(representation_id: str, contract: dict[str, Any]) -> BaseEstimator:
    definitions = contract["representations"]
    if representation_id in {"WORD_TFIDF", "CHAR_TFIDF"}:
        definition = definitions[representation_id]
        parameters = dict(definition["parameters"])
        parameters["ngram_range"] = tuple(parameters["ngram_range"])
        return TfidfVectorizer(**parameters)
    if representation_id == "WORD_CHAR_TFIDF":
        definition = definitions[representation_id]
        members = definition["members"]
        return FeatureUnion(
            [(member.lower(), build_representation(member, contract)) for member in members]
        )
    if representation_id == "BGE_SMALL":
        raise ValueError("BGE_SMALL uses the separately validated frozen feature cache")
    raise ValueError(f"unknown representation: {representation_id}")


def build_classifier(candidate: dict[str, Any], contract: dict[str, Any]) -> BaseEstimator:
    definition = contract["classifiers"][candidate["classifier_id"]]
    parameters = {**definition["fixed_parameters"], **candidate["hyperparameters"]}
    classes: dict[str, type[BaseEstimator]] = {
        "LinearSVC": LinearSVC,
        "LogisticRegression": LogisticRegression,
    }
    try:
        classifier_class = classes[definition["class"]]
    except KeyError as exc:
        raise ValueError(f"unknown frozen classifier: {definition['class']}") from exc
    return classifier_class(**parameters)


def fit_classifier(classifier: BaseEstimator, features: Any, labels: np.ndarray) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        classifier.fit(features, labels)


def serialized_model_size(model: Any) -> int:
    return len(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))


def measure_local_prediction_latency(
    predict_full_validation: Callable[[], Any],
    example_count: int,
    *,
    warmup_runs: int,
    clock: Callable[[], float] = time.perf_counter,
) -> float:
    if example_count <= 0:
        raise ValueError("validation fold cannot be empty")
    for _ in range(warmup_runs):
        predict_full_validation()
    started = clock()
    predictions = predict_full_validation()
    elapsed = clock() - started
    if len(predictions) != example_count:
        raise ValueError("timed prediction count changed")
    return elapsed * 1000.0 / example_count


def _ratio(numerator: int, denominator: int, name: str) -> float:
    if denominator <= 0:
        raise ValueError(f"metric denominator is zero: {name}")
    return numerator / denominator


def safety_metrics(
    expected: Sequence[str], predicted: Sequence[str], protected_intents: Sequence[str]
) -> dict[str, float | int]:
    if len(expected) != len(predicted):
        raise ValueError("gold and predicted lengths differ")
    protected = set(protected_intents)
    pairs = list(zip(expected, predicted, strict=True))
    non_protected_total = sum(gold not in protected for gold, _ in pairs)
    protected_false_positives = sum(
        gold not in protected and guess in protected for gold, guess in pairs
    )
    protected_total = sum(gold in protected for gold, _ in pairs)
    protected_exact = sum(gold in protected and guess == gold for gold, guess in pairs)
    unsupported_total = sum(gold == UNSUPPORTED_INTENT for gold, _ in pairs)
    unsupported_exact = sum(
        gold == UNSUPPORTED_INTENT and guess == UNSUPPORTED_INTENT
        for gold, guess in pairs
    )
    return {
        "protected_write_false_positive_count": protected_false_positives,
        "protected_write_non_protected_denominator": non_protected_total,
        "protected_write_false_positive_rate": _ratio(
            protected_false_positives,
            non_protected_total,
            "protected_write_false_positive_rate",
        ),
        "exact_protected_write_correct_count": protected_exact,
        "protected_write_gold_denominator": protected_total,
        "exact_protected_write_recall": _ratio(
            protected_exact, protected_total, "exact_protected_write_recall"
        ),
        "unsupported_or_uncertain_correct_count": unsupported_exact,
        "unsupported_or_uncertain_gold_denominator": unsupported_total,
        "unsupported_or_uncertain_recall": _ratio(
            unsupported_exact,
            unsupported_total,
            "unsupported_or_uncertain_recall",
        ),
    }


def pooled_classification_metrics(
    expected: Sequence[str], predicted: Sequence[str], contract: dict[str, Any]
) -> dict[str, Any]:
    labels = contract["taxonomy"]["intent_label_order"]
    precision, recall, f1, support = precision_recall_fscore_support(
        expected,
        predicted,
        labels=labels,
        zero_division=0,
    )
    per_intent = {
        label: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(labels)
    }

    def subset_macro_f1(subset: Sequence[str]) -> float:
        indices = [index for index, gold in enumerate(expected) if gold in subset]
        subset_expected = [expected[index] for index in indices]
        subset_predicted = [predicted[index] for index in indices]
        return float(
            f1_score(
                subset_expected,
                subset_predicted,
                labels=list(subset),
                average="macro",
                zero_division=0,
            )
        )

    return {
        "pooled_oof_macro_f1": float(
            f1_score(
                expected,
                predicted,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "accuracy": float(accuracy_score(expected, predicted)),
        "balanced_accuracy": float(np.mean(recall)),
        "per_intent": per_intent,
        "confusion_matrix": {
            "label_order": list(labels),
            "values": confusion_matrix(expected, predicted, labels=labels).tolist(),
        },
        "new_seven_intent_macro_f1": subset_macro_f1(
            contract["metrics"]["new_seven_intents"]
        ),
        "historical_nine_label_macro_f1": subset_macro_f1(
            contract["metrics"]["historical_nine_labels"]
        ),
    }


def apply_safety_gates(
    metrics: dict[str, Any], contract: dict[str, Any]
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for gate in contract["safety_gates"]["gates"]:
        metric = gate["metric"]
        value = metrics[metric]
        if gate["comparison"] == "less_than_or_equal":
            passed = value <= gate["threshold"]
            operator = "<="
        elif gate["comparison"] == "greater_than_or_equal":
            passed = value >= gate["threshold"]
            operator = ">="
        else:
            raise ValueError(f"unsupported safety-gate comparison: {gate['comparison']}")
        results[metric] = {
            "value": value,
            "operator": operator,
            "threshold": gate["threshold"],
            "passed": bool(passed),
        }
    return {
        "all_gates_pass": all(result["passed"] for result in results.values()),
        "gates": results,
    }


def oof_prediction_sha256(
    examples: Sequence[dict[str, Any]], predicted: Sequence[str]
) -> str:
    rows = [
        {"example_id": row["example_id"], "predicted_intent": predicted[index]}
        for index, row in enumerate(examples)
    ]
    return sha256_bytes(stable_json_bytes(rows))


def run_candidate_cv(
    candidate: dict[str, Any],
    examples: Sequence[dict[str, Any]],
    folds: np.ndarray,
    contract: dict[str, Any],
    embeddings: np.ndarray | None,
    bge_model: Any | None,
    *,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    labels = np.asarray([row["intent"] for row in examples], dtype=object)
    texts = [row["text"] for row in examples]
    label_set = set(contract["taxonomy"]["intent_label_order"])
    oof = np.full(len(examples), "", dtype=object)
    fold_metrics: list[dict[str, Any]] = []
    fold_latencies: list[float] = []
    started_candidate = clock()
    is_bge = candidate["representation_id"] == "BGE_SMALL"
    if is_bge and (embeddings is None or bge_model is None):
        raise ValueError("BGE candidate requires validated embeddings and model")
    if not is_bge and embeddings is not None:
        embeddings = None

    for fold in range(contract["cross_validation"]["n_splits"]):
        train_indices = np.flatnonzero(folds != fold)
        validation_indices = np.flatnonzero(folds == fold)
        train_labels = labels[train_indices]
        validation_labels = labels[validation_indices]
        validation_texts = [texts[int(index)] for index in validation_indices]
        representation_fit_seconds = 0.0
        representation_validation_transform_seconds = 0.0

        if is_bge:
            train_features = embeddings[train_indices]
            validation_features = embeddings[validation_indices]
            representation: BaseEstimator | None = None
        else:
            representation = build_representation(candidate["representation_id"], contract)
            started = clock()
            train_features = representation.fit_transform(
                [texts[int(index)] for index in train_indices]
            )
            representation_fit_seconds = clock() - started
            started = clock()
            validation_features = representation.transform(validation_texts)
            representation_validation_transform_seconds = clock() - started

        classifier = build_classifier(candidate, contract)
        started = clock()
        fit_classifier(classifier, train_features, train_labels)
        classifier_fit_seconds = clock() - started
        started = clock()
        predictions = np.asarray(classifier.predict(validation_features), dtype=object)
        prediction_seconds = clock() - started
        if predictions.shape != (len(validation_indices),):
            raise ValueError("classifier prediction shape changed")
        if not set(predictions.tolist()).issubset(label_set):
            raise ValueError("classifier predicted outside the frozen taxonomy")
        oof[validation_indices] = predictions

        if is_bge:

            def predict_full_validation(
                current_validation_texts: Sequence[str] = validation_texts,
                current_classifier: BaseEstimator = classifier,
                current_bge_model: Any = bge_model,
            ) -> np.ndarray:
                live_embeddings = embed_texts(
                    current_bge_model, current_validation_texts, contract
                )
                return np.asarray(
                    current_classifier.predict(live_embeddings), dtype=object
                )
        else:

            def predict_full_validation(
                current_validation_texts: Sequence[str] = validation_texts,
                current_classifier: BaseEstimator = classifier,
                current_representation: Any = representation,
            ) -> np.ndarray:
                live_features = current_representation.transform(
                    current_validation_texts
                )
                return np.asarray(
                    current_classifier.predict(live_features), dtype=object
                )

        latency = measure_local_prediction_latency(
            predict_full_validation,
            len(validation_indices),
            warmup_runs=contract["metrics"]["latency_measurement"][
                "warmup_full_validation_prediction_runs"
            ],
            clock=clock,
        )
        fold_latencies.append(latency)
        fold_macro_f1 = float(
            f1_score(
                validation_labels,
                predictions,
                labels=contract["taxonomy"]["intent_label_order"],
                average="macro",
                zero_division=0,
            )
        )
        fold_metrics.append(
            {
                "validation_fold": fold,
                "validation_example_count": len(validation_indices),
                "macro_f1": fold_macro_f1,
                "accuracy": float(accuracy_score(validation_labels, predictions)),
                "representation_fit_seconds": representation_fit_seconds,
                "representation_validation_transform_seconds": (
                    representation_validation_transform_seconds
                ),
                "classifier_fit_seconds": classifier_fit_seconds,
                "prediction_seconds": prediction_seconds,
                "local_prediction_latency_ms_per_example": latency,
                "serialized_fitted_fold_artifact_bytes": serialized_model_size(
                    (representation, classifier)
                ),
            }
        )

    if np.any(oof == ""):
        raise ValueError("candidate did not produce exactly one OOF prediction per record")
    candidate_seconds = clock() - started_candidate
    gold = labels.tolist()
    predicted = oof.tolist()
    aggregate = pooled_classification_metrics(gold, predicted, contract)
    macro_values = [row["macro_f1"] for row in fold_metrics]
    aggregate["mean_fold_macro_f1"] = float(np.mean(macro_values))
    aggregate["worst_fold_macro_f1"] = float(min(macro_values))
    safety = safety_metrics(
        gold, predicted, contract["safety_gates"]["protected_write_intents"]
    )
    gates = apply_safety_gates(safety, contract)
    representation_seconds = sum(
        row["representation_fit_seconds"] for row in fold_metrics
    )
    classifier_seconds = sum(row["classifier_fit_seconds"] for row in fold_metrics)
    shared_embedding_seconds = 0.0
    if is_bge:
        shared_embedding_seconds = read_json(BGE_CACHE_MANIFEST_PATH)[
            "generation_elapsed_seconds"
        ]
    timing = {
        "representation_fit_or_shared_embedding_generation_seconds": (
            representation_seconds + shared_embedding_seconds
        ),
        "representation_fit_seconds_sum": representation_seconds,
        "representation_fit_seconds_mean": representation_seconds / len(fold_metrics),
        "classifier_fit_seconds_sum": classifier_seconds,
        "classifier_fit_seconds_mean": classifier_seconds / len(fold_metrics),
        "local_fit_time_seconds_sum": (
            representation_seconds + shared_embedding_seconds + classifier_seconds
        ),
        "local_fit_time_seconds_mean": (
            representation_seconds + shared_embedding_seconds + classifier_seconds
        )
        / len(fold_metrics),
        "representation_validation_transform_seconds_sum": sum(
            row["representation_validation_transform_seconds"] for row in fold_metrics
        ),
        "prediction_seconds_sum": sum(row["prediction_seconds"] for row in fold_metrics),
        "total_candidate_cv_seconds": candidate_seconds,
        "local_prediction_latency": float(np.median(fold_latencies)),
        "local_prediction_latency_ms_per_example": float(np.median(fold_latencies)),
        "local_prediction_latency_fold_values": fold_latencies,
    }
    artifact_sizes = [row["serialized_fitted_fold_artifact_bytes"] for row in fold_metrics]
    configuration = candidate_configuration(contract, candidate)
    return {
        "candidate_id": candidate["candidate_id"],
        "family_id": candidate["family_id"],
        "representation_id": candidate["representation_id"],
        "classifier_id": candidate["classifier_id"],
        "configuration": configuration,
        "configuration_sha256": sha256_bytes(stable_json_bytes(configuration)),
        "fold_metrics": fold_metrics,
        "aggregate_metrics": aggregate,
        "safety_metrics": safety,
        "safety_gate_results": gates,
        "timing": timing,
        "resulting_artifact_size": {
            "unit": "bytes",
            "scope": (
                "serialized_fold_classifier_only_bge_model_not_included"
                if is_bge
                else "serialized_fold_representation_and_classifier"
            ),
            "fold_serialized_sizes": artifact_sizes,
            "sum": int(sum(artifact_sizes)),
            "mean": float(np.mean(artifact_sizes)),
        },
        "oof_prediction_count": len(predicted),
        "oof_prediction_sha256": oof_prediction_sha256(examples, predicted),
        "oof_predictions_persisted": False,
        "raw_text_persisted": False,
    }


def selection_metric(result: dict[str, Any], metric: str) -> float:
    if metric in {"mean_fold_macro_f1", "worst_fold_macro_f1"}:
        return float(result["aggregate_metrics"][metric])
    if metric in {
        "protected_write_false_positive_rate",
        "unsupported_or_uncertain_recall",
    }:
        return float(result["safety_metrics"][metric])
    if metric == "local_prediction_latency":
        timing = result["timing"]
        if "local_prediction_latency" in timing:
            return float(timing["local_prediction_latency"])
        return float(timing["local_prediction_latency_ms_per_example"])
    raise ValueError(f"unknown numerical selection metric: {metric}")


def candidate_is_better(
    candidate: dict[str, Any], incumbent: dict[str, Any], contract: dict[str, Any]
) -> bool:
    tolerance = contract["selection_rule"]["numerical_tie_tolerance"]["absolute"]
    for tie_break in contract["selection_rule"]["ordered_tie_breaks"]:
        metric = tie_break["metric"]
        direction = tie_break["direction"]
        if metric == "candidate_id":
            return candidate["candidate_id"] < incumbent["candidate_id"]
        left = selection_metric(candidate, metric)
        right = selection_metric(incumbent, metric)
        if abs(left - right) <= tolerance:
            continue
        if direction == "highest":
            return left > right
        if direction == "lowest":
            return left < right
        raise ValueError(f"unknown selection direction: {direction}")
    return False


def select_candidate(
    candidate_results: Sequence[dict[str, Any]], contract: dict[str, Any]
) -> dict[str, Any]:
    eligible = [
        result
        for result in candidate_results
        if result["safety_gate_results"]["all_gates_pass"] is True
    ]
    if not eligible:
        return {
            "eligible_candidate_count": 0,
            "selected_candidate_id": None,
            "selected_candidate_configuration": None,
            "model_selection_success": False,
            "gates_weakened": False,
            "step23_permitted": False,
        }
    selected = eligible[0]
    for result in eligible[1:]:
        if candidate_is_better(result, selected, contract):
            selected = result
    return {
        "eligible_candidate_count": len(eligible),
        "selected_candidate_id": selected["candidate_id"],
        "selected_candidate_configuration": selected["configuration"],
        "model_selection_success": True,
        "gates_weakened": False,
        "step23_permitted": True,
    }


def all_string_values(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from all_string_values(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from all_string_values(child)


def validate_text_free(payload: dict[str, Any], examples: Sequence[dict[str, Any]]) -> None:
    strings = set(all_string_values(payload))
    raw_texts = {row["text"] for row in examples}
    overlap = strings & raw_texts
    if overlap:
        raise ValueError("tracked model-selection artifact contains raw development text")
    prohibited_keys = {"text", "utterance", "raw_text", "texts", "utterances"}
    if strings & prohibited_keys:
        raise ValueError("tracked model-selection artifact contains a raw-text field")


def build_results_payload(
    examples: Sequence[dict[str, Any]],
    fold_payload: dict[str, Any],
    candidate_results: list[dict[str, Any]],
    contract: dict[str, Any],
) -> dict[str, Any]:
    expected_candidates = expand_candidates(contract)
    if [result["candidate_id"] for result in candidate_results] != [
        candidate["candidate_id"] for candidate in expected_candidates
    ]:
        raise ValueError("complete ordered 27-candidate results are required")
    selection = select_candidate(candidate_results, contract)
    payload = {
        "schema_version": RESULTS_SCHEMA_VERSION,
        "phase": "V2-C5 Step 22B1",
        "contract": {
            "path": str(CONTRACT_PATH.relative_to(REPOSITORY_ROOT)),
            "sha256": sha256_file(CONTRACT_PATH),
        },
        "source_development_dataset": contract["source_artifacts"][
            "expanded_development_dataset"
        ],
        "source_artifacts": contract["source_artifacts"],
        "fold_artifact": {
            "path": str(FOLDS_PATH.relative_to(REPOSITORY_ROOT)),
            "sha256": sha256_file(FOLDS_PATH),
            "assignment_sha256": fold_payload["assignment_sha256"],
        },
        "candidate_count": len(expected_candidates),
        "completed_candidate_count": len(candidate_results),
        "development_record_count": len(examples),
        "candidate_results": candidate_results,
        "selection": selection,
        "governance": {
            "development_only": True,
            "embeddings_generated_for_development_only": True,
            "final_holdout_accessed": False,
            "final_holdout_evaluated": False,
            "final_holdout_inference_performed": False,
            "threshold_tuning_performed": False,
            "runtime_behavior_changed": False,
            "final_model_acceptance_claimed": False,
            "step23_permitted": selection["step23_permitted"],
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "fastembed": importlib.metadata.version("fastembed"),
        },
    }
    validate_results_payload(payload, examples, contract, fold_payload)
    return payload


def _require_finite_nonnegative(value: Any, field: str) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"result metric must be numeric: {field}")
    if not math.isfinite(float(value)) or value < 0:
        raise ValueError(f"result metric must be finite and nonnegative: {field}")


def validate_candidate_result(
    result: dict[str, Any],
    candidate: dict[str, Any],
    contract: dict[str, Any],
    development_count: int,
) -> None:
    if result.get("candidate_id") != candidate["candidate_id"]:
        raise ValueError("candidate result ID or order changed")
    expected_configuration = candidate_configuration(contract, candidate)
    if result.get("configuration") != expected_configuration:
        raise ValueError("candidate result configuration changed")
    if result.get("configuration_sha256") != sha256_bytes(
        stable_json_bytes(expected_configuration)
    ):
        raise ValueError("candidate configuration digest changed")
    folds = result.get("fold_metrics")
    if not isinstance(folds, list) or len(folds) != 5:
        raise ValueError("candidate result must contain five fold metrics")
    if [row.get("validation_fold") for row in folds] != list(range(5)):
        raise ValueError("candidate fold metric order changed")
    if sum(row.get("validation_example_count", 0) for row in folds) != development_count:
        raise ValueError("candidate fold metrics do not cover development records")
    macro_values = [row["macro_f1"] for row in folds]
    aggregate = result.get("aggregate_metrics", {})
    required_aggregate_metrics = {
        "mean_fold_macro_f1",
        "worst_fold_macro_f1",
        "pooled_oof_macro_f1",
        "accuracy",
        "balanced_accuracy",
        "per_intent",
        "confusion_matrix",
        "new_seven_intent_macro_f1",
        "historical_nine_label_macro_f1",
    }
    if set(aggregate) != required_aggregate_metrics:
        raise ValueError("candidate aggregate metric schema changed")
    if not math.isclose(
        aggregate.get("mean_fold_macro_f1", -1),
        float(np.mean(macro_values)),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("mean fold macro-F1 is inconsistent")
    if not math.isclose(
        aggregate.get("worst_fold_macro_f1", -1),
        min(macro_values),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("worst fold macro-F1 is inconsistent")
    labels = contract["taxonomy"]["intent_label_order"]
    if list(aggregate.get("per_intent", {})) != labels:
        raise ValueError("per-intent metric label order changed")
    confusion = aggregate.get("confusion_matrix", {})
    if confusion.get("label_order") != labels:
        raise ValueError("confusion-matrix label order changed")
    values = confusion.get("values")
    if not isinstance(values, list) or len(values) != len(labels) or any(
        not isinstance(row, list) or len(row) != len(labels) for row in values
    ):
        raise ValueError("confusion-matrix dimensions changed")
    safety = result.get("safety_metrics", {})
    expected_gates = apply_safety_gates(safety, contract)
    if result.get("safety_gate_results") != expected_gates:
        raise ValueError("candidate safety-gate results are inconsistent")
    if result.get("oof_prediction_count") != development_count:
        raise ValueError("candidate OOF prediction count changed")
    digest = result.get("oof_prediction_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("candidate OOF prediction digest is invalid")
    _require_finite_nonnegative(
        result.get("timing", {}).get("local_prediction_latency_ms_per_example"),
        "local_prediction_latency",
    )
    if result.get("oof_predictions_persisted") is not False:
        raise ValueError("individual OOF predictions must not be persisted")
    if result.get("raw_text_persisted") is not False:
        raise ValueError("raw development text must not be persisted")


def validate_results_payload(
    payload: dict[str, Any],
    examples: Sequence[dict[str, Any]],
    contract: dict[str, Any],
    fold_payload: dict[str, Any],
) -> None:
    if payload.get("schema_version") != RESULTS_SCHEMA_VERSION:
        raise ValueError("model-selection result schema changed")
    if payload.get("contract", {}).get("sha256") != sha256_file(CONTRACT_PATH):
        raise ValueError("model-selection result contract lineage changed")
    if payload.get("source_development_dataset") != contract["source_artifacts"][
        "expanded_development_dataset"
    ]:
        raise ValueError("model-selection result development lineage changed")
    if payload.get("source_artifacts") != contract["source_artifacts"]:
        raise ValueError("model-selection result source lineage changed")
    if payload.get("fold_artifact") != {
        "path": str(FOLDS_PATH.relative_to(REPOSITORY_ROOT)),
        "sha256": sha256_file(FOLDS_PATH),
        "assignment_sha256": fold_payload["assignment_sha256"],
    }:
        raise ValueError("model-selection result fold lineage changed")
    candidates = expand_candidates(contract)
    results = payload.get("candidate_results")
    if not isinstance(results, list) or len(results) != len(candidates):
        raise ValueError("complete 27-candidate matrix is required")
    if payload.get("candidate_count") != 27 or payload.get("completed_candidate_count") != 27:
        raise ValueError("result candidate counts changed")
    if payload.get("development_record_count") != len(examples):
        raise ValueError("result development count changed")
    for result, candidate in zip(results, candidates, strict=True):
        validate_candidate_result(result, candidate, contract, len(examples))
    if payload.get("selection") != select_candidate(results, contract):
        raise ValueError("model-selection decision is inconsistent")
    governance = payload.get("governance", {})
    expected_governance = {
        "development_only": True,
        "embeddings_generated_for_development_only": True,
        "final_holdout_accessed": False,
        "final_holdout_evaluated": False,
        "final_holdout_inference_performed": False,
        "threshold_tuning_performed": False,
        "runtime_behavior_changed": False,
        "final_model_acceptance_claimed": False,
        "step23_permitted": payload["selection"]["step23_permitted"],
    }
    if governance != expected_governance:
        raise ValueError("result governance fields changed")
    validate_text_free(payload, examples)


def build_results_manifest(payload: dict[str, Any], result_bytes: bytes) -> dict[str, Any]:
    selection = payload["selection"]
    return {
        "schema_version": RESULTS_MANIFEST_SCHEMA_VERSION,
        "phase": "V2-C5 Step 22B1",
        "results": {
            "path": str(RESULTS_PATH.relative_to(REPOSITORY_ROOT)),
            "schema_version": RESULTS_SCHEMA_VERSION,
            "sha256": sha256_bytes(result_bytes),
        },
        "runner": {
            "path": RUNNER_RELATIVE_PATH,
            "sha256": sha256_file(repository_path(RUNNER_RELATIVE_PATH)),
        },
        "contract": payload["contract"],
        "source_development_dataset": payload["source_development_dataset"],
        "source_artifacts": payload["source_artifacts"],
        "fold_artifact": payload["fold_artifact"],
        "candidate_count": payload["candidate_count"],
        "completed_candidate_count": payload["completed_candidate_count"],
        "development_record_count": payload["development_record_count"],
        "embeddings_generated_for_development_only": True,
        "final_holdout_accessed": False,
        "final_holdout_evaluated": False,
        "final_holdout_inference_performed": False,
        "threshold_tuning_performed": False,
        "runtime_behavior_changed": False,
        "model_selection_success": selection["model_selection_success"],
        "selected_candidate_id": selection["selected_candidate_id"],
        "step23_permitted": selection["step23_permitted"],
        "final_model_acceptance_claimed": False,
    }


def run_model_selection(
    examples: Sequence[dict[str, Any]], contract: dict[str, Any]
) -> tuple[Path, Path]:
    fold_payload, folds = check_folds(examples, contract)
    if RESULTS_PATH.exists() or RESULTS_MANIFEST_PATH.exists():
        raise FileExistsError("model-selection result artifact exists; refusing overwrite")
    embeddings, _ = prepare_bge_cache(examples, contract)
    bge_model = load_fastembed_model(contract)
    results: list[dict[str, Any]] = []
    for candidate in expand_candidates(contract):
        results.append(
            run_candidate_cv(
                candidate,
                examples,
                folds,
                contract,
                embeddings if candidate["representation_id"] == "BGE_SMALL" else None,
                bge_model if candidate["representation_id"] == "BGE_SMALL" else None,
            )
        )
    payload = build_results_payload(examples, fold_payload, results, contract)
    result_bytes = stable_json_bytes(payload)
    manifest_bytes = stable_json_bytes(build_results_manifest(payload, result_bytes))
    write_bytes_atomically(RESULTS_PATH, result_bytes)
    write_bytes_atomically(RESULTS_MANIFEST_PATH, manifest_bytes)
    return RESULTS_PATH, RESULTS_MANIFEST_PATH


def check_results(
    examples: Sequence[dict[str, Any]], contract: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    fold_payload, _ = check_folds(examples, contract)
    if not RESULTS_PATH.is_file() or not RESULTS_MANIFEST_PATH.is_file():
        raise FileNotFoundError("complete model-selection result artifact pair is required")
    payload = read_json(RESULTS_PATH)
    validate_results_payload(payload, examples, contract, fold_payload)
    if RESULTS_PATH.read_bytes() != stable_json_bytes(payload):
        raise ValueError("model-selection result serialization changed")
    manifest = read_json(RESULTS_MANIFEST_PATH)
    expected_manifest = build_results_manifest(payload, RESULTS_PATH.read_bytes())
    if manifest != expected_manifest:
        raise ValueError("model-selection result manifest changed")
    if RESULTS_MANIFEST_PATH.read_bytes() != stable_json_bytes(expected_manifest):
        raise ValueError("model-selection result manifest serialization changed")
    return payload, manifest


def preflight(contract: dict[str, Any]) -> dict[str, Any]:
    examples, taxonomy = load_and_validate_sources(contract)
    group_counts = validate_group_feasibility(examples, contract)
    fold_presence = (FOLDS_PATH.is_file(), FOLDS_MANIFEST_PATH.is_file())
    if any(fold_presence) and not all(fold_presence):
        raise FileNotFoundError("incomplete existing fold artifact pair")
    if all(fold_presence):
        check_folds(examples, contract)
    packages: dict[str, str | None] = {}
    for package in ("numpy", "scikit-learn", "fastembed"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return {
        "status": "ready" if all(packages.values()) else "missing_required_package",
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "development_dataset_sha256": contract["source_artifacts"][
            "expanded_development_dataset"
        ]["sha256"],
        "development_record_count": len(examples),
        "taxonomy_version": taxonomy["taxonomy_version"],
        "intent_count": len(contract["taxonomy"]["intent_label_order"]),
        "candidate_family_count": contract["candidate_matrix"][
            "candidate_family_count"
        ],
        "candidate_count": len(expand_candidates(contract)),
        "independent_group_counts_by_intent": group_counts,
        "fold_artifacts_present": all(fold_presence),
        "result_artifacts_present": RESULTS_PATH.is_file()
        and RESULTS_MANIFEST_PATH.is_file(),
        "packages": packages,
        "model_work_performed": False,
        "files_written": False,
        "final_holdout_accessed": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run frozen V2-C5 development-only model selection"
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--prepare-folds", action="store_true")
    modes.add_argument("--check-folds", action="store_true")
    modes.add_argument("--run", action="store_true")
    modes.add_argument("--check-results", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    contract = load_contract()
    if args.preflight:
        print(json.dumps(preflight(contract), indent=2, sort_keys=True))
        return
    examples, _ = load_and_validate_sources(contract)
    if args.prepare_folds:
        paths = prepare_folds(examples, contract)
        print("\n".join(str(path.relative_to(REPOSITORY_ROOT)) for path in paths))
    elif args.check_folds:
        payload, _ = check_folds(examples, contract)
        print(
            json.dumps(
                {
                    "status": "valid",
                    "record_count": payload["record_count"],
                    "files_written": False,
                    "model_work_performed": False,
                    "final_holdout_accessed": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
    elif args.run:
        paths = run_model_selection(examples, contract)
        print("\n".join(str(path.relative_to(REPOSITORY_ROOT)) for path in paths))
    elif args.check_results:
        payload, _ = check_results(examples, contract)
        print(
            json.dumps(
                {
                    "status": "valid",
                    "completed_candidate_count": payload["completed_candidate_count"],
                    "model_selection_success": payload["selection"][
                        "model_selection_success"
                    ],
                    "selected_candidate_id": payload["selection"][
                        "selected_candidate_id"
                    ],
                    "files_written": False,
                    "final_holdout_accessed": False,
                },
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
