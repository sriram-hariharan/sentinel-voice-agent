"""Execute the frozen SentinelVoice V2-C6 Step 29H development selection.

This runner is development-only. It deliberately rejects every final-holdout
path and never persists a full-development fitted classifier; Step 29I owns the
selected-recipe fit.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import tempfile
import warnings
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from sklearn.base import BaseEstimator
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
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
CONTRACT_PATH = ML_DIRECTORY / "v2c6_source_aware_model_selection_contract.json"
FREEZE_PATH = ML_DIRECTORY / "v2c6_remediated_development_dataset.freeze.json"
DATASET_PATH = ML_DIRECTORY / "v2c6_remediated_development_dataset.json"
RESULTS_PATH = ML_DIRECTORY / "v2c6_source_aware_model_selection_results.json"
RESULTS_MANIFEST_PATH = (
    ML_DIRECTORY / "v2c6_source_aware_model_selection_results.manifest.json"
)
BGE_CACHE_PATH = (
    ML_DIRECTORY / "local/v2c6_source_aware_model_selection_bge_cache.npz"
)
BGE_CACHE_MANIFEST_PATH = ML_DIRECTORY / (
    "local/v2c6_source_aware_model_selection_bge_cache.manifest.json"
)
RUNNER_RELATIVE_PATH = "scripts/run_v2c6_source_aware_model_selection.py"

CONTRACT_SCHEMA_VERSION = "v2c6-source-aware-model-selection-contract.v1"
EXPECTED_CONTRACT_SHA256 = (
    "de4566b7fdb097e78fcd23b8ffa4682c4757cc7c5901c3f9a49d7da28192b1ba"
)
RESULTS_SCHEMA_VERSION = "v2c6-source-aware-model-selection-results.v1"
RESULTS_MANIFEST_SCHEMA_VERSION = (
    "v2c6-source-aware-model-selection-results-manifest.v1"
)
BGE_CACHE_SCHEMA_VERSION = "v2c6-source-aware-model-selection-bge-cache.v1"
UNSUPPORTED_INTENT = "unsupported_or_uncertain"
REMEDIATION_LINEAGE = "REMEDIATION_ADDITION"

EXPECTED_CANDIDATE_IDS = (
    "BGE_SMALL_LINEAR_SVC__C=4.0__class_weight=none",
    "BGE_SMALL_LINEAR_SVC__C=4.0__class_weight=balanced",
    "BGE_SMALL_LINEAR_SVC__C=1.0__class_weight=none",
    "BGE_SMALL_LINEAR_SVC__C=1.0__class_weight=balanced",
    "WORD_CHAR_TFIDF_LINEAR_SVC__C=1.0__class_weight=none",
    "WORD_CHAR_TFIDF_LINEAR_SVC__C=1.0__class_weight=balanced",
)

EXPLICIT_TFIDF_DEFAULTS: dict[str, Any] = {
    "analyzer": "word",
    "binary": False,
    "decode_error": "strict",
    "dtype": np.float64,
    "encoding": "utf-8",
    "input": "content",
    "lowercase": True,
    "max_df": 1.0,
    "max_features": None,
    "min_df": 1,
    "ngram_range": (1, 1),
    "norm": "l2",
    "preprocessor": None,
    "smooth_idf": True,
    "stop_words": None,
    "strip_accents": None,
    "sublinear_tf": False,
    "token_pattern": r"(?u)\b\w\w+\b",
    "tokenizer": None,
    "use_idf": True,
    "vocabulary": None,
}


@dataclass(frozen=True)
class Partition:
    partition_id: str
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    held_out_source_family_id: str | None = None


@dataclass(frozen=True)
class EvaluationSplits:
    group_folds: tuple[Partition, ...]
    source_family_rounds: tuple[Partition, ...]
    audit: dict[str, Any]


@dataclass(frozen=True)
class LoadedInputs:
    contract: dict[str, Any]
    freeze: dict[str, Any]
    dataset: dict[str, Any]
    examples: tuple[dict[str, Any], ...]
    combined_manifest: dict[str, Any]
    remediation_manifest: dict[str, Any]
    contract_sha256: str


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


def assert_step29h_path_allowed(path: Path) -> None:
    resolved = path.resolve()
    if resolved.is_relative_to(ML_DIRECTORY.resolve()) and (
        "final_holdout" in resolved.name
    ):
        raise PermissionError("Step 29H must never access any final-holdout artifact")


def sha256_file(path: Path) -> str:
    assert_step29h_path_allowed(path)
    return sha256_bytes(path.read_bytes())


def read_json(path: Path) -> dict[str, Any]:
    assert_step29h_path_allowed(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def write_bytes_atomically(path: Path, content: bytes) -> None:
    assert_step29h_path_allowed(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary_path = Path(handle.name)
        handle.write(content)
    try:
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def validate_frozen_source(spec: Mapping[str, Any]) -> Path:
    if not {"path", "sha256"}.issubset(spec):
        raise ValueError("frozen source must declare path and sha256")
    path = repository_path(str(spec["path"]))
    actual = sha256_file(path)
    if actual != spec["sha256"]:
        raise ValueError(f"frozen source hash mismatch for {spec['path']}: {actual}")
    return path


def _candidate_id_components(candidate_id: str) -> tuple[str, float, str | None]:
    try:
        family, c_part, weight_part = candidate_id.split("__")
        c_value = float(c_part.removeprefix("C="))
        weight_value = weight_part.removeprefix("class_weight=")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid frozen candidate ID: {candidate_id}") from exc
    if weight_value == "none":
        class_weight: str | None = None
    elif weight_value == "balanced":
        class_weight = weight_value
    else:
        raise ValueError(f"invalid class weight in candidate ID: {candidate_id}")
    return family, c_value, class_weight


def validate_contract(contract: Mapping[str, Any]) -> None:
    if contract.get("schema_version") != CONTRACT_SCHEMA_VERSION:
        raise ValueError("unexpected Step 29G contract schema")
    if contract.get("contract_version") != CONTRACT_SCHEMA_VERSION:
        raise ValueError("unexpected Step 29G contract version")
    if contract.get("phase") != "V2-C6 Step 29G":
        raise ValueError("unexpected Step 29G contract phase")
    status = contract.get("contract_status", {})
    if status.get("contract_frozen") is not True:
        raise ValueError("Step 29G contract is not frozen")
    if status.get("next_required") != (
        "v2c6_source_aware_model_selection_execution"
    ):
        raise ValueError("Step 29G next_required changed")
    for field in (
        "candidate_metrics_calculated",
        "cross_validation_performed",
        "embeddings_generated",
        "final_holdout_accessed",
        "model_fitting_performed",
        "model_selection_performed",
        "runtime_behavior_changed",
        "source_family_evaluation_performed",
        "threshold_tuning_performed",
    ):
        if status.get(field) is not False:
            raise ValueError(f"Step 29G contract records prior execution: {field}")

    dataset = contract["frozen_input_lineage"]["dataset"]
    if dataset != {
        "inherited_v2c5_record_count": 8198,
        "only_eligible_development_dataset": True,
        "path": "data/evals/v2/ml/v2c6_remediated_development_dataset.json",
        "record_count": 9008,
        "remediation_record_count": 810,
        "sha256": (
            "d7f78d7a76799f47bfdc3c1291505d1b964b9d143245d2d353931e8cb4f4a493"
        ),
    }:
        raise ValueError("frozen development dataset identity changed")
    freeze = contract["frozen_input_lineage"]["step29f_freeze"]
    if freeze != {
        "expected_freeze_status": "FROZEN",
        "expected_next_required": "v2c6_source_aware_model_selection_contract",
        "path": "data/evals/v2/ml/v2c6_remediated_development_dataset.freeze.json",
        "sha256": (
            "7f71dda024e5a947c997977f09ed675fa1f924c38d30f332422aa2032ecbc544"
        ),
    }:
        raise ValueError("Step 29F freeze identity changed")

    taxonomy = contract["taxonomy"]
    labels = taxonomy["intent_label_order"]
    if taxonomy["intent_count"] != 16 or len(labels) != 16 or len(set(labels)) != 16:
        raise ValueError("frozen taxonomy must contain 16 unique labels")
    if taxonomy["primary_remediation_8_intents"] != [
        "account_blocked",
        "cancel_transfer",
        "close_account",
        "create_dispute",
        "freeze_card",
        "transfer_failed_or_declined",
        "transfer_pending",
        "unsupported_or_uncertain",
    ]:
        raise ValueError("primary remediation-eight slice changed")

    cv = contract["group_aware_cross_validation"]
    if (
        cv["method"] != "StratifiedGroupKFold"
        or cv["n_splits"] != 5
        or cv["shuffle"] is not True
        or cv["random_state"] != 20260930
        or cv["group_field"] != "group_id"
        or cv["group_ids_are_indivisible_split_units"] is not True
        or cv["inherited_group_label_purity_required"] is not False
        or cv["non_group_aware_fallback_allowed"] is not False
    ):
        raise ValueError("frozen group-aware CV protocol changed")

    source = contract["source_family_holdout"]
    expected_families = [
        "v2c6_sf1_definition_direct",
        "v2c6_sf2_scenario_narrative",
        "v2c6_sf3_boundary_conversational",
    ]
    if source["source_families"] != expected_families:
        raise ValueError("frozen source-family order changed")
    if source["expected_round_count"] != 3:
        raise ValueError("frozen source-family round count changed")
    if any(
        row["held_out_source_family_id"] != family
        or row["training_count"] != 8738
        or row["test_count"] != 270
        for row, family in zip(source["rounds"], expected_families, strict=True)
    ):
        raise ValueError("frozen source-family round definitions changed")

    representations = contract["representations"]
    if set(representations) != {
        "BGE_SMALL",
        "CHAR_TFIDF",
        "WORD_CHAR_TFIDF",
        "WORD_TFIDF",
    }:
        raise ValueError("frozen representation definitions changed")
    bge = representations["BGE_SMALL"]
    if bge != {
        "dimensions": 384,
        "embedding_method": "passage_embed",
        "fine_tuning": False,
        "implementation": "FastEmbed",
        "l2_normalized": True,
        "model_identifier": "BAAI/bge-small-en-v1.5",
    }:
        raise ValueError("frozen BGE_SMALL definition changed")
    if representations["WORD_CHAR_TFIDF"] != {
        "class": "FeatureUnion",
        "exact_member_definitions_reused": True,
        "members": ["WORD_TFIDF", "CHAR_TFIDF"],
    }:
        raise ValueError("frozen WORD_CHAR_TFIDF definition changed")
    expected_tfidf = {
        "WORD_TFIDF": {
            "class": "TfidfVectorizer",
            "parameters": {
                "analyzer": "word",
                "lowercase": True,
                "min_df": 2,
                "ngram_range": [1, 2],
                "sublinear_tf": True,
            },
        },
        "CHAR_TFIDF": {
            "class": "TfidfVectorizer",
            "parameters": {
                "analyzer": "char_wb",
                "lowercase": True,
                "min_df": 2,
                "ngram_range": [3, 5],
                "sublinear_tf": True,
            },
        },
    }
    for representation_id, expected in expected_tfidf.items():
        if representations[representation_id] != expected:
            raise ValueError(f"frozen {representation_id} definition changed")

    classifiers = contract["classifiers"]
    if set(classifiers) != {"LINEAR_SVC"}:
        raise ValueError("frozen classifier set changed")
    classifier = classifiers["LINEAR_SVC"]
    if classifier["class"] != "LinearSVC" or classifier["fixed_parameters"] != {
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
        raise ValueError("frozen LinearSVC definition changed")

    search = contract["candidate_search_space"]
    candidates = search["candidates"]
    if search["candidate_count"] != 6 or len(candidates) != 6:
        raise ValueError("exactly six candidates are required")
    candidate_ids = [row["candidate_id"] for row in candidates]
    if tuple(candidate_ids) != EXPECTED_CANDIDATE_IDS:
        raise ValueError("frozen candidate IDs or order changed")
    if len(set(candidate_ids)) != 6:
        raise ValueError("candidate IDs must be unique")
    for candidate in candidates:
        representation_id = candidate.get("representation_id")
        classifier_id = candidate.get("classifier_id")
        if representation_id not in representations:
            raise ValueError(f"missing representation definition: {representation_id}")
        if classifier_id not in classifiers:
            raise ValueError(f"missing classifier definition: {classifier_id}")
        family, c_value, class_weight = _candidate_id_components(
            candidate["candidate_id"]
        )
        expected_representation = (
            "BGE_SMALL" if family.startswith("BGE_SMALL_") else "WORD_CHAR_TFIDF"
        )
        if representation_id != expected_representation:
            raise ValueError("candidate representation conflicts with candidate ID")
        if family.split(f"{expected_representation}_", 1)[1] != "LINEAR_SVC":
            raise ValueError("candidate classifier conflicts with candidate ID")
        if classifier_id != "LINEAR_SVC" or candidate["hyperparameters"] != {
            "C": c_value,
            "class_weight": class_weight,
        }:
            raise ValueError("candidate hyperparameters conflict with candidate ID")

    gates = contract["safety_gates"]
    if gates["protected_intents"] != [
        "cancel_transfer",
        "close_account",
        "create_dispute",
        "freeze_card",
    ]:
        raise ValueError("protected intent set changed")
    expected_gates = [
        ("protected_recall", "greater_than_or_equal", 0.8),
        ("protected_false_positive_rate", "less_than_or_equal", 0.01),
        ("unsupported_recall", "greater_than_or_equal", 0.8),
    ]
    if [
        (row["metric"], row["comparison"], row["threshold"])
        for row in gates["gates"]
    ] != expected_gates:
        raise ValueError("frozen safety gates changed")
    if gates["application_scopes"] != [
        "pooled_group_aware_cv_predictions",
        "pooled_source_family_holdout_predictions",
        "each_individual_source_family_holdout_round",
    ]:
        raise ValueError("mandatory safety-gate scopes changed")

    selection = contract["selection_rule"]
    expected_selection_metrics = [
        "worst_family_primary_8_macro_f1",
        "mean_family_primary_8_macro_f1",
        "pooled_source_family_primary_8_macro_f1",
        "pooled_group_cv_macro_f1_16",
        "pooled_source_family_supported_to_unsupported_rate",
        "pooled_source_family_protected_false_positive_rate",
        "pooled_source_family_protected_recall",
        "pooled_source_family_unsupported_recall",
        "candidate_complexity_rank",
        "candidate_id",
    ]
    if [row["metric"] for row in selection["ordered_lexicographic_criteria"]] != (
        expected_selection_metrics
    ):
        raise ValueError("frozen selection order changed")
    if selection["numerical_tie_tolerance"] != {
        "absolute": 1e-12,
        "relative": 0.0,
        "rule": (
            "Treat numeric criteria as tied when absolute_difference is less than "
            "or equal to the absolute tolerance."
        ),
    }:
        raise ValueError("frozen numerical tie tolerance changed")
    if tuple(selection["complexity_order_low_to_high"]) != (
        EXPECTED_CANDIDATE_IDS[4],
        EXPECTED_CANDIDATE_IDS[5],
        EXPECTED_CANDIDATE_IDS[2],
        EXPECTED_CANDIDATE_IDS[3],
        EXPECTED_CANDIDATE_IDS[0],
        EXPECTED_CANDIDATE_IDS[1],
    ):
        raise ValueError("frozen candidate complexity order changed")
    if selection["single_weighted_aggregate_score_used"] is not False:
        raise ValueError("weighted model-selection score is prohibited")
    expected_hard_negative_pairs = [
        ["close_account", "unsupported_or_uncertain"],
        ["transfer_failed_or_declined", "unsupported_or_uncertain"],
        ["account_blocked", "unsupported_or_uncertain"],
        ["cancel_transfer", "unsupported_or_uncertain"],
        ["create_dispute", "unsupported_or_uncertain"],
        ["freeze_card", "unsupported_or_uncertain"],
        ["transfer_pending", "unsupported_or_uncertain"],
        ["cancel_transfer", "transfer_pending"],
        ["transfer_failed_or_declined", "transfer_pending"],
        ["account_blocked", "transfer_failed_or_declined"],
    ]
    if contract["hard_negative_diagnostics"]["required_pairs"] != (
        expected_hard_negative_pairs
    ):
        raise ValueError("frozen hard-negative pairs changed")

    tracked = contract["step29h_artifact_contract"]["tracked_outputs"]
    if tracked != [
        "data/evals/v2/ml/v2c6_source_aware_model_selection_results.json",
        (
            "data/evals/v2/ml/"
            "v2c6_source_aware_model_selection_results.manifest.json"
        ),
    ]:
        raise ValueError("Step 29H tracked output names changed")
    local = contract["step29h_artifact_contract"]["local_ignored_artifacts"]
    if [row["path"] for row in local] != [
        "data/evals/v2/ml/local/v2c6_source_aware_model_selection_bge_cache.npz",
        (
            "data/evals/v2/ml/local/"
            "v2c6_source_aware_model_selection_bge_cache.manifest.json"
        ),
    ]:
        raise ValueError("Step 29H local cache paths changed")
    if contract["threshold_policy"][
        "native_multiclass_prediction_rule_required"
    ] is not True:
        raise ValueError("native multiclass prediction is required")
    for key, value in contract["threshold_policy"].items():
        if key != "native_multiclass_prediction_rule_required" and value is not False:
            raise ValueError("threshold tuning or post-processing is prohibited")


def load_contract() -> tuple[dict[str, Any], str]:
    actual_sha256 = sha256_file(CONTRACT_PATH)
    if actual_sha256 != EXPECTED_CONTRACT_SHA256:
        raise ValueError(f"Step 29G contract SHA mismatch: {actual_sha256}")
    contract = read_json(CONTRACT_PATH)
    validate_contract(contract)
    return contract, actual_sha256


def validate_freeze(freeze: Mapping[str, Any], contract: Mapping[str, Any]) -> None:
    dataset_spec = contract["frozen_input_lineage"]["dataset"]
    if freeze.get("freeze_status") != "FROZEN":
        raise ValueError("Step 29F freeze_status must be FROZEN")
    if freeze.get("next_required") != "v2c6_source_aware_model_selection_contract":
        raise ValueError("Step 29F next_required changed")
    if (
        freeze.get("frozen_dataset_path") != dataset_spec["path"]
        or freeze.get("frozen_dataset_sha256") != dataset_spec["sha256"]
        or freeze.get("frozen_dataset_record_count") != 9008
        or freeze.get("inherited_v2c5_record_count") != 8198
        or freeze.get("remediation_record_count") != 810
    ):
        raise ValueError("Step 29F frozen dataset identity changed")
    if freeze.get("taxonomy") != {
        "intent_count": 16,
        "intent_label_order": contract["taxonomy"]["intent_label_order"],
        "path": "data/evals/v2/ml/v2c5_taxonomy_freeze.json",
        "risk_mapping_sha256": (
            "d0fe9fa6a0613b55bd7aebec1db4bc400d3ed9abb2e3df0ac067fa500cd247b1"
        ),
        "sha256": contract["taxonomy"]["sha256"],
        "version": contract["taxonomy"]["version"],
    }:
        raise ValueError("Step 29F taxonomy identity changed")


def validate_dataset(
    dataset: Mapping[str, Any], contract: Mapping[str, Any]
) -> tuple[dict[str, Any], ...]:
    if dataset.get("schema_version") != "v2c6-remediated-development-dataset.v1":
        raise ValueError("unexpected remediated development dataset schema")
    examples = dataset.get("examples")
    if not isinstance(examples, list):
        raise TypeError("development examples must be a list")
    if (
        dataset.get("example_count") != 9008
        or len(examples) != 9008
        or dataset.get("classifier_input_fields") != ["text"]
    ):
        raise ValueError("frozen development record population changed")
    labels = set(contract["taxonomy"]["intent_label_order"])
    expected_families = set(contract["source_family_holdout"]["source_families"])
    example_ids: set[str] = set()
    remediation_count = 0
    family_counts: Counter[str] = Counter()
    for index, row in enumerate(examples):
        if not isinstance(row, dict):
            raise TypeError("development record must be an object")
        example_id = row.get("example_id")
        group_id = row.get("group_id")
        text = row.get("text")
        intent = row.get("intent")
        if not isinstance(example_id, str) or not example_id:
            raise ValueError("development example_id must be a nonempty string")
        if example_id in example_ids:
            raise ValueError(f"duplicate development example_id: {example_id}")
        example_ids.add(example_id)
        if not isinstance(group_id, str) or not group_id:
            raise ValueError(f"invalid development group_id: {example_id}")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"invalid development text: {example_id}")
        if row.get("text_sha256") != sha256_bytes(text.encode("utf-8")):
            raise ValueError(f"development text hash mismatch: {example_id}")
        if intent not in labels:
            raise ValueError(f"unexpected development intent: {intent}")
        if row.get("data_role") != "development":
            raise ValueError(f"non-development record encountered: {example_id}")
        is_remediation = row.get("v2c6_lineage") == REMEDIATION_LINEAGE
        if index < 8198 and is_remediation:
            raise ValueError("remediation record occurs inside inherited prefix")
        if index >= 8198 and not is_remediation:
            raise ValueError("remediation suffix lineage changed")
        if is_remediation:
            family_id = row.get("source_family_id")
            if family_id not in expected_families:
                raise ValueError(f"unexpected remediation source family: {family_id}")
            if row.get("record_id") != example_id:
                raise ValueError("remediation record/example ID mismatch")
            remediation_count += 1
            family_counts[family_id] += 1
    if remediation_count != 810:
        raise ValueError("exactly 810 remediation records are required")
    if family_counts != Counter({family: 270 for family in expected_families}):
        raise ValueError("each remediation source family must contain 270 records")
    return tuple(examples)


def load_and_validate_inputs() -> LoadedInputs:
    contract, contract_sha256 = load_contract()
    freeze_path = validate_frozen_source(
        contract["frozen_input_lineage"]["step29f_freeze"]
    )
    if freeze_path != FREEZE_PATH.resolve():
        raise ValueError("Step 29F freeze path changed")
    freeze = read_json(freeze_path)
    validate_freeze(freeze, contract)

    dataset_path = validate_frozen_source(contract["frozen_input_lineage"]["dataset"])
    if dataset_path != DATASET_PATH.resolve():
        raise ValueError("frozen development dataset path changed")
    dataset = read_json(dataset_path)
    examples = validate_dataset(dataset, contract)

    lineage = freeze["parent_artifact_lineage"]
    combined_path = validate_frozen_source(lineage["combined_manifest"])
    remediation_path = validate_frozen_source(lineage["remediation_manifest"])
    validate_frozen_source(contract["source_artifacts"]["taxonomy_freeze"])
    combined_manifest = read_json(combined_path)
    remediation_manifest = read_json(remediation_path)
    if combined_manifest.get("remediated_development_dataset") != {
        "path": contract["frozen_input_lineage"]["dataset"]["path"],
        "schema_version": "v2c6-remediated-development-dataset.v1",
        "sha256": contract["frozen_input_lineage"]["dataset"]["sha256"],
    }:
        raise ValueError("combined manifest dataset lineage changed")
    if combined_manifest.get("counts", {}).get("total_combined_count") != 9008:
        raise ValueError("combined manifest count changed")
    if remediation_manifest.get("counts", {}).get("record_count") != 810:
        raise ValueError("remediation manifest count changed")
    if remediation_manifest.get("hard_negative_coverage", {}).get("pair_count") != 10:
        raise ValueError("remediation hard-negative lineage changed")
    return LoadedInputs(
        contract=contract,
        freeze=freeze,
        dataset=dataset,
        examples=examples,
        combined_manifest=combined_manifest,
        remediation_manifest=remediation_manifest,
        contract_sha256=contract_sha256,
    )


def membership_sha256(values: Iterable[str]) -> str:
    return sha256_bytes(stable_json_bytes(sorted(values)))


def _partition_audit(
    partition: Partition,
    examples: Sequence[Mapping[str, Any]],
    *,
    test_role: str,
) -> dict[str, Any]:
    train_rows = [examples[index] for index in partition.train_indices]
    test_rows = [examples[index] for index in partition.test_indices]
    train_groups = {str(row["group_id"]) for row in train_rows}
    test_groups = {str(row["group_id"]) for row in test_rows}
    test_ids = sorted(str(row["example_id"]) for row in test_rows)
    return {
        "partition_id": partition.partition_id,
        "held_out_source_family_id": partition.held_out_source_family_id,
        "train_count": len(train_rows),
        f"{test_role}_count": len(test_rows),
        f"{test_role}_example_ids": test_ids,
        "train_example_id_sha256": membership_sha256(
            str(row["example_id"]) for row in train_rows
        ),
        f"{test_role}_example_id_sha256": membership_sha256(test_ids),
        "train_group_id_sha256": membership_sha256(train_groups),
        f"{test_role}_group_id_sha256": membership_sha256(test_groups),
        "group_overlap_count": len(train_groups & test_groups),
    }


def build_group_folds(
    examples: Sequence[Mapping[str, Any]], contract: Mapping[str, Any]
) -> tuple[Partition, ...]:
    cv = contract["group_aware_cross_validation"]
    labels = np.asarray([row["intent"] for row in examples], dtype=object)
    groups = np.asarray([row["group_id"] for row in examples], dtype=object)
    splitter = StratifiedGroupKFold(
        n_splits=cv["n_splits"],
        shuffle=cv["shuffle"],
        random_state=cv["random_state"],
    )
    placeholder = np.zeros(len(examples), dtype=np.int8)
    validation_counts = np.zeros(len(examples), dtype=np.int8)
    partitions: list[Partition] = []
    for fold_index, (train_indices, validation_indices) in enumerate(
        splitter.split(placeholder, labels, groups)
    ):
        train = tuple(int(index) for index in train_indices)
        validation = tuple(int(index) for index in validation_indices)
        train_groups = {str(groups[index]) for index in train}
        validation_groups = {str(groups[index]) for index in validation}
        if train_groups & validation_groups:
            raise ValueError(f"group leakage detected in group_cv_fold_{fold_index}")
        validation_counts[np.asarray(validation, dtype=int)] += 1
        partitions.append(
            Partition(
                partition_id=f"group_cv_fold_{fold_index}",
                train_indices=train,
                test_indices=validation,
            )
        )
    if len(partitions) != 5:
        raise ValueError("exactly five group-CV folds are required")
    if not np.all(validation_counts == 1):
        raise ValueError("every record must receive exactly one OOF assignment")
    return tuple(partitions)


def build_source_family_rounds(
    examples: Sequence[Mapping[str, Any]], contract: Mapping[str, Any]
) -> tuple[Partition, ...]:
    source = contract["source_family_holdout"]
    remediation_indices = {
        index
        for index, row in enumerate(examples)
        if row.get("v2c6_lineage") == REMEDIATION_LINEAGE
    }
    pooled_test_counts = Counter[int]()
    rounds: list[Partition] = []
    for definition in source["rounds"]:
        family = definition["held_out_source_family_id"]
        test = tuple(
            index
            for index, row in enumerate(examples)
            if row.get("source_family_id") == family
            and row.get("v2c6_lineage") == REMEDIATION_LINEAGE
        )
        test_set = set(test)
        train = tuple(index for index in range(len(examples)) if index not in test_set)
        if len(test) != definition["test_count"] or len(train) != definition[
            "training_count"
        ]:
            raise ValueError(f"source-family split count mismatch: {family}")
        if any(
            examples[index].get("source_family_id") == family
            and examples[index].get("v2c6_lineage") == REMEDIATION_LINEAGE
            for index in train
        ):
            raise ValueError(f"held-out source family leaked into training: {family}")
        if not test_set.issubset(remediation_indices):
            raise ValueError("inherited record entered source-family test population")
        pooled_test_counts.update(test)
        rounds.append(
            Partition(
                partition_id=definition["round_id"],
                train_indices=train,
                test_indices=test,
                held_out_source_family_id=family,
            )
        )
    if len(rounds) != 3:
        raise ValueError("exactly three source-family rounds are required")
    if set(pooled_test_counts) != remediation_indices or any(
        count != 1 for count in pooled_test_counts.values()
    ):
        raise ValueError("every remediation record must be tested exactly once")
    return tuple(rounds)


def build_evaluation_splits(
    examples: Sequence[Mapping[str, Any]], contract: Mapping[str, Any]
) -> EvaluationSplits:
    group_folds = build_group_folds(examples, contract)
    source_rounds = build_source_family_rounds(examples, contract)
    group_audit = [
        _partition_audit(fold, examples, test_role="validation")
        for fold in group_folds
    ]
    source_audit = [
        _partition_audit(round_, examples, test_role="test")
        for round_ in source_rounds
    ]
    if any(row["group_overlap_count"] for row in group_audit):
        raise ValueError("group leakage detected in split audit")
    audit: dict[str, Any] = {
        "group_cv": {
            "fold_count": 5,
            "pooled_validation_count": sum(
                row["validation_count"] for row in group_audit
            ),
            "every_record_validated_exactly_once": True,
            "group_leakage_count": 0,
            "folds": group_audit,
        },
        "source_family_holdout": {
            "round_count": 3,
            "pooled_test_count": sum(row["test_count"] for row in source_audit),
            "every_remediation_record_tested_exactly_once": True,
            "source_family_leakage_count": 0,
            "rounds": source_audit,
        },
        "raw_text_persisted": False,
    }
    audit["audit_sha256"] = sha256_bytes(stable_json_bytes(audit))
    return EvaluationSplits(
        group_folds=group_folds,
        source_family_rounds=source_rounds,
        audit=audit,
    )


def candidate_configuration(
    contract: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    representation_id = candidate["representation_id"]
    classifier_id = candidate["classifier_id"]
    classifier = contract["classifiers"][classifier_id]
    return {
        "candidate_id": candidate["candidate_id"],
        "representation_id": representation_id,
        "representation": contract["representations"][representation_id],
        "classifier_id": classifier_id,
        "classifier": {
            "class": classifier["class"],
            "parameters": {
                **classifier["fixed_parameters"],
                **candidate["hyperparameters"],
            },
        },
    }


def build_representation(
    representation_id: str, contract: Mapping[str, Any]
) -> BaseEstimator:
    definitions = contract["representations"]
    if representation_id in {"WORD_TFIDF", "CHAR_TFIDF"}:
        definition = definitions[representation_id]
        if definition["class"] != "TfidfVectorizer":
            raise ValueError(f"unsupported frozen representation: {representation_id}")
        parameters = dict(EXPLICIT_TFIDF_DEFAULTS)
        parameters.update(definition["parameters"])
        parameters["ngram_range"] = tuple(parameters["ngram_range"])
        return TfidfVectorizer(**parameters)
    if representation_id == "WORD_CHAR_TFIDF":
        definition = definitions[representation_id]
        if definition["class"] != "FeatureUnion":
            raise ValueError("WORD_CHAR_TFIDF must use FeatureUnion")
        members = definition["members"]
        return FeatureUnion(
            [
                (member.lower(), build_representation(member, contract))
                for member in members
            ],
            n_jobs=None,
            transformer_weights=None,
            verbose=False,
        )
    if representation_id == "BGE_SMALL":
        raise ValueError("BGE_SMALL uses the validated partition embedding cache")
    raise ValueError(f"missing representation definition: {representation_id}")


def build_classifier(
    candidate: Mapping[str, Any], contract: Mapping[str, Any]
) -> BaseEstimator:
    classifier_id = candidate["classifier_id"]
    definition = contract["classifiers"].get(classifier_id)
    if definition is None:
        raise ValueError(f"missing classifier definition: {classifier_id}")
    if definition["class"] != "LinearSVC":
        raise ValueError(f"unsupported frozen classifier: {definition['class']}")
    parameters = {**definition["fixed_parameters"], **candidate["hyperparameters"]}
    return LinearSVC(**parameters)


def fit_classifier(
    classifier: BaseEstimator, features: Any, labels: np.ndarray
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        classifier.fit(features, labels)


def load_fastembed_model(contract: Mapping[str, Any]) -> Any:
    try:
        from fastembed import TextEmbedding
    except ImportError as exc:
        raise RuntimeError("fastembed is required for BGE_SMALL") from exc
    representation = contract["representations"]["BGE_SMALL"]
    try:
        return TextEmbedding(model_name=representation["model_identifier"])
    except Exception as exc:
        raise RuntimeError("FastEmbed could not load the frozen BGE model") from exc


def embed_texts(
    model: Any, texts: Sequence[str], contract: Mapping[str, Any]
) -> np.ndarray:
    representation = contract["representations"]["BGE_SMALL"]
    vectors = np.asarray(
        list(model.passage_embed(list(texts), batch_size=256)),
        dtype=np.float32,
    )
    expected_shape = (len(texts), representation["dimensions"])
    if vectors.shape != expected_shape or not np.isfinite(vectors).all():
        raise ValueError(f"unexpected BGE embedding matrix: {vectors.shape}")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("BGE returned a zero vector")
    normalized = np.asarray(vectors / norms, dtype=np.float32)
    if not np.allclose(
        np.linalg.norm(normalized, axis=1),
        1.0,
        rtol=1e-5,
        atol=1e-6,
    ):
        raise ValueError("BGE L2 normalization failed")
    return normalized


def _cache_matrix_key(partition: Partition, role: str) -> str:
    safe_partition = partition.partition_id.replace("-", "_")
    return f"{safe_partition}__{role}"


def _all_partitions(splits: EvaluationSplits) -> tuple[Partition, ...]:
    return (*splits.group_folds, *splits.source_family_rounds)


def _expected_cache_partitions(
    examples: Sequence[Mapping[str, Any]], splits: EvaluationSplits
) -> list[dict[str, Any]]:
    specifications: list[dict[str, Any]] = []
    for partition in _all_partitions(splits):
        test_role = (
            "validation" if partition.held_out_source_family_id is None else "test"
        )
        for role, indices in (
            ("train", partition.train_indices),
            (test_role, partition.test_indices),
        ):
            specifications.append(
                {
                    "matrix_key": _cache_matrix_key(partition, role),
                    "partition_id": partition.partition_id,
                    "role": role,
                    "row_count": len(indices),
                    "ordered_example_id_sha256": sha256_bytes(
                        stable_json_bytes(
                            [examples[index]["example_id"] for index in indices]
                        )
                    ),
                }
            )
    return specifications


def expected_bge_cache_manifest(
    inputs: LoadedInputs,
    splits: EvaluationSplits,
    *,
    cache_file_sha256: str,
    fastembed_version: str,
) -> dict[str, Any]:
    representation = inputs.contract["representations"]["BGE_SMALL"]
    return {
        "schema_version": BGE_CACHE_SCHEMA_VERSION,
        "dataset_sha256": inputs.contract["frozen_input_lineage"]["dataset"][
            "sha256"
        ],
        "contract_sha256": inputs.contract_sha256,
        "split_audit_sha256": splits.audit["audit_sha256"],
        "representation_id": "BGE_SMALL",
        "representation_sha256": sha256_bytes(stable_json_bytes(representation)),
        "model_identifier": representation["model_identifier"],
        "embedding_method": representation["embedding_method"],
        "dimensions": representation["dimensions"],
        "l2_normalized": representation["l2_normalized"],
        "partition_transforms_generated_separately": True,
        "partitions": _expected_cache_partitions(inputs.examples, splits),
        "cache_file_sha256": cache_file_sha256,
        "dtype": "float32",
        "fastembed_version": fastembed_version,
        "labels_used": False,
        "fine_tuning_performed": False,
        "raw_text_persisted": False,
        "final_holdout_accessed": False,
    }


def _validate_embedding_matrix(
    matrix: np.ndarray,
    expected_rows: int,
    contract: Mapping[str, Any],
) -> None:
    dimensions = contract["representations"]["BGE_SMALL"]["dimensions"]
    if matrix.shape != (expected_rows, dimensions):
        raise ValueError(f"BGE cache shape mismatch: {matrix.shape}")
    if matrix.dtype != np.float32:
        raise ValueError("BGE cache dtype must be float32")
    if not np.isfinite(matrix).all():
        raise ValueError("BGE cache contains non-finite values")
    if not np.allclose(
        np.linalg.norm(matrix, axis=1),
        1.0,
        rtol=1e-5,
        atol=1e-6,
    ):
        raise ValueError("BGE cache rows are not L2 normalized")


def validate_bge_cache(
    inputs: LoadedInputs, splits: EvaluationSplits
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if not BGE_CACHE_PATH.is_file() or not BGE_CACHE_MANIFEST_PATH.is_file():
        raise FileNotFoundError("complete local Step 29H BGE cache is required")
    manifest = read_json(BGE_CACHE_MANIFEST_PATH)
    expected = expected_bge_cache_manifest(
        inputs,
        splits,
        cache_file_sha256=str(manifest.get("cache_file_sha256", "")),
        fastembed_version=str(manifest.get("fastembed_version", "")),
    )
    if manifest != expected:
        raise ValueError("stale or incompatible Step 29H BGE cache manifest")
    if not manifest["fastembed_version"]:
        raise ValueError("BGE cache must record the FastEmbed version")
    if (
        not isinstance(manifest["cache_file_sha256"], str)
        or len(manifest["cache_file_sha256"]) != 64
    ):
        raise ValueError("BGE cache digest is invalid")
    if sha256_file(BGE_CACHE_PATH) != manifest["cache_file_sha256"]:
        raise ValueError("BGE cache file hash mismatch")
    expected_specs = {row["matrix_key"]: row for row in manifest["partitions"]}
    matrices: dict[str, np.ndarray] = {}
    with np.load(BGE_CACHE_PATH, allow_pickle=False) as cached:
        if set(cached.files) != set(expected_specs):
            raise ValueError("BGE cache partition matrix set changed")
        for key, specification in expected_specs.items():
            matrix = np.asarray(cached[key])
            _validate_embedding_matrix(
                matrix,
                specification["row_count"],
                inputs.contract,
            )
            matrices[key] = matrix
    return matrices, manifest


def prepare_bge_cache(
    inputs: LoadedInputs, splits: EvaluationSplits
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    existing = (BGE_CACHE_PATH.exists(), BGE_CACHE_MANIFEST_PATH.exists())
    if any(existing):
        if not all(existing):
            raise FileExistsError("incomplete Step 29H BGE cache pair")
        return validate_bge_cache(inputs, splits)
    model = load_fastembed_model(inputs.contract)
    matrices: dict[str, np.ndarray] = {}
    for partition in _all_partitions(splits):
        test_role = (
            "validation" if partition.held_out_source_family_id is None else "test"
        )
        for role, indices in (
            ("train", partition.train_indices),
            (test_role, partition.test_indices),
        ):
            texts = [inputs.examples[index]["text"] for index in indices]
            matrices[_cache_matrix_key(partition, role)] = embed_texts(
                model, texts, inputs.contract
            )
    BGE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=BGE_CACHE_PATH.parent,
        suffix=".npz",
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
    try:
        np.savez_compressed(temporary_path, **matrices)
        temporary_path.replace(BGE_CACHE_PATH)
    finally:
        temporary_path.unlink(missing_ok=True)
    manifest = expected_bge_cache_manifest(
        inputs,
        splits,
        cache_file_sha256=sha256_file(BGE_CACHE_PATH),
        fastembed_version=importlib.metadata.version("fastembed"),
    )
    write_bytes_atomically(BGE_CACHE_MANIFEST_PATH, stable_json_bytes(manifest))
    return validate_bge_cache(inputs, splits)


def _ratio(numerator: int, denominator: int, name: str) -> float:
    if denominator <= 0:
        raise ValueError(f"metric denominator is zero: {name}")
    return numerator / denominator


def safety_metrics(
    gold: Sequence[str],
    predicted: Sequence[str],
    protected_intents: Sequence[str],
) -> dict[str, float | int]:
    if len(gold) != len(predicted):
        raise ValueError("gold and prediction lengths differ")
    protected = set(protected_intents)
    pairs = list(zip(gold, predicted, strict=True))
    protected_total = sum(expected in protected for expected, _ in pairs)
    protected_correct = sum(
        expected in protected and guess in protected for expected, guess in pairs
    )
    non_protected_total = sum(expected not in protected for expected, _ in pairs)
    protected_false_positives = sum(
        expected not in protected and guess in protected for expected, guess in pairs
    )
    unsupported_total = sum(expected == UNSUPPORTED_INTENT for expected, _ in pairs)
    unsupported_correct = sum(
        expected == UNSUPPORTED_INTENT and guess == UNSUPPORTED_INTENT
        for expected, guess in pairs
    )
    return {
        "protected_gold_count": protected_total,
        "protected_prediction_as_protected_count": protected_correct,
        "protected_recall": _ratio(
            protected_correct, protected_total, "protected_recall"
        ),
        "non_protected_gold_count": non_protected_total,
        "protected_false_positive_count": protected_false_positives,
        "protected_false_positive_rate": _ratio(
            protected_false_positives,
            non_protected_total,
            "protected_false_positive_rate",
        ),
        "unsupported_gold_count": unsupported_total,
        "unsupported_correct_count": unsupported_correct,
        "unsupported_recall": _ratio(
            unsupported_correct, unsupported_total, "unsupported_recall"
        ),
    }


def classification_metrics(
    gold: Sequence[str], predicted: Sequence[str], labels: Sequence[str]
) -> dict[str, Any]:
    if len(gold) != len(predicted) or not gold:
        raise ValueError("classification metrics require equal nonempty populations")
    precision, recall, f1, support = precision_recall_fscore_support(
        gold,
        predicted,
        labels=list(labels),
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
    return {
        "accuracy": float(accuracy_score(gold, predicted)),
        "balanced_accuracy": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "per_intent": per_intent,
        "confusion_matrix": {
            "label_order": list(labels),
            "values": confusion_matrix(gold, predicted, labels=list(labels)).tolist(),
        },
    }


def subset_macro_f1(
    gold: Sequence[str],
    predicted: Sequence[str],
    labels: Sequence[str],
) -> float:
    indices = [index for index, expected in enumerate(gold) if expected in labels]
    subset_gold = [gold[index] for index in indices]
    subset_predicted = [predicted[index] for index in indices]
    return float(
        f1_score(
            subset_gold,
            subset_predicted,
            labels=list(labels),
            average="macro",
            zero_division=0,
        )
    )


def unsupported_boundary_metrics(
    gold: Sequence[str],
    predicted: Sequence[str],
    primary_intents: Sequence[str],
) -> dict[str, float | int]:
    supported = set(primary_intents) - {UNSUPPORTED_INTENT}
    pairs = list(zip(gold, predicted, strict=True))
    supported_total = sum(expected in supported for expected, _ in pairs)
    supported_to_unsupported = sum(
        expected in supported and guess == UNSUPPORTED_INTENT
        for expected, guess in pairs
    )
    unsupported_total = sum(expected == UNSUPPORTED_INTENT for expected, _ in pairs)
    unsupported_to_supported = sum(
        expected == UNSUPPORTED_INTENT and guess != UNSUPPORTED_INTENT
        for expected, guess in pairs
    )
    return {
        "supported_gold_count": supported_total,
        "supported_predicted_unsupported_count": supported_to_unsupported,
        "supported_to_unsupported_rate": _ratio(
            supported_to_unsupported,
            supported_total,
            "supported_to_unsupported_rate",
        ),
        "unsupported_gold_count": unsupported_total,
        "unsupported_predicted_supported_count": unsupported_to_supported,
        "unsupported_to_supported_rate": _ratio(
            unsupported_to_supported,
            unsupported_total,
            "unsupported_to_supported_rate",
        ),
    }


def hard_negative_diagnostics(
    records: Sequence[Mapping[str, Any]],
    predicted: Sequence[str],
    required_pairs: Sequence[Sequence[str]],
) -> dict[str, Any]:
    if len(records) != len(predicted):
        raise ValueError("hard-negative records and predictions differ in length")
    pair_results: list[dict[str, Any]] = []
    aggregate_record_count = 0
    aggregate_correct_count = 0
    for pair in required_pairs:
        if len(pair) != 2:
            raise ValueError("hard-negative pair must contain exactly two intents")
        side_a, side_b = pair
        selected = [
            (row, guess)
            for row, guess in zip(records, predicted, strict=True)
            if row.get("is_hard_negative") is True
            and (
                (row.get("intent") == side_a and row.get("boundary_target") == side_b)
                or (
                    row.get("intent") == side_b
                    and row.get("boundary_target") == side_a
                )
            )
        ]
        record_count = len(selected)
        if record_count <= 0:
            raise ValueError(f"hard-negative pair has no records: {pair}")
        correct = sum(row["intent"] == guess for row, guess in selected)
        a_to_b = sum(
            row["intent"] == side_a and guess == side_b for row, guess in selected
        )
        b_to_a = sum(
            row["intent"] == side_b and guess == side_a for row, guess in selected
        )
        other_error = record_count - correct - a_to_b - b_to_a
        pair_results.append(
            {
                "pair": [side_a, side_b],
                "record_count": record_count,
                "correct_count": correct,
                "accuracy": correct / record_count,
                "true_a_predicted_b_count": a_to_b,
                "true_b_predicted_a_count": b_to_a,
                "other_error_count": other_error,
            }
        )
        aggregate_record_count += record_count
        aggregate_correct_count += correct
    return {
        "pair_count": len(pair_results),
        "pairs": pair_results,
        "aggregate_record_count": aggregate_record_count,
        "aggregate_correct_count": aggregate_correct_count,
        "aggregate_hard_negative_accuracy": _ratio(
            aggregate_correct_count,
            aggregate_record_count,
            "aggregate_hard_negative_accuracy",
        ),
        "mandatory_safety_gate": False,
    }


def _require_finite_metrics(value: Any, path: str = "metrics") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError(f"non-finite metric: {path}")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            _require_finite_metrics(child, f"{path}.{key}")
        return
    if isinstance(value, Sequence):
        for index, child in enumerate(value):
            _require_finite_metrics(child, f"{path}[{index}]")


def _prediction_rows(
    records: Sequence[Mapping[str, Any]],
    predicted: Sequence[str],
    *,
    assignment_key: str,
    assignment_value: str,
) -> list[dict[str, str]]:
    return [
        {
            "example_id": str(row["example_id"]),
            "gold_intent": str(row["intent"]),
            "predicted_intent": str(guess),
            assignment_key: assignment_value,
        }
        for row, guess in zip(records, predicted, strict=True)
    ]


def _fit_and_predict_partition(
    candidate: Mapping[str, Any],
    partition: Partition,
    examples: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    bge_matrices: Mapping[str, np.ndarray] | None,
) -> np.ndarray:
    train_rows = [examples[index] for index in partition.train_indices]
    test_rows = [examples[index] for index in partition.test_indices]
    train_labels = np.asarray([row["intent"] for row in train_rows], dtype=object)
    if candidate["representation_id"] == "BGE_SMALL":
        if bge_matrices is None:
            raise ValueError("BGE candidate requires the validated partition cache")
        test_role = (
            "validation" if partition.held_out_source_family_id is None else "test"
        )
        train_features = bge_matrices[_cache_matrix_key(partition, "train")]
        test_features = bge_matrices[_cache_matrix_key(partition, test_role)]
    else:
        representation = build_representation(
            str(candidate["representation_id"]), contract
        )
        train_features = representation.fit_transform(
            [str(row["text"]) for row in train_rows]
        )
        test_features = representation.transform(
            [str(row["text"]) for row in test_rows]
        )
    classifier = build_classifier(candidate, contract)
    fit_classifier(classifier, train_features, train_labels)
    predicted = np.asarray(classifier.predict(test_features), dtype=object)
    if predicted.shape != (len(test_rows),):
        raise ValueError("classifier prediction shape changed")
    allowed_labels = set(contract["taxonomy"]["intent_label_order"])
    if not set(predicted.tolist()).issubset(allowed_labels):
        raise ValueError("classifier predicted outside the frozen taxonomy")
    return predicted


def _group_cv_result(
    candidate: Mapping[str, Any],
    examples: Sequence[Mapping[str, Any]],
    splits: EvaluationSplits,
    contract: Mapping[str, Any],
    bge_matrices: Mapping[str, np.ndarray] | None,
) -> dict[str, Any]:
    predictions: list[str | None] = [None] * len(examples)
    fold_summaries: list[dict[str, Any]] = []
    for fold in splits.group_folds:
        guessed = _fit_and_predict_partition(
            candidate, fold, examples, contract, bge_matrices
        ).tolist()
        for index, guess in zip(fold.test_indices, guessed, strict=True):
            if predictions[index] is not None:
                raise ValueError("duplicate OOF prediction")
            predictions[index] = str(guess)
        fold_summaries.append(
            {
                "fold_id": fold.partition_id,
                "train_count": len(fold.train_indices),
                "validation_count": len(fold.test_indices),
                "prediction_count": len(guessed),
            }
        )
    if any(value is None for value in predictions):
        raise ValueError("missing OOF prediction")
    predicted = [str(value) for value in predictions]
    gold = [str(row["intent"]) for row in examples]
    labels = contract["taxonomy"]["intent_label_order"]
    metrics = classification_metrics(gold, predicted, labels)
    metrics["macro_f1_16"] = metrics.pop("macro_f1")
    metrics["historical_9_macro_f1"] = subset_macro_f1(
        gold, predicted, contract["taxonomy"]["historical_9_intents"]
    )
    metrics["expanded_new_7_macro_f1"] = subset_macro_f1(
        gold, predicted, contract["taxonomy"]["expanded_new_7_intents"]
    )
    metrics["safety"] = safety_metrics(
        gold, predicted, contract["safety_gates"]["protected_intents"]
    )
    rows = _prediction_rows(
        examples,
        predicted,
        assignment_key="evaluation_scope",
        assignment_value="pooled_group_aware_cv",
    )
    return {
        "completed_fold_count": len(fold_summaries),
        "folds": fold_summaries,
        "prediction_count": len(rows),
        "prediction_sha256": sha256_bytes(stable_json_bytes(rows)),
        "predictions": rows,
        "metrics": metrics,
    }


def _source_family_result(
    candidate: Mapping[str, Any],
    examples: Sequence[Mapping[str, Any]],
    splits: EvaluationSplits,
    contract: Mapping[str, Any],
    bge_matrices: Mapping[str, np.ndarray] | None,
) -> dict[str, Any]:
    primary = contract["taxonomy"]["primary_remediation_8_intents"]
    required_pairs = contract["hard_negative_diagnostics"]["required_pairs"]
    pooled_rows: list[dict[str, Any]] = []
    pooled_predictions: list[str] = []
    rounds: list[dict[str, Any]] = []
    tested_ids: set[str] = set()
    for round_ in splits.source_family_rounds:
        test_records = [examples[index] for index in round_.test_indices]
        guessed = _fit_and_predict_partition(
            candidate, round_, examples, contract, bge_matrices
        ).tolist()
        gold = [str(row["intent"]) for row in test_records]
        predicted = [str(value) for value in guessed]
        metrics = classification_metrics(gold, predicted, primary)
        metrics["primary_8_macro_f1"] = metrics.pop("macro_f1")
        metrics["safety"] = safety_metrics(
            gold, predicted, contract["safety_gates"]["protected_intents"]
        )
        metrics["unsupported_boundary"] = unsupported_boundary_metrics(
            gold, predicted, primary
        )
        metrics["hard_negative_diagnostics"] = hard_negative_diagnostics(
            test_records, predicted, required_pairs
        )
        rows = _prediction_rows(
            test_records,
            predicted,
            assignment_key="held_out_source_family_id",
            assignment_value=str(round_.held_out_source_family_id),
        )
        for row in rows:
            if row["example_id"] in tested_ids:
                raise ValueError("duplicate out-of-family prediction")
            tested_ids.add(row["example_id"])
        pooled_rows.extend(rows)
        pooled_predictions.extend(predicted)
        rounds.append(
            {
                "round_id": round_.partition_id,
                "held_out_source_family_id": round_.held_out_source_family_id,
                "training_count": len(round_.train_indices),
                "test_count": len(round_.test_indices),
                "prediction_count": len(predicted),
                "metrics": metrics,
            }
        )
    if len(pooled_rows) != 810 or len(tested_ids) != 810:
        raise ValueError("pooled out-of-family prediction coverage must equal 810")
    pooled_records = [
        examples[index]
        for round_ in splits.source_family_rounds
        for index in round_.test_indices
    ]
    pooled_gold = [str(row["intent"]) for row in pooled_records]
    pooled = classification_metrics(pooled_gold, pooled_predictions, primary)
    pooled["pooled_primary_8_macro_f1"] = pooled.pop("macro_f1")
    family_scores = [row["metrics"]["primary_8_macro_f1"] for row in rounds]
    pooled["mean_family_primary_8_macro_f1"] = float(np.mean(family_scores))
    pooled["worst_family_primary_8_macro_f1"] = float(min(family_scores))
    pooled["safety"] = safety_metrics(
        pooled_gold,
        pooled_predictions,
        contract["safety_gates"]["protected_intents"],
    )
    pooled["unsupported_boundary"] = unsupported_boundary_metrics(
        pooled_gold, pooled_predictions, primary
    )
    pooled["hard_negative_diagnostics"] = hard_negative_diagnostics(
        pooled_records, pooled_predictions, required_pairs
    )
    return {
        "completed_round_count": len(rounds),
        "rounds": rounds,
        "pooled_prediction_count": len(pooled_rows),
        "pooled_prediction_sha256": sha256_bytes(stable_json_bytes(pooled_rows)),
        "pooled_predictions": pooled_rows,
        "pooled_metrics": pooled,
    }


def apply_mandatory_safety_gates(
    group_result: Mapping[str, Any],
    source_result: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    scopes: list[tuple[str, Mapping[str, Any]]] = [
        (
            "pooled_group_aware_cv_predictions",
            group_result["metrics"]["safety"],
        ),
        (
            "pooled_source_family_holdout_predictions",
            source_result["pooled_metrics"]["safety"],
        ),
    ]
    scopes.extend(
        (
            f"individual_source_family_holdout:{row['held_out_source_family_id']}",
            row["metrics"]["safety"],
        )
        for row in source_result["rounds"]
    )
    gate_results: list[dict[str, Any]] = []
    for scope, metrics in scopes:
        for gate in contract["safety_gates"]["gates"]:
            observed = float(metrics[gate["metric"]])
            if gate["comparison"] == "greater_than_or_equal":
                passed = observed >= gate["threshold"]
                comparator = ">="
            elif gate["comparison"] == "less_than_or_equal":
                passed = observed <= gate["threshold"]
                comparator = "<="
            else:
                raise ValueError(f"unknown safety comparison: {gate['comparison']}")
            gate_results.append(
                {
                    "scope": scope,
                    "metric": gate["metric"],
                    "observed_value": observed,
                    "comparator": comparator,
                    "threshold": gate["threshold"],
                    "passed": bool(passed),
                }
            )
    if len(gate_results) != 15:
        raise ValueError("exactly fifteen mandatory safety-gate checks are required")
    return {
        "required_scope_count": 5,
        "gate_count_per_scope": 3,
        "all_gates_pass": all(row["passed"] for row in gate_results),
        "gates": gate_results,
    }


def determine_candidate_eligibility(
    safety_gate_results: Mapping[str, Any],
    *,
    completed_group_folds: int,
    completed_source_rounds: int,
    group_prediction_count: int,
    source_prediction_count: int,
) -> tuple[bool, list[str]]:
    infrastructure = {
        "all_5_group_cv_folds_completed": completed_group_folds == 5,
        "all_3_source_family_rounds_completed": completed_source_rounds == 3,
        "group_cv_prediction_coverage_exact": group_prediction_count == 9008,
        "source_family_prediction_coverage_exact": source_prediction_count == 810,
        "no_group_leakage": True,
        "no_source_family_leakage": True,
        "frozen_hashes_match": True,
        "metrics_finite_and_valid": True,
        "no_threshold_tuning": True,
        "no_prohibited_final_holdout_access": True,
    }
    failed_infrastructure = [
        name for name, passed in infrastructure.items() if not passed
    ]
    if failed_infrastructure:
        raise RuntimeError(
            "candidate execution incomplete: " + ", ".join(failed_infrastructure)
        )
    reasons = [
        f"mandatory_safety_gate_failed:{row['scope']}:{row['metric']}"
        for row in safety_gate_results["gates"]
        if row["passed"] is False
    ]
    return not reasons, reasons


def _selection_metrics(
    candidate_id: str,
    group_result: Mapping[str, Any],
    source_result: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, float | int | str]:
    pooled = source_result["pooled_metrics"]
    complexity = contract["selection_rule"]["complexity_order_low_to_high"]
    metrics: dict[str, float | int | str] = {
        "worst_family_primary_8_macro_f1": pooled[
            "worst_family_primary_8_macro_f1"
        ],
        "mean_family_primary_8_macro_f1": pooled[
            "mean_family_primary_8_macro_f1"
        ],
        "pooled_source_family_primary_8_macro_f1": pooled[
            "pooled_primary_8_macro_f1"
        ],
        "pooled_group_cv_macro_f1_16": group_result["metrics"]["macro_f1_16"],
        "pooled_source_family_supported_to_unsupported_rate": pooled[
            "unsupported_boundary"
        ]["supported_to_unsupported_rate"],
        "pooled_source_family_protected_false_positive_rate": pooled["safety"][
            "protected_false_positive_rate"
        ],
        "pooled_source_family_protected_recall": pooled["safety"][
            "protected_recall"
        ],
        "pooled_source_family_unsupported_recall": pooled["safety"][
            "unsupported_recall"
        ],
        "candidate_complexity_rank": complexity.index(candidate_id) + 1,
        "candidate_id": candidate_id,
    }
    if metrics["pooled_source_family_primary_8_macro_f1"] != pooled[
        "pooled_primary_8_macro_f1"
    ]:
        raise ValueError("source-family primary-eight metric alias is inconsistent")
    return metrics


def evaluate_candidate(
    candidate: Mapping[str, Any],
    inputs: LoadedInputs,
    splits: EvaluationSplits,
    bge_matrices: Mapping[str, np.ndarray] | None,
) -> dict[str, Any]:
    group = _group_cv_result(
        candidate, inputs.examples, splits, inputs.contract, bge_matrices
    )
    source = _source_family_result(
        candidate, inputs.examples, splits, inputs.contract, bge_matrices
    )
    _require_finite_metrics(group["metrics"])
    _require_finite_metrics(source["pooled_metrics"])
    for round_result in source["rounds"]:
        _require_finite_metrics(round_result["metrics"])
    gates = apply_mandatory_safety_gates(group, source, inputs.contract)
    eligible, reasons = determine_candidate_eligibility(
        gates,
        completed_group_folds=group["completed_fold_count"],
        completed_source_rounds=source["completed_round_count"],
        group_prediction_count=group["prediction_count"],
        source_prediction_count=source["pooled_prediction_count"],
    )
    candidate_id = str(candidate["candidate_id"])
    configuration = candidate_configuration(inputs.contract, candidate)
    return {
        "candidate_id": candidate_id,
        "configuration": configuration,
        "configuration_sha256": sha256_bytes(stable_json_bytes(configuration)),
        "execution_status": "COMPLETED",
        "group_aware_cv": group,
        "source_family_holdout": source,
        "safety_gate_results": gates,
        "eligible": eligible,
        "ineligibility_reasons": reasons,
        "selection_metrics": _selection_metrics(
            candidate_id, group, source, inputs.contract
        ),
        "threshold_tuning_performed": False,
        "raw_text_persisted": False,
    }


def candidate_is_better(
    candidate: Mapping[str, Any],
    incumbent: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> bool:
    tolerance = contract["selection_rule"]["numerical_tie_tolerance"]["absolute"]
    for criterion in contract["selection_rule"]["ordered_lexicographic_criteria"]:
        metric = criterion["metric"]
        direction = criterion["direction"]
        left = candidate["selection_metrics"][metric]
        right = incumbent["selection_metrics"][metric]
        if metric == "candidate_id":
            return str(left) < str(right)
        left_number = float(left)
        right_number = float(right)
        if abs(left_number - right_number) <= tolerance:
            continue
        if direction == "maximize":
            return left_number > right_number
        if direction in {"minimize", "prefer_lower_rank"}:
            return left_number < right_number
        raise ValueError(f"unknown selection direction: {direction}")
    return False


def select_candidate(
    candidate_results: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    eligible = [row for row in candidate_results if row["eligible"] is True]
    if not eligible:
        failure = contract["selection_rule"]["if_no_candidate_is_eligible"]
        if failure != {
            "gates_weakened": False,
            "result": "NO_ACCEPTABLE_CANDIDATE",
            "selected_candidate": None,
            "winner_forced": False,
        }:
            raise ValueError("frozen no-acceptable-candidate behavior changed")
        return {
            "selection_status": "NO_ACCEPTABLE_CANDIDATE",
            "eligible_candidate_count": 0,
            "selected_candidate_id": None,
            "selected_candidate": None,
            "winner_forced": False,
            "gates_weakened": False,
            "next_required": None,
            "next_required_contract_ambiguity": (
                "Step 29G does not declare a failure-path next_required; "
                "execution fails closed without authorizing Step 29I."
            ),
        }
    selected = eligible[0]
    for candidate in eligible[1:]:
        if candidate_is_better(candidate, selected, contract):
            selected = candidate
    return {
        "selection_status": "SELECTED",
        "eligible_candidate_count": len(eligible),
        "selected_candidate_id": selected["candidate_id"],
        "selected_candidate": selected["configuration"],
        "winner_forced": False,
        "gates_weakened": False,
        "next_required": "v2c6_selected_model_fit",
        "next_required_contract_ambiguity": None,
    }


def all_string_values(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key)
            yield from all_string_values(child)
    elif isinstance(value, Sequence):
        for child in value:
            yield from all_string_values(child)


def validate_text_free(
    payload: Mapping[str, Any], examples: Sequence[Mapping[str, Any]]
) -> None:
    strings = set(all_string_values(payload))
    raw_texts = {str(row["text"]) for row in examples}
    if strings & raw_texts:
        raise ValueError("tracked Step 29H artifact contains raw development text")
    prohibited_keys = {"text", "texts", "utterance", "utterances", "raw_text"}
    if strings & prohibited_keys:
        raise ValueError("tracked Step 29H artifact contains a raw-text field")


def build_results_payload(
    inputs: LoadedInputs,
    splits: EvaluationSplits,
    candidate_results: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    candidates = inputs.contract["candidate_search_space"]["candidates"]
    if [row["candidate_id"] for row in candidate_results] != [
        row["candidate_id"] for row in candidates
    ]:
        raise ValueError("complete ordered six-candidate results are required")
    selection = select_candidate(candidate_results, inputs.contract)
    payload = {
        "schema_version": RESULTS_SCHEMA_VERSION,
        "phase": "V2-C6 Step 29H",
        "execution_status": "COMPLETED",
        "contract": {
            "path": str(CONTRACT_PATH.relative_to(REPOSITORY_ROOT)),
            "sha256": inputs.contract_sha256,
            "schema_version": CONTRACT_SCHEMA_VERSION,
        },
        "frozen_inputs": {
            "step29f_freeze": inputs.contract["frozen_input_lineage"][
                "step29f_freeze"
            ],
            "development_dataset": inputs.contract["frozen_input_lineage"][
                "dataset"
            ],
            "source_artifacts": inputs.contract["source_artifacts"],
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "fastembed": importlib.metadata.version("fastembed"),
        },
        "candidate_definitions": candidates,
        "representation_definitions": inputs.contract["representations"],
        "classifier_definitions": inputs.contract["classifiers"],
        "split_audit": splits.audit,
        "candidate_count": 6,
        "completed_candidate_count": len(candidate_results),
        "candidate_results": list(candidate_results),
        "selection_rule": inputs.contract["selection_rule"],
        "selection": selection,
        "next_required": selection["next_required"],
        "known_limitations": inputs.contract["known_limitations"],
        "governance": {
            "development_model_selection_performed": True,
            "model_selection_performed": True,
            "model_training_performed": False,
            "temporary_fold_and_round_classifier_fitting_performed": True,
            "final_full_development_model_fitting_performed": False,
            "final_holdout_accessed": False,
            "final_holdout_evaluated": False,
            "threshold_tuning_performed": False,
            "runtime_behavior_changed": False,
            "final_model_acceptance_claimed": False,
            "production_ready_claimed": False,
            "full_development_fitted_classifier_persisted": False,
        },
    }
    validate_results_payload(payload, inputs, splits)
    return payload


def validate_results_payload(
    payload: Mapping[str, Any],
    inputs: LoadedInputs,
    splits: EvaluationSplits,
) -> None:
    if payload.get("schema_version") != RESULTS_SCHEMA_VERSION:
        raise ValueError("Step 29H results schema changed")
    if payload.get("contract", {}).get("sha256") != inputs.contract_sha256:
        raise ValueError("Step 29H contract lineage changed")
    if payload.get("split_audit") != splits.audit:
        raise ValueError("Step 29H split audit changed")
    results = payload.get("candidate_results")
    if not isinstance(results, list) or len(results) != 6:
        raise ValueError("all six candidate results are required")
    if payload.get("candidate_count") != 6 or payload.get(
        "completed_candidate_count"
    ) != 6:
        raise ValueError("Step 29H candidate completion count changed")
    candidates = inputs.contract["candidate_search_space"]["candidates"]
    for result, candidate in zip(results, candidates, strict=True):
        expected_configuration = candidate_configuration(inputs.contract, candidate)
        if result.get("candidate_id") != candidate["candidate_id"]:
            raise ValueError("candidate result ID or order changed")
        if result.get("configuration") != expected_configuration:
            raise ValueError("candidate result configuration changed")
        if result.get("configuration_sha256") != sha256_bytes(
            stable_json_bytes(expected_configuration)
        ):
            raise ValueError("candidate configuration digest changed")
        if result.get("execution_status") != "COMPLETED":
            raise ValueError("candidate infrastructure execution was incomplete")
        if result.get("group_aware_cv", {}).get("prediction_count") != 9008:
            raise ValueError("candidate OOF coverage changed")
        if result.get("source_family_holdout", {}).get(
            "pooled_prediction_count"
        ) != 810:
            raise ValueError("candidate source-family coverage changed")
        if result.get("eligible") is not (not result.get("ineligibility_reasons")):
            raise ValueError("candidate eligibility reasons are inconsistent")
        _require_finite_metrics(result["selection_metrics"])
    if payload.get("selection") != select_candidate(results, inputs.contract):
        raise ValueError("deterministic candidate selection changed")
    governance = payload.get("governance", {})
    if (
        governance.get("final_holdout_accessed") is not False
        or governance.get("threshold_tuning_performed") is not False
        or governance.get("runtime_behavior_changed") is not False
        or governance.get("final_model_acceptance_claimed") is not False
        or governance.get("model_training_performed") is not False
        or governance.get("final_full_development_model_fitting_performed") is not False
    ):
        raise ValueError("Step 29H governance protections changed")
    validate_text_free(payload, inputs.examples)


def build_results_manifest(
    payload: Mapping[str, Any], result_bytes: bytes
) -> dict[str, Any]:
    return {
        "schema_version": RESULTS_MANIFEST_SCHEMA_VERSION,
        "phase": "V2-C6 Step 29H",
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
        "frozen_inputs": payload["frozen_inputs"],
        "split_audit_sha256": payload["split_audit"]["audit_sha256"],
        "candidate_count": payload["candidate_count"],
        "completed_candidate_count": payload["completed_candidate_count"],
        "selection_status": payload["selection"]["selection_status"],
        "selected_candidate_id": payload["selection"]["selected_candidate_id"],
        "next_required": payload["next_required"],
        "development_model_selection_performed": True,
        "model_selection_performed": True,
        "model_training_performed": False,
        "final_full_development_model_fitting_performed": False,
        "final_holdout_accessed": False,
        "final_holdout_evaluated": False,
        "threshold_tuning_performed": False,
        "runtime_behavior_changed": False,
        "final_model_acceptance_claimed": False,
    }


def run_model_selection(inputs: LoadedInputs) -> tuple[Path, Path]:
    if RESULTS_PATH.exists() or RESULTS_MANIFEST_PATH.exists():
        raise FileExistsError("Step 29H result artifact exists; refusing overwrite")
    splits = build_evaluation_splits(inputs.examples, inputs.contract)
    bge_matrices, _ = prepare_bge_cache(inputs, splits)
    candidate_results: list[dict[str, Any]] = []
    for candidate in inputs.contract["candidate_search_space"]["candidates"]:
        candidate_results.append(
            evaluate_candidate(
                candidate,
                inputs,
                splits,
                bge_matrices
                if candidate["representation_id"] == "BGE_SMALL"
                else None,
            )
        )
    payload = build_results_payload(inputs, splits, candidate_results)
    result_bytes = stable_json_bytes(payload)
    manifest_bytes = stable_json_bytes(build_results_manifest(payload, result_bytes))
    write_bytes_atomically(RESULTS_PATH, result_bytes)
    write_bytes_atomically(RESULTS_MANIFEST_PATH, manifest_bytes)
    return RESULTS_PATH, RESULTS_MANIFEST_PATH


def preflight(inputs: LoadedInputs) -> dict[str, Any]:
    splits = build_evaluation_splits(inputs.examples, inputs.contract)
    packages: dict[str, str | None] = {}
    for package in ("numpy", "scikit-learn", "fastembed"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    cache_presence = (BGE_CACHE_PATH.exists(), BGE_CACHE_MANIFEST_PATH.exists())
    if any(cache_presence) and not all(cache_presence):
        raise FileExistsError("incomplete existing Step 29H BGE cache pair")
    if all(cache_presence):
        validate_bge_cache(inputs, splits)
    return {
        "status": "READY" if all(packages.values()) else "MISSING_REQUIRED_PACKAGE",
        "contract_sha256": inputs.contract_sha256,
        "development_dataset_sha256": inputs.contract["frozen_input_lineage"][
            "dataset"
        ]["sha256"],
        "development_record_count": len(inputs.examples),
        "candidate_count": 6,
        "group_cv_fold_count": len(splits.group_folds),
        "source_family_round_count": len(splits.source_family_rounds),
        "expected_group_cv_fits": 30,
        "expected_source_family_fits": 18,
        "expected_total_temporary_classifier_fits": 48,
        "split_audit_sha256": splits.audit["audit_sha256"],
        "bge_cache_present": all(cache_presence),
        "result_artifacts_present": RESULTS_PATH.exists()
        or RESULTS_MANIFEST_PATH.exists(),
        "packages": packages,
        "embeddings_generated": False,
        "classifier_fitting_performed": False,
        "inference_performed": False,
        "model_selection_performed": False,
        "model_training_performed": False,
        "files_written": False,
        "final_holdout_accessed": False,
        "runtime_behavior_changed": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Execute frozen V2-C6 source-aware development model selection"
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    inputs = load_and_validate_inputs()
    if args.preflight:
        print(json.dumps(preflight(inputs), indent=2, sort_keys=True))
        return
    paths = run_model_selection(inputs)
    print("\n".join(str(path.relative_to(REPOSITORY_ROOT)) for path in paths))


if __name__ == "__main__":
    main()
