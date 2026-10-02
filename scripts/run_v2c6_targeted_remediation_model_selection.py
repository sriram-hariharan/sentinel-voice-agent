"""Run the frozen V2-C6 targeted-remediation development model selection.

The experiment is development-only. Fresh source-evaluation records are never
used to fit a representation or classifier, and every final-holdout path is
rejected. No fitted model is persisted and Step 29I remains blocked.
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
from scipy import sparse
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
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ML_DIRECTORY = REPOSITORY_ROOT / "data/evals/v2/ml"
CONTRACT_PATH = ML_DIRECTORY / "v2c6_targeted_remediation_design_contract.json"
STEP29G_CONTRACT_PATH = ML_DIRECTORY / "v2c6_source_aware_model_selection_contract.json"
DEVELOPMENT_DATASET_PATH = (
    ML_DIRECTORY / "v2c6_targeted_remediated_development_dataset.json"
)
DEVELOPMENT_MANIFEST_PATH = (
    ML_DIRECTORY / "v2c6_targeted_remediated_development_dataset.manifest.json"
)
FRESH_DATASET_PATH = ML_DIRECTORY / "v2c6_fresh_source_evaluation_dataset.json"
FRESH_MANIFEST_PATH = (
    ML_DIRECTORY / "v2c6_fresh_source_evaluation_dataset.manifest.json"
)
RESULTS_PATH = (
    ML_DIRECTORY / "v2c6_targeted_remediation_model_selection_results.json"
)
RESULTS_MANIFEST_PATH = ML_DIRECTORY / (
    "v2c6_targeted_remediation_model_selection_results.manifest.json"
)
BGE_CACHE_PATH = ML_DIRECTORY / (
    "local/v2c6_targeted_remediation_model_selection_bge_cache.npz"
)
BGE_CACHE_MANIFEST_PATH = ML_DIRECTORY / (
    "local/v2c6_targeted_remediation_model_selection_bge_cache.manifest.json"
)
RUNNER_RELATIVE_PATH = (
    "scripts/run_v2c6_targeted_remediation_model_selection.py"
)

CONTRACT_SCHEMA_VERSION = "v2c6-targeted-remediation-design-contract.v1"
EXPECTED_CONTRACT_SHA256 = (
    "339bd20f4bb94e73a42bd297c053aef8deaf55ee42a3b2145b8792172e7d5e91"
)
STEP29G_CONTRACT_SCHEMA_VERSION = "v2c6-source-aware-model-selection-contract.v1"
DEVELOPMENT_SCHEMA_VERSION = "v2c6-targeted-remediated-development-dataset.v1"
DEVELOPMENT_MANIFEST_SCHEMA_VERSION = (
    "v2c6-targeted-remediated-development-dataset-manifest.v1"
)
FRESH_SCHEMA_VERSION = "v2c6-fresh-source-evaluation-dataset.v1"
FRESH_MANIFEST_SCHEMA_VERSION = "v2c6-fresh-source-evaluation-dataset-manifest.v1"
RESULTS_SCHEMA_VERSION = "v2c6-targeted-remediation-model-selection-results.v1"
RESULTS_MANIFEST_SCHEMA_VERSION = (
    "v2c6-targeted-remediation-model-selection-results-manifest.v1"
)
BGE_CACHE_SCHEMA_VERSION = (
    "v2c6-targeted-remediation-model-selection-bge-cache.v1"
)

EXPECTED_DEVELOPMENT_SHA256 = (
    "133456aa10552058fe67ed2eed37acbe583e31331656174cd2e018430cca799d"
)
EXPECTED_DEVELOPMENT_MANIFEST_SHA256 = (
    "d30ead62c1370e4f50760d6a159f38d0a6118c9cdce2542f97a09086e49d3c58"
)
EXPECTED_FRESH_SHA256 = (
    "df3c1eef7e262260754a1a10908b540d479fe7dc3d6f228d1a9ba432ae9f1a6e"
)
EXPECTED_FRESH_MANIFEST_SHA256 = (
    "5e2ad040155fcc05f37854d1dd10960b7dab5a840f4c42a34075f4c86c6f77da"
)

UNSUPPORTED_INTENT = "unsupported_or_uncertain"
SUPPORTED_BINARY_LABEL = "supported"
DEVELOPMENT_RECORD_COUNT = 9608
FRESH_RECORD_COUNT = 640
FRESH_FAMILY_RECORD_COUNT = 320

EXPECTED_CANDIDATE_IDS = (
    "BGE_SMALL_LINEAR_SVC__C=4.0__class_weight=none",
    "WORD_CHAR_TFIDF_LINEAR_SVC__C=1.0__class_weight=none",
    "HYBRID_BGE_TFIDF_LINEAR_SVC__C=1.0__class_weight=none",
    "HIERARCHICAL_TFIDF_LINEAR_SVC__C=1.0__class_weight=none",
)
PRIMARY_INTENTS = (
    "account_blocked",
    "cancel_transfer",
    "close_account",
    "create_dispute",
    "freeze_card",
    "transfer_failed_or_declined",
    "transfer_pending",
    "unsupported_or_uncertain",
)
PROTECTED_INTENTS = (
    "cancel_transfer",
    "close_account",
    "create_dispute",
    "freeze_card",
)
FRESH_FAMILY_IDS = (
    "v2c6_r2_eval_sf1_independent_casework",
    "v2c6_r2_eval_sf2_independent_naturalistic",
)
COMPLEXITY_ORDER = (
    "WORD_CHAR_TFIDF_LINEAR_SVC",
    "BGE_SMALL_LINEAR_SVC",
    "HYBRID_BGE_TFIDF_LINEAR_SVC",
    "HIERARCHICAL_TFIDF_LINEAR_SVC",
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
    validation_indices: tuple[int, ...]


@dataclass(frozen=True)
class EvaluationSplits:
    group_folds: tuple[Partition, ...]
    audit: dict[str, Any]


@dataclass(frozen=True)
class LoadedInputs:
    contract: dict[str, Any]
    step29g_contract: dict[str, Any]
    development_dataset: dict[str, Any]
    fresh_dataset: dict[str, Any]
    development_examples: tuple[dict[str, Any], ...]
    fresh_examples: tuple[dict[str, Any], ...]
    development_manifest: dict[str, Any]
    fresh_manifest: dict[str, Any]
    contract_sha256: str
    development_sha256: str
    fresh_sha256: str


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


def assert_path_allowed(path: Path) -> None:
    resolved = path.resolve()
    if "final_holdout" in resolved.name:
        raise PermissionError(
            "targeted-remediation selection must never access a final-holdout artifact"
        )


def sha256_file(path: Path) -> str:
    assert_path_allowed(path)
    return sha256_bytes(path.read_bytes())


def read_json(path: Path) -> dict[str, Any]:
    assert_path_allowed(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def write_bytes_atomically(path: Path, content: bytes) -> None:
    assert_path_allowed(path)
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


def _expected_candidate_signature() -> list[tuple[str, str, str, float, None]]:
    return [
        (EXPECTED_CANDIDATE_IDS[0], "BGE_SMALL", "LinearSVC", 4.0, None),
        (EXPECTED_CANDIDATE_IDS[1], "WORD_CHAR_TFIDF", "LinearSVC", 1.0, None),
        (EXPECTED_CANDIDATE_IDS[2], "HYBRID_BGE_TFIDF", "LinearSVC", 1.0, None),
        (
            EXPECTED_CANDIDATE_IDS[3],
            "WORD_CHAR_TFIDF",
            "two_stage_LinearSVC",
            1.0,
            None,
        ),
    ]


def validate_contract(contract: Mapping[str, Any]) -> None:
    if contract.get("schema_version") != CONTRACT_SCHEMA_VERSION:
        raise ValueError("unexpected Step 29H-C contract schema")
    if contract.get("contract_version") != CONTRACT_SCHEMA_VERSION:
        raise ValueError("unexpected Step 29H-C contract version")
    if contract.get("phase") != "V2-C6 Step 29H-C" or contract.get("status") != "FROZEN":
        raise ValueError("Step 29H-C contract identity changed")
    status = contract.get("contract_status", {})
    if status.get("contract_frozen") is not True or status.get("design_only") is not True:
        raise ValueError("Step 29H-C design contract is not frozen")
    for field in (
        "embeddings_generated",
        "final_holdout_accessed",
        "model_fitting_performed",
        "model_inference_performed",
        "model_selection_performed",
        "runtime_behavior_changed",
        "step29i_authorized",
        "threshold_tuning_performed",
    ):
        if status.get(field) is not False:
            raise ValueError(f"Step 29H-C contract records prohibited execution: {field}")

    search = contract.get("candidate_search_space", {})
    candidates = search.get("candidates")
    if not isinstance(candidates, list) or search.get("candidate_count") != 4:
        raise ValueError("exactly four frozen candidates are required")
    signature = [
        (
            row.get("candidate_id"),
            row.get("representation"),
            row.get("classifier"),
            row.get("C"),
            row.get("class_weight"),
        )
        for row in candidates
    ]
    if signature != _expected_candidate_signature():
        raise ValueError("frozen candidate definitions or ordering changed")
    if tuple(search.get("exact_candidate_ids", ())) != EXPECTED_CANDIDATE_IDS:
        raise ValueError("frozen candidate ID list changed")
    if (
        search.get("balanced_class_weighting_carried_forward") is not False
        or search.get("search_additional_c_values") is not False
        or search.get("new_embedding_model_introduced") is not False
    ):
        raise ValueError("frozen bounded candidate search changed")

    implementations = contract.get("candidate_implementation_semantics", {})
    common = implementations.get("common", {})
    if common != {
        "calibration_permitted": False,
        "class_weight": None,
        "classifier_random_state": 20260930,
        "post_hoc_rule_override_permitted": False,
        "threshold_tuning_permitted": False,
    }:
        raise ValueError("common candidate semantics changed")
    bge = implementations.get(EXPECTED_CANDIDATE_IDS[0], {})
    if (
        bge.get("embedding_model") != "BAAI/bge-small-en-v1.5"
        or bge.get("dimensions") != 384
        or bge.get("final_row_l2_normalized") is not True
        or bge.get("fine_tuning") is not False
    ):
        raise ValueError("frozen BGE candidate semantics changed")
    hybrid = implementations.get(EXPECTED_CANDIDATE_IDS[2], {})
    if (
        hybrid.get("bge_block", {}).get("embedding_model")
        != "BAAI/bge-small-en-v1.5"
        or hybrid.get("bge_block", {}).get("dimensions") != 384
        or hybrid.get("tfidf_block", {}).get("members")
        != ["WORD_TFIDF", "CHAR_TFIDF"]
        or hybrid.get("concatenation")
        != "deterministic_sparse_hstack_of_word_char_tfidf_and_dense_bge_block_converted_to_sparse"
        or hybrid.get("final_combined_row_l2_normalization") is not True
        or hybrid.get("learned_fusion_layer_permitted") is not False
        or hybrid.get("threshold_tuning_permitted") is not False
    ):
        raise ValueError("frozen hybrid candidate semantics changed")
    hierarchy = implementations.get(EXPECTED_CANDIDATE_IDS[3], {})
    expected_supported = [
        "account_balance",
        "account_blocked",
        "cancel_transfer",
        "card_status",
        "close_account",
        "create_dispute",
        "escalation",
        "freeze_card",
        "informational_policy",
        "lost_or_stolen_phone",
        "passcode_recovery",
        "recent_transactions",
        "transaction_details",
        "transfer_failed_or_declined",
        "transfer_pending",
    ]
    if (
        hierarchy.get("stage_1", {}).get("classes")
        != [UNSUPPORTED_INTENT, SUPPORTED_BINARY_LABEL]
        or hierarchy.get("stage_1", {}).get("representation") != "WORD_CHAR_TFIDF"
        or hierarchy.get("stage_1", {}).get("classifier") != "LinearSVC"
        or hierarchy.get("stage_1", {}).get("C") != 1.0
        or hierarchy.get("stage_1", {}).get("class_weight") is not None
        or hierarchy.get("stage_2", {}).get("class_count") != 15
        or hierarchy.get("stage_2", {}).get("classes") != expected_supported
        or hierarchy.get("stage_2", {}).get("excluded_intent") != UNSUPPORTED_INTENT
        or hierarchy.get("stage_2", {}).get("representation") != "WORD_CHAR_TFIDF"
        or hierarchy.get("stage_2", {}).get("classifier") != "LinearSVC"
        or hierarchy.get("stage_2", {}).get("C") != 1.0
        or hierarchy.get("stage_2", {}).get("class_weight") is not None
        or hierarchy.get("calibration_permitted") is not False
        or hierarchy.get("confidence_override_permitted") is not False
        or hierarchy.get("protected_intent_rule_special_casing_permitted") is not False
        or hierarchy.get("threshold_tuning_permitted") is not False
    ):
        raise ValueError("frozen hierarchical candidate semantics changed")

    cv = contract.get("group_aware_cross_validation", {})
    if cv != {
        "group_field": "group_id",
        "group_id_is_atomic": True,
        "method": "StratifiedGroupKFold",
        "n_splits": 5,
        "non_group_aware_fallback_allowed": False,
        "random_state": 20260930,
        "report_same_broad_metrics_as_step29g_step29h": True,
        "shuffle": True,
        "use_when_feasible_under_frozen_taxonomy_and_group_constraints": True,
    }:
        raise ValueError("frozen group-aware CV protocol changed")
    if tuple(contract.get("targeted_intents", ())) != PRIMARY_INTENTS:
        raise ValueError("primary-eight intent definition changed")

    gates = contract.get("safety_gates", {})
    if tuple(gates.get("protected_intents", ())) != PROTECTED_INTENTS:
        raise ValueError("protected intent definition changed")
    if gates.get("application_scopes") != [
        "pooled_group_aware_cv",
        *FRESH_FAMILY_IDS,
        "pooled_fresh_evaluation",
    ]:
        raise ValueError("mandatory safety-gate scopes changed")
    expected_gates = [
        ("protected_recall", "greater_than_or_equal", 0.8),
        ("protected_false_positive_rate", "less_than_or_equal", 0.01),
        ("unsupported_recall", "greater_than_or_equal", 0.8),
    ]
    if [
        (row.get("metric"), row.get("comparison"), row.get("threshold"))
        for row in gates.get("gates", [])
    ] != expected_gates:
        raise ValueError("frozen safety thresholds changed")
    if (
        gates.get("gates_mandatory_for_candidate_eligibility") is not True
        or gates.get("thresholds_may_be_weakened_after_results") is not False
        or gates.get("old_consumed_source_families_are_mandatory_fresh_gates")
        is not False
    ):
        raise ValueError("frozen safety-gate policy changed")

    fresh = contract.get("fresh_evaluation_specification", {})
    if (
        fresh.get("record_count") != FRESH_RECORD_COUNT
        or fresh.get("family_count") != 2
        or fresh.get("per_family_record_count") != FRESH_FAMILY_RECORD_COUNT
        or fresh.get("per_intent_per_family_count") != 40
        or tuple(fresh.get("source_family_ids", ())) != FRESH_FAMILY_IDS
        or fresh.get("excluded_from_candidate_fitting") is not True
    ):
        raise ValueError("frozen fresh-evaluation population changed")
    for specification, family in zip(
        fresh.get("family_specifications", []), FRESH_FAMILY_IDS, strict=True
    ):
        if (
            specification.get("source_family_id") != family
            or specification.get("record_count") != 320
            or specification.get("protected_record_count") != 160
            or specification.get("non_protected_record_count") != 160
            or specification.get("intent_counts") != dict.fromkeys(PRIMARY_INTENTS, 40)
        ):
            raise ValueError("fresh family specification changed")
    if fresh.get("protected_false_positive_allowance_per_family") != {
        "formula": "floor(0.01 * non_protected_record_count)",
        "maximum_passing_count": 1,
        "non_protected_record_count": 160,
        "threshold": 0.01,
    }:
        raise ValueError("fresh protected-FP allowance changed")

    protocol = contract.get("training_and_evaluation_protocol", {})
    training = protocol.get("training_and_model_development_population", {})
    fresh_protocol = protocol.get("fresh_source_evaluation", {})
    if training.get("total_record_count") != DEVELOPMENT_RECORD_COUNT:
        raise ValueError("frozen development population changed")
    if (
        fresh_protocol.get("candidate_fit_population")
        != "entire_9608_record_training_and_model_development_population"
        or fresh_protocol.get("each_candidate_fit_once_for_fresh_evaluation") is not True
        or fresh_protocol.get("fresh_records_excluded_from_fitting") is not True
        or fresh_protocol.get("evaluation_scopes")
        != [*FRESH_FAMILY_IDS, "pooled_fresh_evaluation"]
    ):
        raise ValueError("frozen fresh-evaluation execution protocol changed")

    selection = contract.get("selection_rule", {})
    expected_criteria = [
        ("worst_fresh_family_primary_8_macro_f1", "maximize"),
        ("mean_fresh_family_primary_8_macro_f1", "maximize"),
        ("pooled_fresh_primary_8_macro_f1", "maximize"),
        ("pooled_group_cv_macro_f1_16", "maximize"),
        ("pooled_fresh_protected_false_positive_rate", "minimize"),
        ("pooled_fresh_protected_recall", "maximize"),
        ("pooled_fresh_unsupported_recall", "maximize"),
        ("model_complexity_rank", "prefer_lower_rank"),
        ("candidate_id", "ascending_lexical"),
    ]
    if [
        (row.get("metric"), row.get("direction"))
        for row in selection.get("ordered_lexicographic_criteria", [])
    ] != expected_criteria:
        raise ValueError("frozen lexicographic selection rule changed")
    if (
        selection.get("selection_population") != "eligible_candidates_only"
        or selection.get("single_weighted_composite_score_used") is not False
        or selection.get("if_no_candidate_is_eligible")
        != {
            "gates_weakened": False,
            "selected_candidate": None,
            "selection_status": "NO_ACCEPTABLE_CANDIDATE",
            "winner_forced": False,
        }
    ):
        raise ValueError("frozen selection governance changed")
    if tuple(
        contract.get("complexity_ordering", {}).get("simplest_to_more_complex", ())
    ) != COMPLEXITY_ORDER:
        raise ValueError("frozen model-complexity ordering changed")

    final_holdout = contract.get("final_holdout_policy", {})
    if (
        final_holdout.get("existing_final_holdout_access_permitted") is not False
        or final_holdout.get("future_v2c6_final_holdout_access_permitted") is not False
        or final_holdout.get("final_holdout_created_by_this_step") is not False
        or final_holdout.get("prohibited_path")
        != "data/evals/v2/ml/v2c5_final_holdout.json"
    ):
        raise ValueError("frozen final-holdout policy changed")
    if contract.get("step29i_policy", {}).get("blocked") is not True:
        raise ValueError("Step 29I must remain blocked")


def validate_step29g_conventions(contract: Mapping[str, Any]) -> None:
    if (
        contract.get("schema_version") != STEP29G_CONTRACT_SCHEMA_VERSION
        or contract.get("phase") != "V2-C6 Step 29G"
    ):
        raise ValueError("unexpected frozen Step 29G convention source")
    taxonomy = contract.get("taxonomy", {})
    labels = taxonomy.get("intent_label_order")
    if (
        not isinstance(labels, list)
        or len(labels) != 16
        or len(set(labels)) != 16
        or tuple(taxonomy.get("primary_remediation_8_intents", ())) != PRIMARY_INTENTS
    ):
        raise ValueError("frozen Step 29G taxonomy changed")
    representations = contract.get("representations", {})
    if representations.get("BGE_SMALL") != {
        "dimensions": 384,
        "embedding_method": "passage_embed",
        "fine_tuning": False,
        "implementation": "FastEmbed",
        "l2_normalized": True,
        "model_identifier": "BAAI/bge-small-en-v1.5",
    }:
        raise ValueError("frozen Step 29G BGE convention changed")
    if representations.get("WORD_CHAR_TFIDF") != {
        "class": "FeatureUnion",
        "exact_member_definitions_reused": True,
        "members": ["WORD_TFIDF", "CHAR_TFIDF"],
    }:
        raise ValueError("frozen Step 29G TF-IDF union changed")
    expected_members = {
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
    for name, expected in expected_members.items():
        if representations.get(name) != expected:
            raise ValueError(f"frozen Step 29G {name} convention changed")
    classifier = contract.get("classifiers", {}).get("LINEAR_SVC", {})
    if classifier.get("class") != "LinearSVC" or classifier.get(
        "fixed_parameters"
    ) != {
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
        raise ValueError("frozen Step 29G LinearSVC convention changed")


def _validate_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_schema: str,
    artifact_path: Path,
    artifact_sha256: str,
    expected_count: int,
) -> None:
    if manifest.get("schema_version") != expected_schema:
        raise ValueError("unexpected frozen dataset manifest schema")
    artifact = manifest.get("artifact", {})
    if artifact != {
        "path": str(artifact_path.relative_to(REPOSITORY_ROOT)),
        "schema_version": (
            DEVELOPMENT_SCHEMA_VERSION
            if artifact_path == DEVELOPMENT_DATASET_PATH
            else FRESH_SCHEMA_VERSION
        ),
        "sha256": artifact_sha256,
    }:
        raise ValueError("frozen dataset manifest identity changed")
    if manifest.get("contract", {}) != {
        "path": str(CONTRACT_PATH.relative_to(REPOSITORY_ROOT)),
        "sha256": EXPECTED_CONTRACT_SHA256,
    }:
        raise ValueError("frozen dataset contract lineage changed")
    counts = manifest.get("counts", {})
    count = counts.get("total_record_count", counts.get("record_count"))
    if count != expected_count:
        raise ValueError("frozen dataset manifest record count changed")
    governance = manifest.get("governance", {})
    for field in (
        "final_holdout_accessed",
        "model_fitting_performed",
        "model_inference_performed",
        "model_selection_performed",
        "runtime_behavior_changed",
        "step29i_authorized",
        "threshold_tuning_performed",
    ):
        if governance.get(field) is not False:
            raise ValueError(f"frozen dataset governance changed: {field}")


def _validate_text_record(
    row: Mapping[str, Any],
    *,
    identifier_field: str,
    allowed_intents: set[str],
) -> tuple[str, str]:
    identifier = row.get(identifier_field)
    group_id = row.get("group_id")
    text = row.get("text")
    if not isinstance(identifier, str) or not identifier:
        raise ValueError(f"{identifier_field} must be a nonempty string")
    if not isinstance(group_id, str) or not group_id:
        raise ValueError(f"group_id must be a nonempty string: {identifier}")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"text must be nonempty: {identifier}")
    if row.get("text_sha256") != sha256_bytes(text.encode("utf-8")):
        raise ValueError(f"text hash mismatch: {identifier}")
    if row.get("intent") not in allowed_intents:
        raise ValueError(f"unexpected intent: {identifier}")
    return identifier, group_id


def validate_development_dataset(
    dataset: Mapping[str, Any], step29g_contract: Mapping[str, Any]
) -> tuple[dict[str, Any], ...]:
    if dataset.get("schema_version") != DEVELOPMENT_SCHEMA_VERSION:
        raise ValueError("unexpected targeted-remediated development schema")
    examples = dataset.get("examples")
    if not isinstance(examples, list):
        raise TypeError("development examples must be a list")
    if (
        dataset.get("example_count") != DEVELOPMENT_RECORD_COUNT
        or len(examples) != DEVELOPMENT_RECORD_COUNT
        or dataset.get("classifier_input_fields") != ["text"]
        or dataset.get("metadata_fields_are_classifier_features") is not False
        or dataset.get("governance", {}).get("fresh_evaluation_records_included")
        is not False
    ):
        raise ValueError("frozen development population or feature policy changed")
    allowed = set(step29g_contract["taxonomy"]["intent_label_order"])
    identifiers: set[str] = set()
    for row in examples:
        if not isinstance(row, dict):
            raise TypeError("development record must be an object")
        identifier, _ = _validate_text_record(
            row, identifier_field="example_id", allowed_intents=allowed
        )
        if identifier in identifiers:
            raise ValueError(f"duplicate development example_id: {identifier}")
        identifiers.add(identifier)
        if row.get("data_role") != "development":
            raise ValueError(f"non-development row in fit population: {identifier}")
    return tuple(examples)


def validate_fresh_dataset(
    dataset: Mapping[str, Any], development_examples: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, Any], ...]:
    if dataset.get("schema_version") != FRESH_SCHEMA_VERSION:
        raise ValueError("unexpected fresh source-evaluation schema")
    examples = dataset.get("examples")
    if not isinstance(examples, list):
        raise TypeError("fresh evaluation examples must be a list")
    if (
        dataset.get("example_count") != FRESH_RECORD_COUNT
        or len(examples) != FRESH_RECORD_COUNT
        or dataset.get("classifier_input_fields") != ["text"]
        or dataset.get("metadata_fields_are_classifier_features") is not False
        or dataset.get("dataset_role") != "fresh_source_evaluation"
    ):
        raise ValueError("frozen fresh-evaluation population or feature policy changed")
    governance = dataset.get("fresh_evaluation_governance", {})
    if governance != {
        "candidate_outputs_inspected_during_authoring": False,
        "consumed_old_source_family_used_as_paraphrase_template": False,
        "excluded_from_candidate_fitting": True,
        "independently_authored": True,
        "not_training_data": True,
        "prediction_informed_authoring": False,
    }:
        raise ValueError("fresh-evaluation governance changed")

    identifiers: set[str] = set()
    groups: set[str] = set()
    family_counts: Counter[str] = Counter()
    family_intents: dict[str, Counter[str]] = {
        family: Counter() for family in FRESH_FAMILY_IDS
    }
    protected_counts: Counter[str] = Counter()
    allowed = set(PRIMARY_INTENTS)
    for row in examples:
        if not isinstance(row, dict):
            raise TypeError("fresh evaluation record must be an object")
        identifier, group_id = _validate_text_record(
            row, identifier_field="record_id", allowed_intents=allowed
        )
        if identifier in identifiers:
            raise ValueError(f"duplicate fresh record_id: {identifier}")
        if group_id in groups:
            raise ValueError(f"duplicate fresh group_id: {group_id}")
        identifiers.add(identifier)
        groups.add(group_id)
        family = row.get("source_family_id")
        if family not in FRESH_FAMILY_IDS:
            raise ValueError(f"unexpected fresh source family: {family}")
        if (
            row.get("dataset_role") != "fresh_source_evaluation"
            or row.get("excluded_from_candidate_fitting") is not True
            or row.get("not_training_data") is not True
        ):
            raise ValueError(f"fresh row is not excluded from fitting: {identifier}")
        family_counts[str(family)] += 1
        family_intents[str(family)][str(row["intent"])] += 1
        if row["intent"] in PROTECTED_INTENTS:
            protected_counts[str(family)] += 1
    if family_counts != Counter(dict.fromkeys(FRESH_FAMILY_IDS, 320)):
        raise ValueError("fresh family counts changed")
    expected_intents = Counter(dict.fromkeys(PRIMARY_INTENTS, 40))
    if any(family_intents[family] != expected_intents for family in FRESH_FAMILY_IDS):
        raise ValueError("fresh per-family intent counts changed")
    if protected_counts != Counter(dict.fromkeys(FRESH_FAMILY_IDS, 160)):
        raise ValueError("fresh per-family protected composition changed")

    development_ids = {str(row["example_id"]) for row in development_examples}
    development_groups = {str(row["group_id"]) for row in development_examples}
    if identifiers & development_ids:
        raise ValueError("development/fresh record-ID collision detected")
    if groups & development_groups:
        raise ValueError("development/fresh group-ID collision detected")
    return tuple(examples)


def load_and_validate_inputs() -> LoadedInputs:
    contract_sha256 = sha256_file(CONTRACT_PATH)
    if contract_sha256 != EXPECTED_CONTRACT_SHA256:
        raise ValueError(f"Step 29H-C contract SHA mismatch: {contract_sha256}")
    contract = read_json(CONTRACT_PATH)
    validate_contract(contract)

    step29g_spec = contract["source_artifacts"]["step29g_contract"]
    step29g_path = validate_frozen_source(step29g_spec)
    if step29g_path != STEP29G_CONTRACT_PATH.resolve():
        raise ValueError("Step 29G convention-source path changed")
    step29g_contract = read_json(step29g_path)
    validate_step29g_conventions(step29g_contract)

    if sha256_file(DEVELOPMENT_MANIFEST_PATH) != EXPECTED_DEVELOPMENT_MANIFEST_SHA256:
        raise ValueError("targeted development manifest SHA mismatch")
    if sha256_file(FRESH_MANIFEST_PATH) != EXPECTED_FRESH_MANIFEST_SHA256:
        raise ValueError("fresh evaluation manifest SHA mismatch")
    development_manifest = read_json(DEVELOPMENT_MANIFEST_PATH)
    fresh_manifest = read_json(FRESH_MANIFEST_PATH)

    development_sha256 = sha256_file(DEVELOPMENT_DATASET_PATH)
    fresh_sha256 = sha256_file(FRESH_DATASET_PATH)
    if development_sha256 != EXPECTED_DEVELOPMENT_SHA256:
        raise ValueError("targeted development dataset SHA mismatch")
    if fresh_sha256 != EXPECTED_FRESH_SHA256:
        raise ValueError("fresh evaluation dataset SHA mismatch")
    _validate_manifest(
        development_manifest,
        expected_schema=DEVELOPMENT_MANIFEST_SCHEMA_VERSION,
        artifact_path=DEVELOPMENT_DATASET_PATH,
        artifact_sha256=development_sha256,
        expected_count=DEVELOPMENT_RECORD_COUNT,
    )
    _validate_manifest(
        fresh_manifest,
        expected_schema=FRESH_MANIFEST_SCHEMA_VERSION,
        artifact_path=FRESH_DATASET_PATH,
        artifact_sha256=fresh_sha256,
        expected_count=FRESH_RECORD_COUNT,
    )
    development_dataset = read_json(DEVELOPMENT_DATASET_PATH)
    development_examples = validate_development_dataset(
        development_dataset, step29g_contract
    )
    fresh_dataset = read_json(FRESH_DATASET_PATH)
    fresh_examples = validate_fresh_dataset(fresh_dataset, development_examples)
    return LoadedInputs(
        contract=contract,
        step29g_contract=step29g_contract,
        development_dataset=development_dataset,
        fresh_dataset=fresh_dataset,
        development_examples=development_examples,
        fresh_examples=fresh_examples,
        development_manifest=development_manifest,
        fresh_manifest=fresh_manifest,
        contract_sha256=contract_sha256,
        development_sha256=development_sha256,
        fresh_sha256=fresh_sha256,
    )


def membership_sha256(values: Iterable[str]) -> str:
    return sha256_bytes(stable_json_bytes(sorted(values)))


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
        if set(train) & set(validation):
            raise ValueError(f"record leakage detected in group_cv_fold_{fold_index}")
        train_groups = {str(groups[index]) for index in train}
        validation_groups = {str(groups[index]) for index in validation}
        if train_groups & validation_groups:
            raise ValueError(f"group leakage detected in group_cv_fold_{fold_index}")
        validation_counts[np.asarray(validation, dtype=int)] += 1
        partitions.append(
            Partition(
                partition_id=f"group_cv_fold_{fold_index}",
                train_indices=train,
                validation_indices=validation,
            )
        )
    if len(partitions) != 5:
        raise ValueError("exactly five group-CV folds are required")
    if not np.all(validation_counts == 1):
        raise ValueError("every development record must receive exactly one OOF assignment")
    return tuple(partitions)


def build_evaluation_splits(
    development_examples: Sequence[Mapping[str, Any]],
    fresh_examples: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> EvaluationSplits:
    folds = build_group_folds(development_examples, contract)
    fold_audit: list[dict[str, Any]] = []
    for fold in folds:
        train_rows = [development_examples[index] for index in fold.train_indices]
        validation_rows = [
            development_examples[index] for index in fold.validation_indices
        ]
        train_groups = {str(row["group_id"]) for row in train_rows}
        validation_groups = {str(row["group_id"]) for row in validation_rows}
        fold_audit.append(
            {
                "fold_id": fold.partition_id,
                "train_count": len(train_rows),
                "validation_count": len(validation_rows),
                "train_example_id_sha256": membership_sha256(
                    str(row["example_id"]) for row in train_rows
                ),
                "validation_example_id_sha256": membership_sha256(
                    str(row["example_id"]) for row in validation_rows
                ),
                "train_group_id_sha256": membership_sha256(train_groups),
                "validation_group_id_sha256": membership_sha256(validation_groups),
                "record_overlap_count": len(
                    set(fold.train_indices) & set(fold.validation_indices)
                ),
                "group_overlap_count": len(train_groups & validation_groups),
            }
        )
    fresh_ids = {str(row["record_id"]) for row in fresh_examples}
    development_ids = {str(row["example_id"]) for row in development_examples}
    fresh_groups = {str(row["group_id"]) for row in fresh_examples}
    development_groups = {str(row["group_id"]) for row in development_examples}
    if any(
        row["record_overlap_count"] or row["group_overlap_count"] for row in fold_audit
    ):
        raise ValueError("group-CV split audit detected leakage")
    if fresh_ids & development_ids or fresh_groups & development_groups:
        raise ValueError("fresh evaluation overlaps the fit population")
    audit: dict[str, Any] = {
        "fold_count": len(folds),
        "pooled_validation_count": sum(row["validation_count"] for row in fold_audit),
        "every_development_record_validated_exactly_once": True,
        "record_leakage_count": 0,
        "group_leakage_count": 0,
        "fresh_record_count": len(fresh_examples),
        "fresh_records_in_group_cv": 0,
        "fresh_record_id_overlap_count": 0,
        "fresh_group_id_overlap_count": 0,
        "folds": fold_audit,
        "raw_text_persisted": False,
    }
    audit["audit_sha256"] = sha256_bytes(stable_json_bytes(audit))
    return EvaluationSplits(group_folds=folds, audit=audit)


def build_representation(
    representation_id: str, step29g_contract: Mapping[str, Any]
) -> BaseEstimator:
    definitions = step29g_contract["representations"]
    if representation_id in {"WORD_TFIDF", "CHAR_TFIDF"}:
        definition = definitions[representation_id]
        parameters = dict(EXPLICIT_TFIDF_DEFAULTS)
        parameters.update(definition["parameters"])
        parameters["ngram_range"] = tuple(parameters["ngram_range"])
        return TfidfVectorizer(**parameters)
    if representation_id == "WORD_CHAR_TFIDF":
        definition = definitions[representation_id]
        return FeatureUnion(
            [
                (member.lower(), build_representation(member, step29g_contract))
                for member in definition["members"]
            ],
            n_jobs=None,
            transformer_weights=None,
            verbose=False,
        )
    if representation_id == "BGE_SMALL":
        raise ValueError("BGE_SMALL uses the validated fixed embedding cache")
    raise ValueError(f"unsupported frozen representation: {representation_id}")


def build_classifier(step29g_contract: Mapping[str, Any], *, c_value: float) -> LinearSVC:
    definition = step29g_contract["classifiers"]["LINEAR_SVC"]
    return LinearSVC(
        **definition["fixed_parameters"],
        C=c_value,
        class_weight=None,
    )


def fit_classifier(
    classifier: BaseEstimator, features: Any, labels: np.ndarray
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        classifier.fit(features, labels)


def load_fastembed_model(step29g_contract: Mapping[str, Any]) -> Any:
    try:
        from fastembed import TextEmbedding
    except ImportError as exc:
        raise RuntimeError("fastembed is required for BGE_SMALL") from exc
    model_name = step29g_contract["representations"]["BGE_SMALL"]["model_identifier"]
    try:
        return TextEmbedding(model_name=model_name)
    except Exception as exc:
        raise RuntimeError("FastEmbed could not load the frozen BGE model") from exc


def embed_texts(
    model: Any, texts: Sequence[str], step29g_contract: Mapping[str, Any]
) -> np.ndarray:
    definition = step29g_contract["representations"]["BGE_SMALL"]
    vectors = np.asarray(
        list(model.passage_embed(list(texts), batch_size=256)), dtype=np.float32
    )
    expected_shape = (len(texts), definition["dimensions"])
    if vectors.shape != expected_shape or not np.isfinite(vectors).all():
        raise ValueError(f"unexpected BGE embedding matrix: {vectors.shape}")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("BGE returned a zero vector")
    normalized = np.asarray(vectors / norms, dtype=np.float32)
    if not np.allclose(
        np.linalg.norm(normalized, axis=1), 1.0, rtol=1e-5, atol=1e-6
    ):
        raise ValueError("BGE L2 normalization failed")
    return normalized


def _population_cache_spec(
    name: str,
    records: Sequence[Mapping[str, Any]],
    identifier_field: str,
) -> dict[str, Any]:
    return {
        "matrix_key": name,
        "row_count": len(records),
        "ordered_record_id_sha256": sha256_bytes(
            stable_json_bytes([str(row[identifier_field]) for row in records])
        ),
        "ordered_text_population_sha256": sha256_bytes(
            stable_json_bytes([str(row["text_sha256"]) for row in records])
        ),
    }


def expected_bge_cache_manifest(
    inputs: LoadedInputs,
    *,
    cache_file_sha256: str,
    fastembed_version: str,
) -> dict[str, Any]:
    representation = inputs.step29g_contract["representations"]["BGE_SMALL"]
    return {
        "schema_version": BGE_CACHE_SCHEMA_VERSION,
        "runner": {
            "path": RUNNER_RELATIVE_PATH,
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "contract_sha256": inputs.contract_sha256,
        "development_dataset_sha256": inputs.development_sha256,
        "fresh_evaluation_dataset_sha256": inputs.fresh_sha256,
        "representation_id": "BGE_SMALL",
        "representation_sha256": sha256_bytes(stable_json_bytes(representation)),
        "model_identifier": representation["model_identifier"],
        "embedding_method": representation["embedding_method"],
        "dimensions": representation["dimensions"],
        "l2_normalized": representation["l2_normalized"],
        "populations": [
            _population_cache_spec(
                "development", inputs.development_examples, "example_id"
            ),
            _population_cache_spec(
                "fresh_evaluation", inputs.fresh_examples, "record_id"
            ),
        ],
        "cache_file_sha256": cache_file_sha256,
        "dtype": "float32",
        "fastembed_version": fastembed_version,
        "labels_used": False,
        "fine_tuning_performed": False,
        "fresh_evaluation_used_for_fitting": False,
        "raw_text_persisted": False,
        "final_holdout_accessed": False,
    }


def _validate_embedding_matrix(
    matrix: np.ndarray, expected_rows: int, step29g_contract: Mapping[str, Any]
) -> None:
    dimensions = step29g_contract["representations"]["BGE_SMALL"]["dimensions"]
    if matrix.shape != (expected_rows, dimensions):
        raise ValueError(f"BGE cache shape mismatch: {matrix.shape}")
    if matrix.dtype != np.float32:
        raise ValueError("BGE cache dtype must be float32")
    if not np.isfinite(matrix).all():
        raise ValueError("BGE cache contains non-finite values")
    if not np.allclose(
        np.linalg.norm(matrix, axis=1), 1.0, rtol=1e-5, atol=1e-6
    ):
        raise ValueError("BGE cache rows are not L2 normalized")


def validate_bge_cache(
    inputs: LoadedInputs,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if not BGE_CACHE_PATH.is_file() or not BGE_CACHE_MANIFEST_PATH.is_file():
        raise FileNotFoundError("complete local targeted-remediation BGE cache is required")
    manifest = read_json(BGE_CACHE_MANIFEST_PATH)
    expected = expected_bge_cache_manifest(
        inputs,
        cache_file_sha256=str(manifest.get("cache_file_sha256", "")),
        fastembed_version=str(manifest.get("fastembed_version", "")),
    )
    if manifest != expected:
        raise ValueError("stale or incompatible targeted-remediation BGE cache manifest")
    if not manifest["fastembed_version"]:
        raise ValueError("BGE cache must record the FastEmbed version")
    if sha256_file(BGE_CACHE_PATH) != manifest["cache_file_sha256"]:
        raise ValueError("BGE cache file hash mismatch")
    specifications = {row["matrix_key"]: row for row in manifest["populations"]}
    matrices: dict[str, np.ndarray] = {}
    with np.load(BGE_CACHE_PATH, allow_pickle=False) as cached:
        if set(cached.files) != set(specifications):
            raise ValueError("BGE cache population matrix set changed")
        for key, specification in specifications.items():
            matrix = np.asarray(cached[key])
            _validate_embedding_matrix(
                matrix, specification["row_count"], inputs.step29g_contract
            )
            matrices[key] = matrix
    return matrices, manifest


def prepare_bge_cache(
    inputs: LoadedInputs,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    existing = (BGE_CACHE_PATH.exists(), BGE_CACHE_MANIFEST_PATH.exists())
    if any(existing):
        if not all(existing):
            raise FileExistsError("incomplete targeted-remediation BGE cache pair")
        return validate_bge_cache(inputs)
    model = load_fastembed_model(inputs.step29g_contract)
    matrices = {
        "development": embed_texts(
            model,
            [str(row["text"]) for row in inputs.development_examples],
            inputs.step29g_contract,
        ),
        "fresh_evaluation": embed_texts(
            model,
            [str(row["text"]) for row in inputs.fresh_examples],
            inputs.step29g_contract,
        ),
    }
    BGE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=BGE_CACHE_PATH.parent, suffix=".npz", delete=False
    ) as handle:
        temporary_path = Path(handle.name)
    try:
        np.savez_compressed(temporary_path, **matrices)
        temporary_path.replace(BGE_CACHE_PATH)
    finally:
        temporary_path.unlink(missing_ok=True)
    manifest = expected_bge_cache_manifest(
        inputs,
        cache_file_sha256=sha256_file(BGE_CACHE_PATH),
        fastembed_version=importlib.metadata.version("fastembed"),
    )
    write_bytes_atomically(BGE_CACHE_MANIFEST_PATH, stable_json_bytes(manifest))
    return validate_bge_cache(inputs)


def _hybrid_features(tfidf_features: Any, bge_features: np.ndarray) -> Any:
    combined = sparse.hstack(
        (tfidf_features, sparse.csr_matrix(bge_features)), format="csr"
    )
    normalized = normalize(combined, norm="l2", axis=1, copy=True)
    if not sparse.issparse(normalized):
        raise TypeError("hybrid representation must remain sparse")
    row_norms = np.sqrt(normalized.multiply(normalized).sum(axis=1)).A1
    if not np.allclose(row_norms, 1.0, rtol=1e-5, atol=1e-6):
        raise ValueError("hybrid row L2 normalization failed")
    return normalized


def _validate_predictions(
    predicted: np.ndarray,
    expected_count: int,
    allowed_labels: set[str],
) -> np.ndarray:
    values = np.asarray(predicted, dtype=object)
    if values.shape != (expected_count,):
        raise ValueError("classifier prediction shape changed")
    if not set(values.tolist()).issubset(allowed_labels):
        raise ValueError("classifier predicted outside the frozen taxonomy")
    return values


def fit_predict_candidate(
    candidate_id: str,
    train_rows: Sequence[Mapping[str, Any]],
    evaluation_rows: Sequence[Mapping[str, Any]],
    step29g_contract: Mapping[str, Any],
    *,
    train_bge: np.ndarray | None = None,
    evaluation_bge: np.ndarray | None = None,
) -> np.ndarray:
    if candidate_id not in EXPECTED_CANDIDATE_IDS:
        raise ValueError(f"unexpected candidate: {candidate_id}")
    train_texts = [str(row["text"]) for row in train_rows]
    evaluation_texts = [str(row["text"]) for row in evaluation_rows]
    train_labels = np.asarray([str(row["intent"]) for row in train_rows], dtype=object)
    allowed_labels = set(step29g_contract["taxonomy"]["intent_label_order"])

    if candidate_id == EXPECTED_CANDIDATE_IDS[0]:
        if train_bge is None or evaluation_bge is None:
            raise ValueError("BGE candidate requires validated cached embeddings")
        classifier = build_classifier(step29g_contract, c_value=4.0)
        fit_classifier(classifier, train_bge, train_labels)
        predicted = classifier.predict(evaluation_bge)
    elif candidate_id == EXPECTED_CANDIDATE_IDS[1]:
        representation = build_representation("WORD_CHAR_TFIDF", step29g_contract)
        train_features = representation.fit_transform(train_texts)
        evaluation_features = representation.transform(evaluation_texts)
        classifier = build_classifier(step29g_contract, c_value=1.0)
        fit_classifier(classifier, train_features, train_labels)
        predicted = classifier.predict(evaluation_features)
    elif candidate_id == EXPECTED_CANDIDATE_IDS[2]:
        if train_bge is None or evaluation_bge is None:
            raise ValueError("hybrid candidate requires validated cached embeddings")
        representation = build_representation("WORD_CHAR_TFIDF", step29g_contract)
        train_tfidf = representation.fit_transform(train_texts)
        evaluation_tfidf = representation.transform(evaluation_texts)
        train_features = _hybrid_features(train_tfidf, train_bge)
        evaluation_features = _hybrid_features(evaluation_tfidf, evaluation_bge)
        classifier = build_classifier(step29g_contract, c_value=1.0)
        fit_classifier(classifier, train_features, train_labels)
        predicted = classifier.predict(evaluation_features)
    else:
        stage_1_representation = build_representation(
            "WORD_CHAR_TFIDF", step29g_contract
        )
        stage_1_train = stage_1_representation.fit_transform(train_texts)
        stage_1_evaluation = stage_1_representation.transform(evaluation_texts)
        binary_labels = np.asarray(
            [
                UNSUPPORTED_INTENT
                if label == UNSUPPORTED_INTENT
                else SUPPORTED_BINARY_LABEL
                for label in train_labels
            ],
            dtype=object,
        )
        stage_1_classifier = build_classifier(step29g_contract, c_value=1.0)
        fit_classifier(stage_1_classifier, stage_1_train, binary_labels)
        routing = np.asarray(
            stage_1_classifier.predict(stage_1_evaluation), dtype=object
        )
        if not set(routing.tolist()).issubset(
            {UNSUPPORTED_INTENT, SUPPORTED_BINARY_LABEL}
        ):
            raise ValueError("hierarchical stage 1 predicted an unexpected class")

        supported_indices = [
            index for index, label in enumerate(train_labels) if label != UNSUPPORTED_INTENT
        ]
        if not supported_indices:
            raise ValueError("hierarchical stage 2 has no supported training records")
        stage_2_representation = build_representation(
            "WORD_CHAR_TFIDF", step29g_contract
        )
        stage_2_train = stage_2_representation.fit_transform(
            [train_texts[index] for index in supported_indices]
        )
        stage_2_evaluation = stage_2_representation.transform(evaluation_texts)
        stage_2_classifier = build_classifier(step29g_contract, c_value=1.0)
        fit_classifier(
            stage_2_classifier,
            stage_2_train,
            train_labels[np.asarray(supported_indices, dtype=int)],
        )
        supported_predictions = np.asarray(
            stage_2_classifier.predict(stage_2_evaluation), dtype=object
        )
        if UNSUPPORTED_INTENT in supported_predictions:
            raise ValueError("hierarchical stage 2 predicted its excluded intent")
        predicted = np.where(
            routing == UNSUPPORTED_INTENT,
            UNSUPPORTED_INTENT,
            supported_predictions,
        )
    return _validate_predictions(
        np.asarray(predicted, dtype=object), len(evaluation_rows), allowed_labels
    )


def _ratio(numerator: int, denominator: int, name: str) -> float:
    if denominator <= 0:
        raise ValueError(f"metric denominator is zero: {name}")
    return numerator / denominator


def safety_metrics(
    gold: Sequence[str], predicted: Sequence[str]
) -> dict[str, float | int]:
    if len(gold) != len(predicted):
        raise ValueError("gold and prediction lengths differ")
    protected = set(PROTECTED_INTENTS)
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
        gold, predicted, labels=list(labels), zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(gold, predicted)),
        "balanced_accuracy": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "per_intent": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(labels)
        },
        "confusion_matrix": {
            "label_order": list(labels),
            "values": confusion_matrix(gold, predicted, labels=list(labels)).tolist(),
        },
    }


def subset_macro_f1(
    gold: Sequence[str], predicted: Sequence[str], labels: Sequence[str]
) -> float:
    indices = [index for index, expected in enumerate(gold) if expected in labels]
    return float(
        f1_score(
            [gold[index] for index in indices],
            [predicted[index] for index in indices],
            labels=list(labels),
            average="macro",
            zero_division=0,
        )
    )


def unsupported_boundary_metrics(
    gold: Sequence[str], predicted: Sequence[str]
) -> dict[str, float | int]:
    supported = set(PRIMARY_INTENTS) - {UNSUPPORTED_INTENT}
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
            supported_to_unsupported, supported_total, "supported_to_unsupported_rate"
        ),
        "unsupported_gold_count": unsupported_total,
        "unsupported_predicted_supported_count": unsupported_to_supported,
        "unsupported_to_supported_rate": _ratio(
            unsupported_to_supported,
            unsupported_total,
            "unsupported_to_supported_rate",
        ),
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


def _development_prediction_rows(
    records: Sequence[Mapping[str, Any]],
    predicted: Sequence[str],
    *,
    fold_id: str,
) -> list[dict[str, str]]:
    return [
        {
            "example_id": str(row["example_id"]),
            "gold_intent": str(row["intent"]),
            "predicted_intent": str(guess),
            "fold_id": fold_id,
        }
        for row, guess in zip(records, predicted, strict=True)
    ]


def _fresh_prediction_rows(
    records: Sequence[Mapping[str, Any]], predicted: Sequence[str]
) -> list[dict[str, str]]:
    return [
        {
            "record_id": str(row["record_id"]),
            "source_family_id": str(row["source_family_id"]),
            "gold_intent": str(row["intent"]),
            "predicted_intent": str(guess),
        }
        for row, guess in zip(records, predicted, strict=True)
    ]


def _group_cv_result(
    candidate_id: str,
    inputs: LoadedInputs,
    splits: EvaluationSplits,
    development_bge: np.ndarray,
) -> dict[str, Any]:
    predictions: list[str | None] = [None] * len(inputs.development_examples)
    prediction_rows: list[dict[str, str]] = []
    fold_summaries: list[dict[str, Any]] = []
    for fold in splits.group_folds:
        train_rows = [
            inputs.development_examples[index] for index in fold.train_indices
        ]
        validation_rows = [
            inputs.development_examples[index] for index in fold.validation_indices
        ]
        guessed = fit_predict_candidate(
            candidate_id,
            train_rows,
            validation_rows,
            inputs.step29g_contract,
            train_bge=development_bge[np.asarray(fold.train_indices, dtype=int)],
            evaluation_bge=development_bge[
                np.asarray(fold.validation_indices, dtype=int)
            ],
        ).tolist()
        for index, guess in zip(fold.validation_indices, guessed, strict=True):
            if predictions[index] is not None:
                raise ValueError("duplicate OOF prediction")
            predictions[index] = str(guess)
        prediction_rows.extend(
            _development_prediction_rows(
                validation_rows, [str(value) for value in guessed], fold_id=fold.partition_id
            )
        )
        fold_summaries.append(
            {
                "fold_id": fold.partition_id,
                "train_count": len(train_rows),
                "validation_count": len(validation_rows),
                "prediction_count": len(guessed),
            }
        )
    if any(value is None for value in predictions):
        raise ValueError("missing OOF prediction")
    predicted = [str(value) for value in predictions]
    gold = [str(row["intent"]) for row in inputs.development_examples]
    labels = inputs.step29g_contract["taxonomy"]["intent_label_order"]
    metrics = classification_metrics(gold, predicted, labels)
    metrics["macro_f1_16"] = metrics.pop("macro_f1")
    metrics["primary_8_macro_f1"] = subset_macro_f1(gold, predicted, PRIMARY_INTENTS)
    metrics["safety"] = safety_metrics(gold, predicted)
    metrics["unsupported_boundary"] = unsupported_boundary_metrics(gold, predicted)
    return {
        "completed_fold_count": len(fold_summaries),
        "folds": fold_summaries,
        "prediction_count": len(prediction_rows),
        "unique_prediction_id_count": len(
            {row["example_id"] for row in prediction_rows}
        ),
        "prediction_sha256": sha256_bytes(stable_json_bytes(prediction_rows)),
        "predictions": prediction_rows,
        "metrics": metrics,
    }


def _fresh_evaluation_result(
    candidate_id: str,
    inputs: LoadedInputs,
    development_bge: np.ndarray,
    fresh_bge: np.ndarray,
) -> dict[str, Any]:
    predicted_array = fit_predict_candidate(
        candidate_id,
        inputs.development_examples,
        inputs.fresh_examples,
        inputs.step29g_contract,
        train_bge=development_bge,
        evaluation_bge=fresh_bge,
    )
    predicted = [str(value) for value in predicted_array.tolist()]
    family_results: list[dict[str, Any]] = []
    for family in FRESH_FAMILY_IDS:
        indices = [
            index
            for index, row in enumerate(inputs.fresh_examples)
            if row["source_family_id"] == family
        ]
        rows = [inputs.fresh_examples[index] for index in indices]
        guesses = [predicted[index] for index in indices]
        gold = [str(row["intent"]) for row in rows]
        metrics = classification_metrics(gold, guesses, PRIMARY_INTENTS)
        metrics["primary_8_macro_f1"] = metrics.pop("macro_f1")
        metrics["safety"] = safety_metrics(gold, guesses)
        metrics["unsupported_boundary"] = unsupported_boundary_metrics(gold, guesses)
        family_results.append(
            {
                "source_family_id": family,
                "record_count": len(rows),
                "prediction_count": len(guesses),
                "prediction_sha256": sha256_bytes(
                    stable_json_bytes(_fresh_prediction_rows(rows, guesses))
                ),
                "metrics": metrics,
            }
        )
    if any(row["prediction_count"] != 320 for row in family_results):
        raise ValueError("fresh family prediction coverage changed")
    gold = [str(row["intent"]) for row in inputs.fresh_examples]
    pooled = classification_metrics(gold, predicted, PRIMARY_INTENTS)
    pooled["pooled_fresh_primary_8_macro_f1"] = pooled.pop("macro_f1")
    family_scores = [row["metrics"]["primary_8_macro_f1"] for row in family_results]
    pooled["worst_fresh_family_primary_8_macro_f1"] = float(min(family_scores))
    pooled["mean_fresh_family_primary_8_macro_f1"] = float(np.mean(family_scores))
    pooled["safety"] = safety_metrics(gold, predicted)
    pooled["unsupported_boundary"] = unsupported_boundary_metrics(gold, predicted)
    rows = _fresh_prediction_rows(inputs.fresh_examples, predicted)
    if len(rows) != 640 or len({row["record_id"] for row in rows}) != 640:
        raise ValueError("pooled fresh prediction coverage must equal 640")
    return {
        "full_development_fit_count": 1,
        "full_development_fit_record_count": len(inputs.development_examples),
        "fresh_records_used_for_fitting": 0,
        "same_fitted_candidate_used_for_both_families": True,
        "families": family_results,
        "pooled_prediction_count": len(rows),
        "pooled_prediction_sha256": sha256_bytes(stable_json_bytes(rows)),
        "pooled_predictions": rows,
        "pooled_metrics": pooled,
    }


def apply_mandatory_safety_gates(
    group_result: Mapping[str, Any],
    fresh_result: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    scopes: list[tuple[str, Mapping[str, Any]]] = [
        ("pooled_group_aware_cv", group_result["metrics"]["safety"]),
        *[
            (str(row["source_family_id"]), row["metrics"]["safety"])
            for row in fresh_result["families"]
        ],
        ("pooled_fresh_evaluation", fresh_result["pooled_metrics"]["safety"]),
    ]
    if [scope for scope, _ in scopes] != contract["safety_gates"]["application_scopes"]:
        raise ValueError("required safety scope coverage changed")
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
    if len(gate_results) != 12:
        raise ValueError("exactly twelve mandatory safety-gate checks are required")
    return {
        "required_scope_count": 4,
        "gate_count_per_scope": 3,
        "all_gates_pass": all(row["passed"] for row in gate_results),
        "gates": gate_results,
    }


def determine_candidate_eligibility(
    safety_gate_results: Mapping[str, Any],
    *,
    completed_group_folds: int,
    group_prediction_count: int,
    fresh_prediction_count: int,
    fresh_family_prediction_counts: Sequence[int],
) -> tuple[bool, list[str]]:
    infrastructure = {
        "all_5_group_cv_folds_completed": completed_group_folds == 5,
        "group_cv_prediction_coverage_exact": group_prediction_count
        == DEVELOPMENT_RECORD_COUNT,
        "fresh_prediction_coverage_exact": fresh_prediction_count == FRESH_RECORD_COUNT,
        "fresh_family_prediction_coverage_exact": list(fresh_family_prediction_counts)
        == [FRESH_FAMILY_RECORD_COUNT, FRESH_FAMILY_RECORD_COUNT],
        "no_group_leakage": True,
        "no_fresh_evaluation_training_leakage": True,
        "frozen_hashes_match": True,
        "metrics_finite_and_valid": True,
        "no_threshold_tuning": True,
        "no_prohibited_final_holdout_access": True,
    }
    failed = [name for name, passed in infrastructure.items() if not passed]
    if failed:
        raise RuntimeError("candidate execution incomplete: " + ", ".join(failed))
    reasons = [
        f"mandatory_safety_gate_failed:{row['scope']}:{row['metric']}"
        for row in safety_gate_results["gates"]
        if row["passed"] is False
    ]
    return not reasons, reasons


def model_complexity_rank(candidate_id: str) -> int:
    family = candidate_id.split("__", 1)[0]
    if family not in COMPLEXITY_ORDER:
        raise ValueError(f"candidate has no frozen complexity rank: {candidate_id}")
    return COMPLEXITY_ORDER.index(family) + 1


def candidate_configuration(
    candidate: Mapping[str, Any], inputs: LoadedInputs
) -> dict[str, Any]:
    candidate_id = str(candidate["candidate_id"])
    return {
        "candidate_id": candidate_id,
        "frozen_candidate_definition": dict(candidate),
        "implementation_semantics": inputs.contract[
            "candidate_implementation_semantics"
        ][candidate_id],
        "tfidf_conventions": {
            name: inputs.step29g_contract["representations"][name]
            for name in ("WORD_TFIDF", "CHAR_TFIDF", "WORD_CHAR_TFIDF")
        },
        "bge_convention": inputs.step29g_contract["representations"]["BGE_SMALL"],
        "linear_svc_convention": inputs.step29g_contract["classifiers"]["LINEAR_SVC"],
        "model_complexity_rank": model_complexity_rank(candidate_id),
    }


def _selection_metrics(
    candidate_id: str,
    group_result: Mapping[str, Any],
    fresh_result: Mapping[str, Any],
) -> dict[str, float | int | str]:
    pooled = fresh_result["pooled_metrics"]
    return {
        "worst_fresh_family_primary_8_macro_f1": pooled[
            "worst_fresh_family_primary_8_macro_f1"
        ],
        "mean_fresh_family_primary_8_macro_f1": pooled[
            "mean_fresh_family_primary_8_macro_f1"
        ],
        "pooled_fresh_primary_8_macro_f1": pooled[
            "pooled_fresh_primary_8_macro_f1"
        ],
        "pooled_group_cv_macro_f1_16": group_result["metrics"]["macro_f1_16"],
        "pooled_fresh_protected_false_positive_rate": pooled["safety"][
            "protected_false_positive_rate"
        ],
        "pooled_fresh_protected_recall": pooled["safety"]["protected_recall"],
        "pooled_fresh_unsupported_recall": pooled["safety"]["unsupported_recall"],
        "model_complexity_rank": model_complexity_rank(candidate_id),
        "candidate_id": candidate_id,
    }


def evaluate_candidate(
    candidate: Mapping[str, Any],
    inputs: LoadedInputs,
    splits: EvaluationSplits,
    bge_matrices: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    candidate_id = str(candidate["candidate_id"])
    group = _group_cv_result(
        candidate_id, inputs, splits, bge_matrices["development"]
    )
    fresh = _fresh_evaluation_result(
        candidate_id,
        inputs,
        bge_matrices["development"],
        bge_matrices["fresh_evaluation"],
    )
    _require_finite_metrics(group["metrics"])
    _require_finite_metrics(fresh["pooled_metrics"])
    for family in fresh["families"]:
        _require_finite_metrics(family["metrics"])
    gates = apply_mandatory_safety_gates(group, fresh, inputs.contract)
    eligible, reasons = determine_candidate_eligibility(
        gates,
        completed_group_folds=group["completed_fold_count"],
        group_prediction_count=group["prediction_count"],
        fresh_prediction_count=fresh["pooled_prediction_count"],
        fresh_family_prediction_counts=[
            row["prediction_count"] for row in fresh["families"]
        ],
    )
    configuration = candidate_configuration(candidate, inputs)
    return {
        "candidate_id": candidate_id,
        "configuration": configuration,
        "configuration_sha256": sha256_bytes(stable_json_bytes(configuration)),
        "execution_status": "COMPLETED",
        "group_aware_cv": group,
        "fresh_source_evaluation": fresh,
        "safety_gate_results": gates,
        "eligible": eligible,
        "ineligibility_reasons": reasons,
        "selection_metrics": _selection_metrics(candidate_id, group, fresh),
        "threshold_tuning_performed": False,
        "fresh_evaluation_used_for_fitting": False,
        "raw_text_persisted": False,
    }


def candidate_is_better(
    candidate: Mapping[str, Any],
    incumbent: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> bool:
    for criterion in contract["selection_rule"]["ordered_lexicographic_criteria"]:
        metric = criterion["metric"]
        direction = criterion["direction"]
        left = candidate["selection_metrics"][metric]
        right = incumbent["selection_metrics"][metric]
        if left == right:
            continue
        if direction == "ascending_lexical":
            return str(left) < str(right)
        if direction == "maximize":
            return float(left) > float(right)
        if direction in {"minimize", "prefer_lower_rank"}:
            return float(left) < float(right)
        raise ValueError(f"unknown selection direction: {direction}")
    return False


def select_candidate(
    candidate_results: Sequence[Mapping[str, Any]], contract: Mapping[str, Any]
) -> dict[str, Any]:
    eligible = [row for row in candidate_results if row["eligible"] is True]
    if not eligible:
        frozen = contract["selection_rule"]["if_no_candidate_is_eligible"]
        return {
            **frozen,
            "eligible_candidate_count": 0,
            "selected_candidate_id": None,
            "next_required": None,
            "step29i_authorized": False,
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
        "next_required": None,
        "step29i_authorized": False,
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
    payload: Mapping[str, Any], records: Sequence[Mapping[str, Any]]
) -> None:
    strings = set(all_string_values(payload))
    raw_texts = {str(row["text"]) for row in records}
    if strings & raw_texts:
        raise ValueError("tracked result artifact contains raw example text")
    if strings & {"text", "texts", "utterance", "utterances", "raw_text"}:
        raise ValueError("tracked result artifact contains a raw-text field")


def build_results_payload(
    inputs: LoadedInputs,
    splits: EvaluationSplits,
    candidate_results: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    candidates = inputs.contract["candidate_search_space"]["candidates"]
    if [row["candidate_id"] for row in candidate_results] != list(
        EXPECTED_CANDIDATE_IDS
    ):
        raise ValueError("complete ordered four-candidate results are required")
    selection = select_candidate(candidate_results, inputs.contract)
    payload = {
        "schema_version": RESULTS_SCHEMA_VERSION,
        "phase": "V2-C6 targeted-remediation development model selection",
        "execution_status": "COMPLETED",
        "contract": {
            "path": str(CONTRACT_PATH.relative_to(REPOSITORY_ROOT)),
            "sha256": inputs.contract_sha256,
            "schema_version": CONTRACT_SCHEMA_VERSION,
        },
        "frozen_inputs": {
            "development_dataset": {
                "path": str(DEVELOPMENT_DATASET_PATH.relative_to(REPOSITORY_ROOT)),
                "schema_version": DEVELOPMENT_SCHEMA_VERSION,
                "sha256": inputs.development_sha256,
                "record_count": len(inputs.development_examples),
            },
            "fresh_evaluation_dataset": {
                "path": str(FRESH_DATASET_PATH.relative_to(REPOSITORY_ROOT)),
                "schema_version": FRESH_SCHEMA_VERSION,
                "sha256": inputs.fresh_sha256,
                "record_count": len(inputs.fresh_examples),
                "excluded_from_fitting": True,
            },
            "step29g_convention_source": inputs.contract["source_artifacts"][
                "step29g_contract"
            ],
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "fastembed": importlib.metadata.version("fastembed"),
        },
        "candidate_definitions": candidates,
        "candidate_count": 4,
        "completed_candidate_count": len(candidate_results),
        "fold_configuration": inputs.contract["group_aware_cross_validation"],
        "split_audit": splits.audit,
        "candidate_results": list(candidate_results),
        "selection_rule": inputs.contract["selection_rule"],
        "selection": selection,
        "next_required": None,
        "governance": {
            "development_model_selection_performed": True,
            "model_selection_performed": True,
            "candidate_model_fitting_performed": True,
            "production_model_training_performed": False,
            "full_development_candidate_fits_for_fresh_evaluation": 4,
            "fresh_evaluation_performed": True,
            "fresh_evaluation_used_for_fitting": False,
            "fresh_evaluation_used_for_threshold_tuning": False,
            "persisted_fitted_classifier": False,
            "final_holdout_accessed": False,
            "final_holdout_evaluated": False,
            "threshold_tuning_performed": False,
            "gates_weakened": False,
            "winner_forced": False,
            "runtime_behavior_changed": False,
            "final_model_acceptance_claimed": False,
            "step29i_authorized": False,
        },
    }
    validate_results_payload(payload, inputs, splits)
    return payload


def validate_results_payload(
    payload: Mapping[str, Any], inputs: LoadedInputs, splits: EvaluationSplits
) -> None:
    if payload.get("schema_version") != RESULTS_SCHEMA_VERSION:
        raise ValueError("targeted-remediation result schema changed")
    if payload.get("contract", {}).get("sha256") != inputs.contract_sha256:
        raise ValueError("targeted-remediation contract lineage changed")
    if payload.get("split_audit") != splits.audit:
        raise ValueError("targeted-remediation split audit changed")
    results = payload.get("candidate_results")
    if not isinstance(results, list) or len(results) != 4:
        raise ValueError("all four candidate results are required")
    for result, candidate in zip(
        results,
        inputs.contract["candidate_search_space"]["candidates"],
        strict=True,
    ):
        if result.get("candidate_id") != candidate["candidate_id"]:
            raise ValueError("candidate result ID or order changed")
        if result.get("group_aware_cv", {}).get("prediction_count") != 9608:
            raise ValueError("candidate OOF coverage changed")
        fresh = result.get("fresh_source_evaluation", {})
        if fresh.get("pooled_prediction_count") != 640:
            raise ValueError("candidate pooled fresh coverage changed")
        if [row.get("prediction_count") for row in fresh.get("families", [])] != [
            320,
            320,
        ]:
            raise ValueError("candidate fresh-family coverage changed")
        if fresh.get("fresh_records_used_for_fitting") != 0:
            raise ValueError("fresh records entered candidate fitting")
        if result.get("eligible") is not (not result.get("ineligibility_reasons")):
            raise ValueError("candidate eligibility reasons are inconsistent")
        _require_finite_metrics(result["selection_metrics"])
    if payload.get("selection") != select_candidate(results, inputs.contract):
        raise ValueError("deterministic candidate selection changed")
    governance = payload.get("governance", {})
    required_false = (
        "production_model_training_performed",
        "fresh_evaluation_used_for_fitting",
        "fresh_evaluation_used_for_threshold_tuning",
        "persisted_fitted_classifier",
        "final_holdout_accessed",
        "final_holdout_evaluated",
        "threshold_tuning_performed",
        "gates_weakened",
        "winner_forced",
        "runtime_behavior_changed",
        "final_model_acceptance_claimed",
        "step29i_authorized",
    )
    if any(governance.get(field) is not False for field in required_false):
        raise ValueError("targeted-remediation result governance changed")
    validate_text_free(
        payload, (*inputs.development_examples, *inputs.fresh_examples)
    )


def build_results_manifest(
    payload: Mapping[str, Any], result_bytes: bytes
) -> dict[str, Any]:
    return {
        "schema_version": RESULTS_MANIFEST_SCHEMA_VERSION,
        "phase": payload["phase"],
        "results": {
            "path": str(RESULTS_PATH.relative_to(REPOSITORY_ROOT)),
            "schema_version": RESULTS_SCHEMA_VERSION,
            "sha256": sha256_bytes(result_bytes),
        },
        "runner": {
            "path": RUNNER_RELATIVE_PATH,
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "contract": payload["contract"],
        "frozen_inputs": payload["frozen_inputs"],
        "split_audit_sha256": payload["split_audit"]["audit_sha256"],
        "candidate_count": payload["candidate_count"],
        "completed_candidate_count": payload["completed_candidate_count"],
        "selection_status": payload["selection"]["selection_status"],
        "selected_candidate_id": payload["selection"]["selected_candidate_id"],
        "winner_forced": False,
        "gates_weakened": False,
        "model_selection_performed": True,
        "fresh_evaluation_used_for_fitting": False,
        "fresh_evaluation_used_for_threshold_tuning": False,
        "persisted_fitted_classifier": False,
        "final_holdout_accessed": False,
        "final_holdout_evaluated": False,
        "runtime_behavior_changed": False,
        "final_model_acceptance_claimed": False,
        "step29i_authorized": False,
        "next_required": None,
    }


def run_model_selection(inputs: LoadedInputs) -> tuple[Path, Path]:
    if RESULTS_PATH.exists() or RESULTS_MANIFEST_PATH.exists():
        raise FileExistsError(
            "targeted-remediation result artifact exists; refusing overwrite"
        )
    splits = build_evaluation_splits(
        inputs.development_examples, inputs.fresh_examples, inputs.contract
    )
    bge_matrices, _ = prepare_bge_cache(inputs)
    candidate_results = [
        evaluate_candidate(candidate, inputs, splits, bge_matrices)
        for candidate in inputs.contract["candidate_search_space"]["candidates"]
    ]
    payload = build_results_payload(inputs, splits, candidate_results)
    result_bytes = stable_json_bytes(payload)
    manifest_bytes = stable_json_bytes(build_results_manifest(payload, result_bytes))
    write_bytes_atomically(RESULTS_PATH, result_bytes)
    write_bytes_atomically(RESULTS_MANIFEST_PATH, manifest_bytes)
    return RESULTS_PATH, RESULTS_MANIFEST_PATH


def preflight(inputs: LoadedInputs) -> dict[str, Any]:
    splits = build_evaluation_splits(
        inputs.development_examples, inputs.fresh_examples, inputs.contract
    )
    packages: dict[str, str | None] = {}
    for package in ("numpy", "scipy", "scikit-learn", "fastembed"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    cache_presence = (BGE_CACHE_PATH.exists(), BGE_CACHE_MANIFEST_PATH.exists())
    if any(cache_presence) and not all(cache_presence):
        raise FileExistsError("incomplete targeted-remediation BGE cache pair")
    if all(cache_presence):
        validate_bge_cache(inputs)
    return {
        "status": "READY" if all(packages.values()) else "MISSING_REQUIRED_PACKAGE",
        "contract_sha256": inputs.contract_sha256,
        "development_dataset_sha256": inputs.development_sha256,
        "fresh_evaluation_dataset_sha256": inputs.fresh_sha256,
        "development_record_count": len(inputs.development_examples),
        "fresh_evaluation_record_count": len(inputs.fresh_examples),
        "candidate_count": 4,
        "group_cv_fold_count": len(splits.group_folds),
        "expected_group_cv_classifier_fits": 25,
        "expected_full_development_classifier_fits": 5,
        "expected_total_temporary_classifier_fits": 30,
        "split_audit_sha256": splits.audit["audit_sha256"],
        "bge_cache_present": all(cache_presence),
        "result_artifacts_present": RESULTS_PATH.exists()
        or RESULTS_MANIFEST_PATH.exists(),
        "packages": packages,
        "embeddings_generated": False,
        "classifier_fitting_performed": False,
        "inference_performed": False,
        "model_selection_performed": False,
        "files_written": False,
        "fresh_evaluation_used_for_fitting": False,
        "final_holdout_accessed": False,
        "runtime_behavior_changed": False,
        "step29i_authorized": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Execute frozen V2-C6 targeted-remediation development model selection"
        )
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
