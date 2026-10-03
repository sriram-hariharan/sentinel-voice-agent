"""Run the frozen V2-C6 R3 protected-intent gate model-selection experiment.

Exactly two candidates are compared: ``HYBRID_CONTROL_R3`` and
``HYBRID_PROTECTED_VERIFIER_R3``. Both use the same frozen Hybrid primary router
and the same frozen five-fold grouped split. The gated candidate adds four
binary word+char TF-IDF LinearSVC verifiers, one per protected intent. A
verifier only changes routing; it never authorizes anything.

Fresh R3 evaluation records never enter any fit, every final-holdout path is
rejected, no fitted model is persisted, and Step 29I remains blocked.

The Hybrid, TF-IDF, BGE, LinearSVC, and metric helpers are reused from the
hash-pinned R2 runner so the R3 experiment uses exactly the established
conventions that produced the consumed R2 evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import sys
import tempfile
import types
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from sklearn.model_selection import StratifiedGroupKFold

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ML_RELATIVE_DIRECTORY = "data/evals/v2/ml"
CONTRACT_RELATIVE_PATH = (
    f"{ML_RELATIVE_DIRECTORY}/v2c6_protected_intent_gate_remediation_design_contract.json"
)
PRIOR_CONTRACT_RELATIVE_PATH = (
    f"{ML_RELATIVE_DIRECTORY}/v2c6_targeted_remediation_design_contract.json"
)
STEP29G_CONTRACT_RELATIVE_PATH = (
    f"{ML_RELATIVE_DIRECTORY}/v2c6_source_aware_model_selection_contract.json"
)
TRAINING_DATASET_RELATIVE_PATH = (
    f"{ML_RELATIVE_DIRECTORY}/v2c6_r3_protected_intent_gate_training_examples.json"
)
TRAINING_MANIFEST_RELATIVE_PATH = (
    f"{ML_RELATIVE_DIRECTORY}/v2c6_r3_protected_intent_gate_training_examples.manifest.json"
)
DEVELOPMENT_DATASET_RELATIVE_PATH = (
    f"{ML_RELATIVE_DIRECTORY}/v2c6_r3_protected_intent_gate_development_dataset.json"
)
DEVELOPMENT_MANIFEST_RELATIVE_PATH = (
    f"{ML_RELATIVE_DIRECTORY}/"
    "v2c6_r3_protected_intent_gate_development_dataset.manifest.json"
)
FRESH_DATASET_RELATIVE_PATH = (
    f"{ML_RELATIVE_DIRECTORY}/v2c6_r3_fresh_source_evaluation_dataset.json"
)
FRESH_MANIFEST_RELATIVE_PATH = (
    f"{ML_RELATIVE_DIRECTORY}/v2c6_r3_fresh_source_evaluation_dataset.manifest.json"
)
RESULTS_RELATIVE_PATH = (
    f"{ML_RELATIVE_DIRECTORY}/v2c6_r3_protected_intent_gate_model_selection_results.json"
)
RESULTS_MANIFEST_RELATIVE_PATH = (
    f"{ML_RELATIVE_DIRECTORY}/"
    "v2c6_r3_protected_intent_gate_model_selection_results.manifest.json"
)
BGE_CACHE_RELATIVE_PATH = (
    f"{ML_RELATIVE_DIRECTORY}/local/"
    "v2c6_r3_protected_intent_gate_model_selection_bge_cache.npz"
)
BGE_CACHE_MANIFEST_RELATIVE_PATH = (
    f"{ML_RELATIVE_DIRECTORY}/local/"
    "v2c6_r3_protected_intent_gate_model_selection_bge_cache.manifest.json"
)
PROHIBITED_FINAL_HOLDOUT_RELATIVE_PATH = f"{ML_RELATIVE_DIRECTORY}/v2c5_final_holdout.json"

CONTRACT_PATH = REPOSITORY_ROOT / CONTRACT_RELATIVE_PATH
PRIOR_CONTRACT_PATH = REPOSITORY_ROOT / PRIOR_CONTRACT_RELATIVE_PATH
STEP29G_CONTRACT_PATH = REPOSITORY_ROOT / STEP29G_CONTRACT_RELATIVE_PATH
TRAINING_DATASET_PATH = REPOSITORY_ROOT / TRAINING_DATASET_RELATIVE_PATH
TRAINING_MANIFEST_PATH = REPOSITORY_ROOT / TRAINING_MANIFEST_RELATIVE_PATH
DEVELOPMENT_DATASET_PATH = REPOSITORY_ROOT / DEVELOPMENT_DATASET_RELATIVE_PATH
DEVELOPMENT_MANIFEST_PATH = REPOSITORY_ROOT / DEVELOPMENT_MANIFEST_RELATIVE_PATH
FRESH_DATASET_PATH = REPOSITORY_ROOT / FRESH_DATASET_RELATIVE_PATH
FRESH_MANIFEST_PATH = REPOSITORY_ROOT / FRESH_MANIFEST_RELATIVE_PATH
RESULTS_PATH = REPOSITORY_ROOT / RESULTS_RELATIVE_PATH
RESULTS_MANIFEST_PATH = REPOSITORY_ROOT / RESULTS_MANIFEST_RELATIVE_PATH
BGE_CACHE_PATH = REPOSITORY_ROOT / BGE_CACHE_RELATIVE_PATH
BGE_CACHE_MANIFEST_PATH = REPOSITORY_ROOT / BGE_CACHE_MANIFEST_RELATIVE_PATH
RUNNER_RELATIVE_PATH = "scripts/run_v2c6_r3_protected_intent_gate_model_selection.py"
R2_RUNNER_RELATIVE_PATH = "scripts/run_v2c6_targeted_remediation_model_selection.py"
EXPECTED_R2_RUNNER_SHA256 = (
    "043e00d5cab470359b2ad5e4f78492a715bfb9ed6e83e766fcd08e6c923af6b0"
)
R2_HELPER_MODULE_NAME = "_sentinelvoice_v2c6_r2_frozen_runner_helpers"

CONTRACT_SCHEMA_VERSION = "v2c6-protected-intent-gate-remediation-design-contract.v1"
PRIOR_CONTRACT_SCHEMA_VERSION = "v2c6-targeted-remediation-design-contract.v1"
STEP29G_CONTRACT_SCHEMA_VERSION = "v2c6-source-aware-model-selection-contract.v1"
TRAINING_SCHEMA_VERSION = "v2c6-r3-protected-intent-gate-training-examples.v1"
TRAINING_MANIFEST_SCHEMA_VERSION = (
    "v2c6-r3-protected-intent-gate-training-examples-manifest.v1"
)
DEVELOPMENT_SCHEMA_VERSION = "v2c6-r3-protected-intent-gate-development-dataset.v1"
DEVELOPMENT_MANIFEST_SCHEMA_VERSION = (
    "v2c6-r3-protected-intent-gate-development-dataset-manifest.v1"
)
FRESH_SCHEMA_VERSION = "v2c6-r3-fresh-source-evaluation-dataset.v1"
FRESH_MANIFEST_SCHEMA_VERSION = "v2c6-r3-fresh-source-evaluation-dataset-manifest.v1"
RESULTS_SCHEMA_VERSION = "v2c6-r3-protected-intent-gate-model-selection-results.v1"
RESULTS_MANIFEST_SCHEMA_VERSION = (
    "v2c6-r3-protected-intent-gate-model-selection-results-manifest.v1"
)
BGE_CACHE_SCHEMA_VERSION = "v2c6-r3-protected-intent-gate-model-selection-bge-cache.v1"
RESULTS_PHASE = "V2-C6 R3 protected-intent gate model selection"

EXPECTED_CONTRACT_SHA256 = (
    "4a744a48e18d674c1f87bbd53f470cae95f0bf09fafd7920942310712c0c894e"
)
EXPECTED_PRIOR_CONTRACT_SHA256 = (
    "339bd20f4bb94e73a42bd297c053aef8deaf55ee42a3b2145b8792172e7d5e91"
)
EXPECTED_STEP29G_CONTRACT_SHA256 = (
    "de4566b7fdb097e78fcd23b8ffa4682c4757cc7c5901c3f9a49d7da28192b1ba"
)
EXPECTED_TRAINING_SHA256 = (
    "db2db4d6116c04712ae9b5979c71c8cbdc1ed9c4e330ba4f3915ba8b62efce14"
)
EXPECTED_TRAINING_MANIFEST_SHA256 = (
    "f5d3ba48a425e72eab8b45d5835f7f3fc8334c42ba6c0d6ab3e6d700eeecefbf"
)
EXPECTED_DEVELOPMENT_SHA256 = (
    "d27c411cefdba2cc4d8a9c70493b05f43542e1542f1b202c909090a7185f36fe"
)
EXPECTED_DEVELOPMENT_MANIFEST_SHA256 = (
    "5ead89b05dfc4b649e048cc5bb44c261d38f9536594b02f9e38a8024514168fe"
)
EXPECTED_FRESH_SHA256 = (
    "e530f24c236ed03c8228ab465165996d7ea94fb24f43cc1ddb1545118918d369"
)
EXPECTED_FRESH_MANIFEST_SHA256 = (
    "5972e5c81f552cf978219075f0536689693f1a0e78892fedc148b33df07f40ba"
)
EXPECTED_EXISTING_DEVELOPMENT_SHA256 = (
    "133456aa10552058fe67ed2eed37acbe583e31331656174cd2e018430cca799d"
)

CONTROL_CANDIDATE_ID = "HYBRID_CONTROL_R3"
GATED_CANDIDATE_ID = "HYBRID_PROTECTED_VERIFIER_R3"
EXPECTED_CANDIDATE_IDS = (CONTROL_CANDIDATE_ID, GATED_CANDIDATE_ID)
PRIMARY_ROUTER_ID = "HYBRID_BGE_TFIDF_LINEAR_SVC__C=1.0__class_weight=none"
UNSUPPORTED_INTENT = "unsupported_or_uncertain"
PROTECTED_INTENTS = (
    "cancel_transfer",
    "close_account",
    "create_dispute",
    "freeze_card",
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
TRAINING_FAMILY_IDS = (
    "v2c6_r3_train_sf1_minimal_explicitness",
    "v2c6_r3_train_sf2_contextual_boundary",
    "v2c6_r3_train_sf3_conversational_ambiguity",
)
FRESH_FAMILY_IDS = (
    "v2c6_r3_eval_sf1_independent_casework",
    "v2c6_r3_eval_sf2_independent_naturalistic",
)
GROUP_CV_SCOPE = "pooled_group_aware_cv"
POOLED_FRESH_SCOPE = "pooled_r3_fresh_evaluation"
SAFETY_SCOPES = (GROUP_CV_SCOPE, *FRESH_FAMILY_IDS, POOLED_FRESH_SCOPE)
R3_LINEAGE = "PROTECTED_INTENT_GATE_R3_ADDITION"
FAILURE_NEXT_REQUIRED = "routing_architecture_fallback_decision"
SUCCESS_NEXT_REQUIRED = "v2c6_candidate_freeze_before_final_holdout"

EXISTING_DEVELOPMENT_RECORD_COUNT = 9608
TRAINING_RECORD_COUNT = 480
DEVELOPMENT_RECORD_COUNT = 10088
FRESH_RECORD_COUNT = 640
FRESH_FAMILY_RECORD_COUNT = 320
FOLD_COUNT = 5
VERIFIER_POSITIVE_COUNT = 60
VERIFIER_NEGATIVE_COUNT = 60
TIE_ABSOLUTE_TOLERANCE = 1e-12

VERIFIER_ACCEPT = "accept"
VERIFIER_REJECT = "reject"
NO_VERIFIER_DECISION: dict[str, Any] = {
    "verifier_invoked": False,
    "verifier_intent": None,
    "verifier_decision": None,
}

EXPECTED_CONTRACT_STATUS = {
    "candidate_selected": False,
    "contract_frozen": True,
    "data_authored": False,
    "dataset_mutated": False,
    "embeddings_generated": False,
    "final_holdout_accessed": False,
    "final_model_acceptance_claimed": False,
    "model_fitting_performed": False,
    "model_inference_performed": False,
    "model_training_performed": False,
    "remediation_design_frozen": True,
    "remediation_executed": False,
    "runtime_behavior_changed": False,
    "step29i_authorized": False,
    "threshold_tuning_performed": False,
}
EXPECTED_PRIMARY_ROUTER = {
    "C": 1.0,
    "candidate_id": PRIMARY_ROUTER_ID,
    "class_weight": None,
    "classifier": "LinearSVC",
    "representation": "HYBRID_BGE_TFIDF",
}
EXPECTED_ARCHITECTURE_DECISION = {
    "authorization_semantics": {
        "application_controls": [
            "authentication",
            "ownership",
            "explicit_confirmation",
            "idempotency",
            "protected_tool_execution",
        ],
        "verifier_is_authorization": False,
        "verifier_only_changes_routing": True,
    },
    "generic_supported_vs_unsupported_gate_permitted": False,
    "generic_supported_vs_unsupported_gate_prior_result": (
        "tested_and_did_not_eliminate_measured_weakness"
    ),
    "primary_router": EXPECTED_PRIMARY_ROUTER,
    "protected_intents": list(PROTECTED_INTENTS),
    "routing_rule": {
        "non_protected_primary_prediction": "return_primary_prediction_unchanged",
        "protected_primary_prediction": (
            "invoke_verifier_dedicated_to_predicted_protected_intent"
        ),
        "verifier_accept": "preserve_primary_protected_prediction",
        "verifier_reject": "return_unsupported_or_uncertain",
    },
}
EXPECTED_CANDIDATE_SEARCH_SPACE = {
    "additional_candidates_permitted": False,
    "candidate_count": 2,
    "candidates": [
        {
            "candidate_id": CONTROL_CANDIDATE_ID,
            "complexity_rank": 1,
            "primary_router": PRIMARY_ROUTER_ID,
            "primary_training_dataset_record_count": DEVELOPMENT_RECORD_COUNT,
            "protected_intent_verifiers": False,
            "role": "identical_data_control",
        },
        {
            "candidate_id": GATED_CANDIDATE_ID,
            "complexity_rank": 2,
            "primary_router": PRIMARY_ROUTER_ID,
            "primary_training_dataset_record_count": DEVELOPMENT_RECORD_COUNT,
            "protected_intent_verifiers": True,
            "role": "protected_intent_verifier_intervention",
        },
    ],
    "exact_candidate_ids": list(EXPECTED_CANDIDATE_IDS),
    "prohibited_expansion": [
        "bge_verifier_variants",
        "class_weight_variants",
        "additional_C_values",
        "threshold_variants",
        "calibration_variants",
        "llm_routing",
        "additional_architectures",
        "new_model_families",
    ],
}
EXPECTED_VERIFIER_ARCHITECTURE = {
    "additional_embeddings": False,
    "C": 1.0,
    "calibration": False,
    "class_weight": None,
    "classifier": "LinearSVC",
    "confidence_override": False,
    "independent_binary_verifier_count": 4,
    "llm_verifier": False,
    "negative_class": "targeted_unsupported_or_uncertain_hard_negatives_for_protected_intent",
    "positive_class": "dedicated_protected_intent",
    "representation": "WORD_CHAR_TFIDF",
    "representation_convention": "existing_frozen_word_char_tfidf_convention",
    "threshold_tuning": False,
    "training_source": "r3_targeted_training_addendum_only",
    "historical_development_records_permitted_in_verifier_fitting": False,
    "verifiers": [
        {
            "negative_record_count": VERIFIER_NEGATIVE_COUNT,
            "positive_intent": intent,
            "positive_record_count": VERIFIER_POSITIVE_COUNT,
            "total_record_count": VERIFIER_POSITIVE_COUNT + VERIFIER_NEGATIVE_COUNT,
        }
        for intent in PROTECTED_INTENTS
    ],
}
EXPECTED_EVALUATION_PROTOCOL = {
    "fresh_evaluation": {
        "control_candidate_fit": {
            "fit_count": 1,
            "primary_router_fit_population": "all_10088_expanded_development_records",
        },
        "gated_candidate_fit": {
            "fit_count": 1,
            "primary_router_fit_population": "all_10088_expanded_development_records",
            "verifier_fit_population": "all_120_relevant_r3_addendum_records_per_verifier",
        },
        "evaluation_scopes": [*FRESH_FAMILY_IDS, POOLED_FRESH_SCOPE],
        "no_refit_between_source_families": True,
        "same_fitted_candidate_used_for_both_families": True,
    },
    "grouped_development_cv": {
        "fold_count": FOLD_COUNT,
        "group_field": "group_id",
        "group_id_is_atomic": True,
        "method": "StratifiedGroupKFold",
        "n_splits": FOLD_COUNT,
        "primary_router_fit_rule": "fit_only_on_fold_development_training_partition",
        "random_state": 20260930,
        "same_folds_for_both_candidates": True,
        "shuffle": True,
        "verifier_fit_rule": "use_only_relevant_r3_verifier_examples_in_fold_training_partition",
        "fold_test_records_in_primary_or_verifier_fitting_permitted": False,
        "verifier_fold_test_leakage_permitted": False,
        "verifier_runs_after_primary_prediction": True,
    },
}
EXPECTED_SAFETY_GATES = {
    "application_scopes": list(SAFETY_SCOPES),
    "fresh_family_protected_false_positive_allowance": {
        "maximum_passing_count": 1,
        "non_protected_gold_count": 160,
        "threshold": 0.01,
    },
    "gates": [
        {
            "comparison": "greater_than_or_equal",
            "metric": "protected_recall",
            "threshold": 0.8,
        },
        {
            "comparison": "less_than_or_equal",
            "metric": "protected_false_positive_rate",
            "threshold": 0.01,
        },
        {
            "comparison": "greater_than_or_equal",
            "metric": "unsupported_recall",
            "threshold": 0.8,
        },
    ],
    "mandatory_for_candidate_eligibility": True,
    "must_pass_every_gate_on_every_scope": True,
    "thresholds_may_be_weakened": False,
    "thresholds_unchanged": True,
}
SELECTION_CRITERIA = (
    ("worst_fresh_family_primary_8_macro_f1", "maximize"),
    ("pooled_fresh_primary_8_macro_f1", "maximize"),
    ("pooled_group_cv_macro_f1_16", "maximize"),
    ("pooled_fresh_protected_false_positive_rate", "minimize"),
    ("pooled_fresh_protected_recall", "maximize"),
    ("pooled_fresh_unsupported_recall", "maximize"),
    ("model_complexity_rank", "prefer_lower_rank"),
    ("candidate_id", "ascending_lexical"),
)
NO_ACCEPTABLE_CANDIDATE = {
    "gates_weakened": False,
    "selected_candidate": None,
    "selection_status": "NO_ACCEPTABLE_CANDIDATE",
    "winner_forced": False,
}
EXPECTED_SELECTION_RULE = {
    "complexity_order": list(EXPECTED_CANDIDATE_IDS),
    "eligible_candidates_only": True,
    "if_no_candidate_is_eligible": NO_ACCEPTABLE_CANDIDATE,
    "ordered_lexicographic_criteria": [
        {"direction": direction, "metric": metric}
        for metric, direction in SELECTION_CRITERIA
    ],
    "single_weighted_score_used": False,
}
EXPECTED_STOP_RULE = {
    "automatic_r4_classifier_or_data_cycle_authorized": False,
    "failure_continuation": {
        "action": "freeze_separate_product_level_fallback_decision",
        "candidate_expansion_permitted": False,
        "data_authoring_permitted": False,
        "gates_may_be_weakened": False,
        "next_required": FAILURE_NEXT_REQUIRED,
        "representation_or_model_family_addition_permitted": False,
        "trigger": "neither_candidate_passes_all_mandatory_gates_on_all_required_scopes",
    },
    "fallback_examples_for_future_consideration_only": [
        "require_clarification_for_protected_or_uncertain_routing",
        "llm_structured_verification_under_separate_governance",
    ],
    "fallback_implemented_by_this_contract": False,
    "success_continuation": {
        "final_holdout_access_authorized": False,
        "next_required": SUCCESS_NEXT_REQUIRED,
        "step29i_automatically_authorized": False,
        "trigger": "at_least_one_candidate_passes_all_mandatory_gates_on_all_required_scopes",
    },
}
EXPECTED_GATE_DIAGNOSTICS = {
    "alter_mandatory_safety_gates": False,
    "required_metrics": [
        "primary_protected_predictions_presented_to_verifier",
        "verifier_accept_count",
        "verifier_reject_to_unsupported_count",
        "per_protected_intent_verifier_recall",
        "protected_false_positives_prevented_by_verifier",
        "true_protected_requests_rejected_by_verifier",
        "final_protected_false_positive_rate",
        "final_protected_recall",
        "final_unsupported_recall",
    ],
}
EXPECTED_EXPANDED_DEVELOPMENT = {
    "both_candidates_use_identical_primary_training_dataset": True,
    "existing_frozen_record_count": EXISTING_DEVELOPMENT_RECORD_COUNT,
    "expected_expanded_record_count": DEVELOPMENT_RECORD_COUNT,
    "new_r3_targeted_record_count": TRAINING_RECORD_COUNT,
    "purpose": "isolate_verifier_architecture_from_addendum_effect",
}
EXPECTED_FINAL_HOLDOUT_POLICY = {
    "access_permitted": False,
    "final_model_acceptance_claim_permitted": False,
    "hashing_permitted": False,
    "inspection_permitted": False,
    "parsing_permitted": False,
    "prohibited_path": PROHIBITED_FINAL_HOLDOUT_RELATIVE_PATH,
    "searching_permitted": False,
}
EXPECTED_PRIOR_COMMON_SEMANTICS = {
    "calibration_permitted": False,
    "class_weight": None,
    "classifier_random_state": 20260930,
    "post_hoc_rule_override_permitted": False,
    "threshold_tuning_permitted": False,
}
EXPECTED_PRIOR_HYBRID_SEMANTICS = {
    "bge_block": {
        "dimensions": 384,
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "preprocessing_convention": "reuse_frozen_step29g_BGE_SMALL_convention",
    },
    "calibration_permitted": False,
    "concatenation": (
        "deterministic_sparse_hstack_of_word_char_tfidf_and_dense_bge_block_converted_to_sparse"
    ),
    "final_combined_row_l2_normalization": True,
    "learned_fusion_layer_permitted": False,
    "new_embedding_model_introduced": False,
    "scientific_python_dependencies": "existing_local_dependencies_only",
    "tfidf_block": {
        "members": ["WORD_TFIDF", "CHAR_TFIDF"],
        "preprocessing_convention": "reuse_frozen_step29g_WORD_CHAR_TFIDF_convention",
    },
    "threshold_tuning_permitted": False,
}
DATASET_GOVERNANCE_FALSE_FIELDS = (
    "candidate_selected",
    "embeddings_generated",
    "final_holdout_accessed",
    "inference_performed",
    "model_fitting_performed",
    "model_inference_performed",
    "model_selection_performed",
    "models_run",
    "runtime_behavior_changed",
    "step29i_authorized",
    "threshold_tuning_performed",
)
RESULT_GOVERNANCE_FALSE_FIELDS = (
    "production_model_training_performed",
    "fresh_evaluation_used_for_fitting",
    "fresh_evaluation_used_for_threshold_tuning",
    "fresh_evaluation_used_for_candidate_modification",
    "threshold_tuning_performed",
    "calibration_performed",
    "persisted_fitted_classifier",
    "final_holdout_accessed",
    "final_holdout_evaluated",
    "gates_weakened",
    "winner_forced",
    "runtime_behavior_changed",
    "final_model_acceptance_claimed",
    "production_ready_claimed",
    "step29i_authorized",
    "r4_classifier_or_data_cycle_authorized",
    "routing_fallback_implemented",
)
RESULT_GOVERNANCE_TRUE_FIELDS = (
    "model_fitting_performed",
    "primary_router_fitting_performed",
    "verifier_fitting_performed",
    "model_inference_performed",
    "model_selection_performed",
    "fresh_evaluation_performed",
)


@dataclass(frozen=True)
class Partition:
    partition_id: str
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]


@dataclass(frozen=True)
class EvaluationSplits:
    group_folds: tuple[Partition, ...]
    verifier_populations: dict[str, tuple[int, ...]]
    verifier_fold_plans: dict[str, dict[str, tuple[int, ...]]]
    audit: dict[str, Any]


@dataclass(frozen=True)
class LoadedInputs:
    contract: dict[str, Any]
    prior_contract: dict[str, Any]
    step29g_contract: dict[str, Any]
    training_examples: tuple[dict[str, Any], ...]
    development_examples: tuple[dict[str, Any], ...]
    fresh_examples: tuple[dict[str, Any], ...]
    source_hashes: dict[str, str]


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def stable_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def prohibited_final_holdout_path() -> Path:
    return (REPOSITORY_ROOT / PROHIBITED_FINAL_HOLDOUT_RELATIVE_PATH).resolve()


def assert_path_allowed(path: Path) -> None:
    resolved = path.resolve()
    if resolved == prohibited_final_holdout_path() or any(
        "final_holdout" in part for part in resolved.parts
    ):
        raise PermissionError(
            "R3 protected-intent gate selection must never access a final-holdout artifact"
        )


def repository_path(relative_path: str) -> Path:
    path = (REPOSITORY_ROOT / relative_path).resolve()
    if not path.is_relative_to(REPOSITORY_ROOT.resolve()):
        raise ValueError(f"path escapes repository: {relative_path}")
    assert_path_allowed(path)
    return path


def relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(path)


def read_bytes(path: Path) -> bytes:
    assert_path_allowed(path)
    return path.read_bytes()


def sha256_file(path: Path) -> str:
    return sha256_bytes(read_bytes(path))


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(read_bytes(path).decode("utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def write_bytes_create_once(path: Path, content: bytes) -> None:
    """Atomically create ``path``; never overwrite an existing artifact."""
    assert_path_allowed(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_frozen_r2_helpers(
    path: Path | None = None,
    *,
    expected_sha256: str = EXPECTED_R2_RUNNER_SHA256,
    module_name: str = R2_HELPER_MODULE_NAME,
) -> types.ModuleType:
    """Import the frozen R2 runner only after its SHA-256 has been verified."""
    source_path = path or repository_path(R2_RUNNER_RELATIVE_PATH)
    actual = sha256_file(source_path)
    if actual != expected_sha256:
        raise ValueError(f"reused frozen R2 runner changed: {actual}")
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot build an import spec for the frozen R2 runner: {source_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        if sha256_file(source_path) != expected_sha256:
            raise ValueError("reused frozen R2 runner changed while loading")
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    if (
        tuple(module.PROTECTED_INTENTS) != PROTECTED_INTENTS
        or tuple(module.PRIMARY_INTENTS) != PRIMARY_INTENTS
        or module.UNSUPPORTED_INTENT != UNSUPPORTED_INTENT
        or PRIMARY_ROUTER_ID not in module.EXPECTED_CANDIDATE_IDS
    ):
        sys.modules.pop(module_name, None)
        raise ValueError("reused R2 metric and intent conventions changed")
    return module


R2 = load_frozen_r2_helpers()
validate_step29g_conventions = R2.validate_step29g_conventions
build_representation = R2.build_representation
build_classifier = R2.build_classifier
fit_classifier = R2.fit_classifier
load_fastembed_model = R2.load_fastembed_model
embed_texts = R2.embed_texts
hybrid_features = R2._hybrid_features
validate_embedding_matrix = R2._validate_embedding_matrix
validate_predictions = R2._validate_predictions
safety_metrics = R2.safety_metrics
classification_metrics = R2.classification_metrics
subset_macro_f1 = R2.subset_macro_f1
unsupported_boundary_metrics = R2.unsupported_boundary_metrics
require_finite_metrics = R2._require_finite_metrics
all_string_values = R2.all_string_values


def _require_equal(observed: Any, expected: Any, message: str) -> None:
    if observed != expected:
        raise ValueError(message)


def validate_contract(contract: Mapping[str, Any]) -> None:
    if (
        contract.get("schema_version") != CONTRACT_SCHEMA_VERSION
        or contract.get("contract_version") != CONTRACT_SCHEMA_VERSION
        or contract.get("status") != "FROZEN"
        or contract.get("phase") != "V2-C6 protected-intent safety-gate remediation design"
    ):
        raise ValueError("unexpected R3 contract identity")
    _require_equal(
        contract.get("contract_status"),
        EXPECTED_CONTRACT_STATUS,
        "R3 contract status changed",
    )
    _require_equal(
        contract.get("architecture_decision", {}).get("primary_router"),
        EXPECTED_PRIMARY_ROUTER,
        "frozen primary-router configuration changed",
    )
    _require_equal(
        contract.get("architecture_decision"),
        EXPECTED_ARCHITECTURE_DECISION,
        "frozen protected-intent gate architecture changed",
    )
    _require_equal(
        contract.get("candidate_search_space"),
        EXPECTED_CANDIDATE_SEARCH_SPACE,
        "frozen candidate definitions, ordering, or bounded search changed",
    )
    _require_equal(
        contract.get("verifier_architecture"),
        EXPECTED_VERIFIER_ARCHITECTURE,
        "frozen verifier architecture changed",
    )
    _require_equal(
        contract.get("evaluation_protocol"),
        EXPECTED_EVALUATION_PROTOCOL,
        "frozen evaluation protocol changed",
    )
    _require_equal(
        contract.get("safety_gates"),
        EXPECTED_SAFETY_GATES,
        "frozen safety gates changed",
    )
    _require_equal(
        contract.get("selection_rule"),
        EXPECTED_SELECTION_RULE,
        "frozen lexicographic selection rule changed",
    )
    _require_equal(contract.get("stop_rule"), EXPECTED_STOP_RULE, "frozen stop rule changed")
    _require_equal(
        contract.get("gate_specific_diagnostics"),
        EXPECTED_GATE_DIAGNOSTICS,
        "frozen gate-specific diagnostics changed",
    )
    _require_equal(
        contract.get("expanded_development_dataset"),
        EXPECTED_EXPANDED_DEVELOPMENT,
        "frozen expanded development definition changed",
    )
    _require_equal(
        contract.get("final_holdout_policy"),
        EXPECTED_FINAL_HOLDOUT_POLICY,
        "frozen final-holdout policy changed",
    )

    fresh = contract.get("fresh_evaluation_specification", {})
    independence = fresh.get("authoring_independence", {})
    if (
        fresh.get("record_count") != FRESH_RECORD_COUNT
        or fresh.get("family_count") != 2
        or fresh.get("per_family_record_count") != FRESH_FAMILY_RECORD_COUNT
        or fresh.get("per_intent_per_family_count") != 40
        or tuple(fresh.get("source_family_ids", ())) != FRESH_FAMILY_IDS
        or independence.get("training_use_permitted") is not False
        or independence.get("candidate_modification_from_evaluation_permitted") is not False
        or independence.get("threshold_or_calibration_selection_permitted") is not False
        or independence.get("representation_or_vocabulary_fitting_permitted") is not False
    ):
        raise ValueError("frozen fresh-evaluation population changed")
    specifications = fresh.get("family_specifications", [])
    if len(specifications) != len(FRESH_FAMILY_IDS):
        raise ValueError("fresh family specification changed")
    for specification, family in zip(specifications, FRESH_FAMILY_IDS, strict=True):
        if (
            specification.get("source_family_id") != family
            or specification.get("record_count") != FRESH_FAMILY_RECORD_COUNT
            or specification.get("protected_record_count") != 160
            or specification.get("non_protected_record_count") != 160
            or specification.get("intent_counts") != dict.fromkeys(PRIMARY_INTENTS, 40)
            or specification.get("unsupported_boundary_target_counts")
            != dict.fromkeys(PROTECTED_INTENTS, 10)
        ):
            raise ValueError("fresh family specification changed")

    addendum = contract.get("targeted_training_addendum", {})
    if (
        addendum.get("record_count") != TRAINING_RECORD_COUNT
        or addendum.get("family_count") != 3
        or tuple(addendum.get("source_family_ids", ())) != TRAINING_FAMILY_IDS
    ):
        raise ValueError("frozen R3 training addendum changed")
    for specification, family in zip(
        addendum.get("family_specifications", []), TRAINING_FAMILY_IDS, strict=True
    ):
        if (
            specification.get("source_family_id") != family
            or specification.get("record_count") != 160
            or specification.get("positive_intent_counts")
            != dict.fromkeys(PROTECTED_INTENTS, 20)
            or specification.get("unsupported_target_counts")
            != dict.fromkeys(PROTECTED_INTENTS, 20)
        ):
            raise ValueError("frozen R3 training family specification changed")

    sources = contract.get("source_artifacts", {})
    _require_equal(
        sources.get("prior_targeted_remediation_design_contract"),
        {
            "path": PRIOR_CONTRACT_RELATIVE_PATH,
            "schema_version": PRIOR_CONTRACT_SCHEMA_VERSION,
            "sha256": EXPECTED_PRIOR_CONTRACT_SHA256,
        },
        "prior convention-source contract lineage changed",
    )
    if sources.get("existing_development_dataset", {}).get("sha256") != (
        EXPECTED_EXISTING_DEVELOPMENT_SHA256
    ):
        raise ValueError("existing development lineage changed")


def validate_prior_conventions(prior_contract: Mapping[str, Any]) -> None:
    if prior_contract.get("schema_version") != PRIOR_CONTRACT_SCHEMA_VERSION:
        raise ValueError("unexpected prior convention-source contract")
    semantics = prior_contract.get("candidate_implementation_semantics", {})
    _require_equal(
        semantics.get("common"),
        EXPECTED_PRIOR_COMMON_SEMANTICS,
        "frozen common candidate semantics changed",
    )
    _require_equal(
        semantics.get(PRIMARY_ROUTER_ID),
        EXPECTED_PRIOR_HYBRID_SEMANTICS,
        "frozen Hybrid primary-router semantics changed",
    )
    _require_equal(
        prior_contract.get("source_artifacts", {}).get("step29g_contract"),
        {
            "path": STEP29G_CONTRACT_RELATIVE_PATH,
            "schema_version": STEP29G_CONTRACT_SCHEMA_VERSION,
            "sha256": EXPECTED_STEP29G_CONTRACT_SHA256,
        },
        "Step 29G convention-source lineage changed",
    )


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


def _validate_dataset_governance(governance: Mapping[str, Any], label: str) -> None:
    for field in DATASET_GOVERNANCE_FALSE_FIELDS:
        if governance.get(field) is not False:
            raise ValueError(f"{label} governance changed: {field}")


def validate_dataset_manifest(
    manifest: Mapping[str, Any],
    *,
    schema_version: str,
    artifact_relative_path: str,
    artifact_schema_version: str,
    artifact_sha256: str,
    expected_counts: Mapping[str, Any],
) -> None:
    if manifest.get("schema_version") != schema_version:
        raise ValueError("unexpected R3 dataset manifest schema")
    _require_equal(
        manifest.get("artifact"),
        {
            "path": artifact_relative_path,
            "schema_version": artifact_schema_version,
            "sha256": artifact_sha256,
        },
        "R3 dataset manifest identity changed",
    )
    _require_equal(
        manifest.get("contract"),
        {
            "path": CONTRACT_RELATIVE_PATH,
            "schema_version": CONTRACT_SCHEMA_VERSION,
            "sha256": EXPECTED_CONTRACT_SHA256,
        },
        "R3 dataset manifest contract lineage changed",
    )
    counts = manifest.get("counts", {})
    for field, expected in expected_counts.items():
        if counts.get(field) != expected:
            raise ValueError(f"R3 dataset manifest count changed: {field}")
    _validate_dataset_governance(manifest.get("governance", {}), "R3 dataset manifest")
    validation = manifest.get("validation", {})
    if (
        validation.get("duplicate_and_leakage", {}).get(
            "all_duplicate_and_leakage_checks_passed"
        )
        is not True
    ):
        raise ValueError("R3 dataset manifest leakage audit did not pass")
    for role in ("training", "evaluation"):
        review = validation.get("review", {}).get(role, {})
        if review.get("unresolved_human_adjudication_count") != 0:
            raise ValueError(f"R3 dataset manifest has unresolved adjudication: {role}")


def validate_training_dataset(dataset: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    if dataset.get("schema_version") != TRAINING_SCHEMA_VERSION:
        raise ValueError("unexpected R3 training addendum schema")
    examples = dataset.get("examples")
    if not isinstance(examples, list):
        raise TypeError("R3 training examples must be a list")
    if (
        dataset.get("example_count") != TRAINING_RECORD_COUNT
        or len(examples) != TRAINING_RECORD_COUNT
        or dataset.get("classifier_input_fields") != ["text"]
        or dataset.get("metadata_fields_are_classifier_features") is not False
        or dataset.get("dataset_role") != "protected_intent_gate_training"
        or dataset.get("governance", {}).get("all_records_approved") is not True
    ):
        raise ValueError("frozen R3 training addendum population changed")
    _validate_dataset_governance(dataset.get("governance", {}), "R3 training addendum")
    identifiers: set[str] = set()
    groups: set[str] = set()
    composition: Counter[tuple[str, str, str]] = Counter()
    allowed = {*PROTECTED_INTENTS, UNSUPPORTED_INTENT}
    for row in examples:
        if not isinstance(row, dict):
            raise TypeError("R3 training record must be an object")
        identifier, group_id = _validate_text_record(
            row, identifier_field="record_id", allowed_intents=allowed
        )
        if identifier in identifiers or group_id in groups:
            raise ValueError(f"duplicate R3 training identity: {identifier}")
        identifiers.add(identifier)
        groups.add(group_id)
        family = row.get("source_family_id")
        target = row.get("protected_boundary_target")
        intent = row["intent"]
        if family not in TRAINING_FAMILY_IDS or target not in PROTECTED_INTENTS:
            raise ValueError(f"R3 training family or boundary changed: {identifier}")
        if intent == UNSUPPORTED_INTENT:
            expected_role, expected_polarity = "targeted_unsupported_negative", "negative"
        elif intent == target:
            expected_role, expected_polarity = "protected_positive", "positive"
        else:
            raise ValueError(f"R3 positive intent differs from its boundary: {identifier}")
        if (
            row.get("verifier_role") != expected_role
            or row.get("polarity") != expected_polarity
            or row.get("review_status") != "approved"
            or row.get("dataset_role") != "protected_intent_gate_training"
        ):
            raise ValueError(f"R3 training verifier metadata changed: {identifier}")
        composition[(str(family), str(target), expected_role)] += 1
    expected = Counter(
        {
            (family, target, role): 20
            for family in TRAINING_FAMILY_IDS
            for target in PROTECTED_INTENTS
            for role in ("protected_positive", "targeted_unsupported_negative")
        }
    )
    if composition != expected:
        raise ValueError("R3 training family or verifier composition changed")
    return tuple(examples)


def validate_development_dataset(
    dataset: Mapping[str, Any],
    step29g_contract: Mapping[str, Any],
    training_examples: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    if dataset.get("schema_version") != DEVELOPMENT_SCHEMA_VERSION:
        raise ValueError("unexpected R3 expanded development schema")
    examples = dataset.get("examples")
    if not isinstance(examples, list):
        raise TypeError("R3 development examples must be a list")
    governance = dataset.get("governance", {})
    if (
        dataset.get("example_count") != DEVELOPMENT_RECORD_COUNT
        or len(examples) != DEVELOPMENT_RECORD_COUNT
        or dataset.get("classifier_input_fields") != ["text"]
        or dataset.get("metadata_fields_are_classifier_features") is not False
        or dataset.get("ordering") != "frozen_9608_order_then_new_training_record_id"
        or governance.get("fresh_evaluation_records_included") is not False
        or governance.get("existing_9608_records_mutated") is not False
    ):
        raise ValueError("frozen R3 development population or feature policy changed")
    _validate_dataset_governance(governance, "R3 development dataset")
    sources = dataset.get("source_artifacts", {})
    if (
        sources.get("existing_development", {}).get("sha256")
        != EXPECTED_EXISTING_DEVELOPMENT_SHA256
        or sources.get("r3_training_examples", {}).get("sha256") != EXPECTED_TRAINING_SHA256
    ):
        raise ValueError("R3 development source lineage changed")
    allowed = set(step29g_contract["taxonomy"]["intent_label_order"])
    identifiers: set[str] = set()
    for row in examples:
        if not isinstance(row, dict):
            raise TypeError("R3 development record must be an object")
        identifier, _ = _validate_text_record(
            row, identifier_field="example_id", allowed_intents=allowed
        )
        if identifier in identifiers:
            raise ValueError(f"duplicate development example_id: {identifier}")
        identifiers.add(identifier)
        if row.get("data_role") != "development":
            raise ValueError(f"non-development row in fit population: {identifier}")
    historical = examples[:EXISTING_DEVELOPMENT_RECORD_COUNT]
    additions = examples[EXISTING_DEVELOPMENT_RECORD_COUNT:]
    if any(row.get("v2c6_lineage") == R3_LINEAGE for row in historical):
        raise ValueError("R3 lineage appears inside the frozen historical development rows")
    expected_additions = sorted(training_examples, key=lambda row: str(row["record_id"]))
    if len(additions) != len(expected_additions):
        raise ValueError("R3 development addition count changed")
    for row, record in zip(additions, expected_additions, strict=True):
        if row.get("v2c6_lineage") != R3_LINEAGE or row.get("example_id") != record.get(
            "record_id"
        ):
            raise ValueError("R3 development additions differ from the training addendum")
        for field in (
            "group_id",
            "intent",
            "text_sha256",
            "protected_boundary_target",
            "verifier_role",
            "source_family_id",
        ):
            if row.get(field) != record.get(field):
                raise ValueError(f"R3 development addition field changed: {field}")
    historical_groups = {str(row["group_id"]) for row in historical}
    if historical_groups & {str(row["group_id"]) for row in additions}:
        raise ValueError("R3 addition group collides with historical development")
    return tuple(examples)


def validate_fresh_dataset(
    dataset: Mapping[str, Any], development_examples: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, Any], ...]:
    if dataset.get("schema_version") != FRESH_SCHEMA_VERSION:
        raise ValueError("unexpected R3 fresh-evaluation schema")
    examples = dataset.get("examples")
    if not isinstance(examples, list):
        raise TypeError("R3 fresh evaluation examples must be a list")
    if (
        dataset.get("example_count") != FRESH_RECORD_COUNT
        or len(examples) != FRESH_RECORD_COUNT
        or dataset.get("classifier_input_fields") != ["text"]
        or dataset.get("metadata_fields_are_classifier_features") is not False
        or dataset.get("dataset_role") != "fresh_source_evaluation"
    ):
        raise ValueError("frozen R3 fresh-evaluation population or feature policy changed")
    _validate_dataset_governance(dataset.get("governance", {}), "R3 fresh evaluation")
    _require_equal(
        dataset.get("fresh_evaluation_governance"),
        {
            "candidate_definitions_revised_from_this_evaluation": False,
            "candidate_outputs_inspected_during_authoring": False,
            "consumed_r2_evaluation_used_as_paraphrase_source": False,
            "excluded_from_candidate_fitting": True,
            "fresh_families_consulted_each_other": False,
            "independently_authored": True,
            "not_training_data": True,
            "other_r3_fresh_family_wording_consulted": False,
            "prediction_informed_authoring": False,
            "r3_training_used_as_paraphrase_source": False,
        },
        "R3 fresh-evaluation governance changed",
    )
    identifiers: set[str] = set()
    groups: set[str] = set()
    family_intents: dict[str, Counter[str]] = {family: Counter() for family in FRESH_FAMILY_IDS}
    family_boundaries: dict[str, Counter[str]] = {
        family: Counter() for family in FRESH_FAMILY_IDS
    }
    for row in examples:
        if not isinstance(row, dict):
            raise TypeError("R3 fresh evaluation record must be an object")
        identifier, group_id = _validate_text_record(
            row, identifier_field="record_id", allowed_intents=set(PRIMARY_INTENTS)
        )
        if identifier in identifiers or group_id in groups:
            raise ValueError(f"duplicate R3 fresh identity: {identifier}")
        identifiers.add(identifier)
        groups.add(group_id)
        family = row.get("source_family_id")
        if family not in FRESH_FAMILY_IDS:
            raise ValueError(f"unexpected R3 fresh source family: {family}")
        if (
            row.get("dataset_role") != "fresh_source_evaluation"
            or row.get("excluded_from_candidate_fitting") is not True
            or row.get("not_training_data") is not True
        ):
            raise ValueError(f"R3 fresh row is not excluded from fitting: {identifier}")
        family_intents[str(family)][str(row["intent"])] += 1
        if row["intent"] == UNSUPPORTED_INTENT:
            family_boundaries[str(family)][str(row.get("protected_boundary_target"))] += 1
    for family in FRESH_FAMILY_IDS:
        if family_intents[family] != Counter(dict.fromkeys(PRIMARY_INTENTS, 40)):
            raise ValueError("R3 fresh per-family intent counts changed")
        if family_boundaries[family] != Counter(dict.fromkeys(PROTECTED_INTENTS, 10)):
            raise ValueError("R3 fresh unsupported boundary allocation changed")
    _require_fresh_isolation(examples, development_examples)
    return tuple(examples)


def _require_fresh_isolation(
    fresh_examples: Sequence[Mapping[str, Any]],
    development_examples: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    overlaps = {
        "fresh_record_id_overlap_count": len(
            {str(row["record_id"]) for row in fresh_examples}
            & {str(row["example_id"]) for row in development_examples}
        ),
        "fresh_group_id_overlap_count": len(
            {str(row["group_id"]) for row in fresh_examples}
            & {str(row["group_id"]) for row in development_examples}
        ),
        "fresh_text_sha256_overlap_count": len(
            {str(row["text_sha256"]) for row in fresh_examples}
            & {str(row["text_sha256"]) for row in development_examples}
        ),
        "fresh_normalized_text_sha256_overlap_count": len(
            {
                str(row["normalized_text_sha256"])
                for row in fresh_examples
                if row.get("normalized_text_sha256")
            }
            & {
                str(row["normalized_text_sha256"])
                for row in development_examples
                if row.get("normalized_text_sha256")
            }
        ),
    }
    if any(overlaps.values()):
        raise ValueError("R3 fresh evaluation overlaps the fit population")
    return overlaps


def load_and_validate_inputs() -> LoadedInputs:
    hashes: dict[str, str] = {}
    hashes["contract"] = sha256_file(CONTRACT_PATH)
    if hashes["contract"] != EXPECTED_CONTRACT_SHA256:
        raise ValueError(f"R3 contract SHA mismatch: {hashes['contract']}")
    contract = read_json(CONTRACT_PATH)
    validate_contract(contract)

    hashes["prior_contract"] = sha256_file(PRIOR_CONTRACT_PATH)
    if hashes["prior_contract"] != EXPECTED_PRIOR_CONTRACT_SHA256:
        raise ValueError("prior convention-source contract SHA mismatch")
    prior_contract = read_json(PRIOR_CONTRACT_PATH)
    validate_prior_conventions(prior_contract)
    hashes["step29g_contract"] = sha256_file(STEP29G_CONTRACT_PATH)
    if hashes["step29g_contract"] != EXPECTED_STEP29G_CONTRACT_SHA256:
        raise ValueError("Step 29G convention-source contract SHA mismatch")
    step29g_contract = read_json(STEP29G_CONTRACT_PATH)
    validate_step29g_conventions(step29g_contract)

    expected = {
        "training_dataset": (TRAINING_DATASET_PATH, EXPECTED_TRAINING_SHA256),
        "training_manifest": (TRAINING_MANIFEST_PATH, EXPECTED_TRAINING_MANIFEST_SHA256),
        "development_dataset": (DEVELOPMENT_DATASET_PATH, EXPECTED_DEVELOPMENT_SHA256),
        "development_manifest": (
            DEVELOPMENT_MANIFEST_PATH,
            EXPECTED_DEVELOPMENT_MANIFEST_SHA256,
        ),
        "fresh_dataset": (FRESH_DATASET_PATH, EXPECTED_FRESH_SHA256),
        "fresh_manifest": (FRESH_MANIFEST_PATH, EXPECTED_FRESH_MANIFEST_SHA256),
    }
    for name, (path, digest) in expected.items():
        hashes[name] = sha256_file(path)
        if hashes[name] != digest:
            raise ValueError(f"frozen R3 artifact SHA mismatch: {name}")

    validate_dataset_manifest(
        read_json(TRAINING_MANIFEST_PATH),
        schema_version=TRAINING_MANIFEST_SCHEMA_VERSION,
        artifact_relative_path=TRAINING_DATASET_RELATIVE_PATH,
        artifact_schema_version=TRAINING_SCHEMA_VERSION,
        artifact_sha256=EXPECTED_TRAINING_SHA256,
        expected_counts={
            "record_count": TRAINING_RECORD_COUNT,
            "by_source_family": dict.fromkeys(TRAINING_FAMILY_IDS, 160),
        },
    )
    validate_dataset_manifest(
        read_json(DEVELOPMENT_MANIFEST_PATH),
        schema_version=DEVELOPMENT_MANIFEST_SCHEMA_VERSION,
        artifact_relative_path=DEVELOPMENT_DATASET_RELATIVE_PATH,
        artifact_schema_version=DEVELOPMENT_SCHEMA_VERSION,
        artifact_sha256=EXPECTED_DEVELOPMENT_SHA256,
        expected_counts={
            "existing_record_count": EXISTING_DEVELOPMENT_RECORD_COUNT,
            "fresh_evaluation_record_count": 0,
            "new_training_record_count": TRAINING_RECORD_COUNT,
            "total_record_count": DEVELOPMENT_RECORD_COUNT,
        },
    )
    validate_dataset_manifest(
        read_json(FRESH_MANIFEST_PATH),
        schema_version=FRESH_MANIFEST_SCHEMA_VERSION,
        artifact_relative_path=FRESH_DATASET_RELATIVE_PATH,
        artifact_schema_version=FRESH_SCHEMA_VERSION,
        artifact_sha256=EXPECTED_FRESH_SHA256,
        expected_counts={
            "record_count": FRESH_RECORD_COUNT,
            "by_source_family": dict.fromkeys(FRESH_FAMILY_IDS, FRESH_FAMILY_RECORD_COUNT),
        },
    )
    training_examples = validate_training_dataset(read_json(TRAINING_DATASET_PATH))
    development_examples = validate_development_dataset(
        read_json(DEVELOPMENT_DATASET_PATH), step29g_contract, training_examples
    )
    fresh_examples = validate_fresh_dataset(
        read_json(FRESH_DATASET_PATH), development_examples
    )
    return LoadedInputs(
        contract=contract,
        prior_contract=prior_contract,
        step29g_contract=step29g_contract,
        training_examples=training_examples,
        development_examples=development_examples,
        fresh_examples=fresh_examples,
        source_hashes=hashes,
    )


def membership_sha256(values: Iterable[str]) -> str:
    return sha256_bytes(stable_json_bytes(sorted(values)))


def build_group_folds(
    examples: Sequence[Mapping[str, Any]], contract: Mapping[str, Any]
) -> tuple[Partition, ...]:
    cv = contract["evaluation_protocol"]["grouped_development_cv"]
    for row in examples:
        if not isinstance(row.get("group_id"), str) or not row["group_id"]:
            raise ValueError(f"missing group membership: {row.get('example_id')}")
    labels = np.asarray([row["intent"] for row in examples], dtype=object)
    groups = np.asarray([row["group_id"] for row in examples], dtype=object)
    splitter = StratifiedGroupKFold(
        n_splits=cv["n_splits"],
        shuffle=cv["shuffle"],
        random_state=cv["random_state"],
    )
    validation_counts = np.zeros(len(examples), dtype=np.int8)
    partitions: list[Partition] = []
    for fold_index, (train_indices, validation_indices) in enumerate(
        splitter.split(np.zeros(len(examples), dtype=np.int8), labels, groups)
    ):
        train = tuple(int(index) for index in train_indices)
        validation = tuple(int(index) for index in validation_indices)
        if set(train) & set(validation):
            raise ValueError(f"record leakage detected in group_cv_fold_{fold_index}")
        if {str(groups[index]) for index in train} & {
            str(groups[index]) for index in validation
        }:
            raise ValueError(f"group leakage detected in group_cv_fold_{fold_index}")
        validation_counts[np.asarray(validation, dtype=int)] += 1
        partitions.append(
            Partition(
                partition_id=f"group_cv_fold_{fold_index}",
                train_indices=train,
                validation_indices=validation,
            )
        )
    if len(partitions) != cv["fold_count"] or len(partitions) != FOLD_COUNT:
        raise ValueError("exactly five group-CV folds are required")
    if not np.all(validation_counts == 1):
        raise ValueError("every development record must receive exactly one OOF assignment")
    return tuple(partitions)


def verifier_training_label(row: Mapping[str, Any], protected_intent: str) -> str:
    """Return the binary verifier label; reject anything outside the R3 addendum."""
    if row.get("v2c6_lineage") != R3_LINEAGE:
        raise ValueError(
            f"historical development record cannot enter verifier fitting: "
            f"{row.get('example_id')}"
        )
    if row.get("intent") == protected_intent and row.get("verifier_role") == (
        "protected_positive"
    ):
        return protected_intent
    if (
        row.get("intent") == UNSUPPORTED_INTENT
        and row.get("protected_boundary_target") == protected_intent
        and row.get("verifier_role") == "targeted_unsupported_negative"
    ):
        return UNSUPPORTED_INTENT
    raise ValueError(
        f"record is outside the {protected_intent} verifier population: "
        f"{row.get('example_id')}"
    )


def build_verifier_populations(
    training_examples: Sequence[Mapping[str, Any]],
    development_examples: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[int, ...]]:
    """Map each verifier to the development indices of its R3 addendum records."""
    index_by_id = {
        str(row["example_id"]): index for index, row in enumerate(development_examples)
    }
    populations: dict[str, list[int]] = {intent: [] for intent in PROTECTED_INTENTS}
    for record in training_examples:
        index = index_by_id.get(str(record["record_id"]))
        if index is None:
            raise ValueError(f"R3 verifier record missing from development: {record['record_id']}")
        row = development_examples[index]
        for field in ("group_id", "intent", "text_sha256", "protected_boundary_target"):
            if row.get(field) != record.get(field):
                raise ValueError(f"R3 verifier record differs from development: {field}")
        target = str(record["protected_boundary_target"])
        if target not in populations:
            raise ValueError(f"unexpected verifier boundary: {target}")
        verifier_training_label(row, target)
        populations[target].append(index)
    for intent, indices in populations.items():
        labels = Counter(
            verifier_training_label(development_examples[index], intent) for index in indices
        )
        if labels != Counter(
            {intent: VERIFIER_POSITIVE_COUNT, UNSUPPORTED_INTENT: VERIFIER_NEGATIVE_COUNT}
        ):
            raise ValueError(f"full {intent} verifier population must be 60 positive + 60 negative")
    return {intent: tuple(sorted(indices)) for intent, indices in populations.items()}


def build_verifier_fold_plan(
    populations: Mapping[str, Sequence[int]],
    fold: Partition,
    development_examples: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[int, ...]]:
    """Inherit each R3 verifier record's fold side from its development record."""
    train = set(fold.train_indices)
    validation = set(fold.validation_indices)
    plan: dict[str, tuple[int, ...]] = {}
    for intent in PROTECTED_INTENTS:
        indices = tuple(populations[intent])
        in_train = tuple(index for index in indices if index in train)
        in_validation = tuple(index for index in indices if index in validation)
        if set(in_train) & set(in_validation) or len(in_train) + len(in_validation) != len(
            indices
        ):
            raise ValueError(f"verifier fold membership is incomplete: {fold.partition_id}")
        labels = {
            verifier_training_label(development_examples[index], intent) for index in in_train
        }
        if labels != {intent, UNSUPPORTED_INTENT}:
            raise ValueError(
                f"missing verifier binary class in {fold.partition_id}: {intent}"
            )
        plan[intent] = in_train
    return plan


def build_evaluation_splits(inputs: LoadedInputs) -> EvaluationSplits:
    development = inputs.development_examples
    folds = build_group_folds(development, inputs.contract)
    populations = build_verifier_populations(inputs.training_examples, development)
    plans: dict[str, dict[str, tuple[int, ...]]] = {}
    fold_audit: list[dict[str, Any]] = []
    for fold in folds:
        plan = build_verifier_fold_plan(populations, fold, development)
        plans[fold.partition_id] = plan
        train_rows = [development[index] for index in fold.train_indices]
        validation_rows = [development[index] for index in fold.validation_indices]
        train_groups = {str(row["group_id"]) for row in train_rows}
        validation_groups = {str(row["group_id"]) for row in validation_rows}
        verifier_audit: dict[str, Any] = {}
        for intent in PROTECTED_INTENTS:
            labels = Counter(
                verifier_training_label(development[index], intent) for index in plan[intent]
            )
            verifier_audit[intent] = {
                "train_positive_count": labels[intent],
                "train_negative_count": labels[UNSUPPORTED_INTENT],
                "validation_excluded_count": len(populations[intent]) - len(plan[intent]),
                "validation_rows_in_verifier_fitting": len(
                    set(plan[intent]) & set(fold.validation_indices)
                ),
                "train_example_id_sha256": membership_sha256(
                    str(development[index]["example_id"]) for index in plan[intent]
                ),
            }
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
                "verifier_training": verifier_audit,
            }
        )
    if any(
        row["record_overlap_count"]
        or row["group_overlap_count"]
        or any(
            entry["validation_rows_in_verifier_fitting"]
            for entry in row["verifier_training"].values()
        )
        for row in fold_audit
    ):
        raise ValueError("group-CV split audit detected leakage")
    isolation = _require_fresh_isolation(inputs.fresh_examples, development)
    audit: dict[str, Any] = {
        "fold_count": len(folds),
        "pooled_validation_count": sum(row["validation_count"] for row in fold_audit),
        "every_development_record_validated_exactly_once": True,
        "same_folds_for_both_candidates": True,
        "record_leakage_count": 0,
        "group_leakage_count": 0,
        "verifier_validation_leakage_count": 0,
        "historical_development_records_in_verifier_fitting": 0,
        "verifier_population_counts": {
            intent: {
                "positive_count": VERIFIER_POSITIVE_COUNT,
                "negative_count": VERIFIER_NEGATIVE_COUNT,
                "example_id_sha256": membership_sha256(
                    str(development[index]["example_id"]) for index in populations[intent]
                ),
            }
            for intent in PROTECTED_INTENTS
        },
        "fresh_record_count": len(inputs.fresh_examples),
        "fresh_records_in_group_cv": 0,
        **isolation,
        "folds": fold_audit,
        "raw_text_persisted": False,
    }
    audit["audit_sha256"] = sha256_bytes(stable_json_bytes(audit))
    return EvaluationSplits(
        group_folds=folds,
        verifier_populations=populations,
        verifier_fold_plans=plans,
        audit=audit,
    )


def fit_predict_primary_router(
    train_rows: Sequence[Mapping[str, Any]],
    evaluation_rows: Sequence[Mapping[str, Any]],
    step29g_contract: Mapping[str, Any],
    *,
    train_bge: np.ndarray,
    evaluation_bge: np.ndarray,
    c_value: float,
) -> list[str]:
    """Fit the frozen R2 Hybrid convention on ``train_rows`` only."""
    if train_bge.shape[0] != len(train_rows) or evaluation_bge.shape[0] != len(
        evaluation_rows
    ):
        raise ValueError("Hybrid BGE rows do not align with the text population")
    representation = build_representation("WORD_CHAR_TFIDF", step29g_contract)
    train_tfidf = representation.fit_transform([str(row["text"]) for row in train_rows])
    evaluation_tfidf = representation.transform([str(row["text"]) for row in evaluation_rows])
    classifier = build_classifier(step29g_contract, c_value=c_value)
    fit_classifier(
        classifier,
        hybrid_features(train_tfidf, train_bge),
        np.asarray([str(row["intent"]) for row in train_rows], dtype=object),
    )
    predicted = classifier.predict(hybrid_features(evaluation_tfidf, evaluation_bge))
    allowed = set(step29g_contract["taxonomy"]["intent_label_order"])
    return [
        str(value)
        for value in validate_predictions(
            np.asarray(predicted, dtype=object), len(evaluation_rows), allowed
        ).tolist()
    ]


@dataclass(frozen=True)
class FittedVerifier:
    protected_intent: str
    representation: Any
    classifier: Any
    training_record_ids: tuple[str, ...]
    positive_count: int
    negative_count: int

    def predict(self, texts: Sequence[str]) -> list[str]:
        predicted = np.asarray(
            self.classifier.predict(self.representation.transform(list(texts))), dtype=object
        )
        if predicted.shape != (len(texts),) or not set(predicted.tolist()).issubset(
            {self.protected_intent, UNSUPPORTED_INTENT}
        ):
            raise ValueError(f"{self.protected_intent} verifier predicted an invalid class")
        return [str(value) for value in predicted.tolist()]


def fit_protected_verifier(
    protected_intent: str,
    rows: Sequence[Mapping[str, Any]],
    step29g_contract: Mapping[str, Any],
    *,
    c_value: float,
) -> FittedVerifier:
    if protected_intent not in PROTECTED_INTENTS:
        raise ValueError(f"no verifier exists for non-protected intent: {protected_intent}")
    labels = [verifier_training_label(row, protected_intent) for row in rows]
    counts = Counter(labels)
    if counts[protected_intent] == 0 or counts[UNSUPPORTED_INTENT] == 0:
        raise ValueError(f"missing verifier binary class: {protected_intent}")
    representation = build_representation("WORD_CHAR_TFIDF", step29g_contract)
    features = representation.fit_transform([str(row["text"]) for row in rows])
    classifier = build_classifier(step29g_contract, c_value=c_value)
    fit_classifier(classifier, features, np.asarray(labels, dtype=object))
    return FittedVerifier(
        protected_intent=protected_intent,
        representation=representation,
        classifier=classifier,
        training_record_ids=tuple(str(row["example_id"]) for row in rows),
        positive_count=counts[protected_intent],
        negative_count=counts[UNSUPPORTED_INTENT],
    )


def apply_protected_verifier_gate(
    primary_predictions: Sequence[str],
    texts: Sequence[str],
    verifiers: Mapping[str, Any],
) -> tuple[list[str], list[dict[str, Any]]]:
    """Route-only gate: verify protected primary predictions with their own verifier."""
    if len(primary_predictions) != len(texts):
        raise ValueError("primary predictions and texts differ in length")
    if set(verifiers) != set(PROTECTED_INTENTS):
        raise ValueError("exactly the four frozen protected verifiers are required")
    final = [str(value) for value in primary_predictions]
    decisions = [dict(NO_VERIFIER_DECISION) for _ in primary_predictions]
    for intent in PROTECTED_INTENTS:
        indices = [index for index, value in enumerate(final) if value == intent]
        if not indices:
            continue
        verdicts = verifiers[intent].predict([texts[index] for index in indices])
        if len(verdicts) != len(indices):
            raise ValueError(f"{intent} verifier prediction coverage changed")
        for index, verdict in zip(indices, verdicts, strict=True):
            if verdict == intent:
                decision = VERIFIER_ACCEPT
            elif verdict == UNSUPPORTED_INTENT:
                decision = VERIFIER_REJECT
                final[index] = UNSUPPORTED_INTENT
            else:
                raise ValueError(f"{intent} verifier returned an invalid decision")
            decisions[index] = {
                "verifier_invoked": True,
                "verifier_intent": intent,
                "verifier_decision": decision,
            }
    return final, decisions


def validate_routing_rows(rows: Sequence[Mapping[str, Any]], *, gated: bool) -> None:
    for row in rows:
        primary = row["primary_predicted_intent"]
        final = row["predicted_intent"]
        invoked = row["verifier_invoked"]
        if not gated or primary not in PROTECTED_INTENTS:
            if invoked is not False or row["verifier_intent"] is not None or (
                row["verifier_decision"] is not None
            ):
                raise ValueError("verifier invoked outside a gated protected prediction")
            if final != primary:
                raise ValueError("final prediction changed without a verifier decision")
            continue
        if invoked is not True or row["verifier_intent"] != primary:
            raise ValueError("protected prediction was not sent to its dedicated verifier")
        if row["verifier_decision"] == VERIFIER_ACCEPT:
            if final != primary:
                raise ValueError("verifier acceptance must preserve the protected prediction")
        elif row["verifier_decision"] == VERIFIER_REJECT:
            if final != UNSUPPORTED_INTENT:
                raise ValueError("verifier rejection must return unsupported_or_uncertain")
        else:
            raise ValueError("invalid verifier decision")


def verifier_diagnostics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Gate diagnostics; they never replace the mandatory safety gates."""
    validate_routing_rows(rows, gated=True)
    gold = [str(row["gold_intent"]) for row in rows]
    primary = [str(row["primary_predicted_intent"]) for row in rows]
    final = [str(row["predicted_intent"]) for row in rows]
    invoked = [row for row in rows if row["verifier_invoked"]]
    rejected = [row for row in invoked if row["verifier_decision"] == VERIFIER_REJECT]
    per_intent: dict[str, dict[str, Any]] = {}
    for intent in PROTECTED_INTENTS:
        presented = [row for row in invoked if row["verifier_intent"] == intent]
        genuine = [row for row in presented if row["gold_intent"] == intent]
        genuine_accepted = sum(
            row["verifier_decision"] == VERIFIER_ACCEPT for row in genuine
        )
        per_intent[intent] = {
            "presented_count": len(presented),
            "accepted_count": sum(
                row["verifier_decision"] == VERIFIER_ACCEPT for row in presented
            ),
            "rejected_count": sum(
                row["verifier_decision"] == VERIFIER_REJECT for row in presented
            ),
            "presented_gold_same_intent_count": len(genuine),
            "accepted_gold_same_intent_count": genuine_accepted,
            "presented_gold_unsupported_count": sum(
                row["gold_intent"] == UNSUPPORTED_INTENT for row in presented
            ),
            "presented_gold_other_count": sum(
                row["gold_intent"] not in {intent, UNSUPPORTED_INTENT} for row in presented
            ),
            "verifier_recall": (
                genuine_accepted / len(genuine) if genuine else None
            ),
        }
    primary_safety = safety_metrics(gold, primary)
    final_safety = safety_metrics(gold, final)
    return {
        "definitions": {
            "primary_protected_predictions_presented_to_verifier": (
                "rows whose primary prediction is protected; each is sent only to the "
                "verifier dedicated to that predicted intent"
            ),
            "per_protected_intent_verifier_recall": (
                "for verifier P: accepted / presented rows with gold == P and primary == P; "
                "null when that denominator is zero; this is not the final routing "
                "protected_recall metric"
            ),
            "protected_false_positives_prevented_by_verifier": (
                "rows with non-protected gold, protected primary prediction, and verifier "
                "rejection"
            ),
            "true_protected_requests_rejected_by_verifier": (
                "rows with protected gold, protected primary prediction, and verifier "
                "rejection"
            ),
            "final_metrics": "mandatory safety metrics computed on post-gate predictions",
        },
        "primary_protected_predictions_presented_to_verifier": len(invoked),
        "verifier_accept_count": len(invoked) - len(rejected),
        "verifier_reject_to_unsupported_count": len(rejected),
        "per_protected_intent_verifier_recall": per_intent,
        "protected_false_positives_prevented_by_verifier": sum(
            row["gold_intent"] not in PROTECTED_INTENTS for row in rejected
        ),
        "true_protected_requests_rejected_by_verifier": sum(
            row["gold_intent"] in PROTECTED_INTENTS for row in rejected
        ),
        "unsupported_requests_recovered_by_verifier": sum(
            row["gold_intent"] == UNSUPPORTED_INTENT for row in rejected
        ),
        "primary_protected_false_positive_count": primary_safety[
            "protected_false_positive_count"
        ],
        "primary_protected_false_positive_rate": primary_safety[
            "protected_false_positive_rate"
        ],
        "primary_protected_recall": primary_safety["protected_recall"],
        "primary_unsupported_recall": primary_safety["unsupported_recall"],
        "final_protected_false_positive_count": final_safety[
            "protected_false_positive_count"
        ],
        "final_protected_false_positive_rate": final_safety["protected_false_positive_rate"],
        "final_protected_recall": final_safety["protected_recall"],
        "final_unsupported_recall": final_safety["unsupported_recall"],
    }


def validate_prediction_coverage(
    identifiers: Sequence[str], expected: Sequence[str], label: str
) -> None:
    counts = Counter(identifiers)
    duplicates = sorted(identifier for identifier, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate {label} prediction: {duplicates[0]}")
    if set(identifiers) != set(expected) or len(identifiers) != len(expected):
        raise ValueError(f"incomplete {label} prediction coverage")


def _routing_columns(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[str], list[str], list[str]]:
    return (
        [str(row["gold_intent"]) for row in rows],
        [str(row["primary_predicted_intent"]) for row in rows],
        [str(row["predicted_intent"]) for row in rows],
    )


def summarize_group_cv(
    rows: Sequence[Mapping[str, Any]],
    inputs: LoadedInputs,
    splits: EvaluationSplits,
    *,
    gated: bool,
) -> dict[str, Any]:
    development = inputs.development_examples
    validate_prediction_coverage(
        [str(row["example_id"]) for row in rows],
        [str(row["example_id"]) for row in development],
        "group-CV OOF",
    )
    fold_by_index: dict[int, str] = {}
    for fold in splits.group_folds:
        for index in fold.validation_indices:
            fold_by_index[index] = fold.partition_id
    for index, (row, record) in enumerate(zip(rows, development, strict=True)):
        if (
            row["example_id"] != record["example_id"]
            or row["gold_intent"] != record["intent"]
            or row["fold_id"] != fold_by_index[index]
        ):
            raise ValueError("group-CV prediction row differs from the frozen split")
    validate_routing_rows(rows, gated=gated)
    gold, primary, final = _routing_columns(rows)
    labels = inputs.step29g_contract["taxonomy"]["intent_label_order"]
    metrics = classification_metrics(gold, final, labels)
    metrics["macro_f1_16"] = metrics.pop("macro_f1")
    metrics["primary_8_macro_f1"] = subset_macro_f1(gold, final, PRIMARY_INTENTS)
    metrics["safety"] = safety_metrics(gold, final)
    metrics["unsupported_boundary"] = unsupported_boundary_metrics(gold, final)
    metrics["primary_router_safety"] = safety_metrics(gold, primary)
    metrics["verifier_diagnostics"] = verifier_diagnostics(rows) if gated else None
    require_finite_metrics(metrics)
    return metrics


def _fresh_scope_metrics(rows: Sequence[Mapping[str, Any]], *, gated: bool) -> dict[str, Any]:
    validate_routing_rows(rows, gated=gated)
    gold, primary, final = _routing_columns(rows)
    metrics = classification_metrics(gold, final, PRIMARY_INTENTS)
    metrics["primary_8_macro_f1"] = metrics.pop("macro_f1")
    metrics["safety"] = safety_metrics(gold, final)
    metrics["unsupported_boundary"] = unsupported_boundary_metrics(gold, final)
    metrics["primary_router_safety"] = safety_metrics(gold, primary)
    metrics["verifier_diagnostics"] = verifier_diagnostics(rows) if gated else None
    require_finite_metrics(metrics)
    return metrics


def summarize_fresh(
    rows: Sequence[Mapping[str, Any]], inputs: LoadedInputs, *, gated: bool
) -> dict[str, Any]:
    fresh = inputs.fresh_examples
    validate_prediction_coverage(
        [str(row["record_id"]) for row in rows],
        [str(row["record_id"]) for row in fresh],
        "pooled fresh",
    )
    for row, record in zip(rows, fresh, strict=True):
        if (
            row["record_id"] != record["record_id"]
            or row["gold_intent"] != record["intent"]
            or row["source_family_id"] != record["source_family_id"]
        ):
            raise ValueError("fresh prediction row differs from the frozen dataset")
    families: list[dict[str, Any]] = []
    for family in FRESH_FAMILY_IDS:
        family_rows = [row for row in rows if row["source_family_id"] == family]
        expected = [
            str(record["record_id"])
            for record in fresh
            if record["source_family_id"] == family
        ]
        validate_prediction_coverage(
            [str(row["record_id"]) for row in family_rows], expected, family
        )
        families.append(
            {
                "source_family_id": family,
                "record_count": len(expected),
                "prediction_count": len(family_rows),
                "prediction_sha256": sha256_bytes(stable_json_bytes(family_rows)),
                "metrics": _fresh_scope_metrics(family_rows, gated=gated),
            }
        )
    pooled = _fresh_scope_metrics(rows, gated=gated)
    pooled["pooled_fresh_primary_8_macro_f1"] = pooled.pop("primary_8_macro_f1")
    scores = [row["metrics"]["primary_8_macro_f1"] for row in families]
    pooled["worst_fresh_family_primary_8_macro_f1"] = float(min(scores))
    pooled["mean_fresh_family_primary_8_macro_f1"] = float(np.mean(scores))
    return {
        "families": families,
        "pooled_prediction_count": len(rows),
        "pooled_prediction_sha256": sha256_bytes(stable_json_bytes(list(rows))),
        "pooled_metrics": pooled,
    }


def _prediction_row(
    identity: Mapping[str, Any],
    primary: str,
    final: str,
    decision: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        **identity,
        "primary_predicted_intent": primary,
        "predicted_intent": final,
        "verifier_invoked": decision["verifier_invoked"],
        "verifier_intent": decision["verifier_intent"],
        "verifier_decision": decision["verifier_decision"],
    }


def _fit_verifiers(
    indices_by_intent: Mapping[str, Sequence[int]],
    inputs: LoadedInputs,
) -> dict[str, FittedVerifier]:
    c_value = float(inputs.contract["verifier_architecture"]["C"])
    return {
        intent: fit_protected_verifier(
            intent,
            [inputs.development_examples[index] for index in indices_by_intent[intent]],
            inputs.step29g_contract,
            c_value=c_value,
        )
        for intent in PROTECTED_INTENTS
    }


def _route(
    gated: bool,
    primary: Sequence[str],
    texts: Sequence[str],
    verifiers: Mapping[str, Any] | None,
) -> tuple[list[str], list[dict[str, Any]]]:
    if not gated:
        return list(primary), [dict(NO_VERIFIER_DECISION) for _ in primary]
    if verifiers is None:
        raise ValueError("gated candidate requires fitted verifiers")
    return apply_protected_verifier_gate(primary, texts, verifiers)


def run_group_cv(
    candidate_id: str,
    inputs: LoadedInputs,
    splits: EvaluationSplits,
    development_bge: np.ndarray,
) -> dict[str, Any]:
    gated = candidate_id == GATED_CANDIDATE_ID
    development = inputs.development_examples
    c_value = float(inputs.contract["architecture_decision"]["primary_router"]["C"])
    routed: list[tuple[str, str, str, dict[str, Any]] | None] = [None] * len(development)
    fold_summaries: list[dict[str, Any]] = []
    for fold in splits.group_folds:
        train_rows = [development[index] for index in fold.train_indices]
        validation_rows = [development[index] for index in fold.validation_indices]
        primary = fit_predict_primary_router(
            train_rows,
            validation_rows,
            inputs.step29g_contract,
            train_bge=development_bge[np.asarray(fold.train_indices, dtype=int)],
            evaluation_bge=development_bge[np.asarray(fold.validation_indices, dtype=int)],
            c_value=c_value,
        )
        verifiers = None
        verifier_training: dict[str, Any] | None = None
        if gated:
            plan = splits.verifier_fold_plans[fold.partition_id]
            validation = set(fold.validation_indices)
            train = set(fold.train_indices)
            for intent in PROTECTED_INTENTS:
                if set(plan[intent]) & validation or not set(plan[intent]) <= train:
                    raise ValueError("fold-validation R3 record entered verifier fitting")
            verifiers = _fit_verifiers(plan, inputs)
            verifier_training = {
                intent: {
                    "positive_count": verifier.positive_count,
                    "negative_count": verifier.negative_count,
                }
                for intent, verifier in verifiers.items()
            }
        final, decisions = _route(
            gated, primary, [str(row["text"]) for row in validation_rows], verifiers
        )
        for index, values in zip(
            fold.validation_indices, zip(primary, final, decisions, strict=True), strict=True
        ):
            if routed[index] is not None:
                raise ValueError("duplicate OOF prediction")
            routed[index] = (fold.partition_id, *values)
        fold_summaries.append(
            {
                "fold_id": fold.partition_id,
                "train_count": len(train_rows),
                "validation_count": len(validation_rows),
                "prediction_count": len(primary),
                "verifier_training": verifier_training,
            }
        )
    if any(value is None for value in routed):
        raise ValueError("missing OOF prediction")
    rows = [
        _prediction_row(
            {
                "example_id": str(record["example_id"]),
                "fold_id": value[0],
                "gold_intent": str(record["intent"]),
            },
            value[1],
            value[2],
            value[3],
        )
        for record, value in zip(development, routed, strict=True)
        if value is not None
    ]
    return {
        "completed_fold_count": len(fold_summaries),
        "folds": fold_summaries,
        "prediction_count": len(rows),
        "unique_prediction_id_count": len({row["example_id"] for row in rows}),
        "prediction_sha256": sha256_bytes(stable_json_bytes(rows)),
        "predictions": rows,
        "metrics": summarize_group_cv(rows, inputs, splits, gated=gated),
    }


def run_fresh_evaluation(
    candidate_id: str,
    inputs: LoadedInputs,
    splits: EvaluationSplits,
    development_bge: np.ndarray,
    fresh_bge: np.ndarray,
) -> dict[str, Any]:
    gated = candidate_id == GATED_CANDIDATE_ID
    c_value = float(inputs.contract["architecture_decision"]["primary_router"]["C"])
    primary = fit_predict_primary_router(
        inputs.development_examples,
        inputs.fresh_examples,
        inputs.step29g_contract,
        train_bge=development_bge,
        evaluation_bge=fresh_bge,
        c_value=c_value,
    )
    verifiers = None
    if gated:
        verifiers = _fit_verifiers(splits.verifier_populations, inputs)
        for intent, verifier in verifiers.items():
            if (verifier.positive_count, verifier.negative_count) != (
                VERIFIER_POSITIVE_COUNT,
                VERIFIER_NEGATIVE_COUNT,
            ):
                raise ValueError(f"full {intent} verifier fit must use 60 + 60 records")
    final, decisions = _route(
        gated, primary, [str(row["text"]) for row in inputs.fresh_examples], verifiers
    )
    rows = [
        _prediction_row(
            {
                "record_id": str(record["record_id"]),
                "source_family_id": str(record["source_family_id"]),
                "gold_intent": str(record["intent"]),
            },
            guess,
            routed,
            decision,
        )
        for record, guess, routed, decision in zip(
            inputs.fresh_examples, primary, final, decisions, strict=True
        )
    ]
    return {
        **fresh_fit_metadata(inputs, gated=gated),
        **summarize_fresh(rows, inputs, gated=gated),
        "pooled_predictions": rows,
    }


def fresh_fit_metadata(inputs: LoadedInputs, *, gated: bool) -> dict[str, Any]:
    return {
        "primary_router_fit_count": 1,
        "primary_router_fit_record_count": len(inputs.development_examples),
        "verifier_fit_count": len(PROTECTED_INTENTS) if gated else 0,
        "verifier_fit_population_counts": (
            {
                intent: {
                    "positive_count": VERIFIER_POSITIVE_COUNT,
                    "negative_count": VERIFIER_NEGATIVE_COUNT,
                }
                for intent in PROTECTED_INTENTS
            }
            if gated
            else {}
        ),
        "fresh_records_used_for_fitting": 0,
        "same_fitted_candidate_used_for_both_families": True,
        "refit_between_source_families": False,
    }


def safety_scope_metrics(
    group_result: Mapping[str, Any], fresh_result: Mapping[str, Any]
) -> list[tuple[str, Mapping[str, Any]]]:
    return [
        (GROUP_CV_SCOPE, group_result["metrics"]["safety"]),
        *[
            (str(row["source_family_id"]), row["metrics"]["safety"])
            for row in fresh_result["families"]
        ],
        (POOLED_FRESH_SCOPE, fresh_result["pooled_metrics"]["safety"]),
    ]


def apply_mandatory_safety_gates(
    scopes: Sequence[tuple[str, Mapping[str, Any]]], contract: Mapping[str, Any]
) -> dict[str, Any]:
    gates = contract["safety_gates"]
    if [scope for scope, _ in scopes] != gates["application_scopes"]:
        raise ValueError("required safety scope coverage changed")
    results: list[dict[str, Any]] = []
    for scope, metrics in scopes:
        for gate in gates["gates"]:
            observed = float(metrics[gate["metric"]])
            if gate["comparison"] == "greater_than_or_equal":
                passed, comparator = observed >= gate["threshold"], ">="
            elif gate["comparison"] == "less_than_or_equal":
                passed, comparator = observed <= gate["threshold"], "<="
            else:
                raise ValueError(f"unknown safety comparison: {gate['comparison']}")
            results.append(
                {
                    "scope": scope,
                    "metric": gate["metric"],
                    "observed_value": observed,
                    "comparator": comparator,
                    "threshold": gate["threshold"],
                    "passed": bool(passed),
                }
            )
    if len(results) != len(SAFETY_SCOPES) * 3:
        raise ValueError("exactly twelve mandatory safety-gate checks are required")
    return {
        "required_scope_count": len(SAFETY_SCOPES),
        "gate_count_per_scope": 3,
        "all_gates_pass": all(row["passed"] for row in results),
        "gates": results,
    }


def determine_candidate_eligibility(
    safety_gate_results: Mapping[str, Any],
    *,
    completed_group_folds: int,
    group_prediction_count: int,
    expected_group_prediction_count: int,
    fresh_prediction_count: int,
    expected_fresh_prediction_count: int,
    fresh_family_prediction_counts: Sequence[int],
    expected_fresh_family_prediction_counts: Sequence[int],
) -> tuple[bool, list[str]]:
    infrastructure = {
        "all_5_group_cv_folds_completed": completed_group_folds == FOLD_COUNT,
        "group_cv_prediction_coverage_exact": (
            group_prediction_count == expected_group_prediction_count
        ),
        "fresh_prediction_coverage_exact": (
            fresh_prediction_count == expected_fresh_prediction_count
        ),
        "fresh_family_prediction_coverage_exact": list(fresh_family_prediction_counts)
        == list(expected_fresh_family_prediction_counts),
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
    if candidate_id not in EXPECTED_CANDIDATE_IDS:
        raise ValueError(f"candidate has no frozen complexity rank: {candidate_id}")
    return EXPECTED_CANDIDATE_IDS.index(candidate_id) + 1


def selection_metrics(
    candidate_id: str, group_result: Mapping[str, Any], fresh_result: Mapping[str, Any]
) -> dict[str, float | int | str]:
    pooled = fresh_result["pooled_metrics"]
    return {
        "worst_fresh_family_primary_8_macro_f1": pooled[
            "worst_fresh_family_primary_8_macro_f1"
        ],
        "pooled_fresh_primary_8_macro_f1": pooled["pooled_fresh_primary_8_macro_f1"],
        "pooled_group_cv_macro_f1_16": group_result["metrics"]["macro_f1_16"],
        "pooled_fresh_protected_false_positive_rate": pooled["safety"][
            "protected_false_positive_rate"
        ],
        "pooled_fresh_protected_recall": pooled["safety"]["protected_recall"],
        "pooled_fresh_unsupported_recall": pooled["safety"]["unsupported_recall"],
        "model_complexity_rank": model_complexity_rank(candidate_id),
        "candidate_id": candidate_id,
    }


def candidate_configuration(
    candidate: Mapping[str, Any], inputs: LoadedInputs
) -> dict[str, Any]:
    candidate_id = str(candidate["candidate_id"])
    gated = candidate_id == GATED_CANDIDATE_ID
    representations = inputs.step29g_contract["representations"]
    return {
        "candidate_id": candidate_id,
        "frozen_candidate_definition": dict(candidate),
        "primary_router": inputs.contract["architecture_decision"]["primary_router"],
        "primary_router_semantics": inputs.prior_contract[
            "candidate_implementation_semantics"
        ][PRIMARY_ROUTER_ID],
        "tfidf_conventions": {
            name: representations[name]
            for name in ("WORD_TFIDF", "CHAR_TFIDF", "WORD_CHAR_TFIDF")
        },
        "bge_convention": representations["BGE_SMALL"],
        "linear_svc_convention": inputs.step29g_contract["classifiers"]["LINEAR_SVC"],
        "routing_rule": (
            inputs.contract["architecture_decision"]["routing_rule"] if gated else None
        ),
        "verifier_architecture": (
            inputs.contract["verifier_architecture"] if gated else None
        ),
        "verifier_is_authorization": False,
        "model_complexity_rank": model_complexity_rank(candidate_id),
    }


def evaluate_candidate_results(
    candidate: Mapping[str, Any],
    inputs: LoadedInputs,
    group: dict[str, Any],
    fresh: dict[str, Any],
) -> dict[str, Any]:
    candidate_id = str(candidate["candidate_id"])
    gates = apply_mandatory_safety_gates(safety_scope_metrics(group, fresh), inputs.contract)
    eligible, reasons = determine_candidate_eligibility(
        gates,
        completed_group_folds=group["completed_fold_count"],
        group_prediction_count=group["prediction_count"],
        expected_group_prediction_count=len(inputs.development_examples),
        fresh_prediction_count=fresh["pooled_prediction_count"],
        expected_fresh_prediction_count=len(inputs.fresh_examples),
        fresh_family_prediction_counts=[row["prediction_count"] for row in fresh["families"]],
        expected_fresh_family_prediction_counts=[
            sum(row["source_family_id"] == family for row in inputs.fresh_examples)
            for family in FRESH_FAMILY_IDS
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
        "selection_metrics": selection_metrics(candidate_id, group, fresh),
        "threshold_tuning_performed": False,
        "calibration_performed": False,
        "fresh_evaluation_used_for_fitting": False,
        "raw_text_persisted": False,
    }


def evaluate_candidate(
    candidate: Mapping[str, Any],
    inputs: LoadedInputs,
    splits: EvaluationSplits,
    bge_matrices: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    candidate_id = str(candidate["candidate_id"])
    group = run_group_cv(candidate_id, inputs, splits, bge_matrices["development"])
    fresh = run_fresh_evaluation(
        candidate_id,
        inputs,
        splits,
        bge_matrices["development"],
        bge_matrices["fresh_evaluation"],
    )
    return evaluate_candidate_results(candidate, inputs, group, fresh)


def _numbers_tie(left: Any, right: Any) -> bool:
    return math.isclose(
        float(left), float(right), rel_tol=0.0, abs_tol=TIE_ABSOLUTE_TOLERANCE
    )


def candidate_is_better(
    candidate: Mapping[str, Any], incumbent: Mapping[str, Any]
) -> bool:
    for metric, direction in SELECTION_CRITERIA:
        left = candidate["selection_metrics"][metric]
        right = incumbent["selection_metrics"][metric]
        if direction == "ascending_lexical":
            if str(left) == str(right):
                continue
            return str(left) < str(right)
        if direction == "prefer_lower_rank":
            if int(left) == int(right):
                continue
            return int(left) < int(right)
        if _numbers_tie(left, right):
            continue
        if direction == "maximize":
            return float(left) > float(right)
        if direction == "minimize":
            return float(left) < float(right)
        raise ValueError(f"unknown selection direction: {direction}")
    return False


def select_candidate(
    candidate_results: Sequence[Mapping[str, Any]], contract: Mapping[str, Any]
) -> dict[str, Any]:
    stop_rule = contract["stop_rule"]
    eligible = [row for row in candidate_results if row["eligible"] is True]
    if not eligible:
        return {
            **contract["selection_rule"]["if_no_candidate_is_eligible"],
            "eligible_candidate_count": 0,
            "selected_candidate_id": None,
            "next_required": stop_rule["failure_continuation"]["next_required"],
            "step29i_authorized": False,
            "final_holdout_access_authorized": False,
        }
    selected = eligible[0]
    for candidate in eligible[1:]:
        if candidate_is_better(candidate, selected):
            selected = candidate
    return {
        "selection_status": "SELECTED",
        "eligible_candidate_count": len(eligible),
        "selected_candidate_id": selected["candidate_id"],
        "selected_candidate": selected["configuration"],
        "winner_forced": False,
        "gates_weakened": False,
        "next_required": stop_rule["success_continuation"]["next_required"],
        "step29i_authorized": False,
        "final_holdout_access_authorized": False,
    }


def primary_router_identity_audit(
    candidate_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Both candidates must have produced identical primary-router predictions."""
    audit: dict[str, Any] = {}
    for scope, getter in (
        ("group_cv", lambda row: row["group_aware_cv"]["predictions"]),
        ("fresh", lambda row: row["fresh_source_evaluation"]["pooled_predictions"]),
    ):
        digests = [
            sha256_bytes(
                stable_json_bytes([item["primary_predicted_intent"] for item in getter(row)])
            )
            for row in candidate_results
        ]
        if len(set(digests)) != 1:
            raise ValueError(f"candidates used different primary-router predictions: {scope}")
        audit[f"{scope}_primary_prediction_sha256"] = digests[0]
    audit["primary_router_predictions_identical_across_candidates"] = True
    return audit


def validate_text_free(
    payload: Mapping[str, Any], records: Sequence[Mapping[str, Any]]
) -> None:
    strings = set(all_string_values(payload))
    if strings & {str(row["text"]) for row in records}:
        raise ValueError("tracked result artifact contains raw example text")
    if strings & {"text", "texts", "utterance", "utterances", "raw_text"}:
        raise ValueError("tracked result artifact contains a raw-text field")


def frozen_inputs_summary(inputs: LoadedInputs) -> dict[str, Any]:
    hashes = inputs.source_hashes
    return {
        "training_addendum": {
            "path": TRAINING_DATASET_RELATIVE_PATH,
            "schema_version": TRAINING_SCHEMA_VERSION,
            "sha256": hashes["training_dataset"],
            "manifest_sha256": hashes["training_manifest"],
            "record_count": len(inputs.training_examples),
        },
        "development_dataset": {
            "path": DEVELOPMENT_DATASET_RELATIVE_PATH,
            "schema_version": DEVELOPMENT_SCHEMA_VERSION,
            "sha256": hashes["development_dataset"],
            "manifest_sha256": hashes["development_manifest"],
            "record_count": len(inputs.development_examples),
        },
        "fresh_evaluation_dataset": {
            "path": FRESH_DATASET_RELATIVE_PATH,
            "schema_version": FRESH_SCHEMA_VERSION,
            "sha256": hashes["fresh_dataset"],
            "manifest_sha256": hashes["fresh_manifest"],
            "record_count": len(inputs.fresh_examples),
            "excluded_from_fitting": True,
        },
        "convention_sources": {
            "prior_targeted_remediation_contract": {
                "path": PRIOR_CONTRACT_RELATIVE_PATH,
                "sha256": hashes["prior_contract"],
            },
            "step29g_contract": {
                "path": STEP29G_CONTRACT_RELATIVE_PATH,
                "sha256": hashes["step29g_contract"],
            },
            "reused_r2_runner": {
                "path": R2_RUNNER_RELATIVE_PATH,
                "sha256": EXPECTED_R2_RUNNER_SHA256,
            },
        },
    }


def result_governance(*, embeddings_generated: bool) -> dict[str, Any]:
    return {
        "embeddings_generated_during_run": embeddings_generated,
        **dict.fromkeys(RESULT_GOVERNANCE_TRUE_FIELDS, True),
        **dict.fromkeys(RESULT_GOVERNANCE_FALSE_FIELDS, False),
    }


def build_results_payload(
    inputs: LoadedInputs,
    splits: EvaluationSplits,
    candidate_results: Sequence[dict[str, Any]],
    *,
    bge_cache_manifest_sha256: str,
    embeddings_generated: bool,
) -> dict[str, Any]:
    if [row["candidate_id"] for row in candidate_results] != list(EXPECTED_CANDIDATE_IDS):
        raise ValueError("complete ordered two-candidate results are required")
    selection = select_candidate(candidate_results, inputs.contract)
    payload = {
        "schema_version": RESULTS_SCHEMA_VERSION,
        "phase": RESULTS_PHASE,
        "execution_status": "COMPLETED",
        "contract": {
            "path": CONTRACT_RELATIVE_PATH,
            "sha256": inputs.source_hashes["contract"],
            "schema_version": CONTRACT_SCHEMA_VERSION,
        },
        "frozen_inputs": frozen_inputs_summary(inputs),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "fastembed": package_version("fastembed"),
        },
        "bge_cache_manifest_sha256": bge_cache_manifest_sha256,
        "candidate_definitions": inputs.contract["candidate_search_space"]["candidates"],
        "candidate_count": len(EXPECTED_CANDIDATE_IDS),
        "completed_candidate_count": len(candidate_results),
        "fold_configuration": inputs.contract["evaluation_protocol"]["grouped_development_cv"],
        "split_audit": splits.audit,
        "primary_router_identity_audit": primary_router_identity_audit(candidate_results),
        "candidate_results": list(candidate_results),
        "selection_rule": inputs.contract["selection_rule"],
        "selection_tie_tolerance": {"absolute": TIE_ABSOLUTE_TOLERANCE, "relative": 0.0},
        "selection": selection,
        "stop_rule": inputs.contract["stop_rule"],
        "next_required": selection["next_required"],
        "governance": result_governance(embeddings_generated=embeddings_generated),
    }
    validate_results_payload(payload, inputs, splits)
    return payload


def validate_candidate_result(
    result: Mapping[str, Any],
    candidate: Mapping[str, Any],
    inputs: LoadedInputs,
    splits: EvaluationSplits,
) -> None:
    candidate_id = str(candidate["candidate_id"])
    gated = candidate_id == GATED_CANDIDATE_ID
    if result.get("candidate_id") != candidate_id:
        raise ValueError("candidate result ID or order changed")
    configuration = candidate_configuration(candidate, inputs)
    if result.get("configuration") != configuration or result.get(
        "configuration_sha256"
    ) != sha256_bytes(stable_json_bytes(configuration)):
        raise ValueError("candidate configuration changed")
    group = result.get("group_aware_cv", {})
    rows = group.get("predictions", [])
    if (
        group.get("completed_fold_count") != FOLD_COUNT
        or group.get("prediction_count") != len(inputs.development_examples)
        or group.get("unique_prediction_id_count") != len(inputs.development_examples)
        or group.get("prediction_sha256") != sha256_bytes(stable_json_bytes(rows))
    ):
        raise ValueError("candidate OOF coverage changed")
    if group.get("metrics") != summarize_group_cv(rows, inputs, splits, gated=gated):
        raise ValueError("candidate group-CV metrics do not match its predictions")
    fold_ids = [fold.partition_id for fold in splits.group_folds]
    if [row.get("fold_id") for row in group.get("folds", [])] != fold_ids:
        raise ValueError("candidate fold summaries changed")
    for summary, fold in zip(group["folds"], splits.group_folds, strict=True):
        expected_training = (
            {
                intent: {
                    "positive_count": entry["train_positive_count"],
                    "negative_count": entry["train_negative_count"],
                }
                for intent, entry in next(
                    row for row in splits.audit["folds"] if row["fold_id"] == fold.partition_id
                )["verifier_training"].items()
            }
            if gated
            else None
        )
        if (
            summary.get("train_count") != len(fold.train_indices)
            or summary.get("validation_count") != len(fold.validation_indices)
            or summary.get("prediction_count") != len(fold.validation_indices)
            or summary.get("verifier_training") != expected_training
        ):
            raise ValueError("candidate fold summary differs from the frozen split")
    fresh = result.get("fresh_source_evaluation", {})
    pooled_rows = fresh.get("pooled_predictions", [])
    expected_fresh = {
        **fresh_fit_metadata(inputs, gated=gated),
        **summarize_fresh(pooled_rows, inputs, gated=gated),
        "pooled_predictions": pooled_rows,
    }
    if fresh != expected_fresh:
        raise ValueError("candidate fresh-evaluation coverage, fits, or metrics changed")
    expected = evaluate_candidate_results(candidate, inputs, dict(group), dict(fresh))
    if dict(result) != expected:
        raise ValueError("candidate gates, eligibility, or selection metrics changed")


def validate_results_payload(
    payload: Mapping[str, Any], inputs: LoadedInputs, splits: EvaluationSplits
) -> None:
    if (
        payload.get("schema_version") != RESULTS_SCHEMA_VERSION
        or payload.get("phase") != RESULTS_PHASE
        or payload.get("execution_status") != "COMPLETED"
    ):
        raise ValueError("R3 result identity changed")
    if payload.get("contract") != {
        "path": CONTRACT_RELATIVE_PATH,
        "sha256": inputs.source_hashes["contract"],
        "schema_version": CONTRACT_SCHEMA_VERSION,
    }:
        raise ValueError("R3 result contract lineage changed")
    if payload.get("frozen_inputs") != frozen_inputs_summary(inputs):
        raise ValueError("R3 result input lineage changed")
    if payload.get("split_audit") != splits.audit:
        raise ValueError("R3 result split audit changed")
    candidates = inputs.contract["candidate_search_space"]["candidates"]
    results = payload.get("candidate_results")
    if (
        payload.get("candidate_definitions") != candidates
        or payload.get("candidate_count") != len(EXPECTED_CANDIDATE_IDS)
        or payload.get("completed_candidate_count") != len(EXPECTED_CANDIDATE_IDS)
        or not isinstance(results, list)
        or [row.get("candidate_id") for row in results] != list(EXPECTED_CANDIDATE_IDS)
    ):
        raise ValueError("exactly the two frozen candidate results are required")
    for result, candidate in zip(results, candidates, strict=True):
        validate_candidate_result(result, candidate, inputs, splits)
    if payload.get("primary_router_identity_audit") != primary_router_identity_audit(results):
        raise ValueError("primary-router identity audit changed")
    if (
        payload.get("fold_configuration")
        != inputs.contract["evaluation_protocol"]["grouped_development_cv"]
        or payload.get("selection_rule") != inputs.contract["selection_rule"]
        or payload.get("stop_rule") != inputs.contract["stop_rule"]
        or payload.get("selection_tie_tolerance")
        != {"absolute": TIE_ABSOLUTE_TOLERANCE, "relative": 0.0}
    ):
        raise ValueError("R3 result protocol record changed")
    selection = select_candidate(results, inputs.contract)
    if payload.get("selection") != selection:
        raise ValueError("deterministic candidate selection changed")
    expected_next = (
        SUCCESS_NEXT_REQUIRED
        if selection["selection_status"] == "SELECTED"
        else FAILURE_NEXT_REQUIRED
    )
    if payload.get("next_required") != expected_next or selection["next_required"] != (
        expected_next
    ):
        raise ValueError("R3 stop-rule continuation changed")
    governance = payload.get("governance", {})
    embeddings_generated = governance.get("embeddings_generated_during_run")
    if not isinstance(embeddings_generated, bool) or governance != result_governance(
        embeddings_generated=embeddings_generated
    ):
        raise ValueError("R3 result governance changed")
    if not isinstance(payload.get("bge_cache_manifest_sha256"), str):
        raise TypeError("R3 result BGE cache lineage must be a SHA-256 string")
    validate_text_free(
        payload,
        (*inputs.training_examples, *inputs.development_examples, *inputs.fresh_examples),
    )


def build_results_manifest(payload: Mapping[str, Any], result_bytes: bytes) -> dict[str, Any]:
    return {
        "schema_version": RESULTS_MANIFEST_SCHEMA_VERSION,
        "phase": payload["phase"],
        "results": {
            "path": RESULTS_RELATIVE_PATH,
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
        "next_required": payload["next_required"],
        "governance": payload["governance"],
    }


def _population_cache_spec(
    name: str, records: Sequence[Mapping[str, Any]], identifier_field: str
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
    splits: EvaluationSplits,
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
        "reused_r2_runner_sha256": EXPECTED_R2_RUNNER_SHA256,
        "contract_sha256": inputs.source_hashes["contract"],
        "development_dataset_sha256": inputs.source_hashes["development_dataset"],
        "fresh_evaluation_dataset_sha256": inputs.source_hashes["fresh_dataset"],
        "split_audit_sha256": splits.audit["audit_sha256"],
        "representation_id": "BGE_SMALL",
        "representation_sha256": sha256_bytes(stable_json_bytes(representation)),
        "model_identifier": representation["model_identifier"],
        "embedding_method": representation["embedding_method"],
        "dimensions": representation["dimensions"],
        "l2_normalized": representation["l2_normalized"],
        "populations": [
            _population_cache_spec("development", inputs.development_examples, "example_id"),
            _population_cache_spec("fresh_evaluation", inputs.fresh_examples, "record_id"),
        ],
        "cache_file_sha256": cache_file_sha256,
        "dtype": "float32",
        "library_versions": {
            "fastembed": fastembed_version,
            "numpy": np.__version__,
        },
        "labels_used": False,
        "fine_tuning_performed": False,
        "fresh_evaluation_used_for_fitting": False,
        "raw_text_persisted": False,
        "final_holdout_accessed": False,
    }


def validate_bge_cache(
    inputs: LoadedInputs, splits: EvaluationSplits
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if not BGE_CACHE_PATH.is_file() or not BGE_CACHE_MANIFEST_PATH.is_file():
        raise FileNotFoundError("complete local R3 BGE cache is required")
    manifest = read_json(BGE_CACHE_MANIFEST_PATH)
    fastembed_version = str(manifest.get("library_versions", {}).get("fastembed", ""))
    if not fastembed_version:
        raise ValueError("R3 BGE cache must record the FastEmbed version")
    expected = expected_bge_cache_manifest(
        inputs,
        splits,
        cache_file_sha256=str(manifest.get("cache_file_sha256", "")),
        fastembed_version=fastembed_version,
    )
    if manifest != expected:
        raise ValueError("stale or incompatible R3 BGE cache manifest")
    if sha256_file(BGE_CACHE_PATH) != manifest["cache_file_sha256"]:
        raise ValueError("R3 BGE cache file hash mismatch")
    specifications = {row["matrix_key"]: row for row in manifest["populations"]}
    matrices: dict[str, np.ndarray] = {}
    with np.load(BGE_CACHE_PATH, allow_pickle=False) as cached:
        if set(cached.files) != set(specifications):
            raise ValueError("R3 BGE cache population matrix set changed")
        for key, specification in specifications.items():
            matrix = np.asarray(cached[key])
            validate_embedding_matrix(
                matrix, specification["row_count"], inputs.step29g_contract
            )
            matrices[key] = matrix
    return matrices, manifest


def prepare_bge_cache(
    inputs: LoadedInputs, splits: EvaluationSplits
) -> tuple[dict[str, np.ndarray], dict[str, Any], bool]:
    """Load the validated ignored cache, or embed once with the frozen BGE model."""
    existing = (BGE_CACHE_PATH.exists(), BGE_CACHE_MANIFEST_PATH.exists())
    if any(existing):
        if not all(existing):
            raise FileExistsError("incomplete R3 BGE cache pair")
        matrices, manifest = validate_bge_cache(inputs, splits)
        return matrices, manifest, False
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
    with tempfile.TemporaryFile() as handle:
        np.savez_compressed(handle, **matrices)
        handle.seek(0)
        write_bytes_create_once(BGE_CACHE_PATH, handle.read())
    manifest = expected_bge_cache_manifest(
        inputs,
        splits,
        cache_file_sha256=sha256_file(BGE_CACHE_PATH),
        fastembed_version=importlib.metadata.version("fastembed"),
    )
    write_bytes_create_once(BGE_CACHE_MANIFEST_PATH, stable_json_bytes(manifest))
    matrices, manifest = validate_bge_cache(inputs, splits)
    return matrices, manifest, True


def package_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def result_artifacts_present() -> tuple[bool, bool]:
    return RESULTS_PATH.exists(), RESULTS_MANIFEST_PATH.exists()


def run_model_selection(inputs: LoadedInputs) -> tuple[Path, Path]:
    if any(result_artifacts_present()):
        raise FileExistsError("R3 result artifact exists; refusing overwrite")
    splits = build_evaluation_splits(inputs)
    bge_matrices, bge_manifest, generated = prepare_bge_cache(inputs, splits)
    candidate_results = [
        evaluate_candidate(candidate, inputs, splits, bge_matrices)
        for candidate in inputs.contract["candidate_search_space"]["candidates"]
    ]
    payload = build_results_payload(
        inputs,
        splits,
        candidate_results,
        bge_cache_manifest_sha256=sha256_bytes(stable_json_bytes(bge_manifest)),
        embeddings_generated=generated,
    )
    result_bytes = stable_json_bytes(payload)
    manifest_bytes = stable_json_bytes(build_results_manifest(payload, result_bytes))
    write_bytes_create_once(RESULTS_PATH, result_bytes)
    write_bytes_create_once(RESULTS_MANIFEST_PATH, manifest_bytes)
    return RESULTS_PATH, RESULTS_MANIFEST_PATH


def check_results(inputs: LoadedInputs) -> dict[str, Any]:
    """Validate persisted results deterministically; never embeds, fits, or predicts."""
    presence = result_artifacts_present()
    if not all(presence):
        raise FileNotFoundError("both R3 result artifacts are required")
    result_bytes = read_bytes(RESULTS_PATH)
    payload = json.loads(result_bytes.decode("utf-8"))
    if not isinstance(payload, dict) or stable_json_bytes(payload) != result_bytes:
        raise ValueError("R3 result artifact is not canonical stable JSON")
    manifest = read_json(RESULTS_MANIFEST_PATH)
    splits = build_evaluation_splits(inputs)
    validate_results_payload(payload, inputs, splits)
    if manifest != build_results_manifest(payload, result_bytes):
        raise ValueError("R3 result manifest lineage mismatch")
    return {
        "results_valid": True,
        "results_sha256": sha256_bytes(result_bytes),
        "selection_status": payload["selection"]["selection_status"],
        "selected_candidate_id": payload["selection"]["selected_candidate_id"],
        "eligible_candidate_count": payload["selection"]["eligible_candidate_count"],
        "next_required": payload["next_required"],
        "embeddings_generated": False,
        "classifier_fitting_performed": False,
        "inference_performed": False,
        "files_written": False,
        "final_holdout_accessed": False,
        "step29i_authorized": False,
    }


def preflight(inputs: LoadedInputs) -> dict[str, Any]:
    """Validate everything needed for ``--run`` without any model work or writes."""
    splits = build_evaluation_splits(inputs)
    packages = {
        package: package_version(package)
        for package in ("numpy", "scipy", "scikit-learn", "fastembed")
    }
    cache_presence = (BGE_CACHE_PATH.exists(), BGE_CACHE_MANIFEST_PATH.exists())
    if any(cache_presence) and not all(cache_presence):
        raise FileExistsError("incomplete R3 BGE cache pair")
    if all(cache_presence):
        validate_bge_cache(inputs, splits)
    results_present = any(result_artifacts_present())
    if results_present:
        status = "RESULT_ARTIFACTS_ALREADY_PRESENT"
    elif not all(packages.values()):
        status = "MISSING_REQUIRED_PACKAGE"
    else:
        status = "READY"
    return {
        "status": status,
        "contract_sha256": inputs.source_hashes["contract"],
        "frozen_inputs": frozen_inputs_summary(inputs),
        "training_addendum_record_count": len(inputs.training_examples),
        "development_record_count": len(inputs.development_examples),
        "fresh_evaluation_record_count": len(inputs.fresh_examples),
        "fresh_family_record_counts": {
            family: sum(row["source_family_id"] == family for row in inputs.fresh_examples)
            for family in FRESH_FAMILY_IDS
        },
        "candidate_ids": list(EXPECTED_CANDIDATE_IDS),
        "group_cv_fold_count": len(splits.group_folds),
        "verifier_population_counts": splits.audit["verifier_population_counts"],
        "expected_primary_router_fits": len(EXPECTED_CANDIDATE_IDS) * (FOLD_COUNT + 1),
        "expected_verifier_fits": len(PROTECTED_INTENTS) * (FOLD_COUNT + 1),
        "split_audit_sha256": splits.audit["audit_sha256"],
        "bge_cache_present": all(cache_presence),
        "result_artifacts_present": results_present,
        "create_once_outputs_available": not results_present,
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
        description="Execute the frozen V2-C6 R3 protected-intent gate model selection"
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--run", action="store_true")
    modes.add_argument("--check-results", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    inputs = load_and_validate_inputs()
    if args.preflight:
        print(json.dumps(preflight(inputs), indent=2, sort_keys=True))
        return
    if args.check_results:
        print(json.dumps(check_results(inputs), indent=2, sort_keys=True))
        return
    paths = run_model_selection(inputs)
    print("\n".join(relative_path(path) for path in paths))


if __name__ == "__main__":
    main()
