"""Run the deterministic V2-C6 development-side generalization diagnosis.

The consumed V2-C5 final holdout is prohibited on every code path. This module
uses only frozen development data, existing OOF aggregate evidence, frozen
taxonomy metadata, and the aggregate final context allowed by Step 29A.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import tempfile
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ML_DIRECTORY = REPOSITORY_ROOT / "data/evals/v2/ml"
SCRIPT_RELATIVE_PATH = "scripts/run_v2c6_generalization_diagnosis.py"

CONTRACT_SCHEMA_VERSION = "v2c6-generalization-diagnosis-contract.v1"
RESULT_SCHEMA_VERSION = "v2c6-generalization-diagnosis.v1"
MANIFEST_SCHEMA_VERSION = "v2c6-generalization-diagnosis-manifest.v1"
EXPECTED_CONTRACT_SHA256 = (
    "4276c28eff6158ee947ac26466c7b65ac8830718accba8252de606a49b0493a4"
)
SELECTED_CANDIDATE_ID = "BGE_SMALL_LINEAR_SVC__C=4.0__class_weight=none"
EXPECTED_DEVELOPMENT_RECORD_COUNT = 8198
EXPECTED_INTENT_COUNT = 16
NORMALIZATION_FORM = "NFKC"
TOKEN_PATTERN = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", flags=re.UNICODE)

PROVENANCE_FIELDS = (
    "data_role",
    "source_domain",
    "source_id",
    "source_label",
    "source_revision",
    "source_split",
    "original_split",
    "mapping_status",
    "v2c5_relabel_source",
    "source",
    "source_family",
    "provenance",
    "authoring_method",
    "origin",
    "dataset_family",
    "synthetic",
    "is_synthetic",
    "generation_method",
    "template_family",
)

GOVERNANCE_FLAGS = {
    "embeddings_generated": False,
    "model_inference_performed": False,
    "model_selection_performed": False,
    "model_training_performed": False,
    "raw_v2c5_holdout_accessed": False,
    "remediation_implemented": False,
    "runtime_behavior_changed": False,
    "taxonomy_changed": False,
    "threshold_tuning_performed": False,
}

BytesReader = Callable[[Path], bytes]


@dataclass(frozen=True)
class DiagnosisPaths:
    repository_root: Path
    contract: Path
    development_dataset: Path
    development_manifest: Path
    model_selection_results: Path
    model_selection_manifest: Path
    folds: Path
    folds_manifest: Path
    taxonomy: Path
    taxonomy_manifest: Path
    final_results: Path
    final_results_manifest: Path
    final_state: Path
    prohibited_holdout: Path
    result: Path
    result_manifest: Path
    runner: Path


DEFAULT_PATHS = DiagnosisPaths(
    repository_root=REPOSITORY_ROOT,
    contract=ML_DIRECTORY / "v2c6_generalization_diagnosis_contract.json",
    development_dataset=(
        ML_DIRECTORY / "v2c5_expanded_development_dataset.json"
    ),
    development_manifest=(
        ML_DIRECTORY / "v2c5_expanded_development_dataset.manifest.json"
    ),
    model_selection_results=ML_DIRECTORY / "v2c5_model_selection_results.json",
    model_selection_manifest=(
        ML_DIRECTORY / "v2c5_model_selection_results.manifest.json"
    ),
    folds=ML_DIRECTORY / "v2c5_model_selection_folds.json",
    folds_manifest=ML_DIRECTORY / "v2c5_model_selection_folds.manifest.json",
    taxonomy=ML_DIRECTORY / "v2c5_taxonomy_freeze.json",
    taxonomy_manifest=ML_DIRECTORY / "v2c5_taxonomy_freeze.manifest.json",
    final_results=ML_DIRECTORY / "v2c5_final_evaluation_results.json",
    final_results_manifest=(
        ML_DIRECTORY / "v2c5_final_evaluation_results.manifest.json"
    ),
    final_state=ML_DIRECTORY / "v2c5_final_evaluation_state.json",
    prohibited_holdout=ML_DIRECTORY / "v2c5_final_holdout.json",
    result=ML_DIRECTORY / "v2c6_generalization_diagnosis.json",
    result_manifest=(
        ML_DIRECTORY / "v2c6_generalization_diagnosis.manifest.json"
    ),
    runner=REPOSITORY_ROOT / SCRIPT_RELATIVE_PATH,
)


def stable_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def filesystem_reader(path: Path) -> bytes:
    return path.read_bytes()


def guard_consumed_holdout(path: Path, paths: DiagnosisPaths) -> None:
    if path.resolve() == paths.prohibited_holdout.resolve():
        raise PermissionError("consumed V2-C5 final holdout access is prohibited")


def guarded_read_bytes(
    path: Path,
    paths: DiagnosisPaths,
    reader: BytesReader = filesystem_reader,
) -> bytes:
    guard_consumed_holdout(path, paths)
    return reader(path)


def sha256_file(
    path: Path,
    paths: DiagnosisPaths,
    reader: BytesReader = filesystem_reader,
) -> str:
    return sha256_bytes(guarded_read_bytes(path, paths, reader))


def read_json_object(
    path: Path,
    paths: DiagnosisPaths,
    reader: BytesReader = filesystem_reader,
) -> dict[str, Any]:
    payload = json.loads(guarded_read_bytes(path, paths, reader))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return payload


def display_path(path: Path, paths: DiagnosisPaths) -> str:
    try:
        return str(path.resolve().relative_to(paths.repository_root.resolve()))
    except ValueError:
        return str(path.resolve())


def fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def durable_create(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary_path = Path(handle.name)
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        try:
            os.link(temporary_path, path)
        except FileExistsError as exc:
            raise FileExistsError(f"refusing to overwrite existing file: {path}") from exc
        temporary_path.unlink()
        fsync_directory(path.parent)
    finally:
        temporary_path.unlink(missing_ok=True)


def validate_file_binding(
    path: Path,
    specification: Mapping[str, Any],
    paths: DiagnosisPaths,
    reader: BytesReader,
    label: str,
) -> None:
    if specification.get("path") != display_path(path, paths):
        raise ValueError(f"{label} path binding changed")
    if specification.get("sha256") != sha256_file(path, paths, reader):
        raise ValueError(f"{label} hash binding changed")


def question_by_id(contract: Mapping[str, Any], question_id: str) -> Mapping[str, Any]:
    matches = [
        question
        for question in contract.get("diagnosis_questions", [])
        if question.get("id") == question_id
    ]
    if len(matches) != 1:
        raise ValueError(f"diagnosis question is missing or duplicated: {question_id}")
    return matches[0]


def validate_contract(contract: Mapping[str, Any], paths: DiagnosisPaths) -> None:
    if (
        contract.get("schema_version") != CONTRACT_SCHEMA_VERSION
        or contract.get("contract_version") != CONTRACT_SCHEMA_VERSION
        or contract.get("phase") != "V2-C6 Step 29A"
    ):
        raise ValueError("unexpected V2-C6 diagnosis contract identity")
    status = contract.get("contract_status", {})
    required_status = {
        "contract_frozen": True,
        "diagnosis_performed": False,
        "final_holdout_accessed": False,
        "model_selection_performed": False,
        "model_training_performed": False,
        "next_required": "v2c6_generalization_diagnosis",
        "remediation_implemented": False,
        "runtime_behavior_changed": False,
        "taxonomy_changed": False,
    }
    if status != required_status:
        raise ValueError("V2-C6 diagnosis contract status changed")
    expected_questions = [
        "development_source_composition",
        "group_structure",
        "duplicate_and_near_duplicate_structure",
        "lexical_diversity",
        "class_imbalance",
        "development_oof_behavior",
        "authoring_and_provenance_concentration",
        "intent_boundary_diagnostics",
        "effective_sample_size",
    ]
    questions = contract.get("diagnosis_questions", [])
    if [question.get("id") for question in questions] != expected_questions:
        raise ValueError("frozen diagnosis question order changed")
    if [question.get("ordinal") for question in questions] != list(range(1, 10)):
        raise ValueError("frozen diagnosis question ordinals changed")
    duplicate_question = question_by_id(
        contract, "duplicate_and_near_duplicate_structure"
    )
    if duplicate_question.get("semantic_near_duplicate_threshold") is not None:
        raise ValueError("unfrozen semantic-similarity threshold is prohibited")
    context = contract.get("final_result_context", {})
    expected_context = {
        "allowed_role": (
            "Historical aggregate evidence motivating development-side diagnosis "
            "only."
        ),
        "development_pooled_oof_macro_f1": 0.8850476526181243,
        "final_exact_protected_write_recall": 0.4,
        "final_macro_f1": 0.5632279946795209,
        "final_new_seven_intent_macro_f1": 0.45253940739110227,
        "final_protected_write_false_positive_rate": 0.014583333333333334,
        "final_unsupported_or_uncertain_recall": 0.85,
        "individual_final_examples_permitted": False,
        "selected_candidate_id": SELECTED_CANDIDATE_ID,
    }
    if context != expected_context:
        raise ValueError("frozen aggregate V2-C5 context changed")
    prohibited = contract.get("input_policy", {}).get(
        "consumed_v2c5_final_holdout", {}
    )
    if prohibited != {
        "path": display_path(paths.prohibited_holdout, paths),
        "status": "explicitly_prohibited_raw_input",
    }:
        raise ValueError("consumed V2-C5 holdout prohibition changed")
    if contract.get("input_policy", {}).get(
        "individual_v2c5_final_holdout_examples_permitted"
    ) is not False:
        raise ValueError("individual V2-C5 holdout access became permitted")
    if set(contract.get("prohibited_during_diagnosis", {}).values()) != {True}:
        raise ValueError("diagnosis prohibition changed")
    outputs = contract.get("future_outputs", {})
    if (
        outputs.get("diagnosis", {}).get("path") != display_path(paths.result, paths)
        or outputs.get("manifest", {}).get("path")
        != display_path(paths.result_manifest, paths)
        or outputs.get("generated_by_step29a") is not False
        or outputs.get("diagnosis", {}).get("raw_final_holdout_text_permitted")
        is not False
        or outputs.get("manifest", {}).get("raw_final_holdout_text_permitted")
        is not False
    ):
        raise ValueError("future diagnosis output contract changed")


def validate_taxonomy(taxonomy: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    labels = taxonomy.get("intent_label_order")
    protected = taxonomy.get("protected_write_intents")
    risk_by_intent = taxonomy.get("risk_by_intent")
    if (
        taxonomy.get("schema_version") != "v2c5-taxonomy-freeze.v1"
        or taxonomy.get("final_taxonomy_frozen") is not True
        or not isinstance(labels, list)
        or len(labels) != EXPECTED_INTENT_COUNT
        or len(set(labels)) != EXPECTED_INTENT_COUNT
        or protected
        != ["cancel_transfer", "close_account", "create_dispute", "freeze_card"]
        or not isinstance(risk_by_intent, dict)
        or set(risk_by_intent) != set(labels)
    ):
        raise ValueError("frozen V2-C5 taxonomy changed")
    return list(labels), list(protected)


def validate_development_dataset(
    dataset: Mapping[str, Any],
    manifest: Mapping[str, Any],
    labels: Sequence[str],
    taxonomy: Mapping[str, Any],
) -> list[dict[str, Any]]:
    examples = dataset.get("examples")
    if (
        dataset.get("schema_version") != "v2c5-expanded-development-dataset.v1"
        or dataset.get("example_count") != EXPECTED_DEVELOPMENT_RECORD_COUNT
        or not isinstance(examples, list)
        or len(examples) != EXPECTED_DEVELOPMENT_RECORD_COUNT
        or dataset.get("normalization_version")
        != "unicode-nfkc-lower-whitespace.v1"
    ):
        raise ValueError("expanded development dataset identity changed")
    counts: Counter[str] = Counter()
    identifiers: set[str] = set()
    validated: list[dict[str, Any]] = []
    for index, record in enumerate(examples):
        if not isinstance(record, dict):
            raise TypeError(f"development record {index} must be an object")
        example_id = record.get("example_id")
        group_id = record.get("group_id")
        intent = record.get("intent")
        text = record.get("text")
        if not isinstance(example_id, str) or not example_id:
            raise ValueError(f"development record {index} has an invalid ID")
        if example_id in identifiers:
            raise ValueError(f"duplicate development example ID: {example_id}")
        if not isinstance(group_id, str) or not group_id:
            raise ValueError(f"development record {index} has an invalid group")
        if intent not in labels:
            raise ValueError(f"development record {index} has an unknown intent")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"development record {index} has empty text")
        if record.get("risk") != taxonomy["risk_by_intent"][intent]:
            raise ValueError(f"development record {index} risk changed")
        identifiers.add(example_id)
        counts[str(intent)] += 1
        validated.append(record)
    expected_counts = manifest.get("counts", {}).get("final_intent_counts")
    if dict(counts) != expected_counts or sum(counts.values()) != 8198:
        raise ValueError("development intent counts changed")
    if manifest.get("dataset", {}).get("schema_version") != dataset.get(
        "schema_version"
    ):
        raise ValueError("development dataset schema and manifest differ")
    if manifest.get("taxonomy_intent_labels") != list(labels):
        raise ValueError("development taxonomy labels changed")
    return validated


def selected_candidate_result(
    results: Mapping[str, Any], labels: Sequence[str]
) -> dict[str, Any]:
    if (
        results.get("schema_version") != "v2c5-model-selection-results.v1"
        or results.get("development_record_count") != 8198
        or results.get("selection", {}).get("selected_candidate_id")
        != SELECTED_CANDIDATE_ID
    ):
        raise ValueError("frozen model-selection result changed")
    matches = [
        candidate
        for candidate in results.get("candidate_results", [])
        if candidate.get("candidate_id") == SELECTED_CANDIDATE_ID
    ]
    if len(matches) != 1:
        raise ValueError("selected candidate result is missing or duplicated")
    candidate = matches[0]
    aggregate = candidate.get("aggregate_metrics", {})
    confusion = aggregate.get("confusion_matrix", {})
    if (
        confusion.get("label_order") != list(labels)
        or len(confusion.get("values", [])) != EXPECTED_INTENT_COUNT
        or candidate.get("oof_prediction_count") != 8198
        or candidate.get("oof_predictions_persisted") is not False
        or candidate.get("raw_text_persisted") is not False
    ):
        raise ValueError("selected candidate OOF evidence changed")
    return candidate


def validate_folds(
    folds: Mapping[str, Any],
    folds_manifest: Mapping[str, Any],
    examples: Sequence[Mapping[str, Any]],
    expected_fold_specification: Mapping[str, Any],
) -> None:
    assignments = folds.get("assignments")
    if (
        folds.get("schema_version") != "v2c5-model-selection-folds.v1"
        or folds.get("record_count") != 8198
        or folds.get("assignment_count") != 8198
        or not isinstance(assignments, list)
        or len(assignments) != 8198
    ):
        raise ValueError("frozen fold artifact identity changed")
    if (
        folds_manifest.get("schema_version")
        != "v2c5-model-selection-folds-manifest.v1"
        or folds_manifest.get("record_count") != 8198
        or folds_manifest.get("fold_artifact", {}).get("path")
        != expected_fold_specification.get("path")
        or folds_manifest.get("fold_artifact", {}).get("schema_version")
        != "v2c5-model-selection-folds.v1"
        or folds_manifest.get("fold_artifact", {}).get("sha256")
        != expected_fold_specification.get("sha256")
        or folds_manifest.get("source_development_dataset")
        != folds.get("source_development_dataset")
        or folds_manifest.get("cross_validation") != folds.get("cross_validation")
        or folds_manifest.get("fold_summaries") != folds.get("fold_summaries")
        or folds_manifest.get("integrity") != folds.get("integrity")
        or folds_manifest.get("unique_group_count")
        != folds.get("unique_group_count")
    ):
        raise ValueError("frozen fold manifest changed")
    expected_by_id = {
        record["example_id"]: (record["group_id"], record["intent"])
        for record in examples
    }
    seen: set[str] = set()
    group_folds: dict[str, set[int]] = defaultdict(set)
    for assignment in assignments:
        example_id = assignment.get("example_id")
        if example_id not in expected_by_id or example_id in seen:
            raise ValueError("fold assignment IDs changed")
        group_id, intent = expected_by_id[example_id]
        fold = assignment.get("validation_fold")
        if (
            assignment.get("group_id") != group_id
            or assignment.get("intent") != intent
            or fold not in {0, 1, 2, 3, 4}
        ):
            raise ValueError("fold assignment content changed")
        seen.add(str(example_id))
        group_folds[group_id].add(int(fold))
    if seen != set(expected_by_id) or any(len(values) != 1 for values in group_folds.values()):
        raise ValueError("fold assignments are incomplete or group-leaking")


def validate_final_context(
    final_results: Mapping[str, Any],
    final_manifest: Mapping[str, Any],
    final_state: Mapping[str, Any],
    model_candidate: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> None:
    context = contract["final_result_context"]
    aggregate = model_candidate["aggregate_metrics"]
    safety = final_results.get("safety_metrics", {})
    checks = {
        "development_pooled_oof_macro_f1": aggregate.get("pooled_oof_macro_f1"),
        "final_exact_protected_write_recall": safety.get(
            "exact_protected_write_recall"
        ),
        "final_macro_f1": final_results.get("metrics", {}).get("macro_f1"),
        "final_new_seven_intent_macro_f1": final_results.get("metrics", {}).get(
            "new_seven_intent_macro_f1"
        ),
        "final_protected_write_false_positive_rate": safety.get(
            "protected_write_false_positive_rate"
        ),
        "final_unsupported_or_uncertain_recall": safety.get(
            "unsupported_or_uncertain_recall"
        ),
    }
    if any(context.get(key) != value for key, value in checks.items()):
        raise ValueError("committed aggregate V2-C5 context changed")
    if final_results.get("selected_candidate_id") != SELECTED_CANDIDATE_ID:
        raise ValueError("final-evaluation selected candidate changed")
    if (
        final_manifest.get("results", {}).get("sha256")
        != contract["source_artifacts"]["final_evaluation_results"]["sha256"]
        or final_state.get("state") != "completed"
        or final_state.get("results", {}).get("results_sha256")
        != contract["source_artifacts"]["final_evaluation_results"]["sha256"]
        or final_state.get("results", {}).get("results_manifest_sha256")
        != contract["source_artifacts"]["final_evaluation_results_manifest"][
            "sha256"
        ]
    ):
        raise ValueError("completed final-evaluation aggregate lineage changed")


def source_lineage(
    contract: Mapping[str, Any],
    paths: DiagnosisPaths,
    reader: BytesReader,
) -> dict[str, str]:
    sources = contract["source_artifacts"]
    return {
        "diagnosis_contract_sha256": sha256_file(paths.contract, paths, reader),
        "development_dataset_sha256": sources["expanded_development_dataset"][
            "sha256"
        ],
        "development_manifest_sha256": sources["expanded_development_manifest"][
            "sha256"
        ],
        "final_evaluation_results_manifest_sha256": sources[
            "final_evaluation_results_manifest"
        ]["sha256"],
        "final_evaluation_results_sha256": sources["final_evaluation_results"][
            "sha256"
        ],
        "final_evaluation_state_sha256": sources["final_evaluation_state"][
            "sha256"
        ],
        "fold_artifact_sha256": sha256_file(paths.folds, paths, reader),
        "fold_manifest_sha256": sha256_file(paths.folds_manifest, paths, reader),
        "model_selection_manifest_sha256": sources[
            "model_selection_results_manifest"
        ]["sha256"],
        "model_selection_results_sha256": sources["model_selection_results"][
            "sha256"
        ],
        "taxonomy_manifest_sha256": sources["taxonomy_freeze_manifest"][
            "sha256"
        ],
        "taxonomy_sha256": sources["taxonomy_freeze"]["sha256"],
    }


def ensure_outputs_absent(paths: DiagnosisPaths) -> None:
    if paths.result.exists() or paths.result_manifest.exists():
        raise FileExistsError("V2-C6 diagnosis output artifacts already exist")


def load_preconditions(
    paths: DiagnosisPaths = DEFAULT_PATHS,
    *,
    reader: BytesReader = filesystem_reader,
    require_outputs_absent: bool = True,
) -> dict[str, Any]:
    if sha256_file(paths.contract, paths, reader) != EXPECTED_CONTRACT_SHA256:
        raise ValueError("frozen V2-C6 diagnosis contract hash changed")
    contract = read_json_object(paths.contract, paths, reader)
    validate_contract(contract, paths)
    sources = contract["source_artifacts"]
    path_bindings = {
        "expanded_development_dataset": paths.development_dataset,
        "expanded_development_manifest": paths.development_manifest,
        "final_evaluation_results": paths.final_results,
        "final_evaluation_results_manifest": paths.final_results_manifest,
        "final_evaluation_state": paths.final_state,
        "model_selection_results": paths.model_selection_results,
        "model_selection_results_manifest": paths.model_selection_manifest,
        "taxonomy_freeze": paths.taxonomy,
        "taxonomy_freeze_manifest": paths.taxonomy_manifest,
    }
    for name, path in path_bindings.items():
        validate_file_binding(path, sources[name], paths, reader, name)

    development_manifest = read_json_object(paths.development_manifest, paths, reader)
    development = read_json_object(paths.development_dataset, paths, reader)
    taxonomy_manifest = read_json_object(paths.taxonomy_manifest, paths, reader)
    taxonomy = read_json_object(paths.taxonomy, paths, reader)
    labels, protected = validate_taxonomy(taxonomy)
    if taxonomy_manifest.get("taxonomy_freeze") != sources["taxonomy_freeze"]:
        raise ValueError("taxonomy manifest lineage changed")
    examples = validate_development_dataset(
        development, development_manifest, labels, taxonomy
    )
    if development_manifest.get("dataset", {}).get("sha256") != sources[
        "expanded_development_dataset"
    ]["sha256"]:
        raise ValueError("development manifest dataset hash changed")

    model_manifest = read_json_object(paths.model_selection_manifest, paths, reader)
    model_results = read_json_object(paths.model_selection_results, paths, reader)
    if (
        model_manifest.get("selected_candidate_id") != SELECTED_CANDIDATE_ID
        or model_manifest.get("results") != {
            "path": display_path(paths.model_selection_results, paths),
            "schema_version": "v2c5-model-selection-results.v1",
            "sha256": sources["model_selection_results"]["sha256"],
        }
        or model_manifest.get("source_development_dataset")
        != sources["expanded_development_dataset"]
    ):
        raise ValueError("model-selection manifest lineage changed")
    candidate = selected_candidate_result(model_results, labels)

    fold_specification = model_manifest.get("fold_artifact", {})
    if fold_specification.get("path") != display_path(paths.folds, paths):
        raise ValueError("fold artifact path changed")
    if fold_specification.get("sha256") != sha256_file(paths.folds, paths, reader):
        raise ValueError("fold artifact hash changed")
    if model_results.get("fold_artifact") != fold_specification:
        raise ValueError("model-selection fold lineage differs")
    folds_manifest = read_json_object(paths.folds_manifest, paths, reader)
    folds = read_json_object(paths.folds, paths, reader)
    validate_folds(folds, folds_manifest, examples, fold_specification)

    final_results = read_json_object(paths.final_results, paths, reader)
    final_manifest = read_json_object(paths.final_results_manifest, paths, reader)
    final_state = read_json_object(paths.final_state, paths, reader)
    validate_final_context(
        final_results, final_manifest, final_state, candidate, contract
    )
    if require_outputs_absent:
        ensure_outputs_absent(paths)
    return {
        "candidate": candidate,
        "contract": contract,
        "development": development,
        "development_manifest": development_manifest,
        "examples": examples,
        "folds": folds,
        "folds_manifest": folds_manifest,
        "labels": labels,
        "lineage": source_lineage(contract, paths, reader),
        "model_results": model_results,
        "protected_intents": protected,
        "taxonomy": taxonomy,
    }


def preflight(
    paths: DiagnosisPaths = DEFAULT_PATHS,
    *,
    reader: BytesReader = filesystem_reader,
) -> dict[str, Any]:
    inputs = load_preconditions(paths, reader=reader)
    return {
        "development_record_count": len(inputs["examples"]),
        "embeddings_generated": False,
        "files_written": False,
        "intent_count": len(inputs["labels"]),
        "model_inference_performed": False,
        "model_selection_performed": False,
        "model_training_performed": False,
        "raw_v2c5_holdout_accessed": False,
        "selected_candidate_id": SELECTED_CANDIDATE_ID,
        "status": "ready",
        "threshold_tuning_performed": False,
    }


def _ratio(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def percentile_linear(values: Sequence[int | float], percentile: float) -> float:
    if not values:
        raise ValueError("a non-empty sequence is required")
    if not 0.0 <= percentile <= 1.0:
        raise ValueError("percentile must be between zero and one")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def distribution(values: Sequence[int | float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("a non-empty sequence is required")
    return {
        "count": len(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "p90": percentile_linear(values, 0.90),
        "p95": percentile_linear(values, 0.95),
        "percentile_method": "linear_interpolation_index_n_minus_1",
    }


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize(NORMALIZATION_FORM, text).lower().strip()
    return " ".join(normalized.split())


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(normalize_text(text))


def metadata_concentration(
    examples: Sequence[Mapping[str, Any]], labels: Sequence[str]
) -> dict[str, Any]:
    def summarize(values: Sequence[Any], population: int) -> dict[str, Any]:
        normalized_values = [json.dumps(value, sort_keys=True) for value in values]
        counts = Counter(normalized_values)
        present_count = len(values)
        return {
            "available": bool(values),
            "category_count": len(counts),
            "counts_by_value": dict(sorted(counts.items())),
            "herfindahl_hirschman_index": sum(
                (_ratio(count, present_count)) ** 2 for count in counts.values()
            ),
            "largest_category_share": _ratio(
                max(counts.values(), default=0), present_count
            ),
            "missing_count": population - present_count,
            "missing_rate": _ratio(population - present_count, population),
            "present_count": present_count,
            "raw_count_to_unique_category_ratio": _ratio(
                present_count, len(counts)
            ),
            "shares_by_value": {
                key: _ratio(value, present_count)
                for key, value in sorted(counts.items())
            },
        }

    result: dict[str, Any] = {}
    record_count = len(examples)
    for field in PROVENANCE_FIELDS:
        values = [record.get(field) for record in examples]
        present = [value for value in values if value is not None and value != ""]
        if not present:
            result[field] = {
                "available": False,
                "reason": "metadata_not_present",
            }
            continue
        per_intent: dict[str, Any] = {}
        for label in labels:
            label_records = [
                record
                for record in examples
                if record["intent"] == label
            ]
            label_values = [
                record[field]
                for record in label_records
                if record.get(field) is not None and record.get(field) != ""
            ]
            if label_values:
                per_intent[label] = summarize(label_values, len(label_records))
            else:
                per_intent[label] = {
                    "available": False,
                    "missing_count": len(label_records),
                    "missing_rate": 1.0,
                    "reason": "metadata_not_present",
                }
        summary = summarize(present, record_count)
        result[field] = {
            **summary,
            "distribution_across_intents": per_intent,
        }
    return result


def development_source_composition(
    examples: Sequence[Mapping[str, Any]],
    labels: Sequence[str],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    counts = Counter(str(record["intent"]) for record in examples)
    groups_by_intent: dict[str, set[str]] = defaultdict(set)
    group_counts_by_intent: dict[str, Counter[str]] = defaultdict(Counter)
    all_groups: set[str] = set()
    for record in examples:
        label = str(record["intent"])
        group_id = str(record["group_id"])
        all_groups.add(group_id)
        groups_by_intent[label].add(group_id)
        group_counts_by_intent[label][group_id] += 1
    all_group_counts = Counter(str(record["group_id"]) for record in examples)
    metadata_distributions = {
        field: {
            "available": details["available"],
            **(
                {
                    "counts_by_value": details["counts_by_value"],
                    "shares_by_value": details["shares_by_value"],
                }
                if details["available"]
                else {"reason": details["reason"]}
            ),
        }
        for field, details in provenance.items()
    }
    return {
        "examples_per_group_distribution": distribution(
            list(all_group_counts.values())
        ),
        "examples_per_group_distribution_by_intent": {
            label: distribution(list(group_counts_by_intent[label].values()))
            for label in labels
        },
        "intent_counts": {label: counts[label] for label in labels},
        "intent_percentages": {
            label: _ratio(counts[label], len(examples)) for label in labels
        },
        "metadata_distributions": metadata_distributions,
        "total_record_count": len(examples),
        "total_unique_group_count": len(all_groups),
        "unique_groups_per_intent": {
            label: len(groups_by_intent[label]) for label in labels
        },
    }


def _largest_group_share(
    sizes: Sequence[int], group_fraction: float, record_count: int
) -> dict[str, int | float]:
    group_count = max(1, math.ceil(len(sizes) * group_fraction))
    selected_records = sum(sorted(sizes, reverse=True)[:group_count])
    return {
        "group_count": group_count,
        "record_count": selected_records,
        "record_share": _ratio(selected_records, record_count),
    }


def group_structure(
    examples: Sequence[Mapping[str, Any]], labels: Sequence[str]
) -> dict[str, Any]:
    group_records: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in examples:
        group_records[str(record["group_id"])].append(record)
    sizes = [len(records) for records in group_records.values()]
    singleton_count = sum(size == 1 for size in sizes)
    multi_record_count = sum(size > 1 for size in sizes)
    multi_intent_records = {
        group_id: records
        for group_id, records in group_records.items()
        if len({record["intent"] for record in records}) > 1
    }
    intent_counts_per_multi_group = [
        len({record["intent"] for record in records})
        for records in multi_intent_records.values()
    ]
    dominant_shares = [
        max(Counter(record["intent"] for record in records).values()) / len(records)
        for records in group_records.values()
    ]
    per_intent: dict[str, Any] = {}
    for label in labels:
        label_records = [record for record in examples if record["intent"] == label]
        label_groups = Counter(str(record["group_id"]) for record in label_records)
        per_intent[label] = {
            "largest_group_share": max(label_groups.values()) / len(label_records),
            "raw_records": len(label_records),
            "records_per_group_ratio": len(label_records) / len(label_groups),
            "unique_groups": len(label_groups),
        }
    fixed_largest = {}
    ordered_sizes = sorted(sizes, reverse=True)
    for count in (1, 5, 10, 25):
        selected = sum(ordered_sizes[:count])
        fixed_largest[str(count)] = {
            "record_count": selected,
            "record_share": selected / len(examples),
        }
    return {
        "dominant_intent_share_per_group_distribution": distribution(
            dominant_shares
        ),
        "group_size_distribution": distribution(sizes),
        "groups_with_more_than_one_record_count": multi_record_count,
        "groups_with_more_than_one_record_percentage": _ratio(
            multi_record_count, len(group_records)
        ),
        "largest_group_sizes": ordered_sizes[:25],
        "multi_intent_group_count": len(multi_intent_records),
        "multi_intent_group_percentage": _ratio(
            len(multi_intent_records), len(group_records)
        ),
        "multi_intent_group_intent_count_distribution": (
            distribution(intent_counts_per_multi_group)
            if intent_counts_per_multi_group
            else {"available": False, "reason": "no_multi_intent_groups"}
        ),
        "per_intent": per_intent,
        "raw_rows_are_not_independent_groups": True,
        "records_in_largest_fixed_group_counts": fixed_largest,
        "records_in_multi_intent_groups": sum(
            len(records) for records in multi_intent_records.values()
        ),
        "share_in_top_group_percentages": {
            "1_percent": _largest_group_share(sizes, 0.01, len(examples)),
            "5_percent": _largest_group_share(sizes, 0.05, len(examples)),
            "10_percent": _largest_group_share(sizes, 0.10, len(examples)),
        },
        "singleton_group_count": singleton_count,
        "singleton_group_percentage": _ratio(singleton_count, len(group_records)),
        "total_unique_groups": len(group_records),
    }


def _duplicate_clusters(
    examples: Sequence[Mapping[str, Any]], key: Callable[[Mapping[str, Any]], str]
) -> dict[str, list[Mapping[str, Any]]]:
    clusters: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in examples:
        clusters[key(record)].append(record)
    return {value: records for value, records in clusters.items() if len(records) > 1}


def duplicate_structure(
    examples: Sequence[Mapping[str, Any]], labels: Sequence[str]
) -> dict[str, Any]:
    exact_clusters = _duplicate_clusters(examples, lambda record: str(record["text"]))
    normalized_clusters = _duplicate_clusters(
        examples, lambda record: normalize_text(str(record["text"]))
    )
    exact_within_intent_clusters = {
        key: records
        for key, records in exact_clusters.items()
        if len({record["intent"] for record in records}) == 1
    }
    exact_conflict_clusters = {
        key: records
        for key, records in exact_clusters.items()
        if len({record["intent"] for record in records}) > 1
    }
    normalized_within_intent_clusters = {
        key: records
        for key, records in normalized_clusters.items()
        if len({record["intent"] for record in records}) == 1
    }
    normalized_conflict_clusters = {
        key: records
        for key, records in normalized_clusters.items()
        if len({record["intent"] for record in records}) > 1
    }
    exact_redundant = sum(len(records) - 1 for records in exact_clusters.values())
    normalized_redundant = sum(
        len(records) - 1 for records in normalized_clusters.values()
    )

    def conflict_audit(
        clusters: Mapping[str, Sequence[Mapping[str, Any]]], hash_name: str
    ) -> list[dict[str, Any]]:
        return [
            {
                "example_ids": sorted(
                    str(record["example_id"]) for record in records
                ),
                hash_name: sha256_bytes(key.encode("utf-8")),
                "intents": sorted({str(record["intent"]) for record in records}),
                "record_count": len(records),
            }
            for key, records in sorted(clusters.items())
        ]

    per_intent: dict[str, Any] = {}
    for label in labels:
        label_records = [record for record in examples if record["intent"] == label]
        exact_values = {str(record["text"]) for record in label_records}
        normalized_values = {
            normalize_text(str(record["text"])) for record in label_records
        }
        label_clusters = Counter(
            normalize_text(str(record["text"])) for record in label_records
        )
        exact_label_clusters = Counter(str(record["text"]) for record in label_records)
        per_intent[label] = {
            "exact_duplicate_cluster_count": sum(
                value > 1 for value in exact_label_clusters.values()
            ),
            "exact_unique_to_raw_ratio": len(exact_values) / len(label_records),
            "normalized_duplicate_cluster_count": sum(
                value > 1 for value in label_clusters.values()
            ),
            "normalized_unique_to_raw_ratio": len(normalized_values)
            / len(label_records),
            "raw_records": len(label_records),
            "unique_exact_text_count": len(exact_values),
            "unique_normalized_text_count": len(normalized_values),
        }
    return {
        "cross_intent_exact_conflict_clusters": conflict_audit(
            exact_conflict_clusters, "exact_text_sha256"
        ),
        "cross_intent_normalized_conflict_clusters": conflict_audit(
            normalized_conflict_clusters, "normalized_text_sha256"
        ),
        "cross_intent_exact_duplicate_conflict_cluster_count": len(
            exact_conflict_clusters
        ),
        "cross_intent_normalized_duplicate_conflict_cluster_count": len(
            normalized_conflict_clusters
        ),
        "exact_duplicate_cluster_count": len(exact_clusters),
        "exact_duplicate_rate": exact_redundant / len(examples),
        "exact_duplicate_record_count": exact_redundant,
        "normalization": {
            "case": "lowercase",
            "internal_whitespace": "collapse",
            "leading_trailing_whitespace": "trim",
            "unicode_form": NORMALIZATION_FORM,
            "version": "unicode-nfkc-lower-whitespace.v1",
        },
        "normalized_duplicate_cluster_count": len(normalized_clusters),
        "normalized_duplicate_rate": normalized_redundant / len(examples),
        "normalized_duplicate_record_count": normalized_redundant,
        "per_intent": per_intent,
        "raw_duplicate_text_persisted": False,
        "record_count_definition": "redundant_records_beyond_first_occurrence",
        "records_in_cross_intent_duplicate_conflicts": sum(
            len(records) for records in normalized_conflict_clusters.values()
        ),
        "semantic_similarity_used": False,
        "within_intent_exact_duplicate_cluster_count": len(
            exact_within_intent_clusters
        ),
        "within_intent_normalized_duplicate_cluster_count": len(
            normalized_within_intent_clusters
        ),
    }


def ngram_counter(token_sequences: Sequence[Sequence[str]], size: int) -> Counter[Any]:
    values: Counter[Any] = Counter()
    for tokens in token_sequences:
        values.update(
            tuple(tokens[index : index + size])
            for index in range(len(tokens) - size + 1)
        )
    return values


def concentration_payload(counter: Counter[Any]) -> dict[str, int | float]:
    total = sum(counter.values())
    ordered_counts = sorted(counter.values(), reverse=True)
    return {
        "share_top_10": _ratio(sum(ordered_counts[:10]), total),
        "share_top_25": _ratio(sum(ordered_counts[:25]), total),
        "total_count": total,
        "unique_count": len(counter),
    }


def lexical_scope(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    texts = [str(record["text"]) for record in records]
    token_sequences = [tokenize(text) for text in texts]
    tokens = [token for sequence in token_sequences for token in sequence]
    token_counts = Counter(tokens)
    unique_count = len(token_counts)
    hapax_count = sum(value == 1 for value in token_counts.values())
    return {
        "character_length_distribution": distribution([len(text) for text in texts]),
        "ngram_concentration": {
            "bigrams": concentration_payload(ngram_counter(token_sequences, 2)),
            "trigrams": concentration_payload(ngram_counter(token_sequences, 3)),
            "unigrams": concentration_payload(token_counts),
        },
        "token_length_distribution": distribution(
            [len(sequence) for sequence in token_sequences]
        ),
        "token_statistics": {
            "hapax_rate": _ratio(hapax_count, unique_count),
            "hapax_rate_denominator": "unique_token_count",
            "hapax_token_count": hapax_count,
            "total_token_count": len(tokens),
            "type_token_ratio": _ratio(unique_count, len(tokens)),
            "unique_token_count": unique_count,
        },
    }


def lexical_diversity(
    examples: Sequence[Mapping[str, Any]], labels: Sequence[str]
) -> dict[str, Any]:
    return {
        "global": lexical_scope(examples),
        "per_intent": {
            label: lexical_scope(
                [record for record in examples if record["intent"] == label]
            )
            for label in labels
        },
        "raw_examples_persisted": False,
        "tokenization": {
            "case_and_unicode_normalization": (
                "NFKC, lowercase, trim, collapse whitespace"
            ),
            "regex": TOKEN_PATTERN.pattern,
            "rule": "Unicode alphanumeric words with optional internal apostrophe",
        },
    }


def class_imbalance(
    examples: Sequence[Mapping[str, Any]],
    labels: Sequence[str],
    protected_intents: Sequence[str],
) -> dict[str, Any]:
    counts = Counter(str(record["intent"]) for record in examples)
    risk_counts = Counter(str(record["risk"]) for record in examples)
    ordered_counts = {label: counts[label] for label in labels}
    values = list(ordered_counts.values())
    unsupported_count = counts["unsupported_or_uncertain"]
    protected_count = sum(counts[label] for label in protected_intents)
    return {
        "informational_policy": {
            "count": counts["informational_policy"],
            "share": counts["informational_policy"] / len(examples),
        },
        "intent_counts": ordered_counts,
        "intent_shares": {
            label: counts[label] / len(examples) for label in labels
        },
        "largest_class_count": max(values),
        "largest_to_median_class_ratio": max(values) / statistics.median(values),
        "largest_to_smallest_class_ratio": max(values) / min(values),
        "protected_write_combined": {
            "count": protected_count,
            "share": protected_count / len(examples),
        },
        "risk_levels": {
            risk: {
                "count": count,
                "share": count / len(examples),
            }
            for risk, count in sorted(risk_counts.items())
        },
        "smallest_class_count": min(values),
        "unsupported_or_uncertain": {
            "count": unsupported_count,
            "share": unsupported_count / len(examples),
        },
        "unsupported_or_uncertain_to_each_intent_count_ratio": {
            label: unsupported_count / counts[label] for label in labels
        },
    }


def validate_confusion_matrix(
    matrix: Any, labels: Sequence[str], expected_counts: Mapping[str, int]
) -> list[list[int]]:
    if (
        not isinstance(matrix, list)
        or len(matrix) != len(labels)
        or any(not isinstance(row, list) or len(row) != len(labels) for row in matrix)
    ):
        raise ValueError("existing OOF confusion matrix shape changed")
    validated: list[list[int]] = []
    for row in matrix:
        if any(not isinstance(value, int) or value < 0 for value in row):
            raise ValueError("existing OOF confusion matrix values changed")
        validated.append(list(row))
    if sum(sum(row) for row in validated) != EXPECTED_DEVELOPMENT_RECORD_COUNT:
        raise ValueError("existing OOF confusion matrix population changed")
    if [sum(row) for row in validated] != [expected_counts[label] for label in labels]:
        raise ValueError("existing OOF confusion matrix supports changed")
    return validated


def existing_oof_behavior(
    candidate: Mapping[str, Any],
    labels: Sequence[str],
    protected_intents: Sequence[str],
    expected_counts: Mapping[str, int],
) -> dict[str, Any]:
    aggregate = candidate["aggregate_metrics"]
    confusion = aggregate["confusion_matrix"]
    matrix = validate_confusion_matrix(confusion["values"], labels, expected_counts)
    label_index = {label: index for index, label in enumerate(labels)}
    unsupported_index = label_index["unsupported_or_uncertain"]
    protected_indices = [label_index[label] for label in protected_intents]
    non_protected_indices = [
        index for index, label in enumerate(labels) if label not in protected_intents
    ]
    ordinary_non_protected_indices = [
        index
        for index, label in enumerate(labels)
        if label not in protected_intents and label != "unsupported_or_uncertain"
    ]
    unsupported_predictions = sum(row[unsupported_index] for row in matrix)
    non_unsupported_gold_count = sum(
        sum(matrix[index]) for index in range(len(labels)) if index != unsupported_index
    )
    non_unsupported_to_unsupported = sum(
        matrix[index][unsupported_index]
        for index in range(len(labels))
        if index != unsupported_index
    )
    unsupported_to_non = sum(matrix[unsupported_index]) - matrix[unsupported_index][
        unsupported_index
    ]
    protected_exact = sum(matrix[index][index] for index in protected_indices)
    protected_other = sum(
        matrix[row][column]
        for row in protected_indices
        for column in protected_indices
        if row != column
    )
    protected_to_non = sum(
        matrix[row][column]
        for row in protected_indices
        for column in ordinary_non_protected_indices
    )
    protected_to_unsupported = sum(
        matrix[row][unsupported_index] for row in protected_indices
    )
    non_to_protected = sum(
        matrix[row][column]
        for row in non_protected_indices
        for column in protected_indices
    )
    protected_gold_count = sum(sum(matrix[index]) for index in protected_indices)
    non_protected_gold_count = sum(
        sum(matrix[index]) for index in non_protected_indices
    )
    safety = candidate["safety_metrics"]
    return {
        "balanced_accuracy": aggregate["balanced_accuracy"],
        "confusion_matrix": {
            "label_order": list(labels),
            "values": matrix,
        },
        "development_safety_metrics": {
            "exact_protected_write_recall": safety[
                "exact_protected_write_recall"
            ],
            "protected_write_false_positive_rate": safety[
                "protected_write_false_positive_rate"
            ],
            "unsupported_or_uncertain_recall": safety[
                "unsupported_or_uncertain_recall"
            ],
        },
        "historical_nine_label_macro_f1": aggregate[
            "historical_nine_label_macro_f1"
        ],
        "mean_fold_macro_f1": aggregate["mean_fold_macro_f1"],
        "new_seven_intent_macro_f1": aggregate["new_seven_intent_macro_f1"],
        "per_intent": aggregate["per_intent"],
        "pooled_oof_accuracy": aggregate["accuracy"],
        "pooled_oof_macro_f1": aggregate["pooled_oof_macro_f1"],
        "predictions_generated_by_step29b": False,
        "protected_write_confusion": {
            "gold_protected_predicted_another_protected_intent": protected_other,
            "gold_protected_predicted_exact_protected_intent": protected_exact,
            "gold_protected_predicted_non_protected_intent": protected_to_non,
            "gold_protected_predicted_unsupported_or_uncertain": (
                protected_to_unsupported
            ),
            "gold_protected_record_count": protected_gold_count,
            "non_protected_predicted_any_protected_intent": non_to_protected,
            "non_protected_predicted_any_protected_intent_rate": _ratio(
                non_to_protected, non_protected_gold_count
            ),
            "non_protected_record_count": non_protected_gold_count,
        },
        "unsupported_prediction_behavior": {
            "gold_non_unsupported_count": non_unsupported_gold_count,
            "gold_non_unsupported_to_unsupported_count": (
                non_unsupported_to_unsupported
            ),
            "gold_non_unsupported_to_unsupported_rate": _ratio(
                non_unsupported_to_unsupported, non_unsupported_gold_count
            ),
            "gold_unsupported_count": sum(matrix[unsupported_index]),
            "gold_unsupported_to_non_unsupported_count": unsupported_to_non,
            "gold_unsupported_to_non_unsupported_rate": _ratio(
                unsupported_to_non, sum(matrix[unsupported_index])
            ),
            "total_unsupported_predictions": unsupported_predictions,
            "unsupported_prediction_rate": unsupported_predictions
            / EXPECTED_DEVELOPMENT_RECORD_COUNT,
        },
        "worst_fold_macro_f1": aggregate["worst_fold_macro_f1"],
    }


def metric_variance(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        raise ValueError("fold values are required")
    return {
        "maximum": max(values),
        "mean": statistics.fmean(values),
        "minimum": min(values),
        "population_standard_deviation": statistics.pstdev(values),
        "spread": max(values) - min(values),
        "values": list(values),
    }


def fold_variance(candidate: Mapping[str, Any]) -> dict[str, Any]:
    folds = sorted(candidate["fold_metrics"], key=lambda item: item["validation_fold"])
    if [fold["validation_fold"] for fold in folds] != [0, 1, 2, 3, 4]:
        raise ValueError("existing fold metric order changed")
    return {
        "accuracy": metric_variance([float(fold["accuracy"]) for fold in folds]),
        "macro_f1": metric_variance(
            [float(fold["macro_f1"]) for fold in folds]
        ),
        "standard_deviation_definition": "population_standard_deviation_ddof_0",
    }


def _top_ngrams(tokens: Sequence[str], size: int, limit: int = 25) -> set[Any]:
    counter = ngram_counter([tokens], size)
    return {
        value
        for value, _ in sorted(counter.items(), key=lambda item: (-item[1], item[0]))[
            :limit
        ]
    }


def jaccard_payload(first: set[Any], second: set[Any]) -> dict[str, int | float]:
    intersection = len(first & second)
    union = len(first | second)
    return {
        "intersection_size": intersection,
        "jaccard": _ratio(intersection, union),
        "union_size": union,
    }


def intent_boundary_diagnostics(
    examples: Sequence[Mapping[str, Any]],
    labels: Sequence[str],
    oof: Mapping[str, Any],
    focus_intents: Sequence[str],
) -> dict[str, Any]:
    if len(focus_intents) != 8 or any(label not in labels for label in focus_intents):
        raise ValueError("frozen focus-intent set changed")
    matrix = oof["confusion_matrix"]["values"]
    index = {label: position for position, label in enumerate(labels)}
    focus_matrix = [
        [matrix[index[row]][index[column]] for column in focus_intents]
        for row in focus_intents
    ]
    unsupported = "unsupported_or_uncertain"
    unsupported_index = index[unsupported]
    target_to_unsupported = {}
    unsupported_to_target = {}
    for label in focus_intents:
        if label == unsupported:
            continue
        label_index = index[label]
        target_to_unsupported[label] = {
            "count": matrix[label_index][unsupported_index],
            "denominator": sum(matrix[label_index]),
            "rate": _ratio(
                matrix[label_index][unsupported_index], sum(matrix[label_index])
            ),
        }
        unsupported_to_target[label] = {
            "count": matrix[unsupported_index][label_index],
            "denominator": sum(matrix[unsupported_index]),
            "rate": _ratio(
                matrix[unsupported_index][label_index],
                sum(matrix[unsupported_index]),
            ),
        }

    records_by_intent = {
        label: [record for record in examples if record["intent"] == label]
        for label in focus_intents
    }
    tokens_by_intent = {
        label: [
            token
            for record in records
            for token in tokenize(str(record["text"]))
        ]
        for label, records in records_by_intent.items()
    }
    normalized_by_intent = {
        label: {normalize_text(str(record["text"])) for record in records}
        for label, records in records_by_intent.items()
    }
    pairwise: list[dict[str, Any]] = []
    for first_index, first in enumerate(focus_intents):
        for second in focus_intents[first_index + 1 :]:
            first_tokens = tokens_by_intent[first]
            second_tokens = tokens_by_intent[second]
            first_token_set = set(first_tokens)
            second_token_set = set(second_tokens)
            token_overlap = jaccard_payload(first_token_set, second_token_set)
            pairwise.append(
                {
                    "first_intent": first,
                    "first_to_second_oof_confusion_count": matrix[index[first]][
                        index[second]
                    ],
                    "first_to_second_oof_confusion_rate": _ratio(
                        matrix[index[first]][index[second]],
                        sum(matrix[index[first]]),
                    ),
                    "first_unique_normalized_token_count": len(first_token_set),
                    "normalized_text_collision_count": len(
                        normalized_by_intent[first] & normalized_by_intent[second]
                    ),
                    "second_intent": second,
                    "second_to_first_oof_confusion_count": matrix[index[second]][
                        index[first]
                    ],
                    "second_to_first_oof_confusion_rate": _ratio(
                        matrix[index[second]][index[first]],
                        sum(matrix[index[second]]),
                    ),
                    "second_unique_normalized_token_count": len(second_token_set),
                    "token_set_overlap": token_overlap,
                    "top_25_bigram_overlap": jaccard_payload(
                        _top_ngrams(first_tokens, 2), _top_ngrams(second_tokens, 2)
                    ),
                    "top_25_trigram_overlap": jaccard_payload(
                        _top_ngrams(first_tokens, 3), _top_ngrams(second_tokens, 3)
                    ),
                    "top_25_unigram_overlap": jaccard_payload(
                        _top_ngrams(first_tokens, 1), _top_ngrams(second_tokens, 1)
                    ),
                }
            )
    asymmetry = {}
    for label in focus_intents:
        metrics = oof["per_intent"][label]
        difference = metrics["precision"] - metrics["recall"]
        asymmetry[label] = {
            "absolute_gap": abs(difference),
            "precision": metrics["precision"],
            "precision_minus_recall": difference,
            "recall": metrics["recall"],
        }
    return {
        "focus_confusion_matrix": {
            "label_order": list(focus_intents),
            "values": focus_matrix,
        },
        "focus_intents": list(focus_intents),
        "pairwise_diagnostics": pairwise,
        "precision_recall_asymmetry": asymmetry,
        "semantic_similarity_used": False,
        "target_to_unsupported_or_uncertain": target_to_unsupported,
        "unsupported_or_uncertain_to_target": unsupported_to_target,
    }


def effective_sample_size(
    examples: Sequence[Mapping[str, Any]], labels: Sequence[str]
) -> dict[str, Any]:
    def scope(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        raw_count = len(records)
        unique_groups = len({record["group_id"] for record in records})
        unique_exact = len({record["text"] for record in records})
        unique_normalized = len(
            {normalize_text(str(record["text"])) for record in records}
        )
        return {
            "raw_record_count": raw_count,
            "unique_exact_text_count": unique_exact,
            "unique_exact_text_to_raw_record_ratio": _ratio(
                unique_exact, raw_count
            ),
            "unique_group_count": unique_groups,
            "unique_group_to_raw_record_ratio": _ratio(unique_groups, raw_count),
            "unique_normalized_text_count": unique_normalized,
            "unique_normalized_text_to_raw_record_ratio": _ratio(
                unique_normalized, raw_count
            ),
        }

    return {
        "global": scope(examples),
        "limitation": (
            "These are evidence-independence proxies; unique groups and texts are "
            "not formal statistically independent effective sample sizes."
        ),
        "per_intent": {
            label: scope([record for record in examples if record["intent"] == label])
            for label in labels
        },
        "section_name": "evidence_independence_proxies",
    }


def historical_aggregate_context(contract: Mapping[str, Any]) -> dict[str, Any]:
    context = contract["final_result_context"]
    development = context["development_pooled_oof_macro_f1"]
    final = context["final_macro_f1"]
    return {
        "development_pooled_oof_macro_f1": development,
        "development_to_final_macro_f1_gap": development - final,
        "final_exact_protected_write_recall": context[
            "final_exact_protected_write_recall"
        ],
        "final_macro_f1": final,
        "final_new_seven_intent_macro_f1": context[
            "final_new_seven_intent_macro_f1"
        ],
        "final_protected_write_false_positive_rate": context[
            "final_protected_write_false_positive_rate"
        ],
        "final_unsupported_or_uncertain_recall": context[
            "final_unsupported_or_uncertain_recall"
        ],
        "individual_final_holdout_examples_used": False,
        "role": "historical_aggregate_context_only",
    }


def diagnostic_claims(
    historical: Mapping[str, Any],
    composition: Mapping[str, Any],
    groups: Mapping[str, Any],
    imbalance: Mapping[str, Any],
    oof: Mapping[str, Any],
) -> dict[str, Any]:
    unsupported_share = imbalance["unsupported_or_uncertain"]["share"]
    group_ratio = groups["total_unique_groups"] / composition["total_record_count"]
    focus_confusions = oof["unsupported_prediction_behavior"][
        "gold_non_unsupported_to_unsupported_count"
    ]
    return {
        "causal_claims": [],
        "hypotheses": [
            {
                "evidence_metric_refs": [
                    "class_imbalance.unsupported_or_uncertain.share",
                    "class_imbalance.largest_to_smallest_class_ratio",
                ],
                "statement": (
                    "The development class distribution may contribute to uneven "
                    "boundary learning, but this is unproven."
                ),
                "unproven": True,
            },
            {
                "evidence_metric_refs": [
                    "group_structure.total_unique_groups",
                    "development_source_composition.total_record_count",
                    "lexical_diversity.per_intent",
                ],
                "statement": (
                    "Development group or lexical concentration may indicate "
                    "narrower evidence diversity than raw row counts suggest, but "
                    "it does not establish the final-distribution mechanism."
                ),
                "unproven": True,
            },
        ],
        "observations": [
            {
                "evidence_metric_refs": [
                    "historical_aggregate_context.development_pooled_oof_macro_f1",
                    "historical_aggregate_context.final_macro_f1",
                    "historical_aggregate_context.development_to_final_macro_f1_gap",
                ],
                "statement": (
                    "The frozen development OOF and final macro-F1 values differ "
                    f"by {historical['development_to_final_macro_f1_gap']}."
                ),
            },
            {
                "evidence_metric_refs": [
                    "class_imbalance.unsupported_or_uncertain.count",
                    "class_imbalance.unsupported_or_uncertain.share",
                ],
                "statement": (
                    "unsupported_or_uncertain is the largest development class at "
                    f"share {unsupported_share}."
                ),
            },
            {
                "evidence_metric_refs": [
                    "development_source_composition.total_record_count",
                    "group_structure.total_unique_groups",
                ],
                "statement": (
                    "Development raw rows and unique groups differ; their ratio is "
                    f"{group_ratio}."
                ),
            },
        ],
        "supported_diagnoses": [
            {
                "confidence": "high",
                "evidence_metric_refs": [
                    "class_imbalance.intent_counts",
                    "class_imbalance.largest_to_smallest_class_ratio",
                    "class_imbalance.unsupported_or_uncertain.share",
                ],
                "limitation": (
                    "Class imbalance is measured development evidence, not proof "
                    "that imbalance caused the V2-C5 final result."
                ),
                "statement": "The frozen development label distribution is highly imbalanced.",
            },
            {
                "confidence": "high",
                "evidence_metric_refs": [
                    "development_source_composition.total_record_count",
                    "group_structure.total_unique_groups",
                    "effective_sample_size.global",
                ],
                "limitation": (
                    "Unique groups and normalized texts are independence proxies, "
                    "not formal effective sample-size estimates."
                ),
                "statement": (
                    "Raw development row count exceeds the count of distinct groups, "
                    "so raw rows cannot be treated as independent groups."
                ),
            },
            {
                "confidence": "high",
                "evidence_metric_refs": [
                    "existing_oof_behavior.unsupported_prediction_behavior",
                    "existing_oof_behavior.protected_write_confusion",
                ],
                "limitation": (
                    "Existing development OOF confusions characterize development "
                    "behavior only and do not expose individual final errors."
                ),
                "statement": (
                    "Existing OOF evidence contains measurable unsupported and "
                    f"protected-boundary confusions ({focus_confusions} supported "
                    "gold rows were predicted unsupported)."
                ),
            },
        ],
    }


def remediation_recommendations(
    contract: Mapping[str, Any],
) -> list[dict[str, Any]]:
    recommendations = [
        {
            "category": "rebalance_development_data",
            "evidence_metric_refs": [
                "class_imbalance.intent_counts",
                "class_imbalance.largest_to_smallest_class_ratio",
            ],
            "priority": "high",
            "what_must_be_frozen_before_implementation": (
                "A separate V2-C6 rebalancing experiment with source eligibility, "
                "sampling rules, and evaluation gates."
            ),
            "why": "Measured development class frequencies are strongly unequal.",
        },
        {
            "category": "improve_group_or_source_independence",
            "evidence_metric_refs": [
                "group_structure",
                "provenance_authoring_concentration",
                "effective_sample_size.global",
            ],
            "priority": "medium",
            "what_must_be_frozen_before_implementation": (
                "A separate V2-C6 source/group independence policy and dataset "
                "construction contract."
            ),
            "why": (
                "Group and provenance statistics should govern future evidence "
                "independence rather than relying on raw row count."
            ),
        },
        {
            "category": "improve_training_data_diversity",
            "evidence_metric_refs": [
                "lexical_diversity",
                "duplicate_structure",
                "effective_sample_size.per_intent",
            ],
            "priority": "medium",
            "what_must_be_frozen_before_implementation": (
                "A separate V2-C6 data-diversity contract with eligible sources, "
                "deduplication, and independent evaluation boundaries."
            ),
            "why": (
                "Lexical, duplicate, and independence-proxy measurements provide "
                "the evidence base for controlled diversity improvements."
            ),
        },
        {
            "category": "improve_intent_definitions_or_boundaries",
            "evidence_metric_refs": [
                "intent_boundary_diagnostics.focus_confusion_matrix",
                "intent_boundary_diagnostics.precision_recall_asymmetry",
            ],
            "priority": "high",
            "what_must_be_frozen_before_implementation": (
                "A separate human-reviewed V2-C6 boundary/remapping contract; no "
                "taxonomy change is authorized by Step 29B."
            ),
            "why": "Frozen OOF evidence permits direct measurement of focus boundaries.",
        },
        {
            "category": "hard_negative_generation",
            "evidence_metric_refs": [
                "intent_boundary_diagnostics.pairwise_diagnostics",
                "existing_oof_behavior.protected_write_confusion",
            ],
            "priority": "medium",
            "what_must_be_frozen_before_implementation": (
                "A separately reviewed hard-negative authoring and split-isolation "
                "contract."
            ),
            "why": (
                "Measured development confusions identify boundaries that could be "
                "tested by a future controlled hard-negative experiment."
            ),
        },
        {
            "category": "unsupported_class_redesign",
            "evidence_metric_refs": [
                "class_imbalance.unsupported_or_uncertain",
                "existing_oof_behavior.unsupported_prediction_behavior",
            ],
            "priority": "high",
            "what_must_be_frozen_before_implementation": (
                "A separate V2-C6 unsupported-class design and human-adjudication "
                "contract."
            ),
            "why": (
                "unsupported_or_uncertain dominates development rows and has "
                "measurable two-way OOF boundary errors."
            ),
        },
    ]
    allowed = set(contract["allowed_remediation_recommendations"])
    if any(item["category"] not in allowed for item in recommendations):
        raise ValueError("unfrozen remediation category was generated")
    return recommendations


def build_diagnosis_payload(inputs: Mapping[str, Any]) -> dict[str, Any]:
    examples = inputs["examples"]
    labels = inputs["labels"]
    contract = inputs["contract"]
    provenance = metadata_concentration(examples, labels)
    composition = development_source_composition(
        examples, labels, provenance
    )
    groups = group_structure(examples, labels)
    duplicates = duplicate_structure(examples, labels)
    lexical = lexical_diversity(examples, labels)
    imbalance = class_imbalance(examples, labels, inputs["protected_intents"])
    oof = existing_oof_behavior(
        inputs["candidate"],
        labels,
        inputs["protected_intents"],
        composition["intent_counts"],
    )
    folds = fold_variance(inputs["candidate"])
    focus_intents = question_by_id(
        contract, "intent_boundary_diagnostics"
    )["focus_intents"]
    boundaries = intent_boundary_diagnostics(
        examples, labels, oof, focus_intents
    )
    effective = effective_sample_size(examples, labels)
    historical = historical_aggregate_context(contract)
    claims = diagnostic_claims(
        historical, composition, groups, imbalance, oof
    )
    return {
        "class_imbalance": imbalance,
        "development_source_composition": composition,
        "diagnosis_completed": True,
        "diagnostic_claims": claims,
        "duplicate_structure": duplicates,
        "effective_sample_size": effective,
        "existing_oof_behavior": oof,
        "fold_variance": folds,
        "governance": {
            **GOVERNANCE_FLAGS,
            "diagnosis_completed": True,
        },
        "group_structure": groups,
        "historical_aggregate_context": historical,
        "intent_boundary_diagnostics": boundaries,
        "lexical_diversity": lexical,
        "next_required": "v2c6_remediation_design",
        "phase": "V2-C6 Step 29B",
        "provenance_authoring_concentration": provenance,
        "remediation_recommendations": remediation_recommendations(contract),
        "schema_version": RESULT_SCHEMA_VERSION,
        "selected_candidate_id": SELECTED_CANDIDATE_ID,
        "source_lineage": dict(inputs["lineage"]),
    }


def build_manifest(
    result_bytes: bytes,
    result: Mapping[str, Any],
    paths: DiagnosisPaths,
    reader: BytesReader,
) -> dict[str, Any]:
    return {
        "development_record_count": EXPECTED_DEVELOPMENT_RECORD_COUNT,
        "diagnosis_completed": True,
        "governance": dict(result["governance"]),
        "intent_count": EXPECTED_INTENT_COUNT,
        "next_required": "v2c6_remediation_design",
        "output_contains_raw_consumed_holdout_text": False,
        "phase": "V2-C6 Step 29B",
        "result": {
            "path": display_path(paths.result, paths),
            "schema_version": RESULT_SCHEMA_VERSION,
            "sha256": sha256_bytes(result_bytes),
        },
        "runner": {
            "path": display_path(paths.runner, paths),
            "sha256": sha256_file(paths.runner, paths, reader),
        },
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "selected_candidate_id": SELECTED_CANDIDATE_ID,
        "source_lineage": dict(result["source_lineage"]),
    }


def recursive_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            keys.add(str(key))
            keys.update(recursive_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.update(recursive_keys(child))
    return keys


def _resolve_reference(payload: Mapping[str, Any], reference: str) -> Any:
    value: Any = payload
    for component in reference.split("."):
        if not isinstance(value, Mapping) or component not in value:
            raise ValueError(f"diagnostic evidence reference is invalid: {reference}")
        value = value[component]
    return value


def validate_claims_and_recommendations(
    payload: Mapping[str, Any], contract: Mapping[str, Any]
) -> None:
    claims = payload.get("diagnostic_claims", {})
    if set(claims) != {
        "observations",
        "hypotheses",
        "supported_diagnoses",
        "causal_claims",
    }:
        raise ValueError("diagnostic claim sections changed")
    if claims["causal_claims"] != []:
        raise ValueError("Step 29B cannot emit unsupported causal claims")
    for observation in claims["observations"]:
        if not observation.get("statement") or not observation.get(
            "evidence_metric_refs"
        ):
            raise ValueError("observation must be evidence-backed")
        for reference in observation["evidence_metric_refs"]:
            _resolve_reference(payload, reference)
    for hypothesis in claims["hypotheses"]:
        if hypothesis.get("unproven") is not True:
            raise ValueError("hypotheses must be explicitly unproven")
        for reference in hypothesis.get("evidence_metric_refs", []):
            _resolve_reference(payload, reference)
    for diagnosis in claims["supported_diagnoses"]:
        if (
            diagnosis.get("confidence") not in {"low", "medium", "high"}
            or not diagnosis.get("limitation")
            or not diagnosis.get("evidence_metric_refs")
        ):
            raise ValueError("supported diagnosis lacks required evidence fields")
        for reference in diagnosis["evidence_metric_refs"]:
            _resolve_reference(payload, reference)
    allowed = set(contract["allowed_remediation_recommendations"])
    for recommendation in payload.get("remediation_recommendations", []):
        if (
            recommendation.get("category") not in allowed
            or recommendation.get("priority") not in {"low", "medium", "high"}
            or not recommendation.get("why")
            or not recommendation.get("what_must_be_frozen_before_implementation")
            or not recommendation.get("evidence_metric_refs")
        ):
            raise ValueError("remediation recommendation violates the contract")
        for reference in recommendation["evidence_metric_refs"]:
            _resolve_reference(payload, reference)


def validate_result_payload(
    payload: Mapping[str, Any], inputs: Mapping[str, Any]
) -> None:
    if (
        payload.get("schema_version") != RESULT_SCHEMA_VERSION
        or payload.get("phase") != "V2-C6 Step 29B"
        or payload.get("diagnosis_completed") is not True
        or payload.get("selected_candidate_id") != SELECTED_CANDIDATE_ID
        or payload.get("source_lineage") != inputs["lineage"]
        or payload.get("next_required") != "v2c6_remediation_design"
    ):
        raise ValueError("V2-C6 diagnosis result identity changed")
    expected_governance = {**GOVERNANCE_FLAGS, "diagnosis_completed": True}
    if payload.get("governance") != expected_governance:
        raise ValueError("V2-C6 diagnosis governance changed")
    if payload.get("historical_aggregate_context") != historical_aggregate_context(
        inputs["contract"]
    ):
        raise ValueError("historical aggregate context changed")
    composition = payload.get("development_source_composition", {})
    intent_counts = composition.get("intent_counts", {})
    if (
        composition.get("total_record_count") != EXPECTED_DEVELOPMENT_RECORD_COUNT
        or list(intent_counts) != list(inputs["labels"])
        or sum(intent_counts.values()) != EXPECTED_DEVELOPMENT_RECORD_COUNT
    ):
        raise ValueError("diagnosis development counts do not reconcile")
    prohibited_keys = {
        "raw_holdout_text",
        "final_holdout_examples",
        "holdout_utterance",
    }
    if recursive_keys(payload) & prohibited_keys:
        raise ValueError("diagnosis result contains prohibited final-holdout fields")
    if payload.get("duplicate_structure", {}).get("raw_duplicate_text_persisted") is not False:
        raise ValueError("diagnosis result persisted duplicate raw text")
    validate_claims_and_recommendations(payload, inputs["contract"])


def run_diagnosis(
    paths: DiagnosisPaths = DEFAULT_PATHS,
    *,
    reader: BytesReader = filesystem_reader,
) -> tuple[Path, Path]:
    inputs = load_preconditions(paths, reader=reader)
    result = build_diagnosis_payload(inputs)
    validate_result_payload(result, inputs)
    result_bytes = stable_json_bytes(result)
    manifest = build_manifest(result_bytes, result, paths, reader)
    manifest_bytes = stable_json_bytes(manifest)
    durable_create(paths.result, result_bytes)
    durable_create(paths.result_manifest, manifest_bytes)
    return paths.result, paths.result_manifest


def check_results(
    paths: DiagnosisPaths = DEFAULT_PATHS,
    *,
    reader: BytesReader = filesystem_reader,
) -> dict[str, Any]:
    inputs = load_preconditions(
        paths,
        reader=reader,
        require_outputs_absent=False,
    )
    if not paths.result.is_file() or not paths.result_manifest.is_file():
        raise FileNotFoundError("complete V2-C6 diagnosis result pair is required")
    result_bytes = guarded_read_bytes(paths.result, paths, reader)
    result = json.loads(result_bytes)
    if not isinstance(result, dict):
        raise TypeError("diagnosis result root must be an object")
    if result_bytes != stable_json_bytes(result):
        raise ValueError("diagnosis result serialization changed")
    validate_result_payload(result, inputs)
    expected_result = build_diagnosis_payload(inputs)
    if result != expected_result:
        raise ValueError("diagnosis result differs from deterministic recomputation")

    manifest_bytes = guarded_read_bytes(paths.result_manifest, paths, reader)
    manifest = json.loads(manifest_bytes)
    if not isinstance(manifest, dict):
        raise TypeError("diagnosis manifest root must be an object")
    if manifest_bytes != stable_json_bytes(manifest):
        raise ValueError("diagnosis manifest serialization changed")
    expected_manifest = build_manifest(result_bytes, result, paths, reader)
    if manifest != expected_manifest:
        raise ValueError("diagnosis manifest hash, lineage, or governance changed")
    return {
        "diagnosis_completed": True,
        "files_written": False,
        "model_inference_performed": False,
        "model_training_performed": False,
        "raw_v2c5_holdout_accessed": False,
        "status": "valid",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the deterministic V2-C6 generalization diagnosis"
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--run", action="store_true")
    modes.add_argument("--check-results", action="store_true")
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    if arguments.preflight:
        print(json.dumps(preflight(), indent=2, sort_keys=True))
    elif arguments.run:
        outputs = run_diagnosis()
        print("\n".join(str(path.relative_to(REPOSITORY_ROOT)) for path in outputs))
    elif arguments.check_results:
        print(json.dumps(check_results(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
