"""Execute the frozen V2-C6 Step 29H-A failure-analysis contract.

This module consumes existing development prediction evidence only. It cannot
access a final holdout and contains no model, embedding, or classifier path.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ML_DIRECTORY = REPOSITORY_ROOT / "data/evals/v2/ml"
SCRIPT_RELATIVE_PATH = "scripts/run_v2c6_post_selection_failure_analysis.py"

CONTRACT_SCHEMA_VERSION = "v2c6-post-selection-failure-analysis-contract.v1"
RESULT_SCHEMA_VERSION = "v2c6-post-selection-failure-analysis.v1"
MANIFEST_SCHEMA_VERSION = "v2c6-post-selection-failure-analysis-manifest.v1"
EXPECTED_CONTRACT_SHA256 = (
    "1cd4204593ca5cb51590ff8207d51134f75844ebbceb05afd2ae155298c09979"
)
UNSUPPORTED_INTENT = "unsupported_or_uncertain"
NUMERICAL_TOLERANCE = 1e-12


@dataclass(frozen=True)
class AnalysisPaths:
    repository_root: Path
    contract: Path
    step29h_results: Path
    step29h_manifest: Path
    step29g_contract: Path
    development_dataset: Path
    step29f_freeze: Path
    remediation_contract: Path
    result: Path
    result_manifest: Path
    local_review: Path
    prohibited_holdout: Path
    runner: Path


DEFAULT_PATHS = AnalysisPaths(
    repository_root=REPOSITORY_ROOT,
    contract=ML_DIRECTORY / "v2c6_post_selection_failure_analysis_contract.json",
    step29h_results=(
        ML_DIRECTORY / "v2c6_source_aware_model_selection_results.json"
    ),
    step29h_manifest=(
        ML_DIRECTORY / "v2c6_source_aware_model_selection_results.manifest.json"
    ),
    step29g_contract=(
        ML_DIRECTORY / "v2c6_source_aware_model_selection_contract.json"
    ),
    development_dataset=(
        ML_DIRECTORY / "v2c6_remediated_development_dataset.json"
    ),
    step29f_freeze=(
        ML_DIRECTORY / "v2c6_remediated_development_dataset.freeze.json"
    ),
    remediation_contract=ML_DIRECTORY / "v2c6_remediation_dataset_contract.json",
    result=ML_DIRECTORY / "v2c6_post_selection_failure_analysis.json",
    result_manifest=(
        ML_DIRECTORY / "v2c6_post_selection_failure_analysis.manifest.json"
    ),
    local_review=(
        ML_DIRECTORY / "local/v2c6_post_selection_error_review.csv"
    ),
    prohibited_holdout=ML_DIRECTORY / "v2c5_final_holdout.json",
    runner=REPOSITORY_ROOT / SCRIPT_RELATIVE_PATH,
)


@dataclass(frozen=True)
class AnalysisInputs:
    contract: dict[str, Any]
    contract_sha256: str
    results: dict[str, Any]
    results_manifest: dict[str, Any]
    step29g_contract: dict[str, Any]
    freeze: dict[str, Any]
    dataset: dict[str, Any]
    records_by_id: dict[str, dict[str, Any]]
    remediation_ids: frozenset[str]
    candidate_results: dict[str, dict[str, Any]]
    coverage: dict[str, Any]


def stable_json_bytes(payload: Any) -> bytes:
    serialized = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
    return bytes(serialized + "\n", "utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def display_path(path: Path, paths: AnalysisPaths) -> str:
    try:
        return str(path.resolve().relative_to(paths.repository_root.resolve()))
    except ValueError:
        return str(path.resolve())


def guard_final_holdout(path: Path, paths: AnalysisPaths) -> None:
    resolved = path.resolve()
    if resolved == paths.prohibited_holdout.resolve():
        raise PermissionError("final holdout access is prohibited")
    if (
        resolved.is_relative_to((paths.repository_root / "data/evals/v2/ml").resolve())
        and "final_holdout" in resolved.name
    ):
        raise PermissionError("final holdout access is prohibited")


def read_bytes(path: Path, paths: AnalysisPaths) -> bytes:
    guard_final_holdout(path, paths)
    return path.read_bytes()


def sha256_file(path: Path, paths: AnalysisPaths) -> str:
    return sha256_bytes(read_bytes(path, paths))


def read_json_object(path: Path, paths: AnalysisPaths) -> dict[str, Any]:
    value = json.loads(read_bytes(path, paths))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


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
            raise FileExistsError(
                f"refusing to overwrite existing file: {path}"
            ) from exc
        temporary_path.unlink()
        fsync_directory(path.parent)
    finally:
        temporary_path.unlink(missing_ok=True)


def validate_contract(contract: Mapping[str, Any], paths: AnalysisPaths) -> None:
    if (
        contract.get("schema_version") != CONTRACT_SCHEMA_VERSION
        or contract.get("contract_version") != CONTRACT_SCHEMA_VERSION
        or contract.get("phase") != "V2-C6 Step 29H-A"
    ):
        raise ValueError("unexpected Step 29H-A contract identity")
    status = contract.get("contract_status", {})
    required_false = (
        "candidate_search_performed",
        "dataset_mutated",
        "embeddings_generated",
        "failure_analysis_executed",
        "final_holdout_accessed",
        "model_fitting_performed",
        "model_inference_performed",
        "model_selection_performed",
        "remediation_chosen",
        "runtime_behavior_changed",
        "step29i_authorized",
        "taxonomy_mutated",
        "threshold_tuning_performed",
    )
    if status.get("contract_frozen") is not True or any(
        status.get(field) is not False for field in required_false
    ):
        raise ValueError("Step 29H-A contract status changed")
    if status.get("next_required") != (
        "v2c6_post_selection_failure_analysis_execution"
    ):
        raise ValueError("Step 29H-A next_required changed")

    outputs = contract.get("execution_artifact_contract", {}).get(
        "tracked_outputs"
    )
    if outputs != [
        "data/evals/v2/ml/v2c6_post_selection_failure_analysis.json",
        (
            "data/evals/v2/ml/"
            "v2c6_post_selection_failure_analysis.manifest.json"
        ),
    ]:
        raise ValueError("Step 29H-B output paths changed")
    configured_outputs = [
        display_path(paths.result, paths),
        display_path(paths.result_manifest, paths),
    ]
    if outputs != configured_outputs:
        raise ValueError("Step 29H-B configured output paths changed")

    local = contract.get("error_record_schema", {}).get(
        "optional_local_text_review", {}
    )
    if local.get("path") != display_path(paths.local_review, paths):
        raise ValueError("local review path changed")
    if (
        local.get("tracked") is not False
        or local.get("ignored_by_repository") is not True
        or local.get("creation_is_optional_and_separate") is not True
    ):
        raise ValueError("local review governance changed")

    fp = contract.get("protected_false_positive_analysis", {})
    if fp.get("protected_intents") != [
        "cancel_transfer",
        "close_account",
        "create_dispute",
        "freeze_card",
    ]:
        raise ValueError("protected intent set changed")
    if fp.get("required_scopes") != [
        "pooled_group_aware_cv",
        "pooled_source_family_holdout",
        "v2c6_sf1_definition_direct",
        "v2c6_sf2_scenario_narrative",
        "v2c6_sf3_boundary_conversational",
    ]:
        raise ValueError("required analysis scopes changed")
    allowance = fp.get("allowed_count_calculation", {})
    if allowance != {
        "comparison": "less_than_or_equal",
        "excess_false_positive_count_formula": (
            "max(0, observed_protected_false_positive_count - "
            "maximum_passing_false_positive_count)"
        ),
        "maximum_passing_false_positive_count_formula": (
            "floor(non_protected_denominator * 1 / 100)"
        ),
        "threshold": 0.01,
        "threshold_source": (
            "Step 29G protected_false_positive_rate safety gate"
        ),
    }:
        raise ValueError("protected false-positive allowance changed")

    coverage = contract.get("candidate_coverage", {})
    if coverage.get("candidate_count") != 6:
        raise ValueError("exactly six candidates are required")
    families = contract.get("source_family_analysis", {}).get(
        "required_family_ids"
    )
    if families != [
        "v2c6_sf1_definition_direct",
        "v2c6_sf2_scenario_narrative",
        "v2c6_sf3_boundary_conversational",
    ]:
        raise ValueError("source-family coverage changed")
    if len(contract.get("hard_negative_analysis", {}).get("required_pairs", [])) != 10:
        raise ValueError("exactly ten hard-negative pairs are required")

    prohibited = set(contract.get("prohibited_operations", []))
    required_prohibitions = {
        "embedding_generation",
        "model_fitting",
        "classifier_inference",
        "threshold_tuning",
        "candidate_search",
        "hyperparameter_search",
        "candidate_reranking",
        "model_selection",
        "dataset_mutation",
        "label_mutation",
        "taxonomy_mutation",
        "runtime_change",
        "final_holdout_access",
        "future_final_holdout_authoring",
        "safety_gate_weakening",
        "new_safety_threshold_creation",
        "remediation_implementation",
    }
    if not required_prohibitions.issubset(prohibited):
        raise ValueError("Step 29H-A prohibited operations changed")


def validate_binding(
    path: Path,
    specification: Mapping[str, Any],
    paths: AnalysisPaths,
    label: str,
) -> None:
    if specification.get("path") != display_path(path, paths):
        raise ValueError(f"{label} path identity mismatch")
    actual = sha256_file(path, paths)
    if specification.get("sha256") != actual:
        raise ValueError(f"{label} SHA-256 mismatch: {actual}")


def validate_input_state(
    contract: Mapping[str, Any],
    results: Mapping[str, Any],
    manifest: Mapping[str, Any],
    step29g: Mapping[str, Any],
    freeze: Mapping[str, Any],
) -> None:
    sources = contract["source_artifacts"]
    result_spec = sources["step29h_results"]
    if results.get("schema_version") != result_spec["schema_version"]:
        raise ValueError("Step 29H results schema changed")
    if results.get("execution_status") != result_spec["expected_execution_status"]:
        raise ValueError("Step 29H execution is not completed")
    if results.get("candidate_count") != 6 or results.get(
        "completed_candidate_count"
    ) != 6:
        raise ValueError("Step 29H must contain six completed candidates")
    selection = results.get("selection", {})
    if (
        selection.get("eligible_candidate_count") != 0
        or selection.get("selection_status") != "NO_ACCEPTABLE_CANDIDATE"
        or selection.get("selected_candidate") is not None
        or selection.get("selected_candidate_id") is not None
        or selection.get("winner_forced") is not False
        or selection.get("gates_weakened") is not False
    ):
        raise ValueError("Step 29H failure state changed")
    candidates = results.get("candidate_results")
    if not isinstance(candidates, list) or len(candidates) != 6:
        raise ValueError("Step 29H candidate results are incomplete")
    if any(
        candidate.get("execution_status") != "COMPLETED"
        or candidate.get("eligible") is not False
        for candidate in candidates
    ):
        raise ValueError("Step 29H candidates must be completed and ineligible")
    expected_ids = contract["candidate_coverage"]["candidate_ids"]
    if [candidate.get("candidate_id") for candidate in candidates] != expected_ids:
        raise ValueError("Step 29H candidate coverage changed")

    manifest_spec = sources["step29h_results_manifest"]
    if manifest.get("schema_version") != manifest_spec["schema_version"]:
        raise ValueError("Step 29H manifest schema changed")
    if manifest.get("selection_status") != "NO_ACCEPTABLE_CANDIDATE":
        raise ValueError("Step 29H manifest selection state changed")
    if manifest.get("results", {}).get("sha256") != result_spec["sha256"]:
        raise ValueError("Step 29H result/manifest binding changed")
    if manifest.get("final_holdout_accessed") is not False:
        raise ValueError("Step 29H manifest reports final-holdout access")

    if (
        step29g.get("schema_version")
        != sources["step29g_contract"]["schema_version"]
        or step29g.get("phase") != "V2-C6 Step 29G"
    ):
        raise ValueError("Step 29G contract identity changed")
    if freeze.get("freeze_status") != sources["step29f_freeze"][
        "expected_freeze_status"
    ]:
        raise ValueError("Step 29F freeze status changed")
    if freeze.get("frozen_dataset_sha256") != sources["development_dataset"][
        "sha256"
    ]:
        raise ValueError("Step 29F dataset binding changed")


def validate_dataset(
    contract: Mapping[str, Any], dataset: Mapping[str, Any]
) -> tuple[dict[str, dict[str, Any]], frozenset[str]]:
    examples = dataset.get("examples")
    expected_count = contract["source_artifacts"]["development_dataset"][
        "record_count"
    ]
    if not isinstance(examples, list) or len(examples) != expected_count:
        raise ValueError("frozen development record count changed")
    records: dict[str, dict[str, Any]] = {}
    remediation_ids: set[str] = set()
    families = set(contract["source_family_analysis"]["required_family_ids"])
    family_counts: Counter[str] = Counter()
    for row in examples:
        if not isinstance(row, dict):
            raise TypeError("development record must be an object")
        example_id = row.get("example_id")
        if not isinstance(example_id, str) or not example_id:
            raise ValueError("development example ID must be nonempty")
        if example_id in records:
            raise ValueError(f"duplicate development example ID: {example_id}")
        if not isinstance(row.get("intent"), str):
            raise TypeError(f"missing development intent: {example_id}")
        records[example_id] = row
        family = row.get("source_family_id")
        if family is not None:
            if family not in families or row.get("record_id") != example_id:
                raise ValueError(f"invalid remediation provenance: {example_id}")
            required_metadata = {
                "boundary_target",
                "is_hard_negative",
                "unsupported_subtype",
                "text",
            }
            if not required_metadata.issubset(row):
                raise ValueError(f"missing remediation metadata: {example_id}")
            remediation_ids.add(example_id)
            family_counts[str(family)] += 1
    if len(remediation_ids) != 810:
        raise ValueError("exactly 810 remediation records are required")
    if family_counts != Counter({family: 270 for family in families}):
        raise ValueError("each source family must contain exactly 270 records")
    return records, frozenset(remediation_ids)


def validate_prediction_rows(
    rows: Any,
    *,
    expected_ids: set[str] | frozenset[str],
    records: Mapping[str, Mapping[str, Any]],
    candidate_id: str,
    scope: str,
    source_family: str | None = None,
    allowed_intents: set[str] | None = None,
) -> None:
    if not isinstance(rows, list) or len(rows) != len(expected_ids):
        raise ValueError(f"prediction count mismatch: {candidate_id}:{scope}")
    observed: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise TypeError("prediction row must be an object")
        example_id = row.get("example_id")
        if example_id in observed:
            raise ValueError(
                f"duplicate prediction: {candidate_id}:{scope}:{example_id}"
            )
        if example_id not in expected_ids or example_id not in records:
            raise ValueError(f"unexpected prediction ID: {candidate_id}:{scope}")
        observed.add(str(example_id))
        record = records[str(example_id)]
        if row.get("gold_intent") != record.get("intent"):
            raise ValueError(f"prediction gold intent mismatch: {example_id}")
        if not isinstance(row.get("predicted_intent"), str):
            raise TypeError(f"missing predicted intent: {example_id}")
        if allowed_intents is not None and row.get("predicted_intent") not in (
            allowed_intents
        ):
            raise ValueError(f"prediction outside frozen taxonomy: {example_id}")
        if source_family is not None:
            if row.get("held_out_source_family_id") != source_family:
                raise ValueError(f"prediction source family mismatch: {example_id}")
            if record.get("source_family_id") != source_family:
                raise ValueError(f"dataset source family mismatch: {example_id}")
    if observed != set(expected_ids):
        raise ValueError(f"missing prediction IDs: {candidate_id}:{scope}")


def validate_prediction_coverage(
    contract: Mapping[str, Any],
    results: Mapping[str, Any],
    records: Mapping[str, Mapping[str, Any]],
    remediation_ids: frozenset[str],
    allowed_intents: set[str],
) -> dict[str, Any]:
    all_ids = set(records)
    families = contract["source_family_analysis"]["required_family_ids"]
    family_ids = {
        family: {
            example_id
            for example_id in remediation_ids
            if records[example_id].get("source_family_id") == family
        }
        for family in families
    }
    for candidate in results["candidate_results"]:
        candidate_id = str(candidate["candidate_id"])
        group_rows = candidate.get("group_aware_cv", {}).get("predictions")
        if not isinstance(group_rows, list) or any(
            row.get("evaluation_scope") != "pooled_group_aware_cv"
            for row in group_rows
        ):
            raise ValueError(f"group-CV prediction scope changed: {candidate_id}")
        validate_prediction_rows(
            group_rows,
            expected_ids=all_ids,
            records=records,
            candidate_id=candidate_id,
            scope="pooled_group_aware_cv",
            allowed_intents=allowed_intents,
        )
        source_rows = candidate.get("source_family_holdout", {}).get(
            "pooled_predictions"
        )
        if not isinstance(source_rows, list) or any(
            row.get("held_out_source_family_id") not in family_ids
            for row in source_rows
        ):
            raise ValueError(f"source-family prediction scope changed: {candidate_id}")
        validate_prediction_rows(
            source_rows,
            expected_ids=remediation_ids,
            records=records,
            candidate_id=candidate_id,
            scope="pooled_source_family_holdout",
            allowed_intents=allowed_intents,
        )
        for family in families:
            selected = [
                row
                for row in source_rows
                if row.get("held_out_source_family_id") == family
            ]
            validate_prediction_rows(
                selected,
                expected_ids=family_ids[family],
                records=records,
                candidate_id=candidate_id,
                scope=family,
                source_family=family,
                allowed_intents=allowed_intents,
            )
    return {
        "candidate_count": 6,
        "group_cv_predictions_per_candidate": 9008,
        "group_cv_prediction_count_total": 54048,
        "source_family_predictions_per_candidate": 810,
        "source_family_prediction_count_total": 4860,
        "source_family_predictions_per_candidate_and_family": 270,
        "candidate_family_scope_count": 18,
        "duplicate_prediction_count": 0,
        "missing_prediction_count": 0,
    }


def validate_hard_negative_metadata(
    contract: Mapping[str, Any],
    records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    required_pairs = contract["hard_negative_analysis"]["required_pairs"]
    families = contract["source_family_analysis"]["required_family_ids"]
    resolved: dict[str, dict[str, int]] = {}
    for family in families:
        counts: dict[str, int] = {}
        family_rows = [
            row
            for row in records.values()
            if row.get("source_family_id") == family
            and row.get("is_hard_negative") is True
        ]
        for pair in required_pairs:
            pair_key = "|".join(pair)
            count = sum(
                {row.get("intent"), row.get("boundary_target")} == set(pair)
                for row in family_rows
            )
            if count <= 0:
                raise ValueError(
                    f"hard-negative pair cannot be resolved: {family}:{pair}"
                )
            counts[pair_key] = count
        resolved[family] = counts
    return {
        "pair_count": len(required_pairs),
        "candidate_family_pair_analysis_count": 6 * len(families) * len(required_pairs),
        "record_counts_by_family_and_pair": resolved,
    }


def load_preconditions(
    paths: AnalysisPaths = DEFAULT_PATHS,
    *,
    require_outputs_absent: bool,
) -> AnalysisInputs:
    contract_sha256 = sha256_file(paths.contract, paths)
    if contract_sha256 != EXPECTED_CONTRACT_SHA256:
        raise ValueError(f"Step 29H-A contract SHA-256 mismatch: {contract_sha256}")
    contract = read_json_object(paths.contract, paths)
    validate_contract(contract, paths)
    sources = contract["source_artifacts"]
    bindings = (
        (paths.step29h_results, sources["step29h_results"], "Step 29H results"),
        (
            paths.step29h_manifest,
            sources["step29h_results_manifest"],
            "Step 29H manifest",
        ),
        (paths.step29g_contract, sources["step29g_contract"], "Step 29G contract"),
        (
            paths.development_dataset,
            sources["development_dataset"],
            "development dataset",
        ),
        (paths.step29f_freeze, sources["step29f_freeze"], "Step 29F freeze"),
    )
    for path, specification, label in bindings:
        validate_binding(path, specification, paths, label)

    results = read_json_object(paths.step29h_results, paths)
    results_manifest = read_json_object(paths.step29h_manifest, paths)
    step29g = read_json_object(paths.step29g_contract, paths)
    freeze = read_json_object(paths.step29f_freeze, paths)
    dataset = read_json_object(paths.development_dataset, paths)
    validate_input_state(contract, results, results_manifest, step29g, freeze)
    records, remediation_ids = validate_dataset(contract, dataset)
    coverage = validate_prediction_coverage(
        contract,
        results,
        records,
        remediation_ids,
        set(step29g["taxonomy"]["intent_label_order"]),
    )
    coverage["hard_negative_metadata"] = validate_hard_negative_metadata(
        contract, records
    )
    if require_outputs_absent and (
        paths.result.exists() or paths.result_manifest.exists()
    ):
        raise FileExistsError("Step 29H-B tracked output already exists")
    candidate_results = {
        str(candidate["candidate_id"]): candidate
        for candidate in results["candidate_results"]
    }
    return AnalysisInputs(
        contract=contract,
        contract_sha256=contract_sha256,
        results=results,
        results_manifest=results_manifest,
        step29g_contract=step29g,
        freeze=freeze,
        dataset=dataset,
        records_by_id=records,
        remediation_ids=remediation_ids,
        candidate_results=candidate_results,
        coverage=coverage,
    )


def ratio(numerator: int, denominator: int, metric: str) -> float:
    if denominator <= 0:
        raise ValueError(f"required denominator is zero: {metric}")
    return numerator / denominator


def maximum_passing_false_positive_count(denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("non-protected denominator must be positive")
    return denominator // 100


def _source_family(scope: str, contract: Mapping[str, Any]) -> str | None:
    if scope in contract["source_family_analysis"]["required_family_ids"]:
        return scope
    return None


def prediction_scopes(
    candidate: Mapping[str, Any], contract: Mapping[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    group = list(candidate["group_aware_cv"]["predictions"])
    source = list(candidate["source_family_holdout"]["pooled_predictions"])
    scopes = {
        "pooled_group_aware_cv": group,
        "pooled_source_family_holdout": source,
    }
    for family in contract["source_family_analysis"]["required_family_ids"]:
        scopes[family] = [
            row
            for row in source
            if row["held_out_source_family_id"] == family
        ]
    return scopes


def compute_safety(
    rows: Sequence[Mapping[str, Any]], protected_intents: Sequence[str]
) -> dict[str, float | int]:
    protected = set(protected_intents)
    protected_gold = sum(row["gold_intent"] in protected for row in rows)
    predicted_protected = sum(
        row["gold_intent"] in protected and row["predicted_intent"] in protected
        for row in rows
    )
    non_protected_gold = len(rows) - protected_gold
    false_positives = sum(
        row["gold_intent"] not in protected
        and row["predicted_intent"] in protected
        for row in rows
    )
    unsupported_gold = sum(
        row["gold_intent"] == UNSUPPORTED_INTENT for row in rows
    )
    unsupported_correct = sum(
        row["gold_intent"] == UNSUPPORTED_INTENT
        and row["predicted_intent"] == UNSUPPORTED_INTENT
        for row in rows
    )
    return {
        "protected_gold_count": protected_gold,
        "protected_prediction_as_protected_count": predicted_protected,
        "protected_recall": ratio(
            predicted_protected, protected_gold, "protected_recall"
        ),
        "non_protected_gold_count": non_protected_gold,
        "protected_false_positive_count": false_positives,
        "protected_false_positive_rate": ratio(
            false_positives,
            non_protected_gold,
            "protected_false_positive_rate",
        ),
        "unsupported_gold_count": unsupported_gold,
        "unsupported_correct_count": unsupported_correct,
        "unsupported_recall": ratio(
            unsupported_correct, unsupported_gold, "unsupported_recall"
        ),
    }


def assert_metric_equal(observed: float, expected: Any, label: str) -> None:
    if not isinstance(expected, (int, float)) or isinstance(expected, bool):
        raise TypeError(f"stored metric is not numeric: {label}")
    if not math.isclose(
        float(observed),
        float(expected),
        rel_tol=0.0,
        abs_tol=NUMERICAL_TOLERANCE,
    ):
        raise ValueError(f"recomputed metric mismatch: {label}")


def protected_fp_summary(
    rows: Sequence[Mapping[str, Any]], protected_intents: Sequence[str]
) -> dict[str, float | int]:
    safety = compute_safety(rows, protected_intents)
    denominator = int(safety["non_protected_gold_count"])
    false_positives = int(safety["protected_false_positive_count"])
    maximum = maximum_passing_false_positive_count(denominator)
    return {
        "non_protected_gold_count": denominator,
        "non_protected_denominator": denominator,
        "protected_false_positive_count": false_positives,
        "protected_false_positive_rate": safety[
            "protected_false_positive_rate"
        ],
        "threshold": 0.01,
        "maximum_passing_false_positive_count": maximum,
        "excess_false_positive_count": max(0, false_positives - maximum),
    }


def protected_fp_confusions(
    rows: Sequence[Mapping[str, Any]],
    protected_intents: Sequence[str],
    *,
    candidate_id: str,
    evaluation_scope: str,
    source_family_id: str | None,
) -> list[dict[str, Any]]:
    protected = set(protected_intents)
    supports = Counter(str(row["gold_intent"]) for row in rows)
    counts = Counter(
        (str(row["gold_intent"]), str(row["predicted_intent"]))
        for row in rows
        if row["gold_intent"] not in protected
        and row["predicted_intent"] in protected
    )
    result = [
        {
            "gold_intent": gold,
            "predicted_protected_intent": predicted,
            "count": count,
            "gold_intent_support": supports[gold],
            "rate_among_gold_intent": count / supports[gold],
            "rate_within_gold_intent": count / supports[gold],
            "candidate_id": candidate_id,
            "evaluation_scope": evaluation_scope,
            "source_family_id": source_family_id,
        }
        for (gold, predicted), count in counts.items()
    ]
    return sorted(
        result,
        key=lambda row: (
            -row["count"],
            row["gold_intent"],
            row["predicted_protected_intent"],
        ),
    )


def unsupported_analysis(
    rows: Sequence[Mapping[str, Any]],
    protected_intents: Sequence[str],
    *,
    candidate_id: str,
    evaluation_scope: str,
    source_family_id: str | None,
) -> dict[str, Any]:
    protected = set(protected_intents)
    unsupported_rows = [
        row for row in rows if row["gold_intent"] == UNSUPPORTED_INTENT
    ]
    denominator = len(unsupported_rows)
    correct = sum(
        row["predicted_intent"] == UNSUPPORTED_INTENT for row in unsupported_rows
    )
    to_protected = sum(
        row["predicted_intent"] in protected for row in unsupported_rows
    )
    to_other = sum(
        row["predicted_intent"] != UNSUPPORTED_INTENT
        and row["predicted_intent"] not in protected
        for row in unsupported_rows
    )
    confusions = Counter(
        str(row["predicted_intent"])
        for row in unsupported_rows
        if row["predicted_intent"] != UNSUPPORTED_INTENT
    )
    confusion_rows = [
        {
            "predicted_intent": predicted,
            "count": count,
            "rate": count / denominator,
            "protected_prediction": predicted in protected,
            "candidate_id": candidate_id,
            "evaluation_scope": evaluation_scope,
            "source_family_id": source_family_id,
        }
        for predicted, count in confusions.items()
    ]
    confusion_rows.sort(key=lambda row: (-row["count"], row["predicted_intent"]))
    return {
        "candidate_id": candidate_id,
        "evaluation_scope": evaluation_scope,
        "source_family_id": source_family_id,
        "unsupported_gold_count": denominator,
        "unsupported_denominator": denominator,
        "unsupported_correct_count": correct,
        "unsupported_recall": ratio(correct, denominator, "unsupported_recall"),
        "unsupported_to_protected_count": to_protected,
        "unsupported_to_protected_rate": ratio(
            to_protected, denominator, "unsupported_to_protected_rate"
        ),
        "unsupported_to_other_supported_count": to_other,
        "unsupported_to_other_supported_rate": ratio(
            to_other, denominator, "unsupported_to_other_supported_rate"
        ),
        "confusion_rows": confusion_rows,
    }


def protected_recall_analysis(
    rows: Sequence[Mapping[str, Any]],
    protected_intents: Sequence[str],
    *,
    candidate_id: str,
    evaluation_scope: str,
    source_family_id: str | None,
) -> dict[str, Any]:
    protected = set(protected_intents)
    protected_rows = [row for row in rows if row["gold_intent"] in protected]
    supports = Counter(str(row["gold_intent"]) for row in protected_rows)
    misses = Counter(
        (str(row["gold_intent"]), str(row["predicted_intent"]))
        for row in protected_rows
        if row["predicted_intent"] not in protected
    )
    exact_errors = Counter(
        (str(row["gold_intent"]), str(row["predicted_intent"]))
        for row in protected_rows
        if row["predicted_intent"] in protected
        and row["predicted_intent"] != row["gold_intent"]
    )

    def rows_for(
        counts: Counter[tuple[str, str]], *, recall_failure: bool
    ) -> list[dict[str, Any]]:
        result = [
            {
                "gold_protected_intent": gold,
                "predicted_intent": predicted,
                "predicted_non_protected_intent": (
                    predicted if recall_failure else None
                ),
                "count": count,
                "gold_intent_support": supports[gold],
                "rate_within_gold_protected_intent": count / supports[gold],
                "candidate_id": candidate_id,
                "evaluation_scope": evaluation_scope,
                "source_family_id": source_family_id,
                "protected_recall_failure": recall_failure,
            }
            for (gold, predicted), count in counts.items()
        ]
        return sorted(
            result,
            key=lambda row: (
                -row["count"],
                row["gold_protected_intent"],
                row["predicted_intent"],
            ),
        )

    miss_count = sum(misses.values())
    protected_count = len(protected_rows)
    return {
        "candidate_id": candidate_id,
        "evaluation_scope": evaluation_scope,
        "source_family_id": source_family_id,
        "protected_gold_count": protected_count,
        "protected_recall_miss_count": miss_count,
        "protected_recall": 1.0
        - ratio(miss_count, protected_count, "protected_recall_miss_rate"),
        "protected_recall_miss_rows": rows_for(misses, recall_failure=True),
        "protected_to_different_protected_exact_intent_error_count": sum(
            exact_errors.values()
        ),
        "protected_to_different_protected_exact_intent_error_rows": rows_for(
            exact_errors, recall_failure=False
        ),
    }


def supported_boundary_metrics(
    rows: Sequence[Mapping[str, Any]], supported_intents: Sequence[str]
) -> dict[str, float | int]:
    supported = set(supported_intents) - {UNSUPPORTED_INTENT}
    supported_rows = [row for row in rows if row["gold_intent"] in supported]
    unsupported_rows = [
        row for row in rows if row["gold_intent"] == UNSUPPORTED_INTENT
    ]
    supported_to_unsupported = sum(
        row["predicted_intent"] == UNSUPPORTED_INTENT for row in supported_rows
    )
    unsupported_to_supported = sum(
        row["predicted_intent"] != UNSUPPORTED_INTENT for row in unsupported_rows
    )
    return {
        "supported_gold_count": len(supported_rows),
        "supported_predicted_unsupported_count": supported_to_unsupported,
        "supported_to_unsupported_rate": ratio(
            supported_to_unsupported,
            len(supported_rows),
            "supported_to_unsupported_rate",
        ),
        "unsupported_gold_count": len(unsupported_rows),
        "unsupported_predicted_supported_count": unsupported_to_supported,
        "unsupported_to_supported_rate": ratio(
            unsupported_to_supported,
            len(unsupported_rows),
            "unsupported_to_supported_rate",
        ),
    }


def stored_family_metrics(
    candidate: Mapping[str, Any], family: str
) -> Mapping[str, Any]:
    matches = [
        row["metrics"]
        for row in candidate["source_family_holdout"]["rounds"]
        if row["held_out_source_family_id"] == family
    ]
    if len(matches) != 1:
        raise ValueError(f"missing or duplicate stored source-family metrics: {family}")
    return matches[0]


def validate_recomputed_metrics(
    candidate: Mapping[str, Any],
    scopes: Mapping[str, Sequence[Mapping[str, Any]]],
    contract: Mapping[str, Any],
) -> None:
    protected = contract["protected_false_positive_analysis"]["protected_intents"]
    primary = candidate["source_family_holdout"]["pooled_metrics"][
        "confusion_matrix"
    ]["label_order"]
    stored_group = candidate["group_aware_cv"]["metrics"]["safety"]
    stored_source = candidate["source_family_holdout"]["pooled_metrics"]
    for scope, stored in (
        ("pooled_group_aware_cv", stored_group),
        ("pooled_source_family_holdout", stored_source["safety"]),
    ):
        recomputed = compute_safety(scopes[scope], protected)
        for metric, value in recomputed.items():
            assert_metric_equal(value, stored[metric], f"{scope}:{metric}")
    recomputed_boundary = supported_boundary_metrics(
        scopes["pooled_source_family_holdout"], primary
    )
    for metric, value in recomputed_boundary.items():
        assert_metric_equal(
            value,
            stored_source["unsupported_boundary"][metric],
            f"pooled_source_family_holdout:{metric}",
        )
    for family in contract["source_family_analysis"]["required_family_ids"]:
        stored = stored_family_metrics(candidate, family)
        safety = compute_safety(scopes[family], protected)
        boundary = supported_boundary_metrics(scopes[family], primary)
        for metric, value in safety.items():
            assert_metric_equal(value, stored["safety"][metric], f"{family}:{metric}")
        for metric, value in boundary.items():
            assert_metric_equal(
                value,
                stored["unsupported_boundary"][metric],
                f"{family}:{metric}",
            )


def source_family_summary(
    candidate: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    *,
    candidate_id: str,
    family: str,
) -> dict[str, Any]:
    protected = contract["protected_false_positive_analysis"]["protected_intents"]
    stored = stored_family_metrics(candidate, family)
    safety = compute_safety(rows, protected)
    allowance = maximum_passing_false_positive_count(
        int(safety["non_protected_gold_count"])
    )
    unsupported = unsupported_analysis(
        rows,
        protected,
        candidate_id=candidate_id,
        evaluation_scope=family,
        source_family_id=family,
    )
    recall = protected_recall_analysis(
        rows,
        protected,
        candidate_id=candidate_id,
        evaluation_scope=family,
        source_family_id=family,
    )
    fp_rows = protected_fp_confusions(
        rows,
        protected,
        candidate_id=candidate_id,
        evaluation_scope=family,
        source_family_id=family,
    )
    return {
        "candidate_id": candidate_id,
        "source_family_id": family,
        "record_count": len(rows),
        "primary_8_macro_f1": stored["primary_8_macro_f1"],
        "protected_gold_count": safety["protected_gold_count"],
        "protected_recall": safety["protected_recall"],
        "non_protected_gold_count": safety["non_protected_gold_count"],
        "protected_false_positive_count": safety[
            "protected_false_positive_count"
        ],
        "protected_false_positive_rate": safety[
            "protected_false_positive_rate"
        ],
        "unsupported_gold_count": safety["unsupported_gold_count"],
        "unsupported_recall": safety["unsupported_recall"],
        "supported_to_unsupported_rate": stored["unsupported_boundary"][
            "supported_to_unsupported_rate"
        ],
        "unsupported_to_supported_rate": stored["unsupported_boundary"][
            "unsupported_to_supported_rate"
        ],
        "maximum_passing_protected_fp_count": allowance,
        "excess_protected_fp_count": max(
            0, int(safety["protected_false_positive_count"]) - allowance
        ),
        "protected_fp_confusion_pairs": fp_rows,
        "unsupported_confusion_pairs": unsupported["confusion_rows"],
        "protected_recall_miss_pairs": recall["protected_recall_miss_rows"],
    }


def group_source_delta(
    candidate: Mapping[str, Any], candidate_id: str, contract: Mapping[str, Any]
) -> dict[str, Any]:
    group = candidate["group_aware_cv"]["metrics"]
    source = candidate["source_family_holdout"]["pooled_metrics"]
    deltas = {
        metric: source["safety"][metric] - group["safety"][metric]
        for metric in contract["group_cv_vs_source_shift_delta_analysis"][
            "required_safety_deltas"
        ]
    }
    return {
        "candidate_id": candidate_id,
        "delta_direction": (
            "pooled_source_family_metric_minus_pooled_group_cv_metric"
        ),
        "signed_safety_deltas": deltas,
        "positive_protected_false_positive_rate_delta_means": (
            "higher protected false-positive rate under source-family holdout"
        ),
        "group_cv_macro_f1_16": group["macro_f1_16"],
        "source_family_pooled_primary_8_macro_f1": source[
            "pooled_primary_8_macro_f1"
        ],
        "macro_f1_comparability_warning": contract[
            "group_cv_vs_source_shift_delta_analysis"
        ]["macro_f1_comparison"]["qualification_required"],
        "causal_interpretation_claimed": False,
    }


def metric_bundle(candidate: Mapping[str, Any]) -> dict[str, float]:
    source = candidate["source_family_holdout"]["pooled_metrics"]
    return {
        "protected_recall": source["safety"]["protected_recall"],
        "protected_false_positive_rate": source["safety"][
            "protected_false_positive_rate"
        ],
        "unsupported_recall": source["safety"]["unsupported_recall"],
        "supported_to_unsupported_rate": source["unsupported_boundary"][
            "supported_to_unsupported_rate"
        ],
        "primary_8_macro_f1": source["pooled_primary_8_macro_f1"],
        "worst_family_primary_8_macro_f1": source[
            "worst_family_primary_8_macro_f1"
        ],
    }


def metric_deltas(
    comparison: Mapping[str, float],
    reference: Mapping[str, float],
    required_metrics: Sequence[str],
) -> dict[str, float]:
    return {
        metric: comparison[metric] - reference[metric]
        for metric in required_metrics
    }


def class_weight_comparisons(
    inputs: AnalysisInputs,
) -> list[dict[str, Any]]:
    definition = inputs.contract["class_weight_comparison"]
    results: list[dict[str, Any]] = []
    for pair in definition["matched_pairs"]:
        balanced_id = pair["balanced_candidate_id"]
        unweighted_id = pair["unweighted_candidate_id"]
        balanced = inputs.candidate_results[balanced_id]
        unweighted = inputs.candidate_results[unweighted_id]
        balanced_group = balanced["group_aware_cv"]["metrics"]["safety"]
        unweighted_group = unweighted["group_aware_cv"]["metrics"]["safety"]
        results.append(
            {
                "balanced_candidate_id": balanced_id,
                "unweighted_candidate_id": unweighted_id,
                "delta_direction": definition["delta_direction"],
                "pooled_source_family_deltas": metric_deltas(
                    metric_bundle(balanced),
                    metric_bundle(unweighted),
                    definition["pooled_source_family_delta_metrics"],
                ),
                "pooled_group_cv_safety_deltas": {
                    metric: balanced_group[metric] - unweighted_group[metric]
                    for metric in definition["pooled_group_cv_delta_metrics"]
                },
                "descriptive_only": True,
                "universal_effect_claimed": False,
            }
        )
    return results


def regularization_comparisons(
    inputs: AnalysisInputs,
) -> list[dict[str, Any]]:
    definition = inputs.contract["regularization_comparison"]
    results: list[dict[str, Any]] = []
    for pair in definition["matched_pairs"]:
        c1_id = pair["c1_candidate_id"]
        c4_id = pair["c4_candidate_id"]
        results.append(
            {
                "c1_candidate_id": c1_id,
                "c4_candidate_id": c4_id,
                "delta_direction": definition["delta_direction"],
                "pooled_source_family_deltas": metric_deltas(
                    metric_bundle(inputs.candidate_results[c1_id]),
                    metric_bundle(inputs.candidate_results[c4_id]),
                    definition["delta_metrics"],
                ),
                "descriptive_only": True,
                "winner_selected": False,
            }
        )
    return results


def hard_negative_row(
    rows: Sequence[Mapping[str, Any]],
    records: Mapping[str, Mapping[str, Any]],
    pair: Sequence[str],
    *,
    candidate_id: str,
    family: str,
) -> dict[str, Any]:
    side_a, side_b = pair
    selected = [
        row
        for row in rows
        if records[str(row["example_id"])].get("is_hard_negative") is True
        and {
            records[str(row["example_id"])].get("intent"),
            records[str(row["example_id"])].get("boundary_target"),
        }
        == set(pair)
    ]
    correct = sum(row["gold_intent"] == row["predicted_intent"] for row in selected)
    a_to_b = sum(
        row["gold_intent"] == side_a and row["predicted_intent"] == side_b
        for row in selected
    )
    b_to_a = sum(
        row["gold_intent"] == side_b and row["predicted_intent"] == side_a
        for row in selected
    )
    other = len(selected) - correct - a_to_b - b_to_a
    return {
        "pair": list(pair),
        "source_family_id": family,
        "candidate_id": candidate_id,
        "record_count": len(selected),
        "correct_count": correct,
        "accuracy": ratio(correct, len(selected), "hard_negative_accuracy"),
        "A_to_B_count": a_to_b,
        "B_to_A_count": b_to_a,
        "true_a_predicted_b_count": a_to_b,
        "true_b_predicted_a_count": b_to_a,
        "other_error_count": other,
    }


def aggregate_hard_negative_rows(
    rows: Sequence[Mapping[str, Any]], group_fields: Sequence[str]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(
            tuple(row[field]) if field == "pair" else row[field]
            for field in group_fields
        )
        grouped[key].append(row)
    result: list[dict[str, Any]] = []
    for key, members in grouped.items():
        record_count = sum(int(row["record_count"]) for row in members)
        output = {
            field: list(value) if field == "pair" else value
            for field, value in zip(group_fields, key, strict=True)
        }
        output.update(
            {
                "record_count": record_count,
                "correct_count": sum(int(row["correct_count"]) for row in members),
                "A_to_B_count": sum(int(row["A_to_B_count"]) for row in members),
                "B_to_A_count": sum(int(row["B_to_A_count"]) for row in members),
                "true_a_predicted_b_count": sum(
                    int(row["A_to_B_count"]) for row in members
                ),
                "true_b_predicted_a_count": sum(
                    int(row["B_to_A_count"]) for row in members
                ),
                "other_error_count": sum(
                    int(row["other_error_count"]) for row in members
                ),
            }
        )
        output["accuracy"] = output["correct_count"] / record_count
        result.append(output)
    return sorted(
        result,
        key=lambda row: tuple(
            tuple(row[field]) if field == "pair" else row[field]
            for field in group_fields
        ),
    )


def hard_negative_analysis(inputs: AnalysisInputs) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    families = inputs.contract["source_family_analysis"]["required_family_ids"]
    pairs = inputs.contract["hard_negative_analysis"]["required_pairs"]
    for candidate_id in inputs.contract["candidate_coverage"]["candidate_ids"]:
        candidate = inputs.candidate_results[candidate_id]
        source_rows = candidate["source_family_holdout"]["pooled_predictions"]
        for family in families:
            family_rows = [
                row
                for row in source_rows
                if row["held_out_source_family_id"] == family
            ]
            stored = stored_family_metrics(candidate, family)[
                "hard_negative_diagnostics"
            ]
            stored_by_pair = {
                tuple(row["pair"]): row for row in stored["pairs"]
            }
            for pair in pairs:
                computed = hard_negative_row(
                    family_rows,
                    inputs.records_by_id,
                    pair,
                    candidate_id=candidate_id,
                    family=family,
                )
                expected = stored_by_pair[tuple(pair)]
                comparisons = {
                    "record_count": "record_count",
                    "correct_count": "correct_count",
                    "accuracy": "accuracy",
                    "A_to_B_count": "true_a_predicted_b_count",
                    "B_to_A_count": "true_b_predicted_a_count",
                    "other_error_count": "other_error_count",
                }
                for computed_key, stored_key in comparisons.items():
                    assert_metric_equal(
                        computed[computed_key],
                        expected[stored_key],
                        f"hard_negative:{candidate_id}:{family}:{pair}:{computed_key}",
                    )
                rows.append(computed)
    return {
        "pair_count": 10,
        "candidate_family_pair_rows": rows,
        "by_candidate_across_source_families": aggregate_hard_negative_rows(
            rows, ["candidate_id", "pair"]
        ),
        "by_source_family_across_candidates": aggregate_hard_negative_rows(
            rows, ["source_family_id", "pair"]
        ),
        "across_all_candidates_and_source_families": (
            aggregate_hard_negative_rows(rows, ["pair"])
        ),
        "new_gate_created": False,
    }


def candidate_analyses(inputs: AnalysisInputs) -> list[dict[str, Any]]:
    protected = inputs.contract["protected_false_positive_analysis"][
        "protected_intents"
    ]
    analyses: list[dict[str, Any]] = []
    for candidate_id in inputs.contract["candidate_coverage"]["candidate_ids"]:
        candidate = inputs.candidate_results[candidate_id]
        scopes = prediction_scopes(candidate, inputs.contract)
        validate_recomputed_metrics(candidate, scopes, inputs.contract)
        fp: list[dict[str, Any]] = []
        unsupported: list[dict[str, Any]] = []
        recall: list[dict[str, Any]] = []
        for scope in inputs.contract["protected_false_positive_analysis"][
            "required_scopes"
        ]:
            family = _source_family(scope, inputs.contract)
            fp.append(
                {
                    "candidate_id": candidate_id,
                    "evaluation_scope": scope,
                    "source_family_id": family,
                    **protected_fp_summary(scopes[scope], protected),
                    "confusion_rows": protected_fp_confusions(
                        scopes[scope],
                        protected,
                        candidate_id=candidate_id,
                        evaluation_scope=scope,
                        source_family_id=family,
                    ),
                }
            )
            unsupported.append(
                unsupported_analysis(
                    scopes[scope],
                    protected,
                    candidate_id=candidate_id,
                    evaluation_scope=scope,
                    source_family_id=family,
                )
            )
            recall.append(
                protected_recall_analysis(
                    scopes[scope],
                    protected,
                    candidate_id=candidate_id,
                    evaluation_scope=scope,
                    source_family_id=family,
                )
            )
        summaries = [
            source_family_summary(
                candidate,
                scopes[family],
                inputs.contract,
                candidate_id=candidate_id,
                family=family,
            )
            for family in inputs.contract["source_family_analysis"][
                "required_family_ids"
            ]
        ]
        analyses.append(
            {
                "candidate_id": candidate_id,
                "step29h_eligible": candidate["eligible"],
                "protected_false_positive_analysis": fp,
                "unsupported_error_analysis": unsupported,
                "protected_recall_analysis": recall,
                "source_family_summaries": summaries,
                "group_cv_vs_source_shift_delta": group_source_delta(
                    candidate, candidate_id, inputs.contract
                ),
            }
        )
    return analyses


def _representation_family(candidate_id: str) -> str:
    if candidate_id.startswith("BGE_SMALL_"):
        return "BGE"
    if candidate_id.startswith("WORD_CHAR_TFIDF_"):
        return "TFIDF"
    raise ValueError(f"unknown representation family: {candidate_id}")


def recurrence_rows(
    inputs: AnalysisInputs,
    *,
    error_kind: str,
) -> list[dict[str, Any]]:
    protected = set(
        inputs.contract["protected_false_positive_analysis"]["protected_intents"]
    )
    families = inputs.contract["source_family_analysis"]["required_family_ids"]
    observations: dict[tuple[str, str], dict[str, Any]] = {}
    for candidate_id in inputs.contract["candidate_coverage"]["candidate_ids"]:
        scopes = prediction_scopes(
            inputs.candidate_results[candidate_id], inputs.contract
        )
        for scope in ["pooled_group_aware_cv", *families]:
            for row in scopes[scope]:
                gold = str(row["gold_intent"])
                predicted = str(row["predicted_intent"])
                include = False
                if error_kind == "protected_false_positive":
                    include = gold not in protected and predicted in protected
                elif error_kind == "unsupported_confusion":
                    include = (
                        gold == UNSUPPORTED_INTENT
                        and predicted != UNSUPPORTED_INTENT
                    )
                elif error_kind == "protected_recall_miss":
                    include = gold in protected and predicted not in protected
                else:
                    raise ValueError(f"unknown recurrence kind: {error_kind}")
                if not include:
                    continue
                key = (gold, predicted)
                evidence = observations.setdefault(
                    key,
                    {
                        "candidate_ids": set(),
                        "bge_candidate_ids": set(),
                        "tfidf_candidate_ids": set(),
                        "source_family_ids": set(),
                        "per_source_family_counts": Counter(),
                        "observed_in_group_cv": False,
                        "observed_in_source_family_evaluation": False,
                        "record_ids": set(),
                        "total_error_count_across_candidates": 0,
                    },
                )
                evidence["candidate_ids"].add(candidate_id)
                family_name = _representation_family(candidate_id)
                evidence[f"{family_name.lower()}_candidate_ids"].add(candidate_id)
                evidence["record_ids"].add(str(row["example_id"]))
                evidence["total_error_count_across_candidates"] += 1
                if scope == "pooled_group_aware_cv":
                    evidence["observed_in_group_cv"] = True
                else:
                    evidence["observed_in_source_family_evaluation"] = True
                    evidence["source_family_ids"].add(scope)
                    evidence["per_source_family_counts"][scope] += 1
    result: list[dict[str, Any]] = []
    for (gold, predicted), evidence in observations.items():
        intent_fields = {
            "gold_intent": gold,
            "predicted_intent": predicted,
        }
        if error_kind == "protected_false_positive":
            intent_fields.update(
                {
                    "gold_non_protected_intent": gold,
                    "predicted_protected_intent": predicted,
                }
            )
        result.append(
            {
                **intent_fields,
                "candidate_count_with_confusion": len(evidence["candidate_ids"]),
                "candidate_count": len(evidence["candidate_ids"]),
                "total_error_count_across_candidates": evidence[
                    "total_error_count_across_candidates"
                ],
                "bge_candidate_count_with_confusion": len(
                    evidence["bge_candidate_ids"]
                ),
                "bge_candidate_count": len(evidence["bge_candidate_ids"]),
                "tfidf_candidate_count_with_confusion": len(
                    evidence["tfidf_candidate_ids"]
                ),
                "tfidf_candidate_count": len(evidence["tfidf_candidate_ids"]),
                "source_family_count_with_confusion": len(
                    evidence["source_family_ids"]
                ),
                "source_family_count": len(evidence["source_family_ids"]),
                "per_source_family_counts": {
                    family: evidence["per_source_family_counts"].get(family, 0)
                    for family in families
                },
                "counts_by_source_family": {
                    family: evidence["per_source_family_counts"].get(family, 0)
                    for family in families
                },
                "observed_in_group_cv": evidence["observed_in_group_cv"],
                "observed_in_pooled_group_cv": evidence[
                    "observed_in_group_cv"
                ],
                "observed_in_source_family_evaluation": evidence[
                    "observed_in_source_family_evaluation"
                ],
                "record_ids": sorted(evidence["record_ids"]),
            }
        )
    return sorted(
        result,
        key=lambda row: (
            -row["total_error_count_across_candidates"],
            row["gold_intent"],
            row["predicted_intent"],
        ),
    )


def classify_remediation_evidence(
    contract: Mapping[str, Any],
    *,
    targeted_data_evidence_links: Sequence[str],
    representation_evidence_links: Sequence[str],
) -> dict[str, Any]:
    allowed = contract["remediation_evidence_summary"][
        "descriptive_classifications"
    ]
    targeted = bool(targeted_data_evidence_links)
    representation = bool(representation_evidence_links)
    if targeted and representation:
        classification = "both"
    elif targeted:
        classification = "targeted_data_boundary_evidence"
    elif representation:
        classification = "representation_change_evidence"
    else:
        classification = "insufficient_evidence"
    if classification not in allowed:
        raise ValueError("remediation evidence classification is not frozen")
    return {
        "classification": classification,
        "targeted_data_evidence_links": list(targeted_data_evidence_links),
        "representation_evidence_links": list(representation_evidence_links),
        "automatic_numerical_thresholds_used": False,
        "remediation_selected": False,
        "remediation_implemented": False,
    }


def build_remediation_summary(
    inputs: AnalysisInputs,
    recurrence: Mapping[str, Any],
    hard_negatives: Mapping[str, Any],
    class_weight: Sequence[Mapping[str, Any]],
    regularization: Sequence[Mapping[str, Any]],
    deltas: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    protected_fp = recurrence["protected_false_positive"]
    unsupported = recurrence["unsupported_confusion"]
    protected_miss = recurrence["protected_recall_miss"]
    hard_weaknesses = sorted(
        [
            row
            for row in hard_negatives[
                "across_all_candidates_and_source_families"
            ]
            if row["correct_count"] < row["record_count"]
        ],
        key=lambda row: (row["accuracy"], row["pair"]),
    )
    targeted_sections = {
        "top_protected_false_positive_boundaries": protected_fp,
        "top_unsupported_boundaries": unsupported,
        "top_protected_recall_miss_boundaries": protected_miss,
        "hard_negative_weaknesses": hard_weaknesses,
    }
    targeted_links = [
        name for name, rows in targeted_sections.items() if rows
    ]
    representation_sections = {
        "class_weight_observations": class_weight,
        "regularization_observations": regularization,
        "group_cv_vs_source_shift_deltas": deltas,
    }
    representation_links = [
        name for name, rows in representation_sections.items() if rows
    ]
    classification = classify_remediation_evidence(
        inputs.contract,
        targeted_data_evidence_links=targeted_links,
        representation_evidence_links=representation_links,
    )
    return {
        "top_protected_false_positive_boundaries": protected_fp,
        "top_unsupported_boundaries": unsupported,
        "top_protected_recall_miss_boundaries": protected_miss,
        "cross_family_recurring_boundaries": {
            "protected_false_positive": [
                row
                for row in protected_fp
                if row["source_family_count_with_confusion"] > 1
            ],
            "unsupported_confusion": [
                row
                for row in unsupported
                if row["source_family_count_with_confusion"] > 1
            ],
            "protected_recall_miss": [
                row
                for row in protected_miss
                if row["source_family_count_with_confusion"] > 1
            ],
        },
        "cross_candidate_recurring_boundaries": {
            "protected_false_positive": [
                row for row in protected_fp if row["candidate_count_with_confusion"] > 1
            ],
            "unsupported_confusion": [
                row for row in unsupported if row["candidate_count_with_confusion"] > 1
            ],
            "protected_recall_miss": [
                row
                for row in protected_miss
                if row["candidate_count_with_confusion"] > 1
            ],
        },
        "hard_negative_weaknesses": hard_weaknesses,
        "class_weight_observations": list(class_weight),
        "regularization_observations": list(regularization),
        "group_cv_vs_source_shift_deltas": list(deltas),
        "evidence_classification": classification,
        "classification_is_remediation_selection": False,
        "root_cause_claimed": False,
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
    payload: Mapping[str, Any], records: Mapping[str, Mapping[str, Any]]
) -> None:
    strings = set(all_string_values(payload))
    prohibited_keys = {"text", "utterance", "raw_text", "utterance_text"}
    if strings & prohibited_keys:
        raise ValueError("tracked analysis contains a raw-text field")
    raw_texts = {
        str(record["text"])
        for record in records.values()
        if isinstance(record.get("text"), str)
    }
    if strings & raw_texts:
        raise ValueError("tracked analysis contains raw development text")


def build_results(
    inputs: AnalysisInputs, paths: AnalysisPaths = DEFAULT_PATHS
) -> dict[str, Any]:
    analyses = candidate_analyses(inputs)
    deltas = [row["group_cv_vs_source_shift_delta"] for row in analyses]
    class_weight = class_weight_comparisons(inputs)
    regularization = regularization_comparisons(inputs)
    hard_negatives = hard_negative_analysis(inputs)
    recurrence = {
        "protected_false_positive": recurrence_rows(
            inputs, error_kind="protected_false_positive"
        ),
        "unsupported_confusion": recurrence_rows(
            inputs, error_kind="unsupported_confusion"
        ),
        "protected_recall_miss": recurrence_rows(
            inputs, error_kind="protected_recall_miss"
        ),
        "hard_negative": hard_negatives[
            "across_all_candidates_and_source_families"
        ],
        "systematic_threshold_applied": False,
    }
    summary = build_remediation_summary(
        inputs,
        recurrence,
        hard_negatives,
        class_weight,
        regularization,
        deltas,
    )
    payload = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "phase": "V2-C6 Step 29H-B",
        "execution_status": "COMPLETED",
        "input_artifacts": {
            "failure_analysis_contract": {
                "path": display_path(paths.contract, paths),
                "sha256": inputs.contract_sha256,
                "schema_version": CONTRACT_SCHEMA_VERSION,
            },
            **inputs.contract["source_artifacts"],
        },
        "candidate_coverage": {
            "candidate_count": 6,
            "candidate_ids": inputs.contract["candidate_coverage"][
                "candidate_ids"
            ],
            "all_candidates_analyzed": True,
            "winner_selected": False,
            "diagnostic_references": inputs.contract["candidate_coverage"][
                "diagnostic_references"
            ],
        },
        "prediction_coverage": inputs.coverage,
        "candidate_analyses": analyses,
        "group_cv_vs_source_shift_deltas": deltas,
        "class_weight_comparisons": class_weight,
        "regularization_comparisons": regularization,
        "hard_negative_analysis": hard_negatives,
        "recurrence_analysis": recurrence,
        "remediation_evidence_summary": summary,
        "consumed_evidence_governance": {
            **inputs.contract["consumed_evidence_governance"],
            "diagnostic_error_patterns_consumed_by_this_analysis": True,
            "same_families_are_fresh_after_analysis": False,
        },
        "known_limitations": inputs.contract["known_limitations"],
        "next_required": None,
        "next_required_contract_ambiguity": (
            "Step 29H-A freezes analysis execution but no post-analysis "
            "remediation continuation; Step 29H-B fails closed without "
            "authorizing Step 29I or inventing a next step."
        ),
        "governance": {
            "failure_analysis_performed": True,
            "diagnostic_source_family_evidence_consumed": True,
            "embeddings_generated": False,
            "model_fitting_performed": False,
            "model_inference_performed": False,
            "model_training_performed": False,
            "threshold_tuning_performed": False,
            "candidate_search_performed": False,
            "model_selection_performed": False,
            "dataset_mutated": False,
            "label_mutated": False,
            "taxonomy_mutated": False,
            "runtime_behavior_changed": False,
            "final_holdout_accessed": False,
            "step29i_authorized": False,
            "remediation_selected": False,
            "remediation_implemented": False,
            "raw_text_persisted": False,
        },
    }
    validate_results(payload, inputs)
    return payload


def validate_results(payload: Mapping[str, Any], inputs: AnalysisInputs) -> None:
    if (
        payload.get("schema_version") != RESULT_SCHEMA_VERSION
        or payload.get("phase") != "V2-C6 Step 29H-B"
        or payload.get("execution_status") != "COMPLETED"
    ):
        raise ValueError("Step 29H-B result identity changed")
    if payload.get("candidate_coverage", {}).get("candidate_count") != 6:
        raise ValueError("Step 29H-B candidate coverage changed")
    if len(payload.get("candidate_analyses", [])) != 6:
        raise ValueError("all six candidate analyses are required")
    if payload.get("next_required") is not None:
        raise ValueError("Step 29H-A did not authorize a continuation")
    governance = payload.get("governance", {})
    if governance.get("failure_analysis_performed") is not True:
        raise ValueError("failure analysis completion is not recorded")
    required_false = (
        "embeddings_generated",
        "model_fitting_performed",
        "model_inference_performed",
        "model_training_performed",
        "threshold_tuning_performed",
        "candidate_search_performed",
        "model_selection_performed",
        "dataset_mutated",
        "label_mutated",
        "taxonomy_mutated",
        "runtime_behavior_changed",
        "final_holdout_accessed",
        "step29i_authorized",
        "remediation_selected",
        "remediation_implemented",
        "raw_text_persisted",
    )
    if any(governance.get(field) is not False for field in required_false):
        raise ValueError("Step 29H-B governance changed")
    validate_text_free(payload, inputs.records_by_id)


def build_manifest(
    inputs: AnalysisInputs, result_bytes: bytes, paths: AnalysisPaths
) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "phase": "V2-C6 Step 29H-B",
        "execution_status": "COMPLETED",
        "analysis_contract": {
            "path": display_path(paths.contract, paths),
            "sha256": inputs.contract_sha256,
            "schema_version": CONTRACT_SCHEMA_VERSION,
        },
        "source_artifacts": inputs.contract["source_artifacts"],
        "result": {
            "path": display_path(paths.result, paths),
            "sha256": sha256_bytes(result_bytes),
            "schema_version": RESULT_SCHEMA_VERSION,
        },
        "runner": {
            "path": display_path(paths.runner, paths),
            "sha256": sha256_file(paths.runner, paths),
        },
        "candidate_count": 6,
        "group_cv_prediction_count_total": 54048,
        "source_family_prediction_count_total": 4860,
        "source_family_count": 3,
        "hard_negative_pair_count": 10,
        "tracked_raw_text_persisted": False,
        "failure_analysis_performed": True,
        "diagnostic_source_family_evidence_consumed": True,
        "model_fitting_performed": False,
        "model_inference_performed": False,
        "model_training_performed": False,
        "embeddings_generated": False,
        "threshold_tuning_performed": False,
        "dataset_mutated": False,
        "taxonomy_mutated": False,
        "runtime_behavior_changed": False,
        "final_holdout_accessed": False,
        "step29i_authorized": False,
        "remediation_selected": False,
        "next_required": None,
    }


def preflight(paths: AnalysisPaths = DEFAULT_PATHS) -> dict[str, Any]:
    inputs = load_preconditions(paths, require_outputs_absent=False)
    return {
        "status": "READY",
        "phase": "V2-C6 Step 29H-B",
        "contract_sha256": inputs.contract_sha256,
        "candidate_count": 6,
        "development_record_count": len(inputs.records_by_id),
        "remediation_record_count": len(inputs.remediation_ids),
        "prediction_coverage": inputs.coverage,
        "required_scope_count_per_candidate": 5,
        "expected_candidate_scope_analysis_count": 30,
        "tracked_outputs_present": paths.result.exists()
        or paths.result_manifest.exists(),
        "optional_local_review_present": paths.local_review.exists(),
        "failure_analysis_performed": False,
        "embeddings_generated": False,
        "model_fitting_performed": False,
        "model_inference_performed": False,
        "model_training_performed": False,
        "threshold_tuning_performed": False,
        "dataset_mutated": False,
        "taxonomy_mutated": False,
        "runtime_behavior_changed": False,
        "final_holdout_accessed": False,
        "step29i_authorized": False,
        "files_written": False,
    }


def run_analysis(paths: AnalysisPaths = DEFAULT_PATHS) -> tuple[Path, Path]:
    inputs = load_preconditions(paths, require_outputs_absent=True)
    payload = build_results(inputs, paths)
    result_bytes = stable_json_bytes(payload)
    manifest = build_manifest(inputs, result_bytes, paths)
    manifest_bytes = stable_json_bytes(manifest)
    validate_text_free(manifest, inputs.records_by_id)
    durable_create(paths.result, result_bytes)
    durable_create(paths.result_manifest, manifest_bytes)
    return paths.result, paths.result_manifest


def local_review_rows(inputs: AnalysisInputs) -> list[dict[str, Any]]:
    allowed = inputs.contract["error_record_schema"]["optional_local_text_review"][
        "allowed_columns"
    ]
    rows: list[dict[str, Any]] = []
    for candidate_id in inputs.contract["candidate_coverage"]["candidate_ids"]:
        scopes = prediction_scopes(
            inputs.candidate_results[candidate_id], inputs.contract
        )
        for scope in [
            "pooled_group_aware_cv",
            *inputs.contract["source_family_analysis"]["required_family_ids"],
        ]:
            for prediction in scopes[scope]:
                if prediction["gold_intent"] == prediction["predicted_intent"]:
                    continue
                example_id = str(prediction["example_id"])
                record = inputs.records_by_id[example_id]
                output = {
                    "record_id": example_id,
                    "text": record["text"],
                    "gold_intent": prediction["gold_intent"],
                    "predicted_intent": prediction["predicted_intent"],
                    "candidate_id": candidate_id,
                    "evaluation_scope": scope,
                    "source_family_id": record.get("source_family_id", ""),
                    "boundary_target": record.get("boundary_target", ""),
                    "is_hard_negative": record.get("is_hard_negative", ""),
                    "unsupported_subtype": record.get("unsupported_subtype", ""),
                }
                rows.append({column: output[column] for column in allowed})
    rows.sort(
        key=lambda row: (
            row["candidate_id"],
            row["evaluation_scope"],
            row["record_id"],
            row["predicted_intent"],
        )
    )
    return rows


def write_local_review(paths: AnalysisPaths = DEFAULT_PATHS) -> Path:
    inputs = load_preconditions(paths, require_outputs_absent=False)
    if paths.local_review.exists():
        raise FileExistsError("local error-review file already exists")
    definition = inputs.contract["error_record_schema"]["optional_local_text_review"]
    columns = definition["allowed_columns"]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    provenance = {column: "" for column in columns}
    provenance["record_id"] = "__PROVENANCE__"
    provenance["text"] = definition["provenance_statement_required"]
    writer.writerow(provenance)
    writer.writerows(local_review_rows(inputs))
    durable_create(paths.local_review, bytes(buffer.getvalue(), "utf-8"))
    return paths.local_review


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Execute frozen V2-C6 post-selection failure analysis"
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--run", action="store_true")
    modes.add_argument("--write-local-review", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.preflight:
        print(json.dumps(preflight(), indent=2, sort_keys=True))
        return
    if args.write_local_review:
        path = write_local_review()
        print(display_path(path, DEFAULT_PATHS))
        return
    paths = run_analysis()
    print("\n".join(display_path(path, DEFAULT_PATHS) for path in paths))


if __name__ == "__main__":
    main()
