"""Run the frozen V2-C6 targeted-remediation failure analysis.

This workflow reads only persisted prediction rows and frozen dataset metadata.
It contains no model, embedding, fitting, loading, or inference path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ML_DIRECTORY = REPOSITORY_ROOT / "data/evals/v2/ml"
SCRIPT_RELATIVE_PATH = (
    "scripts/run_v2c6_targeted_remediation_failure_analysis.py"
)

CONTRACT_SCHEMA_VERSION = (
    "v2c6-targeted-remediation-failure-analysis-contract.v1"
)
RESULT_SCHEMA_VERSION = "v2c6-targeted-remediation-failure-analysis.v1"
MANIFEST_SCHEMA_VERSION = (
    "v2c6-targeted-remediation-failure-analysis-manifest.v1"
)
EXPECTED_CONTRACT_SHA256 = (
    "d9cb3fa6fcc6cdbb248ea7221bf0e74029b91dd7a7d74e411a9c2c8d8f0cf652"
)
UNSUPPORTED_INTENT = "unsupported_or_uncertain"
GROUP_SCOPE = "pooled_group_aware_cv"
POOLED_FRESH_SCOPE = "pooled_fresh_evaluation"
HIERARCHICAL_CANDIDATE_ID = (
    "HIERARCHICAL_TFIDF_LINEAR_SVC__C=1.0__class_weight=none"
)


@dataclass(frozen=True)
class AnalysisPaths:
    repository_root: Path
    contract: Path
    results: Path
    results_manifest: Path
    design_contract: Path
    development_dataset: Path
    development_manifest: Path
    fresh_dataset: Path
    fresh_manifest: Path
    result: Path
    result_manifest: Path
    prohibited_holdout: Path
    runner: Path


DEFAULT_PATHS = AnalysisPaths(
    repository_root=REPOSITORY_ROOT,
    contract=(
        ML_DIRECTORY
        / "v2c6_targeted_remediation_failure_analysis_contract.json"
    ),
    results=(
        ML_DIRECTORY
        / "v2c6_targeted_remediation_model_selection_results.json"
    ),
    results_manifest=(
        ML_DIRECTORY
        / "v2c6_targeted_remediation_model_selection_results.manifest.json"
    ),
    design_contract=(
        ML_DIRECTORY / "v2c6_targeted_remediation_design_contract.json"
    ),
    development_dataset=(
        ML_DIRECTORY / "v2c6_targeted_remediated_development_dataset.json"
    ),
    development_manifest=(
        ML_DIRECTORY
        / "v2c6_targeted_remediated_development_dataset.manifest.json"
    ),
    fresh_dataset=(
        ML_DIRECTORY / "v2c6_fresh_source_evaluation_dataset.json"
    ),
    fresh_manifest=(
        ML_DIRECTORY / "v2c6_fresh_source_evaluation_dataset.manifest.json"
    ),
    result=(
        ML_DIRECTORY / "v2c6_targeted_remediation_failure_analysis.json"
    ),
    result_manifest=(
        ML_DIRECTORY
        / "v2c6_targeted_remediation_failure_analysis.manifest.json"
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
    development_records: dict[str, dict[str, Any]]
    fresh_records: dict[str, dict[str, Any]]
    candidate_results: dict[str, dict[str, Any]]
    prediction_coverage: dict[str, Any]


def stable_json_bytes(payload: Any) -> bytes:
    return bytes(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        "utf-8",
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def display_path(path: Path, paths: AnalysisPaths) -> str:
    try:
        return str(path.resolve().relative_to(paths.repository_root.resolve()))
    except ValueError:
        return str(path.resolve())


def guard_final_holdout(path: Path, paths: AnalysisPaths) -> None:
    resolved = path.resolve()
    ml_root = (paths.repository_root / "data/evals/v2/ml").resolve()
    if resolved == paths.prohibited_holdout.resolve() or (
        resolved.is_relative_to(ml_root) and "final_holdout" in resolved.name
    ):
        raise PermissionError("final holdout access is prohibited")


def read_bytes(path: Path, paths: AnalysisPaths) -> bytes:
    guard_final_holdout(path, paths)
    return path.read_bytes()


def sha256_file(path: Path, paths: AnalysisPaths = DEFAULT_PATHS) -> str:
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
    """Create one durable file without ever replacing an existing artifact."""

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


def validate_contract(
    contract: Mapping[str, Any], paths: AnalysisPaths = DEFAULT_PATHS
) -> None:
    if (
        contract.get("schema_version") != CONTRACT_SCHEMA_VERSION
        or contract.get("contract_version") != CONTRACT_SCHEMA_VERSION
        or contract.get("phase")
        != "V2-C6 post-targeted-remediation failure-analysis contract"
        or contract.get("status") != "FROZEN"
    ):
        raise ValueError("unexpected targeted-remediation analysis contract")

    status = contract.get("contract_status", {})
    required_false = (
        "candidate_selected",
        "dataset_mutated",
        "embeddings_generated",
        "failure_analysis_executed",
        "final_holdout_accessed",
        "final_holdout_created",
        "final_model_acceptance_claimed",
        "model_fitting_performed",
        "model_inference_performed",
        "remediation_chosen",
        "runtime_behavior_changed",
        "step29i_authorized",
        "threshold_tuning_performed",
    )
    if status.get("contract_frozen") is not True or any(
        status.get(field) is not False for field in required_false
    ):
        raise ValueError("frozen contract governance changed")
    if status.get("next_required") is not None or contract.get(
        "next_required"
    ) is not None:
        raise ValueError("contract does not authorize a next step")

    coverage = contract.get("candidate_coverage", {})
    if (
        coverage.get("candidate_count") != 4
        or len(coverage.get("candidate_ids", [])) != 4
        or coverage.get("candidate_selection_permitted") is not False
        or coverage.get("winner_may_be_declared") is not False
    ):
        raise ValueError("frozen candidate coverage changed")

    families = contract.get("consumed_evidence_governance", {}).get(
        "fresh_evaluation_source_family_ids"
    )
    if families != contract.get("analysis_dimensions", {}).get(
        "fresh_family_consistency", {}
    ).get("source_family_ids") or len(families or []) != 2:
        raise ValueError("frozen fresh-family identities changed")

    outputs = contract.get("execution_artifact_contract", {}).get(
        "tracked_outputs"
    )
    expected_outputs = [
        {
            "path": display_path(paths.result, paths),
            "schema_version": RESULT_SCHEMA_VERSION,
        },
        {
            "path": display_path(paths.result_manifest, paths),
            "schema_version": MANIFEST_SCHEMA_VERSION,
        },
    ]
    if outputs != expected_outputs:
        raise ValueError("frozen analysis output identities changed")
    if contract["execution_artifact_contract"].get(
        "raw_utterance_text_permitted_in_tracked_outputs"
    ) is not False:
        raise ValueError("tracked raw-text policy changed")

    hierarchy = contract["analysis_dimensions"][
        "hierarchical_architecture_diagnosis"
    ]
    if (
        hierarchy.get("stage_level_predictions_persisted") is not False
        or hierarchy.get("exact_stage_1_error_attribution_available") is not False
        or hierarchy.get("model_reconstruction_permitted") is not False
        or hierarchy.get("model_rerun_permitted") is not False
    ):
        raise ValueError("frozen hierarchy-diagnostic policy changed")

    continuation = contract.get("continuation_policy", {})
    if (
        continuation.get("analysis_execution_authorized_by_this_contract")
        is not True
        or continuation.get("post_analysis_next_required") is not None
        or continuation.get("remediation_choice_pre_authorized") is not False
        or continuation.get("step29i_authorized") is not False
    ):
        raise ValueError("frozen continuation policy changed")


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


def _validate_source_schema(
    source: Mapping[str, Any], specification: Mapping[str, Any], label: str
) -> None:
    if source.get("schema_version") != specification.get("schema_version"):
        raise ValueError(f"{label} schema mismatch")


def validate_input_state(
    contract: Mapping[str, Any],
    results: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    sources = contract["source_artifacts"]
    result_spec = sources["targeted_remediation_model_selection_results"]
    manifest_spec = sources[
        "targeted_remediation_model_selection_results_manifest"
    ]
    _validate_source_schema(results, result_spec, "model-selection results")
    _validate_source_schema(manifest, manifest_spec, "results manifest")
    if results.get("execution_status") != result_spec[
        "expected_execution_status"
    ]:
        raise ValueError("model-selection result is not completed")
    if (
        results.get("candidate_count") != 4
        or results.get("completed_candidate_count") != 4
    ):
        raise ValueError("exactly four completed candidates are required")

    selection = results.get("selection", {})
    expected = {
        "eligible_candidate_count": 0,
        "selection_status": "NO_ACCEPTABLE_CANDIDATE",
        "selected_candidate": None,
        "selected_candidate_id": None,
        "winner_forced": False,
        "gates_weakened": False,
        "step29i_authorized": False,
        "next_required": None,
    }
    if any(selection.get(key) != value for key, value in expected.items()):
        raise ValueError("NO_ACCEPTABLE_CANDIDATE prerequisite changed")
    expected_ids = contract["candidate_coverage"]["candidate_ids"]
    candidate_rows = results.get("candidate_results")
    if not isinstance(candidate_rows, list) or [
        row.get("candidate_id") for row in candidate_rows
    ] != expected_ids:
        raise ValueError("exact four-candidate identity or order changed")
    if any(
        row.get("execution_status") != "COMPLETED"
        or row.get("eligible") is not False
        for row in candidate_rows
    ):
        raise ValueError("all candidates must be completed and ineligible")

    if (
        manifest.get("results", {}).get("sha256") != result_spec["sha256"]
        or manifest.get("selection_status") != "NO_ACCEPTABLE_CANDIDATE"
        or manifest.get("selected_candidate_id") is not None
        or manifest.get("candidate_count") != 4
        or manifest.get("winner_forced") is not False
        or manifest.get("gates_weakened") is not False
        or manifest.get("step29i_authorized") is not False
        or manifest.get("next_required") is not None
        or manifest.get("final_holdout_accessed") is not False
    ):
        raise ValueError("model-selection manifest lineage changed")


def _record_index(
    dataset: Mapping[str, Any],
    *,
    id_field: str,
    expected_count: int,
    expected_schema: str,
    label: str,
) -> dict[str, dict[str, Any]]:
    if dataset.get("schema_version") != expected_schema:
        raise ValueError(f"{label} schema changed")
    rows = dataset.get("examples")
    if not isinstance(rows, list) or len(rows) != expected_count:
        raise ValueError(f"{label} record count changed")
    records: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise TypeError(f"{label} record must be an object")
        record_id = row.get(id_field)
        if not isinstance(record_id, str) or not record_id:
            raise ValueError(f"{label} record ID must be nonempty")
        if record_id in records:
            raise ValueError(f"duplicate {label} record ID: {record_id}")
        if not isinstance(row.get("intent"), str) or not row["intent"]:
            raise ValueError(f"{label} intent must be nonempty: {record_id}")
        records[record_id] = row
    return records


def _validate_prediction_rows(
    rows: Any,
    *,
    records: Mapping[str, Mapping[str, Any]],
    id_field: str,
    candidate_id: str,
    scope: str,
    allowed_intents: set[str],
    source_family_id: str | None = None,
) -> None:
    if not isinstance(rows, list) or len(rows) != len(records):
        raise ValueError(f"prediction count mismatch: {candidate_id}:{scope}")
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise TypeError("prediction row must be an object")
        record_id = row.get(id_field)
        if not isinstance(record_id, str) or record_id not in records:
            raise ValueError(f"unexpected prediction ID: {candidate_id}:{scope}")
        if record_id in seen:
            raise ValueError(f"duplicate prediction: {candidate_id}:{scope}")
        seen.add(record_id)
        record = records[record_id]
        if row.get("gold_intent") != record.get("intent"):
            raise ValueError(f"prediction gold intent mismatch: {record_id}")
        if row.get("predicted_intent") not in allowed_intents:
            raise ValueError(f"prediction outside frozen taxonomy: {record_id}")
        if source_family_id is not None and (
            row.get("source_family_id") != source_family_id
            or record.get("source_family_id") != source_family_id
        ):
            raise ValueError(f"fresh source-family mismatch: {record_id}")
    if seen != set(records):
        raise ValueError(f"missing predictions: {candidate_id}:{scope}")


def validate_prediction_coverage(
    contract: Mapping[str, Any],
    results: Mapping[str, Any],
    development_records: Mapping[str, Mapping[str, Any]],
    fresh_records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    candidate_ids = contract["candidate_coverage"]["candidate_ids"]
    families = contract["consumed_evidence_governance"][
        "fresh_evaluation_source_family_ids"
    ]
    family_records = {
        family: {
            record_id: row
            for record_id, row in fresh_records.items()
            if row.get("source_family_id") == family
        }
        for family in families
    }
    if any(len(rows) != 320 for rows in family_records.values()):
        raise ValueError("each frozen fresh family must contain 320 records")

    taxonomy: set[str] | None = None
    for candidate in results["candidate_results"]:
        candidate_id = str(candidate["candidate_id"])
        if candidate_id not in candidate_ids:
            raise ValueError("unexpected candidate")
        label_order = candidate.get("group_aware_cv", {}).get("metrics", {}).get(
            "confusion_matrix", {}
        ).get("label_order")
        if not isinstance(label_order, list) or not label_order:
            raise ValueError(f"missing taxonomy label order: {candidate_id}")
        current_taxonomy = set(label_order)
        if taxonomy is None:
            taxonomy = current_taxonomy
        elif taxonomy != current_taxonomy:
            raise ValueError("candidate taxonomy label orders differ")

        group = candidate["group_aware_cv"]
        group_rows = group.get("predictions")
        _validate_prediction_rows(
            group_rows,
            records=development_records,
            id_field="example_id",
            candidate_id=candidate_id,
            scope=GROUP_SCOPE,
            allowed_intents=current_taxonomy,
        )
        if (
            group.get("prediction_count") != 9608
            or group.get("unique_prediction_id_count") != 9608
            or group.get("prediction_sha256")
            != sha256_bytes(stable_json_bytes(group_rows))
        ):
            raise ValueError(f"grouped-CV prediction identity changed: {candidate_id}")

        fresh = candidate["fresh_source_evaluation"]
        fresh_rows = fresh.get("pooled_predictions")
        _validate_prediction_rows(
            fresh_rows,
            records=fresh_records,
            id_field="record_id",
            candidate_id=candidate_id,
            scope=POOLED_FRESH_SCOPE,
            allowed_intents=current_taxonomy,
        )
        if (
            fresh.get("pooled_prediction_count") != 640
            or fresh.get("pooled_prediction_sha256")
            != sha256_bytes(stable_json_bytes(fresh_rows))
        ):
            raise ValueError(f"fresh prediction identity changed: {candidate_id}")
        family_summaries = fresh.get("families")
        if not isinstance(family_summaries, list) or [
            row.get("source_family_id") for row in family_summaries
        ] != families:
            raise ValueError(f"fresh family summary changed: {candidate_id}")
        for family, summary in zip(families, family_summaries, strict=True):
            selected = [
                row for row in fresh_rows if row.get("source_family_id") == family
            ]
            _validate_prediction_rows(
                selected,
                records=family_records[family],
                id_field="record_id",
                candidate_id=candidate_id,
                scope=family,
                source_family_id=family,
                allowed_intents=current_taxonomy,
            )
            if (
                summary.get("record_count") != 320
                or summary.get("prediction_count") != 320
                or summary.get("prediction_sha256")
                != sha256_bytes(stable_json_bytes(selected))
            ):
                raise ValueError(
                    f"fresh family prediction identity changed: {candidate_id}:{family}"
                )
    return {
        "candidate_count": 4,
        "grouped_cv_predictions_per_candidate": 9608,
        "grouped_cv_prediction_count_total": 38432,
        "fresh_predictions_per_candidate": 640,
        "fresh_prediction_count_total": 2560,
        "fresh_predictions_per_candidate_and_family": 320,
        "fresh_source_family_count": 2,
        "missing_prediction_count": 0,
        "duplicate_prediction_count": 0,
    }


def load_preconditions(
    paths: AnalysisPaths = DEFAULT_PATHS,
    *,
    require_outputs_absent: bool,
) -> AnalysisInputs:
    contract_sha256 = sha256_file(paths.contract, paths)
    if contract_sha256 != EXPECTED_CONTRACT_SHA256:
        raise ValueError(
            f"failure-analysis contract SHA-256 mismatch: {contract_sha256}"
        )
    contract = read_json_object(paths.contract, paths)
    validate_contract(contract, paths)
    sources = contract["source_artifacts"]
    bindings = (
        (
            paths.results,
            sources["targeted_remediation_model_selection_results"],
            "model-selection results",
        ),
        (
            paths.results_manifest,
            sources[
                "targeted_remediation_model_selection_results_manifest"
            ],
            "model-selection results manifest",
        ),
        (
            paths.design_contract,
            sources["targeted_remediation_design_contract"],
            "targeted-remediation design contract",
        ),
        (
            paths.development_dataset,
            sources["targeted_development_dataset"],
            "targeted development dataset",
        ),
        (
            paths.development_manifest,
            sources["targeted_development_manifest"],
            "targeted development manifest",
        ),
        (
            paths.fresh_dataset,
            sources["fresh_evaluation_dataset"],
            "fresh evaluation dataset",
        ),
        (
            paths.fresh_manifest,
            sources["fresh_evaluation_manifest"],
            "fresh evaluation manifest",
        ),
    )
    for path, specification, label in bindings:
        validate_binding(path, specification, paths, label)

    results = read_json_object(paths.results, paths)
    manifest = read_json_object(paths.results_manifest, paths)
    design = read_json_object(paths.design_contract, paths)
    development = read_json_object(paths.development_dataset, paths)
    development_manifest = read_json_object(paths.development_manifest, paths)
    fresh = read_json_object(paths.fresh_dataset, paths)
    fresh_manifest = read_json_object(paths.fresh_manifest, paths)
    validate_input_state(contract, results, manifest)
    _validate_source_schema(
        design, sources["targeted_remediation_design_contract"], "design contract"
    )
    _validate_source_schema(
        development_manifest,
        sources["targeted_development_manifest"],
        "development manifest",
    )
    _validate_source_schema(
        fresh_manifest,
        sources["fresh_evaluation_manifest"],
        "fresh manifest",
    )
    development_records = _record_index(
        development,
        id_field="example_id",
        expected_count=9608,
        expected_schema=sources["targeted_development_dataset"][
            "schema_version"
        ],
        label="development",
    )
    fresh_records = _record_index(
        fresh,
        id_field="record_id",
        expected_count=640,
        expected_schema=sources["fresh_evaluation_dataset"]["schema_version"],
        label="fresh evaluation",
    )
    prediction_coverage = validate_prediction_coverage(
        contract, results, development_records, fresh_records
    )
    if require_outputs_absent and (
        paths.result.exists() or paths.result_manifest.exists()
    ):
        raise FileExistsError("failure-analysis output already exists")
    candidate_results = {
        str(row["candidate_id"]): row for row in results["candidate_results"]
    }
    return AnalysisInputs(
        contract=contract,
        contract_sha256=contract_sha256,
        results=results,
        results_manifest=manifest,
        development_records=development_records,
        fresh_records=fresh_records,
        candidate_results=candidate_results,
        prediction_coverage=prediction_coverage,
    )


def ratio(numerator: int, denominator: int, label: str) -> float:
    if denominator <= 0:
        raise ValueError(f"{label} denominator must be positive")
    return numerator / denominator


def protected_false_positive_analysis(
    rows: Sequence[Mapping[str, Any]], protected_intents: set[str]
) -> dict[str, Any]:
    non_protected = [
        row for row in rows if row["gold_intent"] not in protected_intents
    ]
    failures = [
        row
        for row in non_protected
        if row["predicted_intent"] in protected_intents
    ]
    by_true = Counter(str(row["gold_intent"]) for row in failures)
    by_predicted = Counter(str(row["predicted_intent"]) for row in failures)
    matrix: dict[str, dict[str, int]] = {}
    for true_intent in sorted(by_true):
        matrix[true_intent] = {
            predicted: sum(
                row["gold_intent"] == true_intent
                and row["predicted_intent"] == predicted
                for row in failures
            )
            for predicted in sorted(protected_intents)
        }
    return {
        "non_protected_gold_count": len(non_protected),
        "protected_false_positive_count": len(failures),
        "protected_false_positive_rate": ratio(
            len(failures), len(non_protected), "protected false-positive rate"
        ),
        "counts_by_true_intent": dict(sorted(by_true.items())),
        "counts_by_predicted_protected_intent": dict(
            sorted(by_predicted.items())
        ),
        "true_intent_by_predicted_protected_intent_matrix": matrix,
    }


def unsupported_miss_analysis(
    rows: Sequence[Mapping[str, Any]], protected_intents: set[str]
) -> dict[str, Any]:
    unsupported = [
        row for row in rows if row["gold_intent"] == UNSUPPORTED_INTENT
    ]
    correct = [
        row
        for row in unsupported
        if row["predicted_intent"] == UNSUPPORTED_INTENT
    ]
    misses = [
        row
        for row in unsupported
        if row["predicted_intent"] != UNSUPPORTED_INTENT
    ]
    destinations = Counter(str(row["predicted_intent"]) for row in misses)
    protected_count = sum(
        row["predicted_intent"] in protected_intents for row in misses
    )
    return {
        "unsupported_gold_count": len(unsupported),
        "unsupported_correct_count": len(correct),
        "unsupported_miss_count": len(misses),
        "unsupported_recall": ratio(
            len(correct), len(unsupported), "unsupported recall"
        ),
        "unsupported_miss_rate": ratio(
            len(misses), len(unsupported), "unsupported miss rate"
        ),
        "counts_by_predicted_intent": dict(sorted(destinations.items())),
        "protected_destination_count": protected_count,
        "non_protected_destination_count": len(misses) - protected_count,
    }


def protected_recall_miss_analysis(
    rows: Sequence[Mapping[str, Any]], protected_intents: set[str]
) -> dict[str, Any]:
    protected = [row for row in rows if row["gold_intent"] in protected_intents]
    correct = [
        row for row in protected if row["predicted_intent"] in protected_intents
    ]
    misses = [
        row
        for row in protected
        if row["predicted_intent"] not in protected_intents
    ]
    matrix: dict[str, dict[str, int]] = {}
    for true_intent in sorted(protected_intents):
        destinations = Counter(
            str(row["predicted_intent"])
            for row in misses
            if row["gold_intent"] == true_intent
        )
        matrix[true_intent] = dict(sorted(destinations.items()))
    return {
        "protected_gold_count": len(protected),
        "protected_prediction_as_protected_count": len(correct),
        "protected_recall_miss_count": len(misses),
        "protected_recall": ratio(
            len(correct), len(protected), "protected recall"
        ),
        "true_protected_intent_by_predicted_intent_matrix": matrix,
    }


def hard_negative_directional_analysis(
    rows: Sequence[Mapping[str, Any]],
    records: Mapping[str, Mapping[str, Any]],
    pairs: Sequence[Sequence[str]],
    *,
    id_field: str,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for pair in pairs:
        intent_a, intent_b = pair
        selected: list[Mapping[str, Any]] = []
        for row in rows:
            record = records[str(row[id_field])]
            if record.get("is_hard_negative") is not True:
                continue
            if {record.get("intent"), record.get("boundary_target")} == {
                intent_a,
                intent_b,
            }:
                selected.append(row)
        true_a = [row for row in selected if row["gold_intent"] == intent_a]
        true_b = [row for row in selected if row["gold_intent"] == intent_b]
        a_to_b = sum(row["predicted_intent"] == intent_b for row in true_a)
        b_to_a = sum(row["predicted_intent"] == intent_a for row in true_b)
        other_error = sum(
            row["predicted_intent"] != row["gold_intent"]
            and not (
                (row["gold_intent"] == intent_a and row["predicted_intent"] == intent_b)
                or (
                    row["gold_intent"] == intent_b
                    and row["predicted_intent"] == intent_a
                )
            )
            for row in selected
        )
        results.append(
            {
                "intent_a": intent_a,
                "intent_b": intent_b,
                "metadata_available": bool(selected),
                "boundary_record_count": len(selected),
                "true_a_total": len(true_a),
                "true_b_total": len(true_b),
                "true_a_predicted_b_count": a_to_b,
                "true_b_predicted_a_count": b_to_a,
                "true_a_predicted_b_rate": (
                    ratio(a_to_b, len(true_a), "hard-negative A-to-B rate")
                    if true_a
                    else None
                ),
                "true_b_predicted_a_rate": (
                    ratio(b_to_a, len(true_b), "hard-negative B-to-A rate")
                    if true_b
                    else None
                ),
                "other_error_count": other_error,
            }
        )
    return {
        "boundaries": results,
        "metadata_join_field": id_field,
        "limitation": (
            None
            if any(row["metadata_available"] for row in results)
            else "Boundary membership is not measurable from persisted metadata."
        ),
    }


def prediction_scopes(
    candidate: Mapping[str, Any], families: Sequence[str]
) -> dict[str, list[dict[str, Any]]]:
    group = list(candidate["group_aware_cv"]["predictions"])
    fresh = list(candidate["fresh_source_evaluation"]["pooled_predictions"])
    return {
        GROUP_SCOPE: group,
        **{
            family: [
                row for row in fresh if row.get("source_family_id") == family
            ]
            for family in families
        },
        POOLED_FRESH_SCOPE: fresh,
    }


FailureKey = tuple[str, str | None, str, str]


def failure_sets_for_scope(
    candidate_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    scope: str,
    source_family_id: str | None,
    protected_intents: set[str],
    records: Mapping[str, Mapping[str, Any]],
    id_field: str,
    hard_negative_pairs: Sequence[Sequence[str]],
) -> dict[str, set[FailureKey]]:
    pair_sets = [set(pair) for pair in hard_negative_pairs]
    output: dict[str, set[FailureKey]] = {}
    for candidate_id, rows in candidate_rows.items():
        failures: set[FailureKey] = set()
        for row in rows:
            gold = str(row["gold_intent"])
            predicted = str(row["predicted_intent"])
            record_id = str(row[id_field])
            if gold not in protected_intents and predicted in protected_intents:
                failures.add(
                    (scope, source_family_id, record_id, "protected_false_positive")
                )
            if gold == UNSUPPORTED_INTENT and predicted != UNSUPPORTED_INTENT:
                failures.add((scope, source_family_id, record_id, "unsupported_miss"))
            if gold in protected_intents and predicted not in protected_intents:
                failures.add(
                    (scope, source_family_id, record_id, "protected_recall_miss")
                )
            record = records[record_id]
            boundary = {record.get("intent"), record.get("boundary_target")}
            if (
                record.get("is_hard_negative") is True
                and boundary in pair_sets
                and predicted in boundary
                and predicted != gold
            ):
                pair = next(pair for pair in pair_sets if pair == boundary)
                pair_name = "|".join(sorted(str(value) for value in pair))
                failures.add(
                    (
                        scope,
                        source_family_id,
                        record_id,
                        f"hard_negative_directional_confusion:{pair_name}",
                    )
                )
        output[candidate_id] = failures
    return output


def _serialize_failure_keys(values: Iterable[FailureKey]) -> list[dict[str, Any]]:
    return [
        {
            "evaluation_scope": scope,
            "source_family_id": family,
            "record_id_or_example_id": record_id,
            "failure_type": failure_type,
        }
        for scope, family, record_id, failure_type in sorted(
            values,
            key=lambda value: (
                value[0],
                value[1] or "",
                value[2],
                value[3],
            ),
        )
    ]


def overlap_analysis(
    error_sets: Mapping[str, set[FailureKey]],
    candidate_ids: Sequence[str],
) -> dict[str, Any]:
    if set(error_sets) != set(candidate_ids):
        raise ValueError("overlap analysis requires every frozen candidate")
    all_shared = set.intersection(*(error_sets[value] for value in candidate_ids))
    first_three = candidate_ids[:3]
    shared_first_three = set.intersection(
        *(error_sets[value] for value in first_three)
    )
    unique = {
        candidate: error_sets[candidate]
        - set().union(
            *(error_sets[other] for other in candidate_ids if other != candidate)
        )
        for candidate in candidate_ids
    }
    pairwise: list[dict[str, Any]] = []
    for first, second in combinations(candidate_ids, 2):
        intersection = error_sets[first] & error_sets[second]
        union = error_sets[first] | error_sets[second]
        pairwise.append(
            {
                "candidate_a": first,
                "candidate_b": second,
                "intersection_count": len(intersection),
                "union_count": len(union),
                "jaccard": len(intersection) / len(union) if union else None,
            }
        )
    result = {
        "intersection_shared_by_all_four_candidates": {
            "count": len(all_shared),
            "records": _serialize_failure_keys(all_shared),
        },
        "intersection_shared_by_bge_tfidf_and_hybrid": {
            "candidate_ids": list(first_three),
            "count": len(shared_first_three),
            "records": _serialize_failure_keys(shared_first_three),
        },
        "failures_unique_to_each_candidate": {
            candidate: {
                "count": len(values),
                "records": _serialize_failure_keys(values),
            }
            for candidate, values in unique.items()
        },
        "pairwise": pairwise,
        "selection_or_ranking_performed": False,
    }
    failure_types = (
        "protected_false_positive",
        "unsupported_miss",
        "protected_recall_miss",
        "hard_negative_directional_confusion",
    )
    result["by_failure_type"] = {
        failure_type: _overlap_counts(
            {
                candidate: {
                    key
                    for key in values
                    if key[3] == failure_type
                    or key[3].startswith(f"{failure_type}:")
                }
                for candidate, values in error_sets.items()
            },
            candidate_ids,
        )
        for failure_type in failure_types
    }
    return result


def _overlap_counts(
    error_sets: Mapping[str, set[FailureKey]],
    candidate_ids: Sequence[str],
) -> dict[str, Any]:
    all_shared = set.intersection(*(error_sets[value] for value in candidate_ids))
    first_three = candidate_ids[:3]
    shared_first_three = set.intersection(
        *(error_sets[value] for value in first_three)
    )
    unique_counts = {
        candidate: len(
            error_sets[candidate]
            - set().union(
                *(
                    error_sets[other]
                    for other in candidate_ids
                    if other != candidate
                )
            )
        )
        for candidate in candidate_ids
    }
    pairwise: list[dict[str, Any]] = []
    for first, second in combinations(candidate_ids, 2):
        intersection = error_sets[first] & error_sets[second]
        union = error_sets[first] | error_sets[second]
        pairwise.append(
            {
                "candidate_a": first,
                "candidate_b": second,
                "intersection_count": len(intersection),
                "union_count": len(union),
                "jaccard": len(intersection) / len(union) if union else None,
            }
        )
    return {
        "intersection_shared_by_all_four_candidates": len(all_shared),
        "intersection_shared_by_bge_tfidf_and_hybrid": len(
            shared_first_three
        ),
        "failures_unique_to_each_candidate": unique_counts,
        "pairwise": pairwise,
    }


def _scope_analysis(
    rows: Sequence[Mapping[str, Any]],
    records: Mapping[str, Mapping[str, Any]],
    *,
    id_field: str,
    protected_intents: set[str],
    hard_negative_pairs: Sequence[Sequence[str]],
) -> dict[str, Any]:
    return {
        "record_count": len(rows),
        "protected_false_positives": protected_false_positive_analysis(
            rows, protected_intents
        ),
        "unsupported_misses": unsupported_miss_analysis(rows, protected_intents),
        "protected_recall_misses": protected_recall_miss_analysis(
            rows, protected_intents
        ),
        "hard_negative_boundaries": hard_negative_directional_analysis(
            rows,
            records,
            hard_negative_pairs,
            id_field=id_field,
        ),
    }


def fresh_family_comparison(
    candidate_analysis: Mapping[str, Mapping[str, Any]], families: Sequence[str]
) -> list[dict[str, Any]]:
    first, second = families
    output: list[dict[str, Any]] = []
    for candidate_id, scopes in candidate_analysis.items():
        left = scopes[first]
        right = scopes[second]
        left_hard = sum(
            row["true_a_predicted_b_count"]
            + row["true_b_predicted_a_count"]
            for row in left["hard_negative_boundaries"]["boundaries"]
        )
        right_hard = sum(
            row["true_a_predicted_b_count"]
            + row["true_b_predicted_a_count"]
            for row in right["hard_negative_boundaries"]["boundaries"]
        )
        left_boundaries = {
            (row["intent_a"], row["intent_b"]): row
            for row in left["hard_negative_boundaries"]["boundaries"]
        }
        right_boundaries = {
            (row["intent_a"], row["intent_b"]): row
            for row in right["hard_negative_boundaries"]["boundaries"]
        }
        boundary_comparisons = []
        for pair, left_row in left_boundaries.items():
            right_row = right_boundaries[pair]
            left_errors = (
                left_row["true_a_predicted_b_count"]
                + left_row["true_b_predicted_a_count"]
            )
            right_errors = (
                right_row["true_a_predicted_b_count"]
                + right_row["true_b_predicted_a_count"]
            )
            boundary_comparisons.append(
                {
                    "intent_a": pair[0],
                    "intent_b": pair[1],
                    "directional_error_count_by_family": {
                        first: left_errors,
                        second: right_errors,
                    },
                    "directional_error_count_difference_b_minus_a": (
                        right_errors - left_errors
                    ),
                }
            )
        output.append(
            {
                "candidate_id": candidate_id,
                "source_family_a": first,
                "source_family_b": second,
                "protected_false_positive_count_difference_b_minus_a": (
                    right["protected_false_positives"][
                        "protected_false_positive_count"
                    ]
                    - left["protected_false_positives"][
                        "protected_false_positive_count"
                    ]
                ),
                "unsupported_miss_count_difference_b_minus_a": (
                    right["unsupported_misses"]["unsupported_miss_count"]
                    - left["unsupported_misses"]["unsupported_miss_count"]
                ),
                "protected_recall_miss_count_difference_b_minus_a": (
                    right["protected_recall_misses"][
                        "protected_recall_miss_count"
                    ]
                    - left["protected_recall_misses"][
                        "protected_recall_miss_count"
                    ]
                ),
                "hard_negative_directional_error_count_by_family": {
                    first: left_hard,
                    second: right_hard,
                },
                "hard_negative_directional_errors_by_boundary": (
                    boundary_comparisons
                ),
                "family_quality_ranking_performed": False,
            }
        )
    return output


def grouped_cv_vs_fresh_comparison(
    candidate_analysis: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for candidate_id, scopes in candidate_analysis.items():
        group = scopes[GROUP_SCOPE]
        fresh = scopes[POOLED_FRESH_SCOPE]
        group_fp = group["protected_false_positives"]
        fresh_fp = fresh["protected_false_positives"]
        group_unsupported = group["unsupported_misses"]
        fresh_unsupported = fresh["unsupported_misses"]
        output.append(
            {
                "candidate_id": candidate_id,
                "protected_false_positive_weakness": (
                    "observed in both"
                    if group_fp["protected_false_positive_count"] > 0
                    and fresh_fp["protected_false_positive_count"] > 0
                    else "not observed in both"
                ),
                "grouped_cv_protected_false_positive_count": group_fp[
                    "protected_false_positive_count"
                ],
                "grouped_cv_non_protected_denominator": group_fp[
                    "non_protected_gold_count"
                ],
                "grouped_cv_protected_false_positive_rate": group_fp[
                    "protected_false_positive_rate"
                ],
                "fresh_protected_false_positive_count": fresh_fp[
                    "protected_false_positive_count"
                ],
                "fresh_non_protected_denominator": fresh_fp[
                    "non_protected_gold_count"
                ],
                "fresh_protected_false_positive_rate": fresh_fp[
                    "protected_false_positive_rate"
                ],
                "protected_false_positive_rate_delta_fresh_minus_grouped_cv": (
                    fresh_fp["protected_false_positive_rate"]
                    - group_fp["protected_false_positive_rate"]
                ),
                "grouped_cv_unsupported_recall": group_unsupported[
                    "unsupported_recall"
                ],
                "fresh_unsupported_recall": fresh_unsupported[
                    "unsupported_recall"
                ],
                "unsupported_recall_delta_fresh_minus_grouped_cv": (
                    fresh_unsupported["unsupported_recall"]
                    - group_unsupported["unsupported_recall"]
                ),
                "unsupported_recall_description": (
                    "lower in fresh evaluation"
                    if fresh_unsupported["unsupported_recall"]
                    < group_unsupported["unsupported_recall"]
                    else "not lower in fresh evaluation"
                ),
                "unsupported_miss_weakness_comparison": (
                    "larger in fresh evaluation"
                    if fresh_unsupported["unsupported_miss_rate"]
                    > group_unsupported["unsupported_miss_rate"]
                    else "not larger in fresh evaluation"
                ),
                "directional_confusion_references": {
                    "grouped_cv": (
                        f"candidate_analyses.{candidate_id}.{GROUP_SCOPE}"
                    ),
                    "fresh_evaluation": (
                        f"candidate_analyses.{candidate_id}."
                        f"{POOLED_FRESH_SCOPE}"
                    ),
                },
                "causal_claim_made": False,
            }
        )
    return output


def hierarchical_analysis(
    contract: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    definition = contract["analysis_dimensions"][
        "hierarchical_architecture_diagnosis"
    ]
    stage_sections = {
        GROUP_SCOPE: candidate.get("group_aware_cv", {}).get(
            "stage_1_predictions"
        ),
        POOLED_FRESH_SCOPE: candidate.get("fresh_source_evaluation", {}).get(
            "stage_1_predictions"
        ),
    }
    has_stage_rows = any(value is not None for value in stage_sections.values())
    if has_stage_rows != definition["stage_level_predictions_persisted"]:
        raise ValueError("hierarchical stage-prediction availability changed")
    if has_stage_rows:
        summaries: dict[str, Any] = {}
        for scope, rows in stage_sections.items():
            if not isinstance(rows, list):
                raise TypeError("persisted Stage-1 predictions must be a list")
            confusion = Counter()
            for row in rows:
                if not isinstance(row, Mapping):
                    raise TypeError("Stage-1 prediction must be an object")
                gold = row.get("gold_label", row.get("gold_intent"))
                predicted = row.get(
                    "predicted_label", row.get("predicted_intent")
                )
                if not isinstance(gold, str) or not isinstance(predicted, str):
                    raise TypeError("Stage-1 prediction labels must be strings")
                confusion[(gold, predicted)] += 1
            summaries[scope] = {
                "prediction_count": len(rows),
                "correct_count": sum(
                    count
                    for (gold, predicted), count in confusion.items()
                    if gold == predicted
                ),
                "confusions": [
                    {
                        "gold_label": gold,
                        "predicted_label": predicted,
                        "count": count,
                    }
                    for (gold, predicted), count in sorted(confusion.items())
                ],
            }
        return {
            "candidate_id": HIERARCHICAL_CANDIDATE_ID,
            "architecture_configuration_persisted": True,
            "final_predictions_analyzed": True,
            "stage_level_predictions_persisted": True,
            "exact_stage_1_error_attribution_available": definition[
                "exact_stage_1_error_attribution_available"
            ],
            "stage_1_analysis": summaries,
            "limitation": None,
            "model_reconstructed": False,
            "model_rerun": False,
        }
    return {
        "candidate_id": HIERARCHICAL_CANDIDATE_ID,
        "architecture_configuration_persisted": True,
        "final_predictions_analyzed": True,
        "stage_level_predictions_persisted": has_stage_rows,
        "exact_stage_1_error_attribution_available": False,
        "limitation": definition["required_limitation_statement"],
        "model_reconstructed": False,
        "model_rerun": False,
    }


def classify_evidence(
    candidate_analysis: Mapping[str, Mapping[str, Any]],
    overlap_by_scope: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    classifications: list[dict[str, Any]] = []
    if any(
        scopes[GROUP_SCOPE]["protected_false_positives"][
            "protected_false_positive_count"
        ]
        or scopes[GROUP_SCOPE]["unsupported_misses"]["unsupported_miss_count"]
        for scopes in candidate_analysis.values()
    ):
        classifications.append(
            {
                "category": "development_distribution_boundary_weakness",
                "evidence_references": [
                    "candidate_analyses.*.pooled_group_aware_cv"
                ],
                "causal_claim": False,
            }
        )
    if any(
        scopes[POOLED_FRESH_SCOPE]["unsupported_misses"]["unsupported_recall"]
        < scopes[GROUP_SCOPE]["unsupported_misses"]["unsupported_recall"]
        or scopes[POOLED_FRESH_SCOPE]["protected_false_positives"][
            "protected_false_positive_rate"
        ]
        > scopes[GROUP_SCOPE]["protected_false_positives"][
            "protected_false_positive_rate"
        ]
        for scopes in candidate_analysis.values()
    ):
        classifications.append(
            {
                "category": "fresh_source_generalization_weakness",
                "evidence_references": ["grouped_cv_vs_fresh_evaluation"],
                "causal_claim": False,
            }
        )
    if any(
        any(
            item["count"] > 0
            for item in scope["failures_unique_to_each_candidate"].values()
        )
        for scope in overlap_by_scope.values()
    ):
        classifications.append(
            {
                "category": "architecture_specific_weakness",
                "evidence_references": [
                    "cross_candidate_overlap.*.failures_unique_to_each_candidate"
                ],
                "causal_claim": False,
            }
        )
    if any(
        scope["intersection_shared_by_all_four_candidates"]["count"] > 0
        for scope in overlap_by_scope.values()
    ):
        classifications.append(
            {
                "category": "cross_architecture_shared_weakness",
                "evidence_references": [
                    (
                        "cross_candidate_overlap.*."
                        "intersection_shared_by_all_four_candidates"
                    )
                ],
                "causal_claim": False,
            }
        )
    if not classifications:
        classifications.append(
            {
                "category": "insufficient_evidence",
                "evidence_references": ["candidate_analyses"],
                "causal_claim": False,
            }
        )
    return classifications


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
    payload: Mapping[str, Any],
    record_collections: Sequence[Mapping[str, Mapping[str, Any]]],
) -> None:
    strings = set(all_string_values(payload))
    if strings & {"text", "raw_text", "utterance", "utterance_text"}:
        raise ValueError("tracked analysis contains a raw-text field")
    raw_texts = {
        str(record["text"])
        for records in record_collections
        for record in records.values()
        if isinstance(record.get("text"), str)
    }
    if strings & raw_texts:
        raise ValueError("tracked analysis contains raw utterance text")


def build_results(
    inputs: AnalysisInputs, paths: AnalysisPaths = DEFAULT_PATHS
) -> dict[str, Any]:
    contract = inputs.contract
    candidate_ids = contract["candidate_coverage"]["candidate_ids"]
    families = contract["consumed_evidence_governance"][
        "fresh_evaluation_source_family_ids"
    ]
    protected = set(
        contract["analysis_dimensions"]["protected_false_positives"][
            "protected_intents"
        ]
    )
    pairs = contract["analysis_dimensions"]["hard_negative_boundaries"][
        "required_pairs"
    ]
    candidate_analysis: dict[str, dict[str, Any]] = {}
    scopes_by_candidate: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for candidate_id in candidate_ids:
        scopes = prediction_scopes(inputs.candidate_results[candidate_id], families)
        scopes_by_candidate[candidate_id] = scopes
        candidate_analysis[candidate_id] = {}
        for scope, rows in scopes.items():
            is_group = scope == GROUP_SCOPE
            records = (
                inputs.development_records if is_group else inputs.fresh_records
            )
            candidate_analysis[candidate_id][scope] = _scope_analysis(
                rows,
                records,
                id_field="example_id" if is_group else "record_id",
                protected_intents=protected,
                hard_negative_pairs=pairs,
            )

    overlap_by_scope: dict[str, dict[str, Any]] = {}
    for scope in [GROUP_SCOPE, *families, POOLED_FRESH_SCOPE]:
        is_group = scope == GROUP_SCOPE
        rows_by_candidate = {
            candidate_id: scopes_by_candidate[candidate_id][scope]
            for candidate_id in candidate_ids
        }
        overlap_by_scope[scope] = overlap_analysis(
            failure_sets_for_scope(
                rows_by_candidate,
                scope=scope,
                source_family_id=scope if scope in families else None,
                protected_intents=protected,
                records=(
                    inputs.development_records
                    if is_group
                    else inputs.fresh_records
                ),
                id_field="example_id" if is_group else "record_id",
                hard_negative_pairs=pairs,
            ),
            candidate_ids,
        )

    payload = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "phase": "V2-C6 targeted-remediation failure analysis",
        "execution_status": "COMPLETED",
        "contract": {
            "path": display_path(paths.contract, paths),
            "sha256": inputs.contract_sha256,
            "schema_version": CONTRACT_SCHEMA_VERSION,
        },
        "frozen_inputs": contract["source_artifacts"],
        "governance": {
            "failure_analysis_executed": True,
            "models_run": False,
            "embeddings_generated": False,
            "model_fitting_performed": False,
            "model_training_performed": False,
            "inference_performed": False,
            "model_inference_performed": False,
            "threshold_tuning_performed": False,
            "training_data_modified": False,
            "dataset_mutated": False,
            "taxonomy_modified": False,
            "candidate_selected": False,
            "candidate_ranked": False,
            "remediation_selected": False,
            "final_holdout_accessed": False,
            "step29i_authorized": False,
            "runtime_behavior_changed": False,
            "raw_utterance_text_persisted": False,
        },
        "candidate_coverage": {
            "candidate_count": 4,
            "candidate_ids": candidate_ids,
            "all_candidates_analyzed": True,
            "candidate_selected": False,
            "winner_declared": False,
        },
        "prediction_coverage": inputs.prediction_coverage,
        "candidate_analyses": candidate_analysis,
        "protected_false_positive_analysis": {
            candidate_id: {
                scope: analysis["protected_false_positives"]
                for scope, analysis in scopes.items()
            }
            for candidate_id, scopes in candidate_analysis.items()
        },
        "unsupported_miss_analysis": {
            candidate_id: {
                scope: analysis["unsupported_misses"]
                for scope, analysis in scopes.items()
            }
            for candidate_id, scopes in candidate_analysis.items()
        },
        "protected_recall_miss_analysis": {
            candidate_id: {
                scope: analysis["protected_recall_misses"]
                for scope, analysis in scopes.items()
            }
            for candidate_id, scopes in candidate_analysis.items()
        },
        "hard_negative_boundary_analysis": {
            candidate_id: {
                scope: analysis["hard_negative_boundaries"]
                for scope, analysis in scopes.items()
            }
            for candidate_id, scopes in candidate_analysis.items()
        },
        "cross_candidate_overlap": overlap_by_scope,
        "fresh_family_comparison": fresh_family_comparison(
            candidate_analysis, families
        ),
        "grouped_cv_vs_fresh_evaluation": grouped_cv_vs_fresh_comparison(
            candidate_analysis
        ),
        "hierarchical_analysis": hierarchical_analysis(
            contract, inputs.candidate_results[HIERARCHICAL_CANDIDATE_ID]
        ),
        "evidence_classification": classify_evidence(
            candidate_analysis, overlap_by_scope
        ),
        "consumed_evidence_governance": contract[
            "consumed_evidence_governance"
        ],
        "known_limitations": contract["known_limitations"],
        "continuation": {
            "remediation_choice_pre_authorized": False,
            "separate_frozen_continuation_design_required": True,
            "step29i_authorized": False,
        },
        "next_required": None,
    }
    validate_results(payload, inputs)
    return payload


def validate_results(payload: Mapping[str, Any], inputs: AnalysisInputs) -> None:
    if (
        payload.get("schema_version") != RESULT_SCHEMA_VERSION
        or payload.get("execution_status") != "COMPLETED"
        or payload.get("next_required") is not None
    ):
        raise ValueError("analysis result identity or continuation changed")
    coverage = payload.get("candidate_coverage", {})
    if (
        coverage.get("candidate_ids")
        != inputs.contract["candidate_coverage"]["candidate_ids"]
        or coverage.get("candidate_count") != 4
        or coverage.get("candidate_selected") is not False
        or coverage.get("winner_declared") is not False
    ):
        raise ValueError("analysis candidate coverage changed")
    governance = payload.get("governance", {})
    if governance.get("failure_analysis_executed") is not True:
        raise ValueError("analysis completion is not recorded")
    for field in (
        "models_run",
        "embeddings_generated",
        "model_fitting_performed",
        "model_training_performed",
        "inference_performed",
        "model_inference_performed",
        "threshold_tuning_performed",
        "training_data_modified",
        "dataset_mutated",
        "taxonomy_modified",
        "candidate_selected",
        "candidate_ranked",
        "remediation_selected",
        "final_holdout_accessed",
        "step29i_authorized",
        "runtime_behavior_changed",
        "raw_utterance_text_persisted",
    ):
        if governance.get(field) is not False:
            raise ValueError(f"prohibited analysis governance changed: {field}")
    categories = {
        row.get("category")
        for row in payload.get("evidence_classification", [])
        if isinstance(row, Mapping)
    }
    allowed_categories = set(
        inputs.contract["evidence_classification"]["allowed_categories"]
    )
    if not categories or not categories.issubset(allowed_categories):
        raise ValueError("analysis evidence classification is not frozen")
    validate_text_free(
        payload, [inputs.development_records, inputs.fresh_records]
    )


def build_manifest(
    inputs: AnalysisInputs,
    result_bytes: bytes,
    paths: AnalysisPaths = DEFAULT_PATHS,
) -> dict[str, Any]:
    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "phase": "V2-C6 targeted-remediation failure analysis",
        "execution_status": "COMPLETED",
        "analysis_contract": {
            "path": display_path(paths.contract, paths),
            "sha256": inputs.contract_sha256,
            "schema_version": CONTRACT_SCHEMA_VERSION,
        },
        "source_result": inputs.contract["source_artifacts"][
            "targeted_remediation_model_selection_results"
        ],
        "source_result_manifest": inputs.contract["source_artifacts"][
            "targeted_remediation_model_selection_results_manifest"
        ],
        "analysis_result": {
            "path": display_path(paths.result, paths),
            "sha256": sha256_bytes(result_bytes),
            "schema_version": RESULT_SCHEMA_VERSION,
        },
        "runner": {
            "path": display_path(paths.runner, paths),
            "sha256": sha256_file(paths.runner, paths),
        },
        "candidate_count": 4,
        "failure_analysis_executed": True,
        "models_run": False,
        "embeddings_generated": False,
        "inference_performed": False,
        "threshold_tuning_performed": False,
        "training_data_modified": False,
        "final_holdout_accessed": False,
        "candidate_selected": False,
        "step29i_authorized": False,
        "runtime_behavior_changed": False,
        "next_required": None,
    }
    validate_manifest(payload, inputs, result_bytes, paths)
    return payload


def validate_manifest(
    payload: Mapping[str, Any],
    inputs: AnalysisInputs,
    result_bytes: bytes,
    paths: AnalysisPaths = DEFAULT_PATHS,
) -> None:
    if (
        payload.get("schema_version") != MANIFEST_SCHEMA_VERSION
        or payload.get("execution_status") != "COMPLETED"
        or payload.get("candidate_count") != 4
        or payload.get("failure_analysis_executed") is not True
        or payload.get("next_required") is not None
    ):
        raise ValueError("analysis manifest identity or continuation changed")
    result = payload.get("analysis_result", {})
    if (
        result.get("path") != display_path(paths.result, paths)
        or result.get("schema_version") != RESULT_SCHEMA_VERSION
        or result.get("sha256") != sha256_bytes(result_bytes)
    ):
        raise ValueError("analysis result/manifest binding changed")
    contract = payload.get("analysis_contract", {})
    if (
        contract.get("sha256") != inputs.contract_sha256
        or contract.get("schema_version") != CONTRACT_SCHEMA_VERSION
    ):
        raise ValueError("analysis contract/manifest binding changed")
    for field in (
        "models_run",
        "embeddings_generated",
        "inference_performed",
        "threshold_tuning_performed",
        "training_data_modified",
        "final_holdout_accessed",
        "candidate_selected",
        "step29i_authorized",
        "runtime_behavior_changed",
    ):
        if payload.get(field) is not False:
            raise ValueError(f"prohibited manifest governance changed: {field}")


def preflight(paths: AnalysisPaths = DEFAULT_PATHS) -> dict[str, Any]:
    inputs = load_preconditions(paths, require_outputs_absent=False)
    families = inputs.contract["consumed_evidence_governance"][
        "fresh_evaluation_source_family_ids"
    ]
    return {
        "status": "READY",
        "phase": "V2-C6 targeted-remediation failure analysis",
        "contract_sha256": inputs.contract_sha256,
        "candidate_ids": inputs.contract["candidate_coverage"]["candidate_ids"],
        "expected_scopes": [GROUP_SCOPE, *families, POOLED_FRESH_SCOPE],
        "prediction_coverage": inputs.prediction_coverage,
        "result_artifact_present": paths.result.exists(),
        "manifest_artifact_present": paths.result_manifest.exists(),
        "prediction_level_analysis_executed": False,
        "files_written": False,
        "models_run": False,
        "embeddings_generated": False,
        "model_fitting_performed": False,
        "inference_performed": False,
        "threshold_tuning_performed": False,
        "final_holdout_accessed": False,
        "step29i_authorized": False,
        "next_required": None,
    }


def run_analysis(paths: AnalysisPaths = DEFAULT_PATHS) -> tuple[Path, Path]:
    inputs = load_preconditions(paths, require_outputs_absent=True)
    payload = build_results(inputs, paths)
    result_bytes = stable_json_bytes(payload)
    manifest = build_manifest(inputs, result_bytes, paths)
    validate_text_free(
        manifest, [inputs.development_records, inputs.fresh_records]
    )
    durable_create(paths.result, result_bytes)
    durable_create(paths.result_manifest, stable_json_bytes(manifest))
    return paths.result, paths.result_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate or execute the frozen read-only V2-C6 targeted-"
            "remediation failure analysis"
        )
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.preflight:
        print(json.dumps(preflight(), indent=2, sort_keys=True))
        return
    created = run_analysis()
    print("\n".join(display_path(path, DEFAULT_PATHS) for path in created))


if __name__ == "__main__":
    main()
