"""Freeze deterministic, text-free CFPB semantic labels for V2-C evaluation."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

try:
    from scripts import annotate_cfpb_semantic_with_codex as workflow
    from scripts import build_cfpb_annotation_workfile as contract
    from scripts import review_cfpb_semantic_annotations as reviewer
except ModuleNotFoundError:  # Direct execution adds scripts/, not the repo root.
    import annotate_cfpb_semantic_with_codex as workflow
    import build_cfpb_annotation_workfile as contract
    import review_cfpb_semantic_annotations as reviewer


DEFAULT_OUTPUT_PATH = (
    contract.EXTERNAL_ROOT
    / "processed/cfpb/cfpb_semantic_final_labels.jsonl"
)
FINAL_LABEL_SCHEMA_VERSION = "cfpb-semantic-final-label.v1"
SCORING_CONTRACT_VERSION = "cfpb-semantic-scoring.v1"
EXPECTED_HOLDOUT_COUNT = 1_800
UNSUPPORTED_TARGET = "unsupported_or_uncertain"
MODEL_INTENTS = (*contract.SUPPORTED_INTENTS, UNSUPPORTED_TARGET)
FINAL_ANNOTATION_SOURCES = frozenset(
    {
        workflow.ANNOTATION_SOURCE,
        "CODEX_DUAL_PASS_AGREEMENT",
        workflow.ADJUDICATION_ANNOTATION_SOURCE,
    }
)
FINAL_LABEL_FIELDS = (
    "schema_version",
    "scoring_contract_version",
    "complaint_id",
    "narrative_sha256",
    "original_mapping_status",
    "final_review_category",
    "final_supported_intents",
    "final_annotation_confidence",
    "final_annotation_source",
    "pass_b_required",
    "pass_c_required",
    "primary_single_label_evaluable",
    "primary_single_label_target",
)


def _validate_final_semantics(
    category: Any,
    supported_intents: Any,
    confidence: Any,
) -> None:
    """Apply the frozen annotation contract without retaining free text."""
    contract.validate_annotation_fields(
        {
            "adjudication_status": "ADJUDICATED",
            "annotation_confidence": confidence,
            "annotation_note": "Final semantic label contract validation.",
            "review_category": category,
            "reviewer_id": "CFPB_FINAL_LABEL_EXPORT",
            "secondary_review_required": False,
            "supported_intents": supported_intents,
        }
    )


def primary_single_label_target(
    final_review_category: str,
    final_supported_intents: Sequence[str],
) -> str | None:
    """Map final semantics to the frozen single-label evaluation target."""
    if final_review_category == "SINGLE_SUPPORTED_INTENT":
        if len(final_supported_intents) != 1:
            raise ValueError("SINGLE_SUPPORTED_INTENT requires exactly one target")
        if final_supported_intents[0] not in contract.SUPPORTED_INTENTS:
            raise ValueError("single-label target is outside the frozen taxonomy")
        return final_supported_intents[0]
    if final_review_category == "MULTI_SUPPORTED_INTENT":
        if len(final_supported_intents) < 2:
            raise ValueError("MULTI_SUPPORTED_INTENT requires at least two intents")
        if set(final_supported_intents) - set(contract.SUPPORTED_INTENTS):
            raise ValueError("multi-intent target is outside the frozen taxonomy")
        return None
    if final_review_category in {
        "UNSUPPORTED",
        "UNCLEAR_OR_INSUFFICIENT",
        "NO_CURRENT_REQUEST",
    }:
        if final_supported_intents:
            raise ValueError(
                f"{final_review_category} cannot have a supported intent target"
            )
        return UNSUPPORTED_TARGET
    raise ValueError(f"invalid final review category: {final_review_category}")


def multi_intent_prediction_is_hit(
    classifier_prediction: str,
    final_supported_intents: Sequence[str],
) -> bool:
    """Return the frozen secondary set-membership result for a MULTI label."""
    if classifier_prediction not in MODEL_INTENTS:
        raise ValueError("classifier prediction is outside the frozen 9-class taxonomy")
    if len(final_supported_intents) < 2:
        raise ValueError("multi-intent membership requires at least two intents")
    if set(final_supported_intents) - set(contract.SUPPORTED_INTENTS):
        raise ValueError("multi-intent membership contains an unsupported intent")
    return classifier_prediction in final_supported_intents


def _validate_source_records(
    source_records: Sequence[Mapping[str, Any]],
    *,
    expected_count: int,
) -> None:
    if len(source_records) != expected_count:
        raise ValueError(
            f"expected {expected_count} source holdout records, got "
            f"{len(source_records)}"
        )
    reviewer.validate_records(source_records)
    complaint_ids = [row.get("complaint_id") for row in source_records]
    if not all(isinstance(value, str) and value.strip() for value in complaint_ids):
        raise ValueError("source holdout records require non-empty complaint_id")
    if len(complaint_ids) != len(set(complaint_ids)):
        raise ValueError("source holdout complaint_id values must be unique")


def _validate_annotation_stores(
    source_records: Sequence[Mapping[str, Any]],
    first_pass_records: Sequence[Mapping[str, Any]],
    second_pass_records: Sequence[Mapping[str, Any]],
    adjudication_records: Sequence[Mapping[str, Any]],
) -> tuple[set[str], dict[str, dict[str, Any]]]:
    for record in first_pass_records:
        workflow.validate_first_pass_record(record)
    workflow.validate_first_pass_source_linkage(source_records, first_pass_records)

    source_hashes = {row["narrative_sha256"] for row in source_records}
    first_hashes = {row["narrative_sha256"] for row in first_pass_records}
    if len(first_pass_records) != len(source_records) or first_hashes != source_hashes:
        raise ValueError("final export requires one Pass-A record per source record")
    if any(row["status"] != "SUCCEEDED" for row in first_pass_records):
        raise ValueError("final export requires every Pass-A record to succeed")

    required_b = set(
        workflow.second_pass_required_hashes(source_records, first_pass_records)
    )
    for record in second_pass_records:
        workflow.validate_second_pass_record(record)
    workflow.validate_second_pass_source_linkage(
        source_records,
        first_pass_records,
        second_pass_records,
    )
    second_hashes = {row["narrative_sha256"] for row in second_pass_records}
    if len(second_pass_records) != len(required_b) or second_hashes != required_b:
        raise ValueError("final export requires one Pass-B record per required hash")
    if any(row["status"] != "SUCCEEDED" for row in second_pass_records):
        raise ValueError("final export requires every required Pass-B record to succeed")

    comparisons, _ = workflow.compare_annotation_passes(
        source_records,
        first_pass_records,
        second_pass_records,
    )
    comparison_by_hash = {
        row["narrative_sha256"]: row for row in comparisons
    }
    required_c = {
        digest
        for digest, comparison in comparison_by_hash.items()
        if comparison["pass_c_required"]
    }

    for record in adjudication_records:
        workflow.validate_adjudication_record(record)
    workflow.validate_adjudication_source_linkage(
        source_records,
        first_pass_records,
        second_pass_records,
        adjudication_records,
    )
    adjudication_hashes = {
        row["narrative_sha256"] for row in adjudication_records
    }
    if (
        len(adjudication_records) != len(required_c)
        or adjudication_hashes != required_c
    ):
        raise ValueError("final export requires one Pass-C record per required hash")
    for record in adjudication_records:
        if record["status"] != "SUCCEEDED":
            raise ValueError("final export requires every required Pass-C record to succeed")
        if record["resolution_status"] != "RESOLVED":
            raise ValueError("UNRESOLVED Pass-C record cannot be finalized")
    return required_b, comparison_by_hash


def validate_final_export_records(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_count: int = EXPECTED_HOLDOUT_COUNT,
) -> None:
    """Validate the complete, allowlisted final-label schema and scoring fields."""
    if len(records) != expected_count:
        raise ValueError(
            f"expected {expected_count} final semantic labels, got {len(records)}"
        )

    complaint_ids: set[str] = set()
    narrative_hashes: set[str] = set()
    for record in records:
        if set(record) != set(FINAL_LABEL_FIELDS):
            raise ValueError("final-label fields differ from schema")
        if record["schema_version"] != FINAL_LABEL_SCHEMA_VERSION:
            raise ValueError("unexpected final-label schema version")
        if record["scoring_contract_version"] != SCORING_CONTRACT_VERSION:
            raise ValueError("unexpected CFPB scoring-contract version")

        complaint_id = record["complaint_id"]
        if not isinstance(complaint_id, str) or not complaint_id.strip():
            raise ValueError("final label requires a non-empty complaint_id")
        if complaint_id in complaint_ids:
            raise ValueError("final-label complaint_id values must be unique")
        complaint_ids.add(complaint_id)

        digest = record["narrative_sha256"]
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("final label requires a narrative SHA-256")
        if digest in narrative_hashes:
            raise ValueError("final-label narrative hashes must be unique")
        narrative_hashes.add(digest)

        if record["original_mapping_status"] not in contract.SEMANTIC_REVIEW_STATUSES:
            raise ValueError("final label has an invalid holdout stratum")
        _validate_final_semantics(
            record["final_review_category"],
            record["final_supported_intents"],
            record["final_annotation_confidence"],
        )
        expected_target = primary_single_label_target(
            record["final_review_category"],
            record["final_supported_intents"],
        )
        if record["primary_single_label_target"] != expected_target:
            raise ValueError("final label has an invalid single-label target")
        if record["primary_single_label_evaluable"] is not (
            expected_target is not None
        ):
            raise ValueError("final label has an invalid primary-evaluable flag")
        if not isinstance(record["pass_b_required"], bool) or not isinstance(
            record["pass_c_required"], bool
        ):
            raise TypeError("final-label pass requirements must be boolean")
        if record["pass_c_required"] and not record["pass_b_required"]:
            raise ValueError("Pass C cannot be required without Pass B")

        annotation_source = record["final_annotation_source"]
        if annotation_source not in FINAL_ANNOTATION_SOURCES:
            raise ValueError("final label has an invalid annotation source")
        expected_source = (
            workflow.ADJUDICATION_ANNOTATION_SOURCE
            if record["pass_c_required"]
            else (
                "CODEX_DUAL_PASS_AGREEMENT"
                if record["pass_b_required"]
                else workflow.ANNOTATION_SOURCE
            )
        )
        if annotation_source != expected_source:
            raise ValueError("final annotation source conflicts with pass routing")


def build_final_labels(
    source_records: Sequence[Mapping[str, Any]],
    first_pass_records: Sequence[Mapping[str, Any]],
    second_pass_records: Sequence[Mapping[str, Any]],
    adjudication_records: Sequence[Mapping[str, Any]],
    *,
    expected_count: int = EXPECTED_HOLDOUT_COUNT,
) -> list[dict[str, Any]]:
    """Derive final labels without making or importing any new semantic judgment."""
    _validate_source_records(source_records, expected_count=expected_count)
    required_b, comparison_by_hash = _validate_annotation_stores(
        source_records,
        first_pass_records,
        second_pass_records,
        adjudication_records,
    )
    first_by_hash = {
        row["narrative_sha256"]: row for row in first_pass_records
    }
    adjudication_by_hash = {
        row["narrative_sha256"]: row for row in adjudication_records
    }

    final_records: list[dict[str, Any]] = []
    for source in source_records:
        digest = source["narrative_sha256"]
        comparison = comparison_by_hash.get(digest)
        pass_b_required = digest in required_b
        pass_c_required = bool(
            comparison is not None and comparison["pass_c_required"]
        )
        if pass_c_required:
            selected = adjudication_by_hash[digest]
            category = selected["final_review_category"]
            supported_intents = list(selected["final_supported_intents"])
            confidence = selected["final_annotation_confidence"]
            annotation_source = workflow.ADJUDICATION_ANNOTATION_SOURCE
        else:
            selected = first_by_hash[digest]
            category = selected["review_category"]
            supported_intents = list(selected["supported_intents"])
            confidence = selected["annotation_confidence"]
            annotation_source = (
                "CODEX_DUAL_PASS_AGREEMENT"
                if pass_b_required
                else workflow.ANNOTATION_SOURCE
            )

        _validate_final_semantics(category, supported_intents, confidence)
        target = primary_single_label_target(category, supported_intents)
        final_records.append(
            {
                "schema_version": FINAL_LABEL_SCHEMA_VERSION,
                "scoring_contract_version": SCORING_CONTRACT_VERSION,
                "complaint_id": source["complaint_id"],
                "narrative_sha256": digest,
                "original_mapping_status": source["mapping_status"],
                "final_review_category": category,
                "final_supported_intents": supported_intents,
                "final_annotation_confidence": confidence,
                "final_annotation_source": annotation_source,
                "pass_b_required": pass_b_required,
                "pass_c_required": pass_c_required,
                "primary_single_label_evaluable": target is not None,
                "primary_single_label_target": target,
            }
        )

    validate_final_export_records(final_records, expected_count=expected_count)
    return final_records


def export_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return label-only coverage and protected-write reporting counts."""
    categories = Counter(row["final_review_category"] for row in records)
    sources = Counter(row["final_annotation_source"] for row in records)
    primary_count = sum(
        bool(row["primary_single_label_evaluable"]) for row in records
    )
    protected_counts = {
        intent: sum(intent in row["final_supported_intents"] for row in records)
        for intent in sorted(contract.PROTECTED_WRITE_INTENTS)
    }
    return {
        "total_holdout_count": len(records),
        "primary_single_label_evaluable_count": primary_count,
        "excluded_multi_intent_count": len(records) - primary_count,
        "primary_evaluation_coverage": primary_count / len(records),
        "category_counts": dict(sorted(categories.items())),
        "final_annotation_source_counts": dict(sorted(sources.items())),
        "protected_write_counts": protected_counts,
    }


def write_final_labels(
    path: Path,
    records: Sequence[Mapping[str, Any]],
    *,
    expected_count: int = EXPECTED_HOLDOUT_COUNT,
) -> bytes:
    """Validate and atomically write deterministic final-label bytes."""
    validate_final_export_records(records, expected_count=expected_count)
    payload = contract.stable_jsonl_bytes(records)
    reviewer.atomic_write_bytes(path, payload)
    if path.read_bytes() != payload:
        raise RuntimeError("final-label artifact differs after atomic write")
    return payload


def _print_summary(summary: Mapping[str, Any]) -> None:
    print(
        " | ".join(
            (
                f"Total holdout: {summary['total_holdout_count']}",
                f"Primary single-label evaluable: {summary['primary_single_label_evaluable_count']}",
                f"Excluded MULTI_SUPPORTED_INTENT: {summary['excluded_multi_intent_count']}",
                f"Primary evaluation coverage: {summary['primary_evaluation_coverage']:.6f}",
            )
        )
    )
    print(f"Final categories: {summary['category_counts']}")
    print(f"Final annotation sources: {summary['final_annotation_source_counts']}")
    print(f"Protected-write labels: {summary['protected_write_counts']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--check",
        action="store_true",
        help="validate inputs and the existing final artifact without rewriting",
    )
    mode.add_argument(
        "--write",
        action="store_true",
        help="atomically write the validated final artifact",
    )
    parser.add_argument("--input", type=Path, default=workflow.DEFAULT_INPUT_PATH)
    parser.add_argument(
        "--first-pass",
        type=Path,
        default=workflow.DEFAULT_FIRST_PASS_PATH,
    )
    parser.add_argument(
        "--second-pass",
        type=Path,
        default=workflow.DEFAULT_SECOND_PASS_PATH,
    )
    parser.add_argument(
        "--adjudication",
        type=Path,
        default=workflow.DEFAULT_ADJUDICATION_PATH,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source_records = reviewer.load_records(
        args.input,
        expected_mapping_counts=reviewer.EXPECTED_MAPPING_COUNTS,
    )
    first_pass_records = workflow.load_first_pass_records(args.first_pass)
    second_pass_records = workflow.load_second_pass_records(args.second_pass)
    adjudication_records = workflow.load_adjudication_records(args.adjudication)
    final_records = build_final_labels(
        source_records,
        first_pass_records,
        second_pass_records,
        adjudication_records,
    )
    payload = contract.stable_jsonl_bytes(final_records)

    if args.write:
        write_final_labels(args.output, final_records)
        print(f"Wrote frozen CFPB semantic labels: {args.output}")
    else:
        if not args.output.exists():
            raise FileNotFoundError(
                f"final-label artifact does not exist: {args.output}; use --write"
            )
        existing_records, existing_payload = contract.load_jsonl(args.output)
        validate_final_export_records(existing_records)
        if existing_payload != payload:
            raise ValueError(
                "existing final-label artifact differs from deterministic inputs"
            )
        print(f"Validated frozen CFPB semantic labels: {args.output}")

    _print_summary(export_summary(final_records))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
