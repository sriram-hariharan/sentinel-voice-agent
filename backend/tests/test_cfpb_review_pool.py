import csv
import hashlib
import io
import json
import zipfile
from collections import Counter
from pathlib import Path

import pytest

from scripts import build_cfpb_review_pool as builder
from scripts import plan_cfpb_narrative_candidates as planner

FIELD_NAMES = [
    "Date received",
    "Product",
    "Sub-product",
    "Issue",
    "Sub-issue",
    "Consumer complaint narrative",
    "Complaint ID",
]


def sized_text(prefix: str, character_length: int) -> str:
    if len(prefix) > character_length:
        raise ValueError("fixture prefix exceeds requested narrative length")
    return prefix + ("x" * (character_length - len(prefix)))


def fixture_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    complaint_number = 1_000
    lengths_by_status = {
        "NEAR_MATCH": [100, 150, 200, 250, 300, 350, 600, 700, 800, 900, 1_200, 1_500],
        "AMBIGUOUS": [100, 150, 200, 250, 300, 600, 700, 800, 900, 1_200, 1_500, 2_500],
        "UNSUPPORTED": [100, 150, 200, 250, 300, 600, 700, 800, 900, 1_200, 1_500, 2_500],
    }
    for status, lengths in lengths_by_status.items():
        for index, character_length in enumerate(lengths):
            complaint_number += 1
            if status == "NEAR_MATCH":
                product = f"Near product {index % 3}"
                prefix = f"near narrative {index}:"
            elif status == "AMBIGUOUS":
                product = f"Ambiguous product {index % 4}"
                prefix = f"ambiguous narrative {index}:"
            else:
                product = f"OOD product {index % 6}"
                prefix = f"freeze my card and create a dispute unsupported {index}:"
            rows.append(
                {
                    "Date received": f"2025-{(index % 12) + 1:02d}-15",
                    "Product": product,
                    "Sub-product": f"{product} subtype",
                    "Issue": f"{status} issue {index}",
                    "Sub-issue": f"{status} sub-issue {index}",
                    "Consumer complaint narrative": sized_text(prefix, character_length),
                    "Complaint ID": str(complaint_number),
                }
            )

    near_first = next(row for row in rows if row["Issue"] == "NEAR_MATCH issue 0")
    duplicate = dict(near_first)
    complaint_number += 1
    duplicate["Complaint ID"] = str(complaint_number)
    rows.append(duplicate)

    shared_narrative = sized_text("shared cross-lane narrative:", 300)
    for status, product in (
        ("NEAR_MATCH", "Near product 0"),
        ("AMBIGUOUS", "Ambiguous product 0"),
        ("UNSUPPORTED", "OOD product 0"),
    ):
        complaint_number += 1
        rows.append(
            {
                "Date received": "2025-06-30",
                "Product": product,
                "Sub-product": f"{product} subtype",
                "Issue": f"{status} shared issue",
                "Sub-issue": f"{status} shared sub-issue",
                "Consumer complaint narrative": shared_narrative,
                "Complaint ID": str(complaint_number),
            }
        )

    complaint_number += 1
    rows.append(
        {
            "Date received": "2025-07-01",
            "Product": "Near product 0",
            "Sub-product": "Near product 0 subtype",
            "Issue": "NEAR_MATCH privacy issue",
            "Sub-issue": "NEAR_MATCH privacy sub-issue",
            "Consumer complaint narrative": "Contact private-fixture@example.com",
            "Complaint ID": str(complaint_number),
        }
    )
    return rows


def status_from_issue(issue: str) -> str:
    return issue.split(" ", maxsplit=1)[0]


def candidates_for(status: str, issue: str) -> list[str]:
    suffix = sum(ord(character) for character in issue) % 3
    if status == "NEAR_MATCH":
        return (
            ["account_balance", "recent_transactions"]
            if suffix == 0
            else ["card_status", "transaction_details"]
        )
    if status == "AMBIGUOUS":
        return (
            ["create_dispute", "escalation"]
            if suffix == 0
            else ["freeze_card", "transaction_details"]
        )
    return ["unsupported_or_uncertain"]


def fixture_mapping(rows: list[dict[str, str]]) -> dict[str, object]:
    assignments: dict[tuple[str, str, str, str], dict[str, object]] = {}
    lane_counts: Counter[str] = Counter()
    for row in rows:
        status = status_from_issue(row["Issue"])
        lane_counts[status] += 1
        taxonomy_tuple = (
            row["Product"],
            row["Sub-product"],
            row["Issue"],
            row["Sub-issue"],
        )
        candidates = candidates_for(status, row["Issue"])
        assignments[taxonomy_tuple] = {
            "candidate_sentinelvoice_intents": candidates,
            "issue": row["Issue"],
            "mapping_status": status,
            "matched_rule_id": f"fixture_{status.lower()}",
            "product": row["Product"],
            "protected_action_risk": (
                "POTENTIAL_PROTECTED_WRITE"
                if {"freeze_card", "create_dispute"}.intersection(candidates)
                else "NONE"
            ),
            "sub_issue": row["Sub-issue"],
            "sub_product": row["Sub-product"],
        }
    return {
        "allowed_mapping_statuses": list(planner.MAPPING_STATUSES),
        "artifact_boundary": {
            "classifier_predictions_run": False,
            "final_evaluation_sample_created": False,
            "individual_narratives_labeled": False,
            "model_training_or_retraining_run": False,
        },
        "coverage": {
            "by_status": {
                status: {"narrative_count": lane_counts[status]}
                for status in planner.MAPPING_STATUSES
            },
            "total_mapped_taxonomy_tuples": len(assignments),
            "total_narrative_bearing_records": len(rows),
        },
        "inventory_sha256": "fixture-inventory-sha256",
        "mapped_taxonomy_combinations": [
            assignments[key] for key in sorted(assignments)
        ],
        "mapping_version": "fixture.v1",
        "schema_version": "cfpb-to-sentinel-intents.v1",
    }


def write_fixture_archive(path: Path, rows: list[dict[str, str]]) -> None:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=FIELD_NAMES)
    writer.writeheader()
    writer.writerows(rows)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("fixture.csv", stream.getvalue())


@pytest.fixture()
def fixture_inputs(tmp_path: Path) -> dict[str, object]:
    rows = fixture_rows()
    archive_path = tmp_path / "cfpb_fixture.zip"
    write_fixture_archive(archive_path, rows)
    mapping = fixture_mapping(rows)
    candidate_plan = planner.build_plan(
        [archive_path],
        mapping,
        manifest_sha256="fixture-source-manifest-sha256",
        mapping_sha256="fixture-taxonomy-mapping-sha256",
    )
    return {
        "archive_path": archive_path,
        "candidate_plan": candidate_plan,
        "mapping": mapping,
        "rows": rows,
        "tmp_path": tmp_path,
    }


def build_fixture_pool(inputs: dict[str, object], suffix: str) -> dict[str, object]:
    tmp_path = inputs["tmp_path"]
    return builder.build_review_pool(
        [inputs["archive_path"]],
        inputs["mapping"],
        inputs["candidate_plan"],
        requested_sizes={
            "NEAR_MATCH": 10,
            "AMBIGUOUS": 10,
            "UNSUPPORTED": 10,
        },
        source_manifest_sha256="fixture-source-manifest-sha256",
        candidate_plan_sha256="fixture-candidate-plan-sha256",
        taxonomy_mapping_sha256="fixture-taxonomy-mapping-sha256",
        builder_script_sha256="fixture-builder-script-sha256",
        manifest_output_path=tmp_path / f"manifest-{suffix}.json",
        local_output_path=tmp_path / f"local-{suffix}.jsonl",
    )


def test_stable_selection_key_behavior() -> None:
    key = planner.TaxonomyKey("Product", "Sub-product", "Issue", "Sub-issue")
    arguments = {
        "narrative_sha256": "a" * 64,
        "mapping_status": "NEAR_MATCH",
        "complaint_id": "123",
        "source_archive": "fixture.zip",
        "date_received": "2025-01-01",
        "key": key,
    }

    first = builder.selection_key_for_record(**arguments)
    second = builder.selection_key_for_record(**arguments)
    changed = builder.selection_key_for_record(**{**arguments, "complaint_id": "124"})

    assert first == second
    assert first != changed
    assert len(first) == hashlib.sha256().digest_size


def test_length_target_allocation_uses_frozen_percentages() -> None:
    assert builder.allocate_length_targets(10) == {
        "lte_500": 4,
        "501_1000": 3,
        "1001_2000": 2,
        "gt_2000": 1,
    }


def test_selection_is_deterministic_and_meets_requested_lane_quotas(
    fixture_inputs: dict[str, object],
) -> None:
    first = build_fixture_pool(fixture_inputs, "deterministic")
    second = build_fixture_pool(fixture_inputs, "deterministic")

    assert planner.stable_json_bytes(first) == planner.stable_json_bytes(second)
    assert first["summary"]["selected_by_mapping_status"] == {
        "NEAR_MATCH": 10,
        "AMBIGUOUS": 10,
        "UNSUPPORTED": 10,
    }
    assert first["summary"]["total_selected"] == 30


def test_global_deduplication_precedence_and_privacy_exclusion(
    fixture_inputs: dict[str, object],
) -> None:
    manifest = build_fixture_pool(fixture_inputs, "dedup")
    selected_hashes = [
        record["narrative_sha256"] for record in manifest["selected_records"]
    ]

    assert len(selected_hashes) == len(set(selected_hashes))
    assert manifest["duplicate_exclusions"]["cross_lane_duplicate_hashes_resolved"] == 1
    assert manifest["duplicate_exclusions"][
        "cross_lane_selected_owner_hashes_by_status"
    ]["AMBIGUOUS"] == 1
    assert manifest["eligibility"]["cross_lane_ownership_precedence"] == [
        "AMBIGUOUS",
        "NEAR_MATCH",
        "UNSUPPORTED",
        "EXACT_MATCH",
    ]
    assert manifest["eligibility"]["privacy_excluded_records_by_status"][
        "NEAR_MATCH"
    ] == 1


def test_length_quota_redistribution_and_issue_cap(
    fixture_inputs: dict[str, object],
) -> None:
    manifest = build_fixture_pool(fixture_inputs, "quotas")
    near_quota = manifest["quota_results"]["NEAR_MATCH"]
    semantic_records = [
        record
        for record in manifest["selected_records"]
        if record["mapping_status"] in builder.SEMANTIC_REVIEW_STATUSES
    ]
    issue_counts = Counter(record["issue"] for record in semantic_records)

    assert near_quota["length_quota_shortfalls_before_reallocation"]["gt_2000"] == 1
    assert near_quota["selected_size"] == 10
    assert near_quota["total_shortfall"] == 0
    assert sum(near_quota["length_quota_excess_after_reallocation"].values()) == 1
    assert near_quota["length_quota_reallocated_count"] == 1
    assert near_quota["issue_cap_exceeded"] is False
    assert max(issue_counts.values()) <= 2


def test_candidate_overlap_and_unsupported_diversity_are_preserved(
    fixture_inputs: dict[str, object],
) -> None:
    manifest = build_fixture_pool(fixture_inputs, "diversity")
    ambiguous = [
        record
        for record in manifest["selected_records"]
        if record["mapping_status"] == "AMBIGUOUS"
    ]
    unsupported = [
        record
        for record in manifest["selected_records"]
        if record["mapping_status"] == "UNSUPPORTED"
    ]
    product_counts = Counter(record["product"] for record in unsupported)

    assert all(len(record["candidate_sentinelvoice_intents"]) == 2 for record in ambiguous)
    assert len(product_counts) == 6
    assert max(product_counts.values()) - min(product_counts.values()) <= 1
    assert manifest["summary"]["by_mapping_status"]["AMBIGUOUS"][
        "candidate_intent_counts_may_overlap"
    ] is True


def test_tracked_manifest_has_no_text_and_local_jsonl_has_text_only(
    fixture_inputs: dict[str, object],
) -> None:
    manifest = build_fixture_pool(fixture_inputs, "boundary")
    manifest_path = fixture_inputs["tmp_path"] / "manifest-boundary.json"
    local_path = fixture_inputs["tmp_path"] / "local-boundary.jsonl"
    manifest_text = manifest_path.read_text(encoding="utf-8")
    local_rows = [
        json.loads(line)
        for line in local_path.read_text(encoding="utf-8").splitlines()
    ]
    source_narratives = {
        row["Consumer complaint narrative"] for row in fixture_inputs["rows"]
    }
    required_local_fields = {
        "candidate_sentinelvoice_intents",
        "complaint_id",
        "date_received",
        "issue",
        "length_bin",
        "mapping_status",
        "narrative",
        "narrative_length",
        "narrative_sha256",
        "product",
        "source_archive",
        "sub_issue",
        "sub_product",
    }

    assert all("narrative" not in record for record in manifest["selected_records"])
    assert all(required_local_fields == set(row) for row in local_rows)
    assert all(row["narrative"] in source_narratives for row in local_rows)
    assert all(row["narrative"] not in manifest_text for row in local_rows)
    assert "private-fixture@example.com" not in local_path.read_text(encoding="utf-8")
    assert len(local_rows) == manifest["summary"]["total_selected"]


def test_no_final_labels_or_protected_action_inference(
    fixture_inputs: dict[str, object],
) -> None:
    manifest = build_fixture_pool(fixture_inputs, "labels")
    local_path = fixture_inputs["tmp_path"] / "local-labels.jsonl"
    local_rows = [
        json.loads(line)
        for line in local_path.read_text(encoding="utf-8").splitlines()
    ]
    serialized = json.dumps(manifest, sort_keys=True)
    protected_keyword_rows = [
        row for row in local_rows if row["narrative"].startswith("freeze my card")
    ]

    assert "ground_truth_intent" not in serialized
    assert "final_intent" not in serialized
    assert "gold_label" not in serialized
    assert protected_keyword_rows
    assert all(row["mapping_status"] == "UNSUPPORTED" for row in protected_keyword_rows)
    assert all(
        row["candidate_sentinelvoice_intents"] == ["unsupported_or_uncertain"]
        for row in protected_keyword_rows
    )
