import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest

from scripts import build_cfpb_taxonomy_mapping as builder

FIELD_NAMES = [
    "Date received",
    "Product",
    "Sub-product",
    "Issue",
    "Sub-issue",
    "Consumer complaint narrative",
    "Complaint ID",
]


def write_fixture_archive(path: Path) -> None:
    rows = [
        {
            "Date received": "01/01/2024",
            "Product": "Checking or savings account",
            "Sub-product": "Checking account",
            "Issue": "Deposits and withdrawals",
            "Sub-issue": "Deposits and withdrawals",
            "Consumer complaint narrative": "First narrative.",
            "Complaint ID": "101",
        },
        {
            "Date received": "01/02/2024",
            "Product": "Checking or savings account",
            "Sub-product": "Checking account",
            "Issue": "Deposits and withdrawals",
            "Sub-issue": "Deposits and withdrawals",
            "Consumer complaint narrative": "Repeated complaint ID fixture.",
            "Complaint ID": "101",
        },
        {
            "Date received": "01/03/2024",
            "Product": "Credit card or prepaid card",
            "Sub-product": "General-purpose credit card or charge card",
            "Issue": "Trouble using your card",
            "Sub-issue": "Can't use card to make purchases",
            "Consumer complaint narrative": "Card topic.",
            "Complaint ID": "102",
        },
        {
            "Date received": "01/04/2024",
            "Product": "Credit card or prepaid card",
            "Sub-product": "General-purpose credit card or charge card",
            "Issue": "Fraud or scam",
            "Sub-issue": "Card was charged for something not purchased",
            "Consumer complaint narrative": "Fraud topic without a requested action.",
            "Complaint ID": "103",
        },
        {
            "Date received": "01/05/2024",
            "Product": "Mortgage",
            "Sub-product": "Conventional home mortgage",
            "Issue": "Struggling to pay mortgage",
            "Sub-issue": "",
            "Consumer complaint narrative": "Unsupported mortgage topic.",
            "Complaint ID": "104",
        },
        {
            "Date received": "01/06/2024",
            "Product": "Novel product",
            "Sub-product": "Novel sub-product",
            "Issue": "Novel issue",
            "Sub-issue": "Novel sub-issue",
            "Consumer complaint narrative": "Unknown taxonomy topic.",
            "Complaint ID": "105",
        },
        {
            "Date received": "01/07/2024",
            "Product": "Mortgage",
            "Sub-product": "Conventional home mortgage",
            "Issue": "Struggling to pay mortgage",
            "Sub-issue": "",
            "Consumer complaint narrative": "",
            "Complaint ID": "106",
        },
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=FIELD_NAMES)
    writer.writeheader()
    writer.writerows(rows)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("fixture.csv", stream.getvalue())


@pytest.fixture()
def fixture_archive(tmp_path: Path) -> Path:
    path = tmp_path / "cfpb_fixture.zip"
    write_fixture_archive(path)
    return path


def artifact_objects(path: Path) -> tuple[dict, dict]:
    inventory_bytes, mapping_bytes = builder.build_artifacts(
        [path], manifest_sha256="manifest-hash", profile_sha256="profile-hash"
    )
    return json.loads(inventory_bytes), json.loads(mapping_bytes)


def test_outputs_are_byte_deterministic(fixture_archive: Path) -> None:
    first = builder.build_artifacts(
        [fixture_archive], manifest_sha256="manifest-hash", profile_sha256="profile-hash"
    )
    second = builder.build_artifacts(
        [fixture_archive], manifest_sha256="manifest-hash", profile_sha256="profile-hash"
    )

    assert first == second


def test_complete_inventory_and_exactly_one_status(fixture_archive: Path) -> None:
    inventory, mapping = artifact_objects(fixture_archive)
    full_tuples = inventory["marginal_counts"][
        "full_product_sub_product_issue_sub_issue"
    ]
    assignments = mapping["mapped_taxonomy_combinations"]

    assert inventory["summary"] == {
        "distinct_complaint_ids": 5,
        "narrative_bearing_rows": 6,
        "narrative_count": 6,
        "total_full_records_rows_scanned": 7,
        "unique_full_taxonomy_tuples": 5,
        "unique_issues": 5,
        "unique_products": 4,
        "unique_sub_issues": 4,
        "unique_sub_products": 4,
    }
    assert len(assignments) == len(full_tuples) == 5
    assert len(
        {
            (
                row["product"],
                row["sub_product"],
                row["issue"],
                row["sub_issue"],
            )
            for row in assignments
        }
    ) == 5
    assert all(row["mapping_status"] in builder.ALLOWED_STATUSES for row in assignments)

    deposit = next(
        row for row in full_tuples if row["issue"] == "Deposits and withdrawals"
    )
    assert deposit["row_count"] == 2
    assert deposit["narrative_count"] == 2
    assert deposit["distinct_complaint_ids"] == 1


def test_intents_counts_and_explicit_default(fixture_archive: Path) -> None:
    inventory, mapping = artifact_objects(fixture_archive)
    assignments = mapping["mapped_taxonomy_combinations"]
    valid_intents = set(builder.SENTINELVOICE_INTENTS)

    assert all(
        set(row["candidate_sentinelvoice_intents"]).issubset(valid_intents)
        for row in assignments
    )
    assert sum(
        lane["narrative_count"]
        for lane in mapping["coverage"]["by_status"].values()
    ) == inventory["summary"]["narrative_count"]
    unknown = next(row for row in assignments if row["product"] == "Novel product")
    assert unknown["mapping_status"] == "UNSUPPORTED"
    assert unknown["matched_rule_id"] == "explicit_default_unsupported"
    assert unknown["sentinelvoice_intent"] == "unsupported_or_uncertain"


def test_same_precedence_rule_conflict_is_rejected() -> None:
    duplicate_selector_rules = (
        builder.MappingRule(
            rule_id="first",
            mapping_status="NEAR_MATCH",
            candidate_intents=("card_status",),
            rationale="fixture",
            product="Credit card",
        ),
        builder.MappingRule(
            rule_id="second",
            mapping_status="AMBIGUOUS",
            candidate_intents=("card_status", "escalation"),
            rationale="fixture",
            product="Credit card",
        ),
        builder.MappingRule(
            rule_id="default",
            mapping_status="UNSUPPORTED",
            candidate_intents=("unsupported_or_uncertain",),
            rationale="fixture",
        ),
    )

    with pytest.raises(ValueError, match="same-precedence rules overlap"):
        builder.validate_rules(duplicate_selector_rules)


def test_protected_write_exact_match_requires_explicit_allowlist() -> None:
    unsafe_rules = (
        builder.MappingRule(
            rule_id="unsafe",
            mapping_status="EXACT_MATCH",
            candidate_intents=("freeze_card",),
            rationale="fixture",
            protected_action_risk="POTENTIAL_PROTECTED_WRITE",
            product="Credit card",
        ),
        builder.MappingRule(
            rule_id="default",
            mapping_status="UNSUPPORTED",
            candidate_intents=("unsupported_or_uncertain",),
            rationale="fixture",
        ),
    )

    assert not builder.PROTECTED_EXACT_RULE_ALLOWLIST
    with pytest.raises(ValueError, match="explicit rule allowlist"):
        builder.validate_rules(unsafe_rules)


def test_builder_does_not_modify_sources_or_existing_mappings(
    fixture_archive: Path,
) -> None:
    existing_paths = (
        builder.EXTERNAL_ROOT / "banking77_intent_mapping.json",
        builder.EXTERNAL_ROOT / "clinc_finance_intent_mapping.json",
    )
    source_hash = hashlib.sha256(fixture_archive.read_bytes()).hexdigest()
    source_stat = fixture_archive.stat()
    existing_bytes = {path: path.read_bytes() for path in existing_paths}

    builder.build_artifacts(
        [fixture_archive], manifest_sha256="manifest-hash", profile_sha256="profile-hash"
    )

    assert hashlib.sha256(fixture_archive.read_bytes()).hexdigest() == source_hash
    assert fixture_archive.stat().st_mtime_ns == source_stat.st_mtime_ns
    assert {path: path.read_bytes() for path in existing_paths} == existing_bytes


def test_generated_complete_archive_artifacts_reconcile() -> None:
    inventory = json.loads(builder.INVENTORY_PATH.read_text(encoding="utf-8"))
    mapping = json.loads(builder.MAPPING_PATH.read_text(encoding="utf-8"))
    full_tuples = inventory["marginal_counts"][
        "full_product_sub_product_issue_sub_issue"
    ]
    assignments = mapping["mapped_taxonomy_combinations"]

    assert len(full_tuples) == len(assignments) == inventory["summary"][
        "unique_full_taxonomy_tuples"
    ]
    assert all(
        row["mapping_status"] in builder.ALLOWED_STATUSES for row in assignments
    )
    assert all(
        set(row["candidate_sentinelvoice_intents"]).issubset(
            builder.SENTINELVOICE_INTENTS
        )
        for row in assignments
    )
    assert sum(
        lane["narrative_count"]
        for lane in mapping["coverage"]["by_status"].values()
    ) == inventory["summary"]["narrative_count"]
    assert mapping["coverage"]["protected_write_exact_match"] == {
        "narrative_count": 0,
        "rule_count": 0,
    }
