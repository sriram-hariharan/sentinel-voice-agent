import copy
import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest

from scripts import plan_cfpb_narrative_candidates as planner

FIELD_NAMES = [
    "Product",
    "Sub-product",
    "Issue",
    "Sub-issue",
    "Consumer complaint narrative",
    "Complaint ID",
]


def fixture_rows() -> list[dict[str, str]]:
    return [
        {
            "Product": "Checking or savings account",
            "Sub-product": "Checking account",
            "Issue": "Deposits and withdrawals",
            "Sub-issue": "Deposits and withdrawals",
            "Consumer complaint narrative": "SENTINEL_SHARED_DUPLICATE",
            "Complaint ID": "101",
        },
        {
            "Product": "Mortgage",
            "Sub-product": "Conventional home mortgage",
            "Issue": "Struggling to pay mortgage",
            "Sub-issue": "",
            "Consumer complaint narrative": "SENTINEL_SHARED_DUPLICATE",
            "Complaint ID": "102",
        },
        {
            "Product": "Checking or savings account",
            "Sub-product": "Checking account",
            "Issue": "Deposits and withdrawals",
            "Sub-issue": "Deposits and withdrawals",
            "Consumer complaint narrative": "SENTINEL_SHARED_DUPLICATE",
            "Complaint ID": "103",
        },
        {
            "Product": "Credit card or prepaid card",
            "Sub-product": "General-purpose credit card or charge card",
            "Issue": "Fraud or scam",
            "Sub-issue": "Card charged without consent",
            "Consumer complaint narrative": "Email SENTINEL_PERSON@example.com",
            "Complaint ID": "104",
        },
        {
            "Product": "Mortgage",
            "Sub-product": "Conventional home mortgage",
            "Issue": "Struggling to pay mortgage",
            "Sub-issue": "",
            "Consumer complaint narrative": (
                "Freeze my card and create a dispute. "
                "See https://sentinel.invalid/resource"
            ),
            "Complaint ID": "105",
        },
        {
            "Product": "Credit card",
            "Sub-product": "General-purpose credit card or charge card",
            "Issue": "APR or interest rate",
            "Sub-issue": "Interest rate",
            "Consumer complaint narrative": "SENTINEL_INFORMATIONAL_POLICY_FIXTURE",
            "Complaint ID": "106",
        },
    ]


def write_fixture_archive(path: Path) -> None:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=FIELD_NAMES)
    writer.writeheader()
    writer.writerows(fixture_rows())
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("fixture.csv", stream.getvalue())


def assignment(
    *,
    product: str,
    sub_product: str,
    issue: str,
    sub_issue: str | None,
    status: str,
    candidates: list[str],
) -> dict[str, object]:
    return {
        "candidate_sentinelvoice_intents": candidates,
        "issue": issue,
        "mapping_status": status,
        "matched_rule_id": f"fixture_{status.lower()}",
        "product": product,
        "protected_action_risk": (
            "POTENTIAL_PROTECTED_WRITE"
            if {"freeze_card", "create_dispute"}.intersection(candidates)
            else "NONE"
        ),
        "sub_issue": sub_issue,
        "sub_product": sub_product,
    }


def fixture_mapping() -> dict[str, object]:
    assignments = [
        assignment(
            product="Checking or savings account",
            sub_product="Checking account",
            issue="Deposits and withdrawals",
            sub_issue="Deposits and withdrawals",
            status="NEAR_MATCH",
            candidates=["recent_transactions", "transaction_details"],
        ),
        assignment(
            product="Mortgage",
            sub_product="Conventional home mortgage",
            issue="Struggling to pay mortgage",
            sub_issue=None,
            status="UNSUPPORTED",
            candidates=["unsupported_or_uncertain"],
        ),
        assignment(
            product="Credit card or prepaid card",
            sub_product="General-purpose credit card or charge card",
            issue="Fraud or scam",
            sub_issue="Card charged without consent",
            status="AMBIGUOUS",
            candidates=["freeze_card", "create_dispute", "escalation"],
        ),
        assignment(
            product="Credit card",
            sub_product="General-purpose credit card or charge card",
            issue="APR or interest rate",
            sub_issue="Interest rate",
            status="EXACT_MATCH",
            candidates=["informational_policy"],
        ),
    ]
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
                "EXACT_MATCH": {"narrative_count": 1},
                "NEAR_MATCH": {"narrative_count": 2},
                "AMBIGUOUS": {"narrative_count": 1},
                "UNSUPPORTED": {"narrative_count": 2},
            },
            "total_mapped_taxonomy_tuples": 4,
            "total_narrative_bearing_records": 6,
        },
        "inventory_sha256": "fixture-inventory-hash",
        "mapped_taxonomy_combinations": assignments,
        "mapping_version": "fixture.v1",
        "schema_version": "cfpb-to-sentinel-intents.v1",
    }


@pytest.fixture()
def fixture_archive(tmp_path: Path) -> Path:
    path = tmp_path / "cfpb_fixture.zip"
    write_fixture_archive(path)
    return path


def build_fixture_plan(path: Path) -> dict[str, object]:
    return planner.build_plan(
        [path],
        fixture_mapping(),
        manifest_sha256="fixture-manifest-hash",
        mapping_sha256="fixture-mapping-hash",
    )


def test_mapping_lookup_uses_frozen_full_tuple_semantics(
    fixture_archive: Path,
) -> None:
    plan = build_fixture_plan(fixture_archive)

    assert plan["by_mapping_status"]["EXACT_MATCH"]["record_count"] == 1
    assert plan["by_mapping_status"]["NEAR_MATCH"]["record_count"] == 2
    assert plan["by_mapping_status"]["AMBIGUOUS"]["record_count"] == 1
    assert plan["by_mapping_status"]["UNSUPPORTED"]["record_count"] == 2


def test_exact_and_cross_lane_duplicate_detection(fixture_archive: Path) -> None:
    plan = build_fixture_plan(fixture_archive)
    overall = plan["overall"]
    near_duplicates = plan["by_mapping_status"]["NEAR_MATCH"][
        "duplicate_statistics"
    ]

    assert overall["unique_exact_narrative_hashes"] == 4
    assert overall["duplicate_narrative_groups"] == 1
    assert overall["repeated_narrative_occurrences_beyond_first"] == 2
    assert overall["hashes_appearing_in_multiple_mapping_lanes"] == 1
    assert overall["records_involved_in_cross_lane_duplicate_hashes"] == 3
    assert near_duplicates["unique_narrative_hashes"] == 1
    assert near_duplicates["duplicate_narrative_groups"] == 1
    assert near_duplicates["repeated_narrative_occurrences_beyond_first"] == 1


@pytest.mark.parametrize(
    ("character_length", "expected"),
    [
        (1, "1-200"),
        (200, "1-200"),
        (201, "201-500"),
        (500, "201-500"),
        (501, "501-1000"),
        (1000, "501-1000"),
        (1001, "1001-2000"),
        (2000, "1001-2000"),
        (2001, "2001-4000"),
        (4000, "2001-4000"),
        (4001, "4001+"),
    ],
)
def test_length_bin_inclusive_boundaries(
    character_length: int, expected: str
) -> None:
    assert planner.length_bin(character_length) == expected


def test_privacy_counts_and_privacy_free_eligibility(fixture_archive: Path) -> None:
    plan = build_fixture_plan(fixture_archive)
    overall = plan["overall"]["privacy_screening"]
    ambiguous = plan["by_mapping_status"]["AMBIGUOUS"]["privacy_screening"]
    unsupported = plan["by_mapping_status"]["UNSUPPORTED"]["privacy_screening"]

    assert overall["any_privacy_screening_signal"] == 2
    assert overall["eligible_without_privacy_signal"] == 4
    assert ambiguous["patterns"]["email_like"]["records_with_pattern"] == 1
    assert unsupported["patterns"]["url_like"]["records_with_pattern"] == 1
    assert plan["potential_evaluation_populations"][
        "exact_deduplicated_without_privacy_signal"
    ]["overall"] == 2


def test_output_is_aggregate_only_and_candidate_counts_overlap(
    fixture_archive: Path,
) -> None:
    plan = build_fixture_plan(fixture_archive)
    serialized = json.dumps(plan, sort_keys=True)
    near_candidates = plan["by_mapping_status"]["NEAR_MATCH"][
        "candidate_sentinelvoice_intent_record_counts"
    ]
    ambiguous_candidates = plan["by_mapping_status"]["AMBIGUOUS"][
        "candidate_sentinelvoice_intent_record_counts"
    ]

    for row in fixture_rows():
        assert row["Consumer complaint narrative"] not in serialized
    assert '"complaint_id":' not in serialized
    assert near_candidates == {
        "recent_transactions": 2,
        "transaction_details": 2,
    }
    assert ambiguous_candidates == {
        "create_dispute": 1,
        "escalation": 1,
        "freeze_card": 1,
    }
    assert plan["artifact_boundary"]["individual_narrative_intents_assigned"] is False
    assert "expected_sentinelvoice_intent" not in serialized


def test_lane_totals_reconcile_and_output_is_deterministic(
    fixture_archive: Path,
) -> None:
    source_hash = hashlib.sha256(fixture_archive.read_bytes()).hexdigest()
    first = build_fixture_plan(fixture_archive)
    second = build_fixture_plan(fixture_archive)

    assert first["reconciliation"] == {
        "frozen_mapping_lane_counts_match": True,
        "lane_record_count_sum": 6,
        "overall_record_count": 6,
        "status_count": 4,
    }
    assert planner.stable_json_bytes(first) == planner.stable_json_bytes(second)
    assert hashlib.sha256(fixture_archive.read_bytes()).hexdigest() == source_hash


def test_protected_write_keywords_do_not_override_frozen_mapping(
    fixture_archive: Path,
) -> None:
    plan = build_fixture_plan(fixture_archive)
    unsupported = plan["by_mapping_status"]["UNSUPPORTED"]

    assert unsupported["record_count"] == 2
    assert unsupported["candidate_sentinelvoice_intent_record_counts"] == {}


def test_protected_write_exact_match_is_rejected() -> None:
    unsafe_mapping = copy.deepcopy(fixture_mapping())
    exact = next(
        row
        for row in unsafe_mapping["mapped_taxonomy_combinations"]
        if row["mapping_status"] == "EXACT_MATCH"
    )
    exact["candidate_sentinelvoice_intents"] = ["freeze_card"]

    with pytest.raises(ValueError, match="protected-write EXACT_MATCH"):
        planner.load_frozen_assignments(unsafe_mapping)
