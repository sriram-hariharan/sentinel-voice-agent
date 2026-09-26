import json
import re
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from uuid import UUID

import pytest

from scripts.generate_v2_banking_data import (
    ANNOTATION_VERSION,
    CONFIG_VERSION,
    DEFAULT_CANONICAL_FIXTURE_PATH,
    DEFAULT_CONFIG_PATH,
    DEFAULT_OUTPUT_DIR,
    GENERATOR_VERSION,
    MANIFEST_VERSION,
    REQUIRED_PATTERN_TYPES,
    SUPPORTED_TRANSACTION_STATUSES,
    build_payloads,
    generate_files,
    sha256_bytes,
    stable_json_bytes,
    stable_uuid,
)
from scripts.seed_database import load_fixtures

CANONICAL_SHA256 = "a1e1a48dcaa0f998975dbe8aea8f04c6f93c668b7a0d5fabc7a1f9275b22540d"
AVERY_CUSTOMER_ID = "11111111-1111-4111-8111-111111111111"
MONEY_PATTERN = re.compile(r"^\d+\.\d{2}$")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _generated() -> dict:
    return _load(DEFAULT_OUTPUT_DIR / "banking.generated.json")


def _annotations() -> dict:
    return _load(DEFAULT_OUTPUT_DIR / "suspicious_patterns.json")


def _manifest() -> dict:
    return _load(DEFAULT_OUTPUT_DIR / "banking.generated.manifest.json")


def _primary_ids(fixture: dict) -> set[str]:
    fields = {
        "customers": "customer_id",
        "accounts": "account_id",
        "cards": "card_id",
        "transactions": "transaction_id",
        "disputes": "dispute_id",
        "support_cases": "case_id",
    }
    return {
        row[field]
        for collection, field in fields.items()
        for row in fixture.get(collection, [])
    }


def test_canonical_fixture_is_frozen_and_generation_does_not_modify_it(
    tmp_path: Path,
) -> None:
    before = DEFAULT_CANONICAL_FIXTURE_PATH.read_bytes()

    generate_files(output_dir=tmp_path)

    assert DEFAULT_CANONICAL_FIXTURE_PATH.read_bytes() == before
    assert sha256_bytes(before) == CANONICAL_SHA256


def test_generation_is_byte_reproducible(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"

    generate_files(output_dir=first)
    generate_files(output_dir=second)

    for filename in (
        "banking.generated.json",
        "suspicious_patterns.json",
        "banking.generated.manifest.json",
    ):
        first_bytes = (first / filename).read_bytes()
        second_bytes = (second / filename).read_bytes()
        assert first_bytes == second_bytes
        assert sha256_bytes(first_bytes) == sha256_bytes(second_bytes)


def test_checked_in_artifacts_match_fresh_generation(tmp_path: Path) -> None:
    generate_files(output_dir=tmp_path)

    for filename in (
        "banking.generated.json",
        "suspicious_patterns.json",
        "banking.generated.manifest.json",
    ):
        assert (tmp_path / filename).read_bytes() == (
            DEFAULT_OUTPUT_DIR / filename
        ).read_bytes()


def test_changing_seed_changes_generated_output() -> None:
    config = _load(DEFAULT_CONFIG_PATH)
    changed = {**config, "seed": config["seed"] + 1}

    original_payloads = build_payloads(config)
    changed_payloads = build_payloads(changed)

    assert original_payloads.banking_bytes != changed_payloads.banking_bytes
    assert sha256_bytes(original_payloads.banking_bytes) != sha256_bytes(
        changed_payloads.banking_bytes
    )


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("account_settings", "minimum_per_customer", 0),
        ("account_settings", "maximum_per_customer", 0),
        ("account_settings", "maximum_per_customer", 3),
        ("account_settings", "multiple_accounts_every", 0),
        ("card_settings", "minimum_per_customer", 0),
        ("card_settings", "maximum_per_customer", 0),
        ("card_settings", "multiple_cards_every", 0),
        ("card_settings", "frozen_primary_every", 0),
        (None, "dispute_count", -1),
        (None, "dispute_count", 9),
    ],
)
def test_unsafe_generation_bounds_fail_with_value_error(
    section: str | None,
    field: str,
    value: int,
) -> None:
    config = deepcopy(_load(DEFAULT_CONFIG_PATH))
    target = config if section is None else config[section]
    target[field] = value

    with pytest.raises(ValueError):
        build_payloads(config)


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("account_settings", "minimum_per_customer"),
        ("account_settings", "multiple_accounts_every"),
        ("card_settings", "maximum_per_customer"),
        ("card_settings", "frozen_primary_every"),
        (None, "dispute_count"),
    ],
)
def test_non_integer_generation_bounds_fail_with_value_error(
    section: str | None,
    field: str,
) -> None:
    config = deepcopy(_load(DEFAULT_CONFIG_PATH))
    target = config if section is None else config[section]
    target[field] = "invalid"

    with pytest.raises(ValueError):
        build_payloads(config)


def test_invalid_config_writes_no_partial_artifacts(tmp_path: Path) -> None:
    config = deepcopy(_load(DEFAULT_CONFIG_PATH))
    config["card_settings"]["multiple_cards_every"] = 0
    config_path = tmp_path / "invalid-config.json"
    output_dir = tmp_path / "generated"
    config_path.write_bytes(stable_json_bytes(config))

    with pytest.raises(ValueError):
        generate_files(config_path=config_path, output_dir=output_dir)

    assert not output_dir.exists()


def test_stable_identifiers_are_uuid5_unique_and_do_not_collide() -> None:
    generated = _generated()
    canonical = _load(DEFAULT_CANONICAL_FIXTURE_PATH)
    generated_ids = _primary_ids(generated)

    assert stable_uuid("customer:v2:0001") == stable_uuid(
        "customer:v2:0001"
    )
    assert stable_uuid("customer:v2:0001").version == 5
    assert len(generated_ids) == (
        len(generated["customers"])
        + len(generated["accounts"])
        + len(generated["cards"])
        + len(generated["transactions"])
        + len(generated["disputes"])
    )
    assert all(UUID(value).version == 5 for value in generated_ids)
    assert generated_ids.isdisjoint(_primary_ids(canonical))


def test_generated_references_resolve_to_the_correct_customer() -> None:
    generated = _generated()
    customers = {row["customer_id"] for row in generated["customers"]}
    accounts = {row["account_id"]: row for row in generated["accounts"]}
    cards = {row["card_id"]: row for row in generated["cards"]}
    transactions = {
        row["transaction_id"]: row for row in generated["transactions"]
    }

    assert len(customers) == len(generated["customers"])
    assert all(row["customer_id"] in customers for row in accounts.values())
    assert all(
        row["customer_id"] in customers
        and row["account_id"] in accounts
        and accounts[row["account_id"]]["customer_id"] == row["customer_id"]
        for row in cards.values()
    )
    assert all(
        row["account_id"] in accounts
        and row["card_id"] in cards
        and cards[row["card_id"]]["account_id"] == row["account_id"]
        for row in transactions.values()
    )
    assert all(
        row["customer_id"] in customers
        and row["transaction_id"] in transactions
        and accounts[transactions[row["transaction_id"]]["account_id"]][
            "customer_id"
        ]
        == row["customer_id"]
        and transactions[row["transaction_id"]]["status"] == "POSTED"
        for row in generated["disputes"]
    )


def test_generated_money_uses_exact_two_decimal_strings() -> None:
    generated = _generated()
    money_values = [
        value
        for row in generated["accounts"]
        for value in (row["current_balance"], row["available_balance"])
    ] + [row["amount"] for row in generated["transactions"]]

    assert all(isinstance(value, str) for value in money_values)
    assert all(MONEY_PATTERN.fullmatch(value) for value in money_values)
    assert all(Decimal(value).as_tuple().exponent == -2 for value in money_values)


def test_generated_timestamps_use_fixed_reference_and_lookback() -> None:
    config = _load(DEFAULT_CONFIG_PATH)
    generated = _generated()
    reference = datetime.fromisoformat(config["reference_datetime"])
    earliest = reference - timedelta(
        days=config["transaction_settings"]["lookback_days"]
    )

    for transaction in generated["transactions"]:
        occurred = datetime.fromisoformat(transaction["transaction_timestamp"])
        assert occurred.tzinfo is not None
        assert earliest <= occurred <= reference
        if transaction["status"] in {"PENDING", "DECLINED"}:
            assert transaction["posted_timestamp"] is None
        else:
            posted = datetime.fromisoformat(transaction["posted_timestamp"])
            assert occurred <= posted <= reference


def test_generated_coverage_includes_ambiguity_status_and_variation() -> None:
    generated = _generated()
    account_counts = Counter(
        row["customer_id"] for row in generated["accounts"]
    )
    card_counts = Counter(row["customer_id"] for row in generated["cards"])
    statuses = {row["status"] for row in generated["transactions"]}
    cards_by_status = Counter(row["status"] for row in generated["cards"])
    account_types = {row["account_type"] for row in generated["accounts"]}
    merchants_by_account: dict[str, Counter[str]] = defaultdict(Counter)
    merchants_by_account_and_amount: dict[
        tuple[str, str], set[str]
    ] = defaultdict(set)
    for transaction in generated["transactions"]:
        merchants_by_account[transaction["account_id"]][
            transaction["merchant_name"]
        ] += 1
        merchants_by_account_and_amount[
            (transaction["account_id"], transaction["amount"])
        ].add(transaction["merchant_name"])

    assert set(account_counts.values()) == {1, 2}
    assert set(card_counts.values()) == {1, 2}
    assert account_types == {"checking", "savings"}
    assert cards_by_status["ACTIVE"] > 0
    assert cards_by_status["FROZEN"] > 0
    assert statuses == SUPPORTED_TRANSACTION_STATUSES
    assert any(max(counts.values()) >= 2 for counts in merchants_by_account.values())
    assert any(
        len(merchants) >= 2
        for merchants in merchants_by_account_and_amount.values()
    )
    assert any(
        row["location"] == "Singapore, SG"
        for row in generated["transactions"]
    )


def test_suspicious_annotations_are_valid_non_authoritative_metadata() -> None:
    generated = _generated()
    sidecar = _annotations()
    transactions = {
        row["transaction_id"]: row for row in generated["transactions"]
    }
    accounts = {row["account_id"]: row for row in generated["accounts"]}

    assert sidecar["schema_version"] == ANNOTATION_VERSION
    assert sidecar["interpretation"] == "not fraud labels or fraud ground truth"
    assert {
        annotation["pattern_type"] for annotation in sidecar["annotations"]
    } == REQUIRED_PATTERN_TYPES
    for annotation in sidecar["annotations"]:
        assert annotation["transaction_ids"]
        for transaction_id in annotation["transaction_ids"]:
            transaction = transactions[transaction_id]
            assert accounts[transaction["account_id"]]["customer_id"] == (
                annotation["customer_id"]
            )


def test_manifest_counts_versions_and_hashes_match_artifacts() -> None:
    config_bytes = DEFAULT_CONFIG_PATH.read_bytes()
    banking_bytes = (DEFAULT_OUTPUT_DIR / "banking.generated.json").read_bytes()
    annotations_bytes = (
        DEFAULT_OUTPUT_DIR / "suspicious_patterns.json"
    ).read_bytes()
    generated = json.loads(banking_bytes)
    annotations = json.loads(annotations_bytes)
    manifest = _manifest()

    assert manifest["schema_version"] == MANIFEST_VERSION
    assert manifest["config_version"] == CONFIG_VERSION
    assert manifest["generator_version"] == GENERATOR_VERSION
    assert manifest["record_counts"] == {
        "customers": len(generated["customers"]),
        "accounts": len(generated["accounts"]),
        "cards": len(generated["cards"]),
        "transactions": len(generated["transactions"]),
        "disputes": len(generated["disputes"]),
        "support_cases": len(generated["support_cases"]),
        "annotations": len(annotations["annotations"]),
    }
    assert manifest["sha256"] == {
        "generation_config": sha256_bytes(config_bytes),
        "generated_banking_fixture": sha256_bytes(banking_bytes),
        "suspicious_patterns": sha256_bytes(annotations_bytes),
        "canonical_v1_banking_fixture": CANONICAL_SHA256,
    }


def test_default_seed_profile_is_unchanged_and_v2_is_additive() -> None:
    canonical = _load(DEFAULT_CANONICAL_FIXTURE_PATH)
    generated = _generated()

    assert load_fixtures() == canonical
    combined = load_fixtures("v2")
    for collection in (
        "customers",
        "accounts",
        "cards",
        "transactions",
        "disputes",
        "support_cases",
    ):
        assert combined[collection][: len(canonical[collection])] == (
            canonical[collection]
        )
        assert combined[collection][len(canonical[collection]) :] == (
            generated[collection]
        )


def test_generated_records_never_extend_avery() -> None:
    generated = _generated()
    canonical = _load(DEFAULT_CANONICAL_FIXTURE_PATH)
    combined = load_fixtures("v2")
    canonical_account_ids = {
        row["account_id"]
        for row in canonical["accounts"]
        if row["customer_id"] == AVERY_CUSTOMER_ID
    }
    canonical_card_ids = {
        row["card_id"]
        for row in canonical["cards"]
        if row["customer_id"] == AVERY_CUSTOMER_ID
    }

    assert AVERY_CUSTOMER_ID not in {
        row["customer_id"] for row in generated["customers"]
    }
    assert all(
        row["customer_id"] != AVERY_CUSTOMER_ID
        for row in generated["accounts"] + generated["cards"]
    )
    assert all(
        row["account_id"] not in canonical_account_ids
        and row["card_id"] not in canonical_card_ids
        for row in generated["transactions"]
    )
    assert [
        row
        for row in combined["accounts"]
        if row["customer_id"] == AVERY_CUSTOMER_ID
    ] == [
        row
        for row in canonical["accounts"]
        if row["customer_id"] == AVERY_CUSTOMER_ID
    ]
    assert [
        row
        for row in combined["cards"]
        if row["customer_id"] == AVERY_CUSTOMER_ID
    ] == [
        row
        for row in canonical["cards"]
        if row["customer_id"] == AVERY_CUSTOMER_ID
    ]
    assert [
        row
        for row in combined["transactions"]
        if row["account_id"] in canonical_account_ids
    ] == [
        row
        for row in canonical["transactions"]
        if row["account_id"] in canonical_account_ids
    ]


def test_stable_serialization_has_no_currency_float_artifacts() -> None:
    payloads = build_payloads(_load(DEFAULT_CONFIG_PATH))

    assert payloads.banking_bytes == stable_json_bytes(payloads.banking)
    assert b"0.30000000000000004" not in payloads.banking_bytes
