"""Generate deterministic, fully synthetic V2 banking evaluation data."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = (
    REPOSITORY_ROOT / "data/generation/v2/banking_generation_config.json"
)
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "data/fixtures/v2"
DEFAULT_CANONICAL_FIXTURE_PATH = REPOSITORY_ROOT / "data/fixtures/banking.json"

CONFIG_VERSION = "v2-banking-config.v1"
GENERATOR_VERSION = "v2-banking-generator.v1"
ANNOTATION_VERSION = "v2-suspicious-patterns.v1"
MANIFEST_VERSION = "v2-banking-manifest.v1"
SUPPORTED_TRANSACTION_STATUSES = {
    "POSTED",
    "PENDING",
    "REVERSED",
    "DECLINED",
}
REQUIRED_PATTERN_TYPES = {
    "unusual_location",
    "near_duplicate_transactions",
    "rapid_transaction_burst",
    "unusual_amount",
    "repeated_merchant_pattern",
}
V2_NAMESPACE = uuid5(NAMESPACE_URL, "sentinelvoice:synthetic:v2-banking")
MONEY_QUANTUM = Decimal("0.01")

FIRST_NAMES = (
    "Alex",
    "Blair",
    "Cameron",
    "Casey",
    "Dakota",
    "Drew",
    "Emerson",
    "Finley",
    "Hayden",
    "Jamie",
    "Jordan",
    "Kai",
    "Logan",
    "Morgan",
    "Parker",
    "Quinn",
)
LAST_NAMES = (
    "Adams",
    "Bennett",
    "Chen",
    "Diaz",
    "Evans",
    "Foster",
    "Gupta",
    "Hughes",
    "Ibrahim",
    "Johnson",
    "Kim",
    "Lopez",
    "Martin",
    "Nguyen",
    "Owens",
    "Patel",
)


@dataclass(frozen=True)
class Merchant:
    name: str
    category: str
    location: str
    minimum_cents: int
    maximum_cents: int


MERCHANTS = (
    Merchant("Brightline Utilities", "Utilities", "Newark, NJ", 4500, 25000),
    Merchant("Cedar Books", "Retail", "Montclair, NJ", 899, 12000),
    Merchant("Cloud Coffee", "Dining", "Newark, NJ", 350, 1800),
    Merchant("Garden State Deli", "Dining", "Jersey City, NJ", 650, 3200),
    Merchant("Harbor Pharmacy", "Health", "Jersey City, NJ", 800, 12000),
    Merchant("Metro Market", "Groceries", "Newark, NJ", 2000, 16000),
    Merchant("Northstar Fuel", "Fuel", "Elizabeth, NJ", 2500, 9000),
    Merchant("Orbit Digital", "Digital Goods", "Online", 499, 19999),
    Merchant("River Transit", "Transportation", "New York, NY", 275, 4000),
    Merchant("Summit Outfitters", "Retail", "Paramus, NJ", 2500, 35000),
)


@dataclass(frozen=True)
class GeneratedPayloads:
    banking: dict[str, Any]
    suspicious_patterns: dict[str, Any]
    banking_bytes: bytes
    suspicious_patterns_bytes: bytes


@dataclass(frozen=True)
class GenerationResult:
    banking_path: Path
    suspicious_patterns_path: Path
    manifest_path: Path
    manifest: dict[str, Any]


def stable_uuid(logical_key: str) -> UUID:
    return uuid5(V2_NAMESPACE, logical_key)


def stable_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _money_from_cents(cents: int) -> Decimal:
    return (Decimal(cents) / Decimal(100)).quantize(MONEY_QUANTUM)


def _money_text(value: Decimal) -> str:
    return format(value.quantize(MONEY_QUANTUM), ".2f")


def _timestamp_text(value: datetime) -> str:
    return value.isoformat(timespec="seconds")


def _integer_setting(
    settings: dict[str, Any],
    field: str,
    *,
    label: str,
) -> int:
    value = settings.get(field)
    if not isinstance(value, int) or isinstance(value, bool):
        # Config schema violations consistently surface as ValueError.
        raise ValueError(f"{label} must be an integer")  # noqa: TRY004
    return value


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("config_version") != CONFIG_VERSION:
        raise ValueError(f"config_version must be {CONFIG_VERSION}")
    if config.get("generator_version") != GENERATOR_VERSION:
        raise ValueError(f"generator_version must be {GENERATOR_VERSION}")

    customer_count = config.get("customer_count")
    if not isinstance(customer_count, int) or not 25 <= customer_count <= 50:
        raise ValueError("customer_count must be between 25 and 50")

    reference = datetime.fromisoformat(config["reference_datetime"])
    if reference.tzinfo is None:
        raise ValueError("reference_datetime must include a timezone")

    account_settings = config.get("account_settings", {})
    account_minimum = _integer_setting(
        account_settings,
        "minimum_per_customer",
        label="account minimum_per_customer",
    )
    account_maximum = _integer_setting(
        account_settings,
        "maximum_per_customer",
        label="account maximum_per_customer",
    )
    multiple_accounts_every = _integer_setting(
        account_settings,
        "multiple_accounts_every",
        label="account multiple_accounts_every",
    )
    if account_minimum < 1:
        raise ValueError("account minimum_per_customer must be at least 1")
    if account_maximum < account_minimum:
        raise ValueError(
            "account maximum_per_customer cannot be less than the minimum"
        )
    if account_maximum > 2:
        raise ValueError("account maximum_per_customer cannot exceed 2")
    if multiple_accounts_every < 1:
        raise ValueError("account multiple_accounts_every must be at least 1")

    card_settings = config.get("card_settings", {})
    card_minimum = _integer_setting(
        card_settings,
        "minimum_per_customer",
        label="card minimum_per_customer",
    )
    card_maximum = _integer_setting(
        card_settings,
        "maximum_per_customer",
        label="card maximum_per_customer",
    )
    multiple_cards_every = _integer_setting(
        card_settings,
        "multiple_cards_every",
        label="card multiple_cards_every",
    )
    frozen_primary_every = _integer_setting(
        card_settings,
        "frozen_primary_every",
        label="card frozen_primary_every",
    )
    if card_minimum < 1:
        raise ValueError("card minimum_per_customer must be at least 1")
    if card_maximum < card_minimum:
        raise ValueError(
            "card maximum_per_customer cannot be less than the minimum"
        )
    if multiple_cards_every < 1:
        raise ValueError("card multiple_cards_every must be at least 1")
    if frozen_primary_every < 1:
        raise ValueError("card frozen_primary_every must be at least 1")

    dispute_count = config.get("dispute_count")
    if not isinstance(dispute_count, int) or isinstance(dispute_count, bool):
        # Config schema violations consistently surface as ValueError.
        raise ValueError("dispute_count must be an integer")  # noqa: TRY004
    if dispute_count < 0:
        raise ValueError("dispute_count cannot be negative")
    maximum_disputes = customer_count // 4
    if dispute_count > maximum_disputes:
        raise ValueError(
            "dispute_count exceeds the customers available to the "
            "every-fourth-customer selection algorithm"
        )

    transaction_settings = config["transaction_settings"]
    transactions_per_customer = _integer_setting(
        transaction_settings,
        "per_customer",
        label="transactions per customer",
    )
    lookback_days = _integer_setting(
        transaction_settings,
        "lookback_days",
        label="transaction lookback",
    )
    if transactions_per_customer < 5:
        raise ValueError(
            "transactions per customer must be at least 5 for pattern coverage"
        )
    if lookback_days <= 0:
        raise ValueError("transaction lookback must be positive")
    status_weights = transaction_settings["status_weights"]
    if set(status_weights) != SUPPORTED_TRANSACTION_STATUSES:
        raise ValueError("status_weights must contain every supported status")
    if any(weight <= 0 for weight in status_weights.values()):
        raise ValueError("transaction status weights must be positive")
    pattern_coverage = config.get("pattern_coverage", {})
    if set(pattern_coverage) != REQUIRED_PATTERN_TYPES or not all(
        pattern_coverage.values()
    ):
        raise ValueError("every required synthetic pattern must be enabled")


def _customer(
    *,
    customer_index: int,
    reference_date: date,
    rng: random.Random,
) -> dict[str, Any]:
    first_name = FIRST_NAMES[(customer_index - 1) % len(FIRST_NAMES)]
    last_name = LAST_NAMES[(customer_index * 7 - 1) % len(LAST_NAMES)]
    age = rng.randint(21, 75)
    birth_month = rng.randint(1, 12)
    birth_day = rng.randint(1, 28)
    return {
        "customer_id": str(
            stable_uuid(f"customer:v2:{customer_index:04d}")
        ),
        "first_name": first_name,
        "last_name": last_name,
        "email": (
            f"{first_name}.{last_name}.v2-{customer_index:04d}"
            "@example.test"
        ).lower(),
        "phone": f"+1-555-{2000 + customer_index:04d}",
        "date_of_birth": date(
            reference_date.year - age,
            birth_month,
            birth_day,
        ).isoformat(),
        "status": "ACTIVE",
        "authentication_profile": {
            "verification_method": "demo_pin",
            "verification_status": "ENROLLED",
        },
    }


def _accounts(
    *,
    customer: dict[str, Any],
    customer_index: int,
    config: dict[str, Any],
    rng: random.Random,
) -> list[dict[str, Any]]:
    settings = config["account_settings"]
    account_count = (
        settings["maximum_per_customer"]
        if customer_index % settings["multiple_accounts_every"] == 0
        else settings["minimum_per_customer"]
    )
    accounts: list[dict[str, Any]] = []
    for account_index in range(1, account_count + 1):
        account_type = "checking" if account_index == 1 else "savings"
        minimum_cents = 50_000 if account_type == "checking" else 100_000
        maximum_cents = 1_500_000 if account_type == "checking" else 7_500_000
        current_balance = _money_from_cents(
            rng.randint(minimum_cents, maximum_cents)
        )
        hold = _money_from_cents(
            rng.randint(0, min(75_000, int(current_balance * 100)))
        )
        accounts.append(
            {
                "account_id": str(
                    stable_uuid(
                        "account:v2:"
                        f"{customer_index:04d}:{account_type}"
                    )
                ),
                "customer_id": customer["customer_id"],
                "account_type": account_type,
                "masked_account_number": (
                    f"****{6000 + customer_index * 10 + account_index:04d}"
                ),
                "current_balance": _money_text(current_balance),
                "available_balance": _money_text(current_balance - hold),
                "currency": "USD",
                "status": "ACTIVE",
            }
        )
    return accounts


def _cards(
    *,
    customer: dict[str, Any],
    checking_account: dict[str, Any],
    customer_index: int,
    reference_date: date,
    config: dict[str, Any],
    rng: random.Random,
) -> list[dict[str, Any]]:
    settings = config["card_settings"]
    card_count = (
        settings["maximum_per_customer"]
        if customer_index % settings["multiple_cards_every"] == 0
        else settings["minimum_per_customer"]
    )
    cards: list[dict[str, Any]] = []
    for card_index in range(1, card_count + 1):
        is_frozen = card_index > 1 or (
            card_index == 1
            and customer_index % settings["frozen_primary_every"] == 0
        )
        cards.append(
            {
                "card_id": str(
                    stable_uuid(
                        f"card:v2:{customer_index:04d}:{card_index:02d}"
                    )
                ),
                "customer_id": customer["customer_id"],
                "account_id": checking_account["account_id"],
                "masked_card_number": (
                    f"****{7000 + customer_index * 10 + card_index:04d}"
                ),
                "card_type": "DEBIT",
                "status": "FROZEN" if is_frozen else "ACTIVE",
                "expiration_month": rng.randint(1, 12),
                "expiration_year": reference_date.year + rng.randint(2, 5),
            }
        )
    return cards


def _posted_timestamp(
    transaction_timestamp: datetime,
    status: str,
    reference: datetime,
    rng: random.Random,
) -> str | None:
    if status in {"PENDING", "DECLINED"}:
        return None
    posted = min(
        transaction_timestamp + timedelta(hours=rng.randint(2, 48)),
        reference,
    )
    return _timestamp_text(posted)


def _transactions(
    *,
    customer_index: int,
    checking_account: dict[str, Any],
    cards: list[dict[str, Any]],
    config: dict[str, Any],
    reference: datetime,
    rng: random.Random,
) -> list[dict[str, Any]]:
    settings = config["transaction_settings"]
    statuses = list(settings["status_weights"])
    weights = [settings["status_weights"][status] for status in statuses]
    transactions: list[dict[str, Any]] = []
    for transaction_index in range(1, settings["per_customer"] + 1):
        merchant = rng.choice(MERCHANTS)
        status = rng.choices(statuses, weights=weights, k=1)[0]
        age = timedelta(
            days=rng.randint(0, settings["lookback_days"] - 1),
            hours=rng.randint(0, 23),
            minutes=rng.randint(0, 59),
        )
        transaction_timestamp = reference - age
        amount = _money_from_cents(
            rng.randint(merchant.minimum_cents, merchant.maximum_cents)
        )
        card = cards[(transaction_index - 1) % len(cards)]
        transactions.append(
            {
                "transaction_id": str(
                    stable_uuid(
                        "transaction:v2:"
                        f"{customer_index:04d}:checking:"
                        f"{transaction_index:04d}"
                    )
                ),
                "account_id": checking_account["account_id"],
                "card_id": card["card_id"],
                "merchant_name": merchant.name,
                "merchant_category": merchant.category,
                "amount": _money_text(amount),
                "currency": "USD",
                "transaction_timestamp": _timestamp_text(
                    transaction_timestamp
                ),
                "posted_timestamp": _posted_timestamp(
                    transaction_timestamp,
                    status,
                    reference,
                    rng,
                ),
                "status": status,
                "transaction_type": "CARD_PURCHASE",
                "location": merchant.location,
            }
        )
    return transactions


def _set_transaction(
    transaction: dict[str, Any],
    *,
    merchant: Merchant,
    amount: str,
    timestamp: datetime,
    status: str,
    reference: datetime,
    location: str | None = None,
) -> None:
    transaction.update(
        {
            "merchant_name": merchant.name,
            "merchant_category": merchant.category,
            "amount": amount,
            "transaction_timestamp": _timestamp_text(timestamp),
            "posted_timestamp": (
                None
                if status in {"PENDING", "DECLINED"}
                else _timestamp_text(min(timestamp + timedelta(hours=12), reference))
            ),
            "status": status,
            "location": location or merchant.location,
        }
    )


def _apply_patterns(
    *,
    customers: list[dict[str, Any]],
    transactions_by_customer: list[list[dict[str, Any]]],
    reference: datetime,
) -> dict[str, Any]:
    annotations: list[dict[str, Any]] = []

    unusual = transactions_by_customer[0][0]
    _set_transaction(
        unusual,
        merchant=MERCHANTS[7],
        amount="187.40",
        timestamp=reference - timedelta(days=3, hours=9),
        status="POSTED",
        reference=reference,
        location="Singapore, SG",
    )
    annotations.append(
        {
            "annotation_id": "v2-pattern-001",
            "pattern_type": "unusual_location",
            "customer_id": customers[0]["customer_id"],
            "transaction_ids": [unusual["transaction_id"]],
            "explanation": (
                "Synthetic purchase location differs from the generated "
                "customer's otherwise regional activity."
            ),
        }
    )

    duplicate_pair = transactions_by_customer[1][:2]
    for offset, transaction in enumerate(duplicate_pair):
        _set_transaction(
            transaction,
            merchant=MERCHANTS[5],
            amount="48.45",
            timestamp=reference - timedelta(days=2 - offset, minutes=offset * 2),
            status="POSTED" if offset == 0 else "PENDING",
            reference=reference,
        )
    annotations.append(
        {
            "annotation_id": "v2-pattern-002",
            "pattern_type": "near_duplicate_transactions",
            "customer_id": customers[1]["customer_id"],
            "transaction_ids": [item["transaction_id"] for item in duplicate_pair],
            "explanation": (
                "Same merchant and amount on nearby dates with posted and "
                "pending statuses creates deliberate selector ambiguity."
            ),
        }
    )

    burst = transactions_by_customer[2][:4]
    for offset, transaction in enumerate(burst):
        merchant = MERCHANTS[(offset + 2) % len(MERCHANTS)]
        _set_transaction(
            transaction,
            merchant=merchant,
            amount=_money_text(_money_from_cents(900 + offset * 175)),
            timestamp=reference - timedelta(hours=20, minutes=offset * 7),
            status="POSTED",
            reference=reference,
        )
    annotations.append(
        {
            "annotation_id": "v2-pattern-003",
            "pattern_type": "rapid_transaction_burst",
            "customer_id": customers[2]["customer_id"],
            "transaction_ids": [item["transaction_id"] for item in burst],
            "explanation": (
                "Four synthetic purchases occur within a short deterministic "
                "time window."
            ),
        }
    )

    large = transactions_by_customer[3][0]
    _set_transaction(
        large,
        merchant=MERCHANTS[9],
        amount="2499.99",
        timestamp=reference - timedelta(days=9, hours=3),
        status="POSTED",
        reference=reference,
    )
    annotations.append(
        {
            "annotation_id": "v2-pattern-004",
            "pattern_type": "unusual_amount",
            "customer_id": customers[3]["customer_id"],
            "transaction_ids": [large["transaction_id"]],
            "explanation": (
                "A deliberately large synthetic purchase exceeds the "
                "customer's surrounding everyday amounts."
            ),
        }
    )

    repeated = transactions_by_customer[4][:5]
    for offset, transaction in enumerate(repeated):
        _set_transaction(
            transaction,
            merchant=MERCHANTS[2],
            amount=_money_text(_money_from_cents(525 + offset * 110)),
            timestamp=reference - timedelta(days=offset * 3 + 1, hours=2),
            status="POSTED",
            reference=reference,
        )
    annotations.append(
        {
            "annotation_id": "v2-pattern-005",
            "pattern_type": "repeated_merchant_pattern",
            "customer_id": customers[4]["customer_id"],
            "transaction_ids": [item["transaction_id"] for item in repeated],
            "explanation": (
                "Repeated synthetic purchases at one merchant support merchant "
                "selector and sequence evaluation."
            ),
        }
    )

    amount_pair = transactions_by_customer[5][:2]
    _set_transaction(
        amount_pair[0],
        merchant=MERCHANTS[1],
        amount="73.21",
        timestamp=reference - timedelta(days=7, hours=4),
        status="POSTED",
        reference=reference,
    )
    _set_transaction(
        amount_pair[1],
        merchant=MERCHANTS[6],
        amount="73.21",
        timestamp=reference - timedelta(days=6, hours=4),
        status="POSTED",
        reference=reference,
    )

    return {
        "schema_version": ANNOTATION_VERSION,
        "provenance": "fully synthetic deterministic evaluation metadata",
        "interpretation": "not fraud labels or fraud ground truth",
        "annotations": annotations,
    }


def _disputes(
    *,
    customers: list[dict[str, Any]],
    transactions_by_customer: list[list[dict[str, Any]]],
    count: int,
) -> list[dict[str, Any]]:
    disputes: list[dict[str, Any]] = []
    statuses = ("RESOLVED", "OPEN", "UNDER_REVIEW", "REJECTED")
    reasons = (
        "DUPLICATE_CHARGE",
        "UNRECOGNIZED_TRANSACTION",
        "MERCHANDISE_NOT_RECEIVED",
        "INCORRECT_AMOUNT",
    )
    for dispute_index in range(1, count + 1):
        customer_index = dispute_index * 4 - 1
        customer = customers[customer_index]
        transaction = next(
            item
            for item in transactions_by_customer[customer_index]
            if item["status"] == "POSTED"
        )
        disputes.append(
            {
                "dispute_id": str(
                    stable_uuid(f"dispute:v2:{dispute_index:04d}")
                ),
                "customer_id": customer["customer_id"],
                "transaction_id": transaction["transaction_id"],
                "reason_code": reasons[(dispute_index - 1) % len(reasons)],
                "status": statuses[(dispute_index - 1) % len(statuses)],
                "notes": "Synthetic historical dispute generated for V2 evaluation.",
            }
        )
    return disputes


def build_payloads(config: dict[str, Any]) -> GeneratedPayloads:
    _validate_config(config)
    rng = random.Random(config["seed"])
    reference = datetime.fromisoformat(config["reference_datetime"])

    customers: list[dict[str, Any]] = []
    accounts: list[dict[str, Any]] = []
    cards: list[dict[str, Any]] = []
    transactions_by_customer: list[list[dict[str, Any]]] = []

    for customer_index in range(1, config["customer_count"] + 1):
        customer = _customer(
            customer_index=customer_index,
            reference_date=reference.date(),
            rng=rng,
        )
        customer_accounts = _accounts(
            customer=customer,
            customer_index=customer_index,
            config=config,
            rng=rng,
        )
        checking_account = customer_accounts[0]
        customer_cards = _cards(
            customer=customer,
            checking_account=checking_account,
            customer_index=customer_index,
            reference_date=reference.date(),
            config=config,
            rng=rng,
        )
        customer_transactions = _transactions(
            customer_index=customer_index,
            checking_account=checking_account,
            cards=customer_cards,
            config=config,
            reference=reference,
            rng=rng,
        )
        customers.append(customer)
        accounts.extend(customer_accounts)
        cards.extend(customer_cards)
        transactions_by_customer.append(customer_transactions)

    suspicious_patterns = _apply_patterns(
        customers=customers,
        transactions_by_customer=transactions_by_customer,
        reference=reference,
    )
    transactions = [
        transaction
        for customer_transactions in transactions_by_customer
        for transaction in customer_transactions
    ]
    banking = {
        "customers": customers,
        "accounts": accounts,
        "cards": cards,
        "transactions": transactions,
        "disputes": _disputes(
            customers=customers,
            transactions_by_customer=transactions_by_customer,
            count=config["dispute_count"],
        ),
        "support_cases": [],
    }
    banking_bytes = stable_json_bytes(banking)
    suspicious_patterns_bytes = stable_json_bytes(suspicious_patterns)
    return GeneratedPayloads(
        banking=banking,
        suspicious_patterns=suspicious_patterns,
        banking_bytes=banking_bytes,
        suspicious_patterns_bytes=suspicious_patterns_bytes,
    )


def generate_files(
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    canonical_fixture_path: Path = DEFAULT_CANONICAL_FIXTURE_PATH,
) -> GenerationResult:
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    payloads = build_payloads(config)
    canonical_bytes = canonical_fixture_path.read_bytes()
    counts = {
        key: len(payloads.banking[key])
        for key in (
            "customers",
            "accounts",
            "cards",
            "transactions",
            "disputes",
            "support_cases",
        )
    }
    counts["annotations"] = len(
        payloads.suspicious_patterns["annotations"]
    )
    manifest = {
        "schema_version": MANIFEST_VERSION,
        "generator_version": config["generator_version"],
        "config_version": config["config_version"],
        "seed": config["seed"],
        "reference_datetime": config["reference_datetime"],
        "record_counts": counts,
        "sha256": {
            "generation_config": sha256_bytes(config_bytes),
            "generated_banking_fixture": sha256_bytes(
                payloads.banking_bytes
            ),
            "suspicious_patterns": sha256_bytes(
                payloads.suspicious_patterns_bytes
            ),
            "canonical_v1_banking_fixture": sha256_bytes(canonical_bytes),
        },
        "provenance": (
            "fully synthetic deterministic data generated without external "
            "or real customer data"
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    banking_path = output_dir / "banking.generated.json"
    suspicious_patterns_path = output_dir / "suspicious_patterns.json"
    manifest_path = output_dir / "banking.generated.manifest.json"
    banking_path.write_bytes(payloads.banking_bytes)
    suspicious_patterns_path.write_bytes(payloads.suspicious_patterns_bytes)
    manifest_path.write_bytes(stable_json_bytes(manifest))
    return GenerationResult(
        banking_path=banking_path,
        suspicious_patterns_path=suspicious_patterns_path,
        manifest_path=manifest_path,
        manifest=manifest,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate deterministic SentinelVoice V2 banking data."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--canonical-fixture",
        type=Path,
        default=DEFAULT_CANONICAL_FIXTURE_PATH,
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    result = generate_files(
        config_path=args.config,
        output_dir=args.output_dir,
        canonical_fixture_path=args.canonical_fixture,
    )
    print(
        "Generated V2 banking fixture: "
        f"{result.manifest['record_counts']}"
    )
    print(
        "Generated fixture SHA-256: "
        f"{result.manifest['sha256']['generated_banking_fixture']}"
    )


if __name__ == "__main__":
    main()
