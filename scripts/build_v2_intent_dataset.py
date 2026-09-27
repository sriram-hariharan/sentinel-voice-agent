"""Build the deterministic, SentinelVoice-only V2-C intent/risk dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any

from backend.app.evaluation.v2_contracts import (
    DatasetSplit,
    IntentLabel,
    IntentRiskDataset,
    RiskLabel,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEED_PATH = REPOSITORY_ROOT / "data/evals/v2/intent_risk_seed.json"
DEFAULT_CONFIG_PATH = (
    REPOSITORY_ROOT
    / "data/evals/v2/ml/intent_risk_expansion_config.json"
)
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "data/evals/v2/ml"
DEFAULT_DATASET_PATH = DEFAULT_OUTPUT_DIR / "intent_risk_dataset.json"
DEFAULT_MANIFEST_PATH = (
    DEFAULT_OUTPUT_DIR / "intent_risk_dataset.manifest.json"
)

CONFIG_VERSION = "v2-intent-expansion-config.v1"
GENERATOR_VERSION = "v2-intent-dataset-builder.v1"
MANIFEST_VERSION = "v2-intent-dataset-manifest.v1"
DATASET_SCHEMA_VERSION = "intent-risk.v1"

RISK_BY_INTENT = {
    IntentLabel.INFORMATIONAL_POLICY.value: RiskLabel.PUBLIC.value,
    IntentLabel.ACCOUNT_BALANCE.value: RiskLabel.PRIVATE_READ.value,
    IntentLabel.RECENT_TRANSACTIONS.value: RiskLabel.PRIVATE_READ.value,
    IntentLabel.TRANSACTION_DETAILS.value: RiskLabel.PRIVATE_READ.value,
    IntentLabel.CARD_STATUS.value: RiskLabel.PRIVATE_READ.value,
    IntentLabel.FREEZE_CARD.value: RiskLabel.PROTECTED_WRITE.value,
    IntentLabel.CREATE_DISPUTE.value: RiskLabel.PROTECTED_WRITE.value,
    IntentLabel.ESCALATION.value: RiskLabel.ESCALATION_OR_UNCERTAIN.value,
    IntentLabel.UNSUPPORTED_OR_UNCERTAIN.value: (
        RiskLabel.ESCALATION_OR_UNCERTAIN.value
    ),
}

TRAIN_TEMPLATES: dict[str, tuple[str, ...]] = {
    "account_balance": (
        "What's my available balance {slot}?",
        "Could you tell me how much I have {slot}?",
        "Uh, read the current balance {slot}.",
        "How much can I spend {slot} right now?",
        "Please check the money available {slot}.",
        "Actually, I only need the balance {slot}.",
        "Give me the total funds {slot}, please.",
    ),
    "recent_transactions": (
        "Show my latest account activity {slot}.",
        "What purchases appeared {slot}?",
        "Could you list the newest transactions {slot}?",
        "Uh, read back my recent charges {slot}.",
        "What money moved in or out {slot}?",
        "Actually, show the transaction history {slot}.",
        "Tell me what has posted lately {slot}.",
    ),
    "transaction_details": (
        "Tell me about the transaction {slot}.",
        "What are the details of the charge {slot}?",
        "Could you identify the purchase {slot}?",
        "Uh, where and when was the payment {slot}?",
        "Read me the merchant information {slot}.",
        "Actually, inspect the transaction {slot}.",
        "What happened with the charge {slot}?",
    ),
    "card_status": (
        "Is the card {slot} currently active?",
        "Check whether the card {slot} is frozen.",
        "What status does my card {slot} have?",
        "Uh, can I still use the card {slot}?",
        "Tell me if the card {slot} is locked.",
        "Actually, just check the card {slot} status.",
        "Has anything disabled the card {slot}?",
    ),
    "freeze_card": (
        "Freeze the card {slot} for me.",
        "Please lock the card {slot} now.",
        "Make my card {slot} unusable.",
        "Uh, disable the card {slot} right away.",
        "Stop purchases on my card {slot}.",
        "Actually, go ahead and freeze the card {slot}.",
        "Block the card {slot} until I find it.",
    ),
    "create_dispute": (
        "Open a dispute {slot}.",
        "Please contest the transaction {slot}.",
        "Start a charge dispute {slot}.",
        "Uh, file a claim for the payment {slot}.",
        "Challenge the purchase {slot} for me.",
        "Actually, report the charge {slot} as unauthorized.",
        "Begin the investigation process {slot}.",
    ),
    "informational_policy": (
        "Explain the bank policy {slot}.",
        "What are the rules {slot}?",
        "How does the process work {slot}?",
        "Uh, what would happen {slot}?",
        "Tell me the requirements {slot}.",
        "Actually, I only want information {slot}.",
        "Can you describe the policy {slot}?",
    ),
    "escalation": (
        "Connect me with a human {slot}.",
        "I need a support representative {slot}.",
        "Please transfer me to a person {slot}.",
        "Uh, can a human take over {slot}?",
        "Let me speak with an agent {slot}.",
        "Actually, escalate this to support {slot}.",
        "I want someone from the support team {slot}.",
    ),
    "unsupported_or_uncertain": (
        "Are you able to {slot}?",
        "Please {slot} for me.",
        "Uh, I need you to {slot}.",
        "Is it possible to {slot}?",
        "Actually, go ahead and {slot}.",
        "I am not sure, but maybe {slot}.",
        "Take care of this: {slot}.",
    ),
}

VALIDATION_REQUESTS: dict[str, tuple[str, ...]] = {
    "account_balance": (
        "what remains in my savings account",
        "read the spendable amount in checking",
        "how much cash is available to me",
        "check my savings total instead of recent activity",
        "tell me whether checking has enough funds",
        "give me only the current account balance",
        "look up what I can spend from savings",
        "say the checking balance out loud",
        "find the available funds in my account",
        "check how much money is left after purchases",
        "show the balance, not the transaction list",
    ),
    "recent_transactions": (
        "list the last few purchases in checking",
        "show what has posted since Monday",
        "read my newest account activity",
        "tell me about recent money coming and going",
        "show the latest charges rather than my balance",
        "find any new transactions in savings",
        "list this week's account movements",
        "read back my most recent purchases",
        "show which payments just appeared",
        "check for new charges on the account",
        "give me the recent transaction list",
    ),
    "transaction_details": (
        "identify the $48.45 Metro Market charge",
        "explain yesterday's Cloud Coffee transaction",
        "show details for the pending Orbit Digital payment",
        "tell me where the Cedar Books purchase happened",
        "inspect the September 22 charge",
        "find the declined Garden State Deli payment",
        "tell me which card paid $73.18",
        "explain the reversed Brightline Utilities transaction",
        "look up the second Metro Market purchase",
        "identify that one hundred eighty-nine ninety-nine charge",
        "show the merchant for the pending transaction",
    ),
    "card_status": (
        "check whether card 4176 is active",
        "tell me if my debit card is currently frozen",
        "see whether I can use card 9031",
        "read the current state of my card",
        "check the status without locking anything",
        "tell me whether card 1842 was disabled",
        "find out if my credit card is active",
        "see why card 6620 may not work",
        "check if the card is already frozen",
        "give me the card state only",
        "verify whether purchases are enabled on my card",
    ),
    "freeze_card": (
        "freeze card 4176 immediately",
        "disable my debit card until I recover it",
        "lock card 9031 against new purchases",
        "stop anyone using my lost card",
        "freeze the credit card, not the debit card",
        "block card 1842 now",
        "turn off purchases for my card",
        "make the missing card unusable",
        "lock the card after all",
        "freeze only the card ending 6620",
        "disable the active card right away",
    ),
    "create_dispute": (
        "dispute the $48.45 Metro Market charge",
        "open a claim for yesterday's Cloud Coffee purchase",
        "contest the posted Orbit Digital transaction",
        "report the Cedar Books payment as unauthorized",
        "start a dispute for the September 22 charge",
        "challenge the $189.99 purchase",
        "file a dispute for the second Metro transaction",
        "investigate that card payment as fraud",
        "open the dispute instead of explaining the policy",
        "contest the Garden State Deli charge",
        "begin a claim for the posted transaction",
    ),
    "informational_policy": (
        "explain what freezing a card changes",
        "tell me how transaction disputes are reviewed",
        "describe the replacement-card fee rules",
        "explain whether pending charges can be disputed",
        "tell me what qualifies as an unauthorized charge",
        "describe the card-freeze process without doing it",
        "explain how long a dispute can take",
        "tell me what happens after a card is locked",
        "describe the information needed for a dispute",
        "explain the transaction-status meanings",
        "tell me the policy, not my private account details",
    ),
    "escalation": (
        "bring a human agent into this conversation",
        "transfer me to customer support",
        "let me speak with a representative about my card",
        "escalate this transaction issue to a person",
        "connect me with someone who can help",
        "hand this conversation over to support",
        "get me a live agent rather than an automated answer",
        "send this problem to a human specialist",
        "let a representative take over",
        "move me to the support team",
        "have a person review this with me",
    ),
    "unsupported_or_uncertain": (
        "increase my card's credit limit",
        "wire money to another bank",
        "close every account I have",
        "change the name on my account",
        "order foreign currency for delivery",
        "do something about the payment, I guess",
        "fix whatever is wrong with everything",
        "buy cryptocurrency from this account",
        "schedule a mortgage appointment",
        "cancel that, I do not know what I want",
        "handle the thing from earlier somehow",
    ),
}

LOCKED_TEXTS: dict[str, tuple[str, ...]] = {
    "account_balance": (
        "Wait, don't list purchases; how much is available in savings?",
        "Uh, before anything else, what's left in checking?",
        "Could you just say how much money I can spend right now?",
        "I asked about charges, but actually I need my balance.",
        "Is there enough in savings for a $73.18 payment?",
        "No transaction details—only the available checking balance.",
        "What does my account total come to today?",
        "Read the balance for savings, please.",
    ),
    "recent_transactions": (
        "Don't give me one charge; show the latest few transactions.",
        "Uh, what has hit my checking account recently?",
        "Skip my balance and list this week's purchases.",
        "Have any new payments posted since yesterday?",
        "Actually, read the recent activity from savings.",
        "What are the last several things I paid for?",
        "Show everything recent, not just the Metro Market charge.",
        "Which transactions appeared most recently?",
    ),
    "transaction_details": (
        "Tell me about the pending Cloud Coffee transaction, not all activity.",
        "What happened with the $189.99 Orbit Digital charge?",
        "I mean the September 22 payment, not yesterday's one.",
        "Which merchant received the spoken forty-eight forty-five payment?",
        "Was the second Metro Market purchase posted or reversed?",
        "Give me details for the card 1842 transaction at Cedar Books.",
        "Where did that declined $12.40 payment occur?",
        "Actually, inspect one charge: the Brightline Utilities one.",
    ),
    "card_status": (
        "Don't freeze the card; tell me its status.",
        "Is card 9031 frozen already?",
        "Wait, I only want to know whether my debit card works.",
        "Has the card ending 4176 been locked?",
        "Check the credit card's state without changing it.",
        "Can card 6620 still be used for purchases?",
        "Tell me if the missing card is active; take no action.",
        "No, don't disable it—just verify its current status.",
    ),
    "freeze_card": (
        "Don't tell me the status; freeze it.",
        "Wait, lock card 4176, not card 9031.",
        "I know it is active; disable the card now.",
        "Actually, stop purchases on my debit card immediately.",
        "Freeze the missing card rather than explaining the policy.",
        "Make card 1842 unusable until I call back.",
        "Go ahead and lock it—I'm done checking the status.",
        "Please block the credit card, but leave debit alone.",
    ),
    "create_dispute": (
        "Open a dispute for that charge.",
        "Don't just explain disputes; file one for the $73.18 purchase.",
        "Actually, contest the second Metro Market transaction.",
        "Report the September 18 Orbit Digital charge as unauthorized.",
        "I know what it is; now start a claim for it.",
        "File the dispute for Cloud Coffee, not Cedar Books.",
        "Please challenge the posted $189.99 payment.",
        "Begin the dispute process for the card 6620 transaction.",
    ),
    "informational_policy": (
        "What happens if I freeze my card?",
        "What happens during a dispute?",
        "Don't freeze anything; explain what a freeze would do.",
        "I'm not contesting the charge yet—what are the dispute rules?",
        "Would a pending transaction qualify for a dispute?",
        "Before I act, how long does a card freeze last?",
        "Explain the difference between pending and posted transactions.",
        "What information would the bank need if I disputed a charge?",
    ),
    "escalation": (
        "Never mind the card action—let me talk to a human.",
        "This is unclear; please connect me to a representative.",
        "I don't want an automated answer. Get support.",
        "Can a person take over this transaction problem?",
        "Actually, escalate me instead of opening the dispute.",
        "Send me to a human agent who can review this.",
        "I need somebody from support on this conversation.",
        "Stop here and transfer me to a live representative.",
    ),
    "unsupported_or_uncertain": (
        "Uh, do something with my account—I don't know what.",
        "Can you transfer $500 to an outside bank?",
        "Change my legal name and mailing address.",
        "Maybe close the account, or don't; I'm not sure.",
        "Apply for a home loan on my behalf.",
        "Take care of that thing from the other conversation.",
        "Buy some digital currency with my savings.",
        "Cancel everything—actually, I don't know what I mean.",
    ),
}


def stable_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def normalize_text(text: str) -> str:
    punctuation = str.maketrans("", "", string.punctuation)
    return re.sub(r"\s+", " ", text.lower().translate(punctuation)).strip()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("config_version") != CONFIG_VERSION:
        raise ValueError(f"config_version must be {CONFIG_VERSION}")
    if config.get("generator_version") != GENERATOR_VERSION:
        raise ValueError(f"generator_version must be {GENERATOR_VERSION}")
    if config.get("external_dataset_sources") != []:
        raise ValueError("V2-C cannot include public or external datasets")
    targets = config.get("target_per_intent", {})
    expected = {"train": 45, "validation": 12, "locked_test": 10}
    if targets != expected:
        raise ValueError(f"target_per_intent must remain frozen at {expected}")
    if not isinstance(config.get("seed"), int):
        # Config schema violations consistently surface as ValueError.
        raise ValueError("seed must be an integer")  # noqa: TRY004


def _intent_slots(config: dict[str, Any]) -> dict[str, tuple[str, ...]]:
    slots = config["slots"]
    merchants = slots["merchants"]
    amounts = slots["amounts"]
    dates = slots["dates"]
    endings = slots["card_endings"]
    return {
        "account_balance": (
            "in checking",
            "in savings",
            "for my checking account",
            "for my savings account",
            "after the latest purchases",
            "before I make another payment",
        ),
        "recent_transactions": (
            "from checking",
            "from savings",
            "since Monday",
            "from the past week",
            "on card ending 1842",
            "before today",
        ),
        "transaction_details": (
            f"at {merchants[0]}",
            f"at {merchants[1]}",
            f"for {amounts[1]}",
            f"from {dates[2]}",
            f"on card ending {endings[2]}",
            f"at {merchants[4]}",
        ),
        "card_status": tuple(f"ending in {ending}" for ending in endings)
        + ("for checking", "for savings"),
        "freeze_card": tuple(f"ending in {ending}" for ending in endings)
        + ("linked to checking", "linked to savings"),
        "create_dispute": (
            f"for the {merchants[0]} charge",
            f"for the {merchants[2]} purchase",
            f"for the {amounts[1]} payment",
            f"for the transaction on {dates[2]}",
            f"for the card ending {endings[0]} charge",
            f"for the {merchants[4]} transaction",
        ),
        "informational_policy": (
            "for freezing a card",
            "for disputing a transaction",
            "for replacing a lost card",
            "for pending transactions",
            "for unauthorized purchases",
            "before I take any action",
        ),
        "escalation": (
            "about my checking account",
            "about a card problem",
            "about a transaction",
            "for this dispute question",
            "instead of the automated assistant",
            "before any action is taken",
        ),
        "unsupported_or_uncertain": (
            "send a wire transfer",
            "increase my credit limit",
            "close all of my accounts",
            "change my legal name",
            "purchase cryptocurrency",
            "fix the vague problem from earlier",
        ),
    }


def _new_example(
    *,
    example_id: str,
    text: str,
    intent: str,
    group_id: str,
    split: str,
    tags: list[str],
) -> dict[str, Any]:
    return {
        "example_id": example_id,
        "text": text,
        "intent": intent,
        "risk": RISK_BY_INTENT[intent],
        "group_id": group_id,
        "split": split,
        "tags": tags,
    }


def _expanded_examples(config: dict[str, Any]) -> list[dict[str, Any]]:
    randomizer = random.Random(config["seed"])
    intent_slots = _intent_slots(config)
    examples: list[dict[str, Any]] = []
    for intent in sorted(RISK_BY_INTENT):
        templates = TRAIN_TEMPLATES[intent]
        slots = list(intent_slots[intent])
        randomizer.shuffle(slots)
        for template_index, template in enumerate(templates, start=1):
            group_id = f"{intent}_v2c_train_{template_index:02d}"
            for slot_index, slot in enumerate(slots, start=1):
                sequence = (template_index - 1) * len(slots) + slot_index
                examples.append(
                    _new_example(
                        example_id=f"{intent}_v2c_train_{sequence:03d}",
                        text=template.format(slot=slot),
                        intent=intent,
                        group_id=group_id,
                        split=DatasetSplit.TRAIN.value,
                        tags=[
                            "deterministic_template_expansion",
                            "synthetic",
                            "voice_like",
                        ],
                    )
                )
        for index, request in enumerate(VALIDATION_REQUESTS[intent], start=1):
            frames = (
                "Could you {request}?",
                "Uh, please {request}.",
                "Actually, {request}.",
                "I need you to {request}.",
                "For me, {request}.",
                "Wait—{request}.",
                "Can you {request}?",
                "Please {request}.",
                "Right now, {request}.",
                "Before we continue, {request}.",
                "Just {request}, please.",
            )
            text = frames[index - 1].format(request=request)
            examples.append(
                _new_example(
                    example_id=f"{intent}_v2c_validation_{index:03d}",
                    text=text,
                    intent=intent,
                    group_id=f"{intent}_v2c_validation_{index:02d}",
                    split=DatasetSplit.VALIDATION.value,
                    tags=["hand_authored_sentinelvoice", "synthetic", "validation"],
                )
            )
        for index, text in enumerate(LOCKED_TEXTS[intent], start=1):
            examples.append(
                _new_example(
                    example_id=f"{intent}_v2c_locked_{index:03d}",
                    text=text,
                    intent=intent,
                    group_id=f"{intent}_v2c_locked_{index:02d}",
                    split=DatasetSplit.LOCKED_TEST.value,
                    tags=["contrast_challenge", "hand_authored_sentinelvoice", "synthetic"],
                )
            )
    return examples


def _count_by(examples: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(example[field]) for example in examples).items()))


def _count_by_split_and_label(
    examples: list[dict[str, Any]], label: str
) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = {}
    for example in examples:
        counts.setdefault(example["split"], Counter())[example[label]] += 1
    return {
        split: dict(sorted(values.items()))
        for split, values in sorted(counts.items())
    }


def _source_category(example: dict[str, Any]) -> str:
    if "_v2c_" not in example["example_id"]:
        return "v2_a_seed_unchanged"
    for category in (
        "contrast_challenge",
        "deterministic_template_expansion",
        "hand_authored_sentinelvoice",
    ):
        if category in example["tags"]:
            return category
    raise ValueError(f"missing source category for {example['example_id']}")


def validate_expanded_dataset(
    payload: dict[str, Any],
    *,
    seed_payload: dict[str, Any],
    config: dict[str, Any],
) -> IntentRiskDataset:
    dataset = IntentRiskDataset.model_validate(payload)
    examples = list(payload["examples"])
    original_by_id = {
        example["example_id"]: example for example in seed_payload["examples"]
    }
    expanded_by_id = {example["example_id"]: example for example in examples}
    if any(expanded_by_id.get(key) != value for key, value in original_by_id.items()):
        raise ValueError("all original V2-A seed examples must remain unchanged")

    normalized = [normalize_text(example["text"]) for example in examples]
    if any(not value for value in normalized):
        raise ValueError("example text cannot be empty after normalization")
    duplicates = sorted(text for text, count in Counter(normalized).items() if count > 1)
    if duplicates:
        raise ValueError(f"normalized duplicate text is not allowed: {duplicates[0]}")

    expected = config["target_per_intent"]
    observed: Counter[tuple[str, str]] = Counter(
        (example["intent"], example["split"]) for example in examples
    )
    for intent in RISK_BY_INTENT:
        for split, minimum in expected.items():
            if observed[(intent, split)] < minimum:
                raise ValueError(f"insufficient {split} coverage for {intent}")
    return dataset


def build_dataset(
    *,
    seed_payload: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    _validate_config(config)
    examples = [*seed_payload["examples"], *_expanded_examples(config)]
    examples.sort(key=lambda example: example["example_id"])
    payload = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "dataset_version": config["dataset_version"],
        "advisory_only": True,
        "examples": examples,
    }
    validate_expanded_dataset(payload, seed_payload=seed_payload, config=config)
    return payload


def build_manifest(
    *,
    dataset: dict[str, Any],
    dataset_bytes: bytes,
    seed_bytes: bytes,
    config_bytes: bytes,
    config: dict[str, Any],
) -> dict[str, Any]:
    examples = dataset["examples"]
    locked = [example for example in examples if example["split"] == "locked_test"]
    source_counts = Counter(_source_category(example) for example in examples)
    return {
        "schema_version": MANIFEST_VERSION,
        "dataset_version": dataset["dataset_version"],
        "generator_version": GENERATOR_VERSION,
        "seed": config["seed"],
        "provenance": "controlled synthetic and hand-reviewed SentinelVoice data only",
        "external_dataset_sources": [],
        "hashes": {
            "config_sha256": sha256_bytes(config_bytes),
            "dataset_sha256": sha256_bytes(dataset_bytes),
            "v2_a_seed_sha256": sha256_bytes(seed_bytes),
        },
        "counts": {
            "examples": len(examples),
            "groups": len({example["group_id"] for example in examples}),
            "by_split": _count_by(examples, "split"),
            "by_intent": _count_by(examples, "intent"),
            "by_risk": _count_by(examples, "risk"),
            "by_source_category": dict(sorted(source_counts.items())),
            "intent_by_split": _count_by_split_and_label(examples, "intent"),
            "risk_by_split": _count_by_split_and_label(examples, "risk"),
            "groups_by_split": {
                split: len(
                    {
                        example["group_id"]
                        for example in examples
                        if example["split"] == split
                    }
                )
                for split in ("train", "validation", "locked_test")
            },
            "locked_test_examples": len(locked),
            "locked_test_groups": len({example["group_id"] for example in locked}),
        },
        "locked_test_policy": (
            "Frozen before model selection; never used for feature, model, "
            "hyperparameter, calibration, or abstention selection."
        ),
    }


def build_files(
    *,
    seed_path: Path = DEFAULT_SEED_PATH,
    config_path: Path = DEFAULT_CONFIG_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> tuple[Path, Path]:
    seed_bytes = seed_path.read_bytes()
    config_bytes = config_path.read_bytes()
    seed_payload = json.loads(seed_bytes)
    config = json.loads(config_bytes)
    dataset = build_dataset(seed_payload=seed_payload, config=config)
    dataset_bytes = stable_json_bytes(dataset)
    manifest = build_manifest(
        dataset=dataset,
        dataset_bytes=dataset_bytes,
        seed_bytes=seed_bytes,
        config_bytes=config_bytes,
        config=config,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / DEFAULT_DATASET_PATH.name
    manifest_path = output_dir / DEFAULT_MANIFEST_PATH.name
    dataset_path.write_bytes(dataset_bytes)
    manifest_path.write_bytes(stable_json_bytes(manifest))
    return dataset_path, manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-path", type=Path, default=DEFAULT_SEED_PATH)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    dataset_path, manifest_path = build_files(
        seed_path=args.seed_path,
        config_path=args.config,
        output_dir=args.output_dir,
    )
    print(f"Wrote {dataset_path}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
