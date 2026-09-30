from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from backend.app.evaluation.v2_contracts import IntentLabel
from scripts import freeze_v2c5_taxonomy as freeze

HISTORICAL_NINE_INTENTS = (
    "informational_policy",
    "account_balance",
    "recent_transactions",
    "transaction_details",
    "card_status",
    "freeze_card",
    "create_dispute",
    "escalation",
    "unsupported_or_uncertain",
)


@pytest.fixture(scope="module")
def step19() -> freeze.Step19Evidence:
    return freeze.load_step19()


@pytest.fixture(scope="module")
def payload(step19: freeze.Step19Evidence) -> dict[str, Any]:
    return freeze.build_freeze_payload(step19.adjudication)


def resolution_by_candidate(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        value["candidate_intent_name"]: value
        for value in payload["candidate_cluster_resolutions"]
    }


def split_by_id(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        value["canonical_cluster_id"]: value
        for value in payload["split_cluster_resolutions"]
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


def test_historical_v2_intent_label_remains_exactly_nine_labels() -> None:
    assert tuple(value.value for value in IntentLabel) == HISTORICAL_NINE_INTENTS


def test_final_taxonomy_contains_exactly_16_unique_sorted_labels(
    payload: dict[str, Any],
) -> None:
    labels = payload["intent_label_order"]

    assert labels == sorted(labels)
    assert len(labels) == 16
    assert len(set(labels)) == 16
    assert tuple(labels) == freeze.INTENT_LABEL_ORDER


def test_exactly_seven_new_labels_are_introduced(
    payload: dict[str, Any],
) -> None:
    assert payload["new_intent_count"] == 7
    assert tuple(payload["new_intents"]) == freeze.NEW_INTENTS
    assert set(payload["new_intents"]).isdisjoint(payload["retained_intents"])


def test_all_candidate_clusters_are_explicitly_resolved(
    payload: dict[str, Any],
) -> None:
    resolutions = payload["candidate_cluster_resolutions"]

    assert len(resolutions) == 9
    assert len({value["canonical_cluster_id"] for value in resolutions}) == 9
    assert len({value["candidate_intent_name"] for value in resolutions}) == 9


def test_seven_candidates_are_accepted_and_two_are_merged(
    payload: dict[str, Any],
) -> None:
    resolutions = resolution_by_candidate(payload)
    accepted = {
        name
        for name, value in resolutions.items()
        if value["resolution"] == "ACCEPT_NEW_INTENT"
    }
    merged = {
        name: value["final_intent"]
        for name, value in resolutions.items()
        if value["resolution"] == "MERGE_TO_EXISTING_INTENT"
    }

    assert accepted == set(freeze.NEW_INTENTS)
    assert merged == {
        "card_retained_by_atm": "informational_policy",
        "transfer_fee_charged": "transaction_details",
    }


def test_all_nine_split_review_clusters_are_explicitly_resolved(
    payload: dict[str, Any],
) -> None:
    resolutions = payload["split_cluster_resolutions"]

    assert len(resolutions) == 9
    assert len({value["canonical_cluster_id"] for value in resolutions}) == 9
    assert all(
        value["resolution"] == "BRANCH_FOR_STEP21_RELABELING"
        for value in resolutions
    )
    assert all(len(value["branches"]) == 2 for value in resolutions)


def test_every_step19_cluster_is_accounted_for_exactly_once(
    step19: freeze.Step19Evidence,
    payload: dict[str, Any],
) -> None:
    resolved_ids = freeze.all_resolution_ids(payload)
    source_ids = {
        value["canonical_cluster_id"]
        for value in step19.adjudication["adjudications"]
    }

    assert len(resolved_ids) == 35
    assert len(set(resolved_ids)) == 35
    assert set(resolved_ids) == source_ids


def test_existing_intent_targets_are_carried_forward_exactly(
    step19: freeze.Step19Evidence,
    payload: dict[str, Any],
) -> None:
    expected = {
        value["canonical_cluster_id"]: value["target_existing_intent"]
        for value in step19.adjudication["adjudications"]
        if value["decision"] == "MAP_TO_EXISTING_INTENT"
    }
    actual = {
        value["canonical_cluster_id"]: value["final_intent"]
        for value in payload["carry_forward_cluster_resolutions"]
        if value["source_decision"] == "MAP_TO_EXISTING_INTENT"
    }

    assert len(expected) == 4
    assert actual == expected


def test_unsupported_clusters_remain_unsupported(
    step19: freeze.Step19Evidence,
    payload: dict[str, Any],
) -> None:
    expected_ids = {
        value["canonical_cluster_id"]
        for value in step19.adjudication["adjudications"]
        if value["decision"] == "REMAIN_UNSUPPORTED"
    }
    actual = {
        value["canonical_cluster_id"]: value["final_intent"]
        for value in payload["carry_forward_cluster_resolutions"]
        if value["source_decision"] == "REMAIN_UNSUPPORTED"
    }

    assert len(expected_ids) == 13
    assert set(actual) == expected_ids
    assert set(actual.values()) == {"unsupported_or_uncertain"}


@pytest.mark.parametrize(
    "decision",
    ["MIXED_OR_INCOHERENT", "INSUFFICIENT_EVIDENCE"],
)
def test_mixed_and_insufficient_source_decisions_are_rejected(
    step19: freeze.Step19Evidence,
    decision: str,
) -> None:
    modified = copy.deepcopy(step19.adjudication)
    modified["adjudications"][0]["decision"] = decision

    with pytest.raises(ValueError, match="mixed or insufficient"):
        freeze.validate_step19_adjudication(modified)


def test_risk_mapping_covers_every_final_label_exactly_once(
    payload: dict[str, Any],
) -> None:
    assert set(payload["risk_by_intent"]) == set(payload["intent_label_order"])
    assert len(payload["risk_by_intent"]) == 16


def test_protected_write_set_is_exact(payload: dict[str, Any]) -> None:
    assert payload["protected_write_intents"] == [
        "cancel_transfer",
        "close_account",
        "create_dispute",
        "freeze_card",
    ]


def test_new_intent_risk_distinctions_are_frozen(
    payload: dict[str, Any],
) -> None:
    risk = payload["risk_by_intent"]
    candidates = resolution_by_candidate(payload)

    assert risk["account_blocked"] == "PRIVATE_READ"
    assert candidates["account_blocked"]["final_intent"] == "account_blocked"
    assert candidates["account_blocked"]["final_intent"] != "freeze_card"
    assert risk["transfer_pending"] == "PRIVATE_READ"
    assert risk["transfer_failed_or_declined"] == "PRIVATE_READ"
    assert risk["passcode_recovery"] == "ESCALATION_OR_UNCERTAIN"
    assert risk["lost_or_stolen_phone"] == "ESCALATION_OR_UNCERTAIN"


def test_no_runtime_tool_mapping_exists(payload: dict[str, Any]) -> None:
    keys = recursive_keys(payload)

    assert all(not ("tool" in key.lower() and "mapping" in key.lower()) for key in keys)
    assert payload["input_scope"]["prohibited_artifacts_consumed"] == []
    assert payload["input_scope"]["consumed_artifacts"] == [
        "data/evals/v2/ml/v2c5_taxonomy_adjudication.json",
        "data/evals/v2/ml/v2c5_taxonomy_adjudication.manifest.json",
    ]
    assert {"text", "normalized_text", "utterance", "examples"}.isdisjoint(keys)


def test_runtime_and_training_governance_remains_non_authoritative(
    payload: dict[str, Any],
) -> None:
    assert payload["runtime_authority"] is False
    assert payload["runtime_behavior_changed"] is False
    assert payload["runtime_tools_changed"] is False
    assert payload["classifier_training_performed"] is False
    assert payload["dataset_relabeling_performed"] is False


def test_fresh_holdout_and_step21_remain_required(
    payload: dict[str, Any],
) -> None:
    assert payload["fresh_holdout_created"] is False
    assert payload["fresh_holdout_required"] is True
    assert payload["step21_required"] is True


def test_deterministic_json_generation_is_stable(
    step19: freeze.Step19Evidence,
) -> None:
    first = freeze.stable_json_bytes(
        freeze.build_freeze_payload(step19.adjudication)
    )
    second = freeze.stable_json_bytes(
        freeze.build_freeze_payload(step19.adjudication)
    )

    assert first == second
    assert first.endswith(b"\n")


def test_existing_outputs_must_match_deterministic_bytes(
    step19: freeze.Step19Evidence,
    tmp_path: Path,
) -> None:
    paths = replace(
        freeze.DEFAULT_PATHS,
        freeze_output=tmp_path / freeze.DEFAULT_PATHS.freeze_output.name,
        freeze_manifest_output=(
            tmp_path / freeze.DEFAULT_PATHS.freeze_manifest_output.name
        ),
    )
    freeze_payload = freeze.build_freeze_payload(step19.adjudication, paths)
    freeze_bytes = freeze.stable_json_bytes(freeze_payload)
    manifest = freeze.build_freeze_manifest(freeze_bytes, paths)
    prepared = freeze.PreparedFreeze(
        freeze=freeze_payload,
        freeze_bytes=freeze_bytes,
        manifest=manifest,
        manifest_bytes=freeze.stable_json_bytes(manifest),
    )
    paths.freeze_output.write_bytes(prepared.freeze_bytes)
    paths.freeze_manifest_output.write_bytes(prepared.manifest_bytes)

    freeze.validate_existing_outputs(prepared, paths)
    paths.freeze_manifest_output.write_bytes(b"modified\n")
    with pytest.raises(ValueError, match="not deterministic"):
        freeze.validate_existing_outputs(prepared, paths)


def test_source_step19_sha_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        freeze.require_sha256(
            b"modified adjudication",
            freeze.STEP19_ADJUDICATION_SHA256,
            "Step 19 adjudication",
        )


@pytest.mark.parametrize("mutation", ["incomplete", "modified"])
def test_exporter_refuses_incomplete_or_modified_step19(
    step19: freeze.Step19Evidence,
    tmp_path: Path,
    mutation: str,
) -> None:
    modified = copy.deepcopy(step19.adjudication)
    if mutation == "incomplete":
        modified["adjudications"].pop()
        modified["cluster_count"] = 34
        modified["reviewed_count"] = 34
    else:
        modified["adjudications"][0]["human_theme"] = "Modified evidence"
    adjudication_path = tmp_path / freeze.DEFAULT_PATHS.step19_adjudication.name
    manifest_path = tmp_path / freeze.DEFAULT_PATHS.step19_manifest.name
    adjudication_path.write_bytes(freeze.stable_json_bytes(modified))
    manifest_path.write_bytes(step19.manifest_bytes)
    paths = replace(
        freeze.DEFAULT_PATHS,
        step19_adjudication=adjudication_path,
        step19_manifest=manifest_path,
        freeze_output=tmp_path / freeze.DEFAULT_PATHS.freeze_output.name,
        freeze_manifest_output=(
            tmp_path / freeze.DEFAULT_PATHS.freeze_manifest_output.name
        ),
    )

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        freeze.export_freeze(paths)
    assert not paths.freeze_output.exists()
    assert not paths.freeze_manifest_output.exists()


def test_manifest_hash_pins_sources_freeze_and_script(
    step19: freeze.Step19Evidence,
    payload: dict[str, Any],
) -> None:
    freeze_bytes = freeze.stable_json_bytes(payload)
    manifest = freeze.build_freeze_manifest(freeze_bytes)

    assert manifest["source_step19_adjudication"]["sha256"] == (
        freeze.STEP19_ADJUDICATION_SHA256
    )
    assert manifest["source_step19_manifest"]["sha256"] == (
        freeze.STEP19_MANIFEST_SHA256
    )
    assert manifest["taxonomy_freeze"]["sha256"] == freeze.sha256_bytes(
        freeze_bytes
    )
    assert len(manifest["step20_script"]["sha256"]) == 64
    assert step19.manifest["adjudication"]["sha256"] == (
        freeze.STEP19_ADJUDICATION_SHA256
    )


def test_split_rules_preserve_safety_distinctions(
    payload: dict[str, Any],
) -> None:
    splits = split_by_id(payload)
    frozen_account = splits["cluster-c29614d54555e30c"]
    verification = splits["cluster-e92864f71372fce3"]
    transfer = splits["cluster-783d0d28bd4c4992"]
    candidate = resolution_by_candidate(payload)["card_retained_by_atm"]

    assert "freeze_card" not in {
        branch["final_intent"] for branch in frozen_account["branches"]
    }
    assert "escalation" not in {
        branch["final_intent"] for branch in verification["branches"]
    }
    assert candidate["final_intent"] == "informational_policy"
    assert "freeze_card" in candidate["rationale"]
    assert "unsupported_or_uncertain" in {
        branch["final_intent"] for branch in transfer["branches"]
    }
    assert "runtime tool" in transfer["safety_constraints"][0]
