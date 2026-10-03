from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from backend.app.conversation.state import (
    ProtectedActionSemanticContext,
    VerifiedDisputeRequest,
)
from backend.app.tools.schemas import CreateDisputeInput

ROOT = Path(__file__).resolve().parents[2]
ML_PATH = ROOT / "data/evals/v2/ml"
AMENDMENT_PATH = ML_PATH / "v2c6_routing_fallback_implementation_contract_amendment.json"
CONTRACT_PATH = ML_PATH / "v2c6_routing_fallback_implementation_evaluation_contract.json"
SCHEMA = "v2c6-routing-fallback-implementation-contract-amendment.v1"


@pytest.fixture(scope="module")
def amendment() -> dict[str, Any]:
    return json.loads(AMENDMENT_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def contract() -> dict[str, Any]:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def test_amendment_identity_and_truthful_status(amendment: dict[str, Any]) -> None:
    assert amendment["schema_version"] == SCHEMA
    assert amendment["amendment_version"] == SCHEMA
    assert amendment["status"] == "FROZEN"
    status = amendment["amendment_status"]
    assert status["design_frozen"] is True
    for field in (
        "original_contract_rewritten",
        "protected_action_semantic_context_fields_changed",
        "groq_calls_performed",
        "fresh_fallback_evaluation_authored",
        "fresh_fallback_evaluation_performed",
        "verifier_max_completion_tokens_changed",
        "production_ready_claimed",
        "final_acceptance_claimed",
        "step29i_authorized",
        "final_holdout_accessed",
    ):
        assert status[field] is False, field


def test_amendment_hash_binds_the_frozen_implementation_contract(
    amendment: dict[str, Any], contract: dict[str, Any]
) -> None:
    bound = amendment["amends_contract"]
    assert bound["path"] == "data/evals/v2/ml/v2c6_routing_fallback_implementation_evaluation_contract.json"
    assert bound["sha256"] == hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest()
    assert bound["schema_version"] == contract["schema_version"]
    assert amendment["discovery_baseline"]["head"] == (
        "25ec5ba74e1b55e0a0130883493308f005cae30d"
    )


def test_discovered_defect_is_recorded(amendment: dict[str, Any]) -> None:
    defect = amendment["discovered_defect"]
    assert defect["original_reason_code_preserved"] is False
    assert defect["original_notes_preserved"] is False
    assert defect["second_model_call_had_conversation_history"] is False


def test_frozen_semantic_context_type_is_unchanged(
    amendment: dict[str, Any], contract: dict[str, Any]
) -> None:
    frozen = contract["protected_action_semantic_context"]["fields"]
    assert set(ProtectedActionSemanticContext.model_fields) == set(frozen)
    assert amendment["unchanged_guarantees"]["protected_action_semantic_context_fields"] == [
        "action",
        "resource_type",
    ]


def test_verified_dispute_request_matches_recorded_narrow_type(
    amendment: dict[str, Any],
) -> None:
    recorded = amendment["verified_dispute_request_state"]
    assert recorded["type_name"] == VerifiedDisputeRequest.__name__
    assert set(VerifiedDisputeRequest.model_fields) == set(recorded["fields"])
    assert recorded["contains_transaction_id"] is False
    assert "transaction_id" not in VerifiedDisputeRequest.model_fields
    assert VerifiedDisputeRequest.model_config.get("extra") == "forbid"
    assert (
        VerifiedDisputeRequest.model_fields["reason_code"].metadata
        == CreateDisputeInput.model_fields["reason_code"].metadata
    )
    for field in (
        "generic_argument_bag",
        "is_confirmation",
        "is_authorization",
        "can_execute",
        "reusable_for_another_protected_action",
        "exposed_as_tool_execution_context_confirmation",
    ):
        assert recorded[field] is False, field
    assert recorded["exists_only_with_matching_create_dispute_semantic_context"] is True


def test_amended_dispute_flow_keeps_application_transaction_authoritative(
    amendment: dict[str, Any],
) -> None:
    flow = amendment["amended_create_dispute_flow"]
    assert flow["selection_turn_reruns_conversational_model"] is False
    assert flow["selection_turn_reverifies_bare_selector"] is False
    assert flow["transaction_id_source"] == "application_selected_resource"
    assert flow["model_supplied_transaction_id_used"] is False
    assert flow["explicit_confirmation_still_required"] is True
    assert flow["non_resource_arguments_validated_before_protected_resource_resolution"]


def test_one_use_marker_is_narrow(amendment: dict[str, Any]) -> None:
    marker = amendment["one_use_continuation_marker"]
    assert marker["storage"] == "local_orchestration_argument_not_conversation_state"
    assert marker["consumed_by_first_tool_call_of_any_kind"] is True
    assert marker["can_exempt_other_protected_action"] is False
    assert marker["is_confirmation"] is False
    assert marker["reaches_tool_executor_directly"] is False


def test_guarantees_budget_holdout_and_step29i_are_unchanged(
    amendment: dict[str, Any], contract: dict[str, Any]
) -> None:
    unchanged = amendment["unchanged_guarantees"]
    assert unchanged["max_completion_tokens"] == 64
    assert unchanged["verifier_timeout_seconds"] == 2.0
    assert unchanged["verifier_decisions"] == contract["verifier_interface"]["allowed_decisions"]
    assert all(value is False for value in unchanged["verifier_authority"].values())
    assert amendment["final_holdout_policy"] == contract["final_holdout_policy"]
    assert amendment["step29i_policy"] == {"blocked": True, "authorized_by_this_amendment": False}
    assert amendment["known_limitations"][0]["open_ended_synonym_rules_added"] is False
    assert amendment["next_required"] == (
        "v2c6_routing_fallback_development_verifier_budget_validation"
    )
