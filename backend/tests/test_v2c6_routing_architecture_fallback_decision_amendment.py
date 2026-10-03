from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
ML_PATH = ROOT / "data/evals/v2/ml"
AMENDMENT_PATH = ML_PATH / "v2c6_routing_architecture_fallback_decision_amendment.json"
ORIGINAL_RELATIVE_PATH = "data/evals/v2/ml/v2c6_routing_architecture_fallback_decision.json"
RESOLVER_RELATIVE_PATH = "backend/app/agent/resource_resolver.py"
ORCHESTRATOR_RELATIVE_PATH = "backend/app/agent/orchestrator.py"
DISCOVERY_HEAD = "6f196eb6063f5e72b58cb7f838302daad0af380d"
SCHEMA = "v2c6-routing-architecture-fallback-decision-amendment.v1"


@pytest.fixture(scope="module")
def amendment() -> dict[str, Any]:
    return json.loads(AMENDMENT_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def original() -> dict[str, Any]:
    return json.loads((ROOT / ORIGINAL_RELATIVE_PATH).read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_at_discovery(amendment: dict[str, Any], relative_path: str) -> str:
    """Return runtime source only while it is byte-identical to the discovery state."""
    recorded = amendment["observed_runtime_behavior"]["observed_file_sha256"][relative_path]
    path = ROOT / relative_path
    if sha256_file(path) != recorded:
        pytest.skip(f"{relative_path} changed after discovery; recorded facts still apply")
    return path.read_text(encoding="utf-8")


def method_source(source: str, name: str) -> str:
    match = re.search(rf"^    (?:async )?def {name}\(", source, flags=re.MULTILINE)
    assert match is not None, name
    start = match.start()
    following = re.search(
        r"^    (?:@staticmethod|(?:async )?def )", source[match.end() :], flags=re.MULTILINE
    )
    return source[start : match.end() + following.start()] if following else source[start:]


def test_amendment_identity_and_design_only_status(amendment: dict[str, Any]) -> None:
    assert amendment["schema_version"] == SCHEMA
    assert amendment["amendment_version"] == SCHEMA
    assert amendment["phase"] == "V2-C6 Routing Architecture Fallback Decision"
    assert amendment["substep"] == "Resolver Safety Amendment"
    assert amendment["status"] == "FROZEN"
    assert amendment["amends_decision_id"] == (
        "CLARIFICATION_GATED_STRUCTURED_LLM_PROTECTED_ROUTING"
    )
    status = amendment["amendment_status"]
    assert status["design_frozen"] is True
    assert status["design_only"] is True
    assert status["original_decision_remains_historical_evidence"] is True
    for field in (
        "original_decision_rewritten",
        "routing_fallback_implemented",
        "resource_resolver_changed",
        "runtime_behavior_changed",
        "step29i_authorized",
        "final_holdout_accessed",
        "production_ready_claimed",
        "final_model_acceptance_claimed",
        "llm_calls_performed",
        "embeddings_generated",
        "model_fitting_performed",
        "model_inference_performed",
        "r4_classifier_or_data_cycle_authorized",
    ):
        assert status[field] is False, field


def test_amendment_hash_binds_the_original_frozen_decision(
    amendment: dict[str, Any], original: dict[str, Any]
) -> None:
    bound = amendment["amended_artifact"]
    assert bound["path"] == ORIGINAL_RELATIVE_PATH
    assert bound["sha256"] == sha256_file(ROOT / ORIGINAL_RELATIVE_PATH)
    assert bound["schema_version"] == original["schema_version"]
    assert bound["committed_at_head"] == DISCOVERY_HEAD
    assert amendment["discovery_baseline"] == {
        "branch": "v2/ml-routing-evaluation",
        "head": DISCOVERY_HEAD,
        "head_subject": "Freeze V2-C6 routing architecture fallback decision",
    }
    assert original["decision"]["placement"]["pre_llm_resource_resolver_unchanged"] is True


def test_pre_llm_resolver_is_not_frozen_as_unchanged_for_protected_actions(
    amendment: dict[str, Any],
) -> None:
    superseded = amendment["superseded_original_assertions"]
    resolver = superseded["decision.placement.pre_llm_resource_resolver_unchanged"]
    assert resolver["original_value"] is True
    assert resolver["amended_value_for_protected_actions"] is False
    invariant = amendment["corrected_invariant"]
    assert invariant["pre_llm_resource_resolver_unchanged_for_protected_actions"] is False
    assert invariant[
        "implementation_must_remove_bypass_or_defer_resolver_protected_pending_action_creation"
    ] is True
    assert invariant["resolver_freeze_card_request_action_may_remain_a_verification_bypass"] is (
        False
    )


def test_recorded_freeze_card_resolver_side_effect(amendment: dict[str, Any]) -> None:
    freeze = amendment["observed_runtime_behavior"]["freeze_card"]
    assert freeze["resolver_creates_pending_action"] is True
    assert freeze["resolver_issues_protected_resource_clarification"] is True
    assert freeze["bypasses_future_semantic_verifier_placement"] is True
    assert freeze["bypasses_explicit_confirmation"] is False
    assert freeze["bypasses_tool_executor_authorization"] is False
    assert "request_action" in freeze["single_matching_card"]


def test_freeze_card_side_effect_matches_discovery_source(amendment: dict[str, Any]) -> None:
    source = source_at_discovery(amendment, RESOLVER_RELATIVE_PATH)
    activate = method_source(source, "_activate_candidate")
    assert 'intent == "freeze_card"' in activate
    assert "state.request_action(" in activate
    assert '"freeze_card",' in activate
    resolve_card = method_source(source, "_resolve_card")
    assert 'if "freeze" in text:' in resolve_card
    assert "self._activate_candidate(" in resolve_card
    assert "state.request_resource_resolution(" in resolve_card
    resolve_pending = method_source(source, "_resolve_pending")
    assert "self._activate_candidate(" in resolve_pending


def test_recorded_create_dispute_behavior(amendment: dict[str, Any]) -> None:
    behavior = amendment["observed_runtime_behavior"]
    dispute = behavior["create_dispute"]
    assert dispute["resolver_creates_pending_action"] is False
    assert dispute["resolver_issues_protected_resource_clarification"] is True
    assert dispute["bypasses_future_semantic_verifier_placement"] == (
        "resource_clarification_only"
    )
    assert "_run_model_loop" in dispute["pending_action_creation_path"]
    assert behavior["symmetry_between_protected_tools"] is False
    assert behavior["authorization_bypass_observed"] is False


def test_create_dispute_behavior_matches_discovery_source(amendment: dict[str, Any]) -> None:
    source = source_at_discovery(amendment, RESOLVER_RELATIVE_PATH)
    activate = method_source(source, "_activate_candidate")
    assert activate.count("state.request_action(") == 1
    assert "create_dispute" not in activate
    resolve_transaction = method_source(source, "_resolve_transaction")
    assert 'if "dispute" in text:' in resolve_transaction
    assert "state.request_resource_resolution(" in resolve_transaction
    assert "request_action(" not in resolve_transaction.replace(
        "request_resource_resolution(", ""
    )


def test_orchestrator_short_circuit_matches_discovery_source(
    amendment: dict[str, Any],
) -> None:
    source = source_at_discovery(amendment, ORCHESTRATOR_RELATIVE_PATH)
    turn = method_source(source, "handle_text_turn")
    resolve = turn.index("self._resource_resolver.resolve(")
    clarification = turn.index("text=resolution.clarification")
    confirmation = turn.index("text=_confirmation_prompt(state.pending_action.action)")
    model_loop = turn.rindex("return await self._run_model_loop(")
    assert resolve < clarification < confirmation < model_loop


def test_semantic_verification_precedes_protected_clarification_and_pending_action(
    amendment: dict[str, Any],
) -> None:
    invariant = amendment["corrected_invariant"]
    assert invariant["applies_to"] == "every_executable_protected_action"
    assert invariant["semantic_verification_must_precede"] == [
        "protected_action_specific_resource_clarification",
        "pending_action_creation",
        "execution_confirmation_prompt",
    ]
    order = amendment["corrected_future_order"]
    verification = order.index("structured_semantic_verification")
    for later in (
        "if_explicit_protected_resource_resolution_and_binding",
        "resource_clarification_if_necessary",
        "create_pending_action",
        "explicit_execution_confirmation",
        "deterministic_tool_executor_authorization",
        "execution",
    ):
        assert verification < order.index(later), later
    assert order.index("if_non_explicit_semantic_clarification_and_stop") == verification + 1
    assert order.index("create_pending_action") < order.index("explicit_execution_confirmation")


def test_resource_resolution_examples(amendment: dict[str, Any]) -> None:
    resolution = amendment["resource_resolution_after_verification"]
    assert resolution["protected_resource_resolution_only_after_explicit_semantic_verification"]
    assert resolution["explicit_request_may_still_need_resource_clarification"] is True
    explicit, informational = resolution["examples"]
    assert explicit["verifier_decision"] == "EXPLICIT_CURRENT_ACTION"
    assert explicit["expected"] == "ask_which_card"
    assert informational["expected"] == "semantic_clarification_only"
    assert informational["ask_which_card"] is False
    assert informational["create_pending_action"] is False


def test_multi_turn_verified_semantic_state_is_non_authoritative(
    amendment: dict[str, Any],
) -> None:
    state = amendment["multi_turn_verified_semantic_state"]
    for field in (
        "is_confirmation",
        "is_authorization",
        "can_execute",
        "reusable_for_different_protected_action",
        "bypasses_explicit_execution_confirmation",
        "state_representation_frozen_by_this_amendment",
    ):
        assert state[field] is False, field
    assert state["action_specific"] is True
    assert set(state["invalidated_by"]) == {"cancellation", "correction", "abandonment"}
    assert state["voice_interruption_handling"] == (
        "must_be_safe_under_existing_interruption_semantics"
    )


def test_resource_selection_replies_cannot_substitute_for_verification(
    amendment: dict[str, Any],
) -> None:
    replies = amendment["resource_selection_replies"]
    assert replies["examples"] == ["the Visa card", "the first one", "ending in 1234"]
    assert replies["independently_count_as_explicit_current_action"] is False
    assert replies["may_continue_only_through_explicit_verified_semantic_state"] is True
    assert (
        "resource_selection_reply_does_not_independently_verify_protected_action"
        in amendment["additional_required_evaluation_scenarios"]
    )


def test_original_authority_guarantees_remain_intact(
    amendment: dict[str, Any], original: dict[str, Any]
) -> None:
    preserved = amendment["preserved_original_guarantees"]
    verifier = original["decision"]["protected_action_semantic_verifier"]
    assert preserved["verifier_decisions"] == [
        "EXPLICIT_CURRENT_ACTION",
        "AMBIGUOUS_OR_INFORMATIONAL",
        "NOT_REQUESTED",
    ]
    assert preserved["verifier_decisions"] == verifier["output_contract"]["allowed_decisions"]
    assert preserved["verifier_authority"] == verifier["authority"]
    assert all(value is False for value in preserved["verifier_authority"].values())
    assert preserved["local_classifier"]["runtime_routing_authority"] is False
    supported = preserved["supported_protected_actions"]
    assert supported["executable_protected_write_tools_at_freeze"] == [
        "create_dispute",
        "freeze_card",
    ]
    assert supported["taxonomy_label_creates_runtime_tool"] is False
    assert preserved["explicit_confirmation_required"] is True
    assert preserved["tool_executor_remains_deterministic_authority"] is True
    assert preserved["threshold_weakening_permitted"] is False
    assert preserved["r4_permitted"] is False
    assert preserved["fresh_fallback_evaluation_required"] is True


def test_holdout_step29i_and_next_step_are_unchanged(
    amendment: dict[str, Any], original: dict[str, Any]
) -> None:
    assert amendment["final_holdout_policy"] == original["final_holdout_policy"]
    assert amendment["final_holdout_policy"]["access_permitted"] is False
    assert amendment["step29i_policy"] == {
        "blocked": True,
        "authorized_by_this_amendment": False,
    }
    assert amendment["next_required"] == original["next_required"]
    assert amendment["orchestration_refactor_owner"] == (
        "separately_governed_implementation_contract"
    )
    serialized = json.dumps(amendment)
    assert serialized.count("v2c5_final_holdout.json") == 1
