from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from backend.app.tools.definitions import PermissionLevel
from backend.app.tools.registry import TOOL_REGISTRY

ROOT = Path(__file__).resolve().parents[2]
DECISION_PATH = ROOT / "data/evals/v2/ml/v2c6_routing_architecture_fallback_decision.json"
R3_CONTRACT_PATH = ROOT / (
    "data/evals/v2/ml/v2c6_protected_intent_gate_remediation_design_contract.json"
)
PROHIBITED_HOLDOUT_PATH = "data/evals/v2/ml/v2c5_final_holdout.json"
SCHEMA = "v2c6-routing-architecture-fallback-decision.v1"
VERIFIER_DECISIONS = [
    "EXPLICIT_CURRENT_ACTION",
    "AMBIGUOUS_OR_INFORMATIONAL",
    "NOT_REQUESTED",
]


@pytest.fixture(scope="module")
def decision() -> dict[str, Any]:
    return json.loads(DECISION_PATH.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_decision_identity_and_design_only_status(decision: dict[str, Any]) -> None:
    assert decision["schema_version"] == SCHEMA
    assert decision["decision_version"] == SCHEMA
    assert decision["phase"] == "V2-C6 Routing Architecture Fallback Decision"
    assert decision["status"] == "FROZEN"
    assert decision["decision_id"] == "CLARIFICATION_GATED_STRUCTURED_LLM_PROTECTED_ROUTING"
    status = decision["decision_status"]
    assert status["design_frozen"] is True
    assert status["design_only"] is True
    for field in (
        "routing_fallback_implemented",
        "runtime_behavior_changed",
        "step29i_authorized",
        "final_holdout_accessed",
        "production_ready_claimed",
        "final_model_acceptance_claimed",
        "safety_claimed_from_design_alone",
        "llm_calls_performed",
        "embeddings_generated",
        "model_fitting_performed",
        "model_inference_performed",
        "threshold_tuning_performed",
        "r4_classifier_or_data_cycle_authorized",
    ):
        assert status[field] is False, field


def test_repository_baseline_is_recorded(decision: dict[str, Any]) -> None:
    assert decision["repository_baseline"] == {
        "branch": "v2/ml-routing-evaluation",
        "head": "d5e1e498775f595b2a9107f95484116cb1c73487",
        "head_subject": "Freeze V2-C6 R3 model selection result",
    }


def test_source_artifacts_are_hash_bound(decision: dict[str, Any]) -> None:
    sources = decision["source_artifacts"]
    assert set(sources) == {
        "r3_model_selection_results",
        "r3_model_selection_results_manifest",
        "r3_remediation_design_contract",
        "taxonomy_freeze",
        "taxonomy_freeze_manifest",
    }
    for specification in sources.values():
        assert specification["path"] != PROHIBITED_HOLDOUT_PATH
        assert "final_holdout" not in specification["path"]
        path = ROOT / specification["path"]
        assert sha256_file(path) == specification["sha256"]
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["schema_version"] == specification["schema_version"]
    assert sources["r3_model_selection_results"]["sha256"] == (
        "883dd954e9ed08c9d2be8a887c803270d09fda11341da9edd8cf67226204cde6"
    )
    assert sources["taxonomy_freeze"]["sha256"] == (
        "c50453617e1b95ead73780597f3f22c00365b88a2de0b32ba508a8d6f56e25c8"
    )


def test_triggering_evidence_matches_frozen_r3_result(decision: dict[str, Any]) -> None:
    results = json.loads(
        (ROOT / decision["source_artifacts"]["r3_model_selection_results"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    selection = results["selection"]
    evidence = decision["triggering_evidence"]
    assert evidence["selection_status"] == selection["selection_status"]
    assert evidence["selection_status"] == "NO_ACCEPTABLE_CANDIDATE"
    assert evidence["eligible_candidate_count"] == selection["eligible_candidate_count"] == 0
    assert evidence["selected_candidate_id"] is None
    assert evidence["winner_forced"] is False
    assert evidence["gates_weakened"] is False
    assert evidence["r3_next_required"] == results["next_required"]
    assert evidence["r3_step29i_authorized"] is False
    assert evidence["causal_root_cause_established"] is False
    expected_failures = {
        row["candidate_id"]: [
            f"{gate['scope']}:{gate['metric']}"
            for gate in row["safety_gate_results"]["gates"]
            if not gate["passed"]
        ]
        for row in results["candidate_results"]
    }
    assert evidence["failed_mandatory_gates"] == expected_failures


def test_r3_stop_rule_requires_this_decision() -> None:
    contract = json.loads(R3_CONTRACT_PATH.read_text(encoding="utf-8"))
    failure = contract["stop_rule"]["failure_continuation"]
    assert failure["next_required"] == "routing_architecture_fallback_decision"
    assert contract["stop_rule"]["automatic_r4_classifier_or_data_cycle_authorized"] is False


def test_local_classifier_has_no_runtime_authority(decision: dict[str, Any]) -> None:
    assert decision["decision"]["local_classifier"] == {
        "role": "evaluation_evidence_only",
        "runtime_routing_authority": False,
        "runtime_integration_permitted": False,
    }
    assert decision["decision"]["runtime_semantic_routing"] == (
        "existing_groq_first_llm_tool_calling"
    )


def test_verifier_is_closed_structured_and_protected_boundary_only(
    decision: dict[str, Any],
) -> None:
    verifier = decision["decision"]["protected_action_semantic_verifier"]
    assert verifier["invoked_on_every_turn"] is False
    assert verifier["invoked_for_non_protected_proposals"] is False
    assert verifier["action_scope"] == "exactly_one_proposed_protected_action"
    assert verifier["output_contract"] == {
        "format": "closed_structured_enum",
        "allowed_decisions": VERIFIER_DECISIONS,
        "free_form_output_authoritative": False,
        "values_outside_enum": "verifier_failure",
    }
    assert verifier["provider"] == {
        "abstraction": "existing_LLMProvider_protocol",
        "preferred_provider": "groq",
        "thin_swappable_interface": True,
        "second_agent": False,
        "multi_agent_system": False,
        "new_service": False,
        "new_infrastructure_component": False,
    }


def test_verifier_is_never_authorization(decision: dict[str, Any]) -> None:
    authority = decision["decision"]["protected_action_semantic_verifier"]["authority"]
    assert set(authority) == {
        "is_authorization",
        "may_authenticate_customer",
        "may_establish_ownership",
        "may_satisfy_confirmation",
        "may_execute_tool",
        "may_bypass_tool_executor",
        "may_bypass_pending_action_state",
        "may_authorize_unsupported_tool",
    }
    assert all(value is False for value in authority.values())
    asymmetric = decision["rationale"]["asymmetric_authority"]
    assert asymmetric["negative_or_uncertain_result_effect"] == "clarification_only"
    assert asymmetric["positive_result_can_execute"] is False
    assert asymmetric["positive_result_bypasses_confirmation"] is False


CURRENT_RUNTIME_SEQUENCE = [
    "model_proposes_tool",
    "registered_and_effective_allowed_tool_check",
    "authoritative_arguments_initialization",
    "resource_bindings_lookup",
    "resource_and_active_intent_validation",
    "possible_resource_clarification_return",
    "pydantic_input_validation",
    "requires_confirmation_branch",
    "conversation_state_request_action",
]


def test_verifier_placement_is_after_allowed_check_and_before_resource_binding(
    decision: dict[str, Any],
) -> None:
    placement = decision["decision"]["placement"]
    assert placement["current_runtime_sequence"] == CURRENT_RUNTIME_SEQUENCE
    future = placement["future_protected_tool_sequence"]
    verifier = future.index("structured_protected_action_semantic_verifier")
    assert future[:verifier] == [
        "model_proposes_tool",
        "registered_and_effective_allowed_tool_check",
    ]
    assert future[verifier + 1 :] == CURRENT_RUNTIME_SEQUENCE[2:]
    for later_step in (
        "authoritative_arguments_initialization",
        "resource_bindings_lookup",
        "possible_resource_clarification_return",
        "pydantic_input_validation",
        "requires_confirmation_branch",
        "conversation_state_request_action",
    ):
        assert verifier < future.index(later_step), later_step
    assert placement["insertion_point"] == (
        "after_the_registered_and_effective_allowed_tool_check_and_before_the_"
        "existing_authoritative_arguments_and_RESOURCE_BINDINGS_block"
    )
    for field in (
        "verifier_runs_after_allowed_tool_check",
        "verifier_runs_before_resource_binding",
        "verifier_runs_before_resource_clarification",
        "verifier_runs_before_input_validation",
        "verifier_runs_before_requires_confirmation_branch",
    ):
        assert placement[field] is True, field
    # Historical value only: the resolver safety amendment supersedes it for
    # protected actions (see test_v2c6_routing_architecture_fallback_decision_amendment).
    assert placement["pre_llm_resource_resolver_unchanged"] is True
    amendment = json.loads(
        (
            ROOT / "data/evals/v2/ml/v2c6_routing_architecture_fallback_decision_amendment.json"
        ).read_text(encoding="utf-8")
    )
    assert amendment["corrected_invariant"][
        "pre_llm_resource_resolver_unchanged_for_protected_actions"
    ] is False
    assert placement["pending_action_created_before_verification"] is False
    assert "existing_branch" not in placement["runtime_reference"]
    steps = decision["decision"]["routing_semantics"]["protected_proposal_ordered_steps"]
    assert steps[0] == "confirm_registered_and_effective_allowed_protected_tool"
    assert steps[1] == "do_not_create_pending_action"


def test_recorded_current_sequence_matches_orchestrator_source(
    decision: dict[str, Any],
) -> None:
    source = (ROOT / "backend/app/agent/orchestrator.py").read_text(encoding="utf-8")
    loop = source[source.index("    async def _run_model_loop(") :]
    loop = loop[: loop.index("\n    async def ", 1)]
    markers = [
        "registered = TOOL_REGISTRY.get(tool_call.name)",
        "authoritative_arguments = dict(tool_call.arguments)",
        "binding = _RESOURCE_BINDINGS.get(tool_call.name)",
        "text=_resource_clarification(tool_call.name)",
        "registered.input_model.model_validate(",
        "if registered.definition.requires_confirmation:",
        "state.request_action(",
    ]
    positions = [loop.index(marker) for marker in markers]
    assert positions == sorted(positions)
    assert decision["decision"]["placement"]["runtime_reference"] == {
        "module": "backend/app/agent/orchestrator.py",
        "method": "AgentOrchestrator._run_model_loop",
        "registered_tool_lookup": "TOOL_REGISTRY.get(tool_call.name)",
        "effective_allowed_tool_check": "AgentOrchestrator._allowed_tool_names",
        "resource_binding_block": "authoritative_arguments_and__RESOURCE_BINDINGS",
        "resource_clarification": "_resource_clarification",
        "input_validation": "registered.input_model.model_validate",
        "confirmation_branch": "registered.definition.requires_confirmation",
        "pending_action_creation": "ConversationState.request_action",
    }


def test_routing_semantics_are_exact(decision: dict[str, Any]) -> None:
    routing = decision["decision"]["routing_semantics"]
    assert routing["non_protected_proposal"] == "preserve_existing_orchestration_path"
    assert routing["on_explicit_current_action"] == [
        "existing_authoritative_arguments_and_resource_binding",
        "existing_resource_and_active_intent_validation",
        "existing_input_validation",
        "existing_requires_confirmation_branch",
        "request_pending_protected_action",
        "explicit_user_confirmation",
        "deterministic_tool_executor_authorization",
        "authentication_and_ownership_checks",
        "execution",
    ]
    clarification = routing["on_ambiguous_or_informational_or_not_requested"]
    assert clarification["create_pending_action"] is False
    assert clarification["ask_execution_confirmation"] is False
    assert clarification["response"] == "deterministic_narrow_clarification_question"
    assert clarification["clarification_text_generated_by_llm"] is False
    assert clarification["user_answer_processing"] == "new_user_turn"
    assert clarification["perform_resource_binding_clarification"] is False
    assert clarification["clarification_answer_counts_as_confirmation"] is False


def test_verifier_failure_fails_closed(decision: dict[str, Any]) -> None:
    failure = decision["decision"]["routing_semantics"]["on_verifier_failure"]
    assert failure["behavior"] == "fail_closed"
    assert failure["create_pending_action"] is False
    assert failure["perform_resource_binding_clarification"] is False
    assert set(failure["failure_modes"]) == {
        "timeout",
        "provider_failure",
        "malformed_structured_output",
        "invalid_enum_value",
        "other_verifier_failure",
    }
    assert failure["human_handoff"] == "existing_escalate_to_human_path_where_appropriate"


def test_deterministic_controls_are_preserved(decision: dict[str, Any]) -> None:
    controls = decision["decision"]["preserved_deterministic_controls"]
    assert controls and all(value is True for value in controls.values())
    assert controls["explicit_confirmation_mandatory_after_positive_verification"] is True
    assert controls["confirmation_one_use_and_stale_safe"] is True


def test_protected_actions_follow_the_runtime_registry(decision: dict[str, Any]) -> None:
    supported = decision["decision"]["supported_protected_actions"]
    registered_protected = sorted(
        name
        for name, tool in TOOL_REGISTRY.items()
        if tool.definition.permission_level == PermissionLevel.PROTECTED_WRITE
    )
    assert supported["executable_protected_write_tools_at_freeze"] == registered_protected
    assert registered_protected == ["create_dispute", "freeze_card"]
    for name in registered_protected:
        definition = TOOL_REGISTRY[name].definition
        assert definition.requires_confirmation is True
        assert definition.requires_authentication is True
    for intent in supported["taxonomy_protected_intents_without_runtime_tool"]:
        assert intent not in TOOL_REGISTRY
    assert supported["taxonomy_label_creates_runtime_tool"] is False
    assert supported["verifier_may_imply_unregistered_capability"] is False
    assert supported["new_tools_introduced"] is False


def test_alternatives_are_recorded(decision: dict[str, Any]) -> None:
    alternatives = {row["id"]: row["status"] for row in decision["alternatives_considered"]}
    assert alternatives == {
        "r4_local_classifier_remediation": "rejected",
        "weaken_protected_fpr_threshold": "rejected",
        "always_clarify_every_protected_request": "not_selected",
        "llm_verifier_on_every_turn": "not_selected",
        "llm_verifier_as_authorization": "rejected",
        "escalate_every_protected_request_to_human": "not_selected",
    }
    assert all(row["reason"] for row in decision["alternatives_considered"])


def test_tradeoffs_and_high_reversibility(decision: dict[str, Any]) -> None:
    tradeoffs = decision["tradeoffs"]
    assert "one_additional_model_call_for_protected_proposals" in tradeoffs["costs"]
    assert "explicit_fail_closed_semantics" in tradeoffs["benefits"]
    assert tradeoffs["reversibility"]["level"] == "HIGH"


def test_evaluation_requirements_are_frozen_without_invented_thresholds(
    decision: dict[str, Any],
) -> None:
    requirements = decision["evaluation_requirements"]
    assert len(requirements["required_scenarios"]) == 16
    assert len(set(requirements["required_scenarios"])) == 16
    for scenario in (
        "malformed_verifier_output_fails_closed",
        "verifier_timeout_fails_closed",
        "provider_failure_fails_closed",
        "no_protected_execution_without_explicit_confirmation",
        "verifier_cannot_directly_execute_a_tool",
        "prompt_injection_cannot_bypass_deterministic_controls",
    ):
        assert scenario in requirements["required_scenarios"]
    assert requirements["tracked_metrics"] == [
        "protected_semantic_false_positive_rate",
        "explicit_protected_request_recall",
        "clarification_rate",
        "clarification_recovery_task_success",
        "verifier_failure_fail_closed_compliance",
        "protected_execution_authorization_compliance",
        "verifier_latency_p50_p90_p95",
        "incremental_cost_per_protected_turn",
    ]
    assert requirements["metric_thresholds_frozen_by_this_decision"] is False


def test_evidence_governance_blocks_reused_or_consumed_evidence(
    decision: dict[str, Any],
) -> None:
    governance = decision["evaluation_requirements"]["evidence_governance"]
    assert governance == {
        "consumed_v2c5_final_holdout_prohibited": True,
        "r3_fresh_evidence_consumed_by_r3_candidate_evaluation": True,
        "r3_fresh_evidence_may_be_presented_as_untouched_final_acceptance_evidence": False,
        "separately_governed_fresh_evaluation_required_before_runtime_acceptance": True,
    }


def test_final_holdout_and_step29i_remain_blocked(decision: dict[str, Any]) -> None:
    policy = decision["final_holdout_policy"]
    assert policy["prohibited_path"] == PROHIBITED_HOLDOUT_PATH
    for field in (
        "access_permitted",
        "hashing_permitted",
        "inspection_permitted",
        "parsing_permitted",
        "searching_permitted",
        "bound_as_source_artifact",
    ):
        assert policy[field] is False
    assert decision["step29i_policy"] == {"blocked": True, "authorized_by_this_decision": False}


def test_next_step_is_separately_governed_implementation(decision: dict[str, Any]) -> None:
    assert decision["next_required"] == "v2c6_routing_fallback_implementation_and_evaluation"
    assert decision["next_required_governance"] == {
        "separately_governed": True,
        "must_freeze_own_implementation_and_evaluation_contract_first": True,
        "implemented_by_this_decision": False,
    }
    for prohibited in (
        "runtime_behavior_change",
        "verifier_implementation",
        "groq_call",
        "r4_classifier_or_data_cycle",
        "v2c5_final_holdout_access",
    ):
        assert prohibited in decision["prohibited_in_this_step"]
