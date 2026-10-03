from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from backend.app.agent.protected_action_verifier import VERIFIER_TIMEOUT_SECONDS

ROOT = Path(__file__).resolve().parents[2]
ML_PATH = ROOT / "data/evals/v2/ml"
AMENDMENT_PATH = ML_PATH / "v2c6_routing_fallback_verifier_budget_provider_amendment.json"
RESULTS_PATH = ML_PATH / "v2c6_routing_fallback_verifier_budget_validation_results.json"
MANIFEST_PATH = ML_PATH / "v2c6_routing_fallback_verifier_budget_validation_results.manifest.json"
FROZEN_RESULTS_SHA256 = "fcc21b6faf947d83b4f1024054dc823299fa0b29318290207f8dbce7a675336b"
FROZEN_MANIFEST_SHA256 = "2da6bf3abee0e1c8e802a8cffb5a3663dab46ae93b2a0bb513cafa589bbe7c71"


@pytest.fixture(scope="module")
def amendment() -> dict[str, Any]:
    return json.loads(AMENDMENT_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def results() -> dict[str, Any]:
    return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_64_token_result_is_immutable_and_hash_bound(amendment: dict[str, Any]) -> None:
    assert sha256(RESULTS_PATH) == FROZEN_RESULTS_SHA256
    assert sha256(MANIFEST_PATH) == FROZEN_MANIFEST_SHA256
    for name, binding in amendment["source_bindings"].items():
        assert "final_holdout" not in binding["path"], name
        if name in {"protected_action_verifier", "dependencies", "groq_provider"}:
            continue  # runtime sources change in the next implementation step
        assert sha256(ROOT / binding["path"]) == binding["sha256"], name
    assert amendment["source_bindings"]["budget_validation_results_64"]["sha256"] == (
        FROZEN_RESULTS_SHA256
    )


def test_recorded_result_matches_the_frozen_artifact(
    amendment: dict[str, Any], results: dict[str, Any]
) -> None:
    frozen = amendment["frozen_64_token_result"]
    cases = results["cases"]
    assert frozen["result_status"] == results["aggregate"]["result_status"] == (
        "TOKEN_BUDGET_AMENDMENT_REQUIRED"
    )
    assert sum(case["finish_reason"] == "length" for case in cases) == 18
    assert sum(case["failure_category"] == "timeout" for case in cases) == 2
    completed = [case for case in cases if case["completion_tokens"] is not None]
    assert len(completed) == 18
    assert all(case["completion_tokens"] == 64 for case in completed)
    assert frozen["all_completed_responses_consumed_exactly_64_completion_tokens"] is True
    assert frozen["structural_success_count"] == 0
    assert all(case["observed_decision"] is None for case in cases)
    assert frozen["total_provider_calls"] == 20
    assert all(case["provider_call_count"] == 1 for case in cases)
    assert frozen["budget_64_rejected"] is True
    assert frozen["failure_class"] == "token_budget_exhaustion"


def test_semantic_zero_is_not_acceptance_or_quality_evidence(amendment: dict[str, Any]) -> None:
    semantic = amendment["semantic_interpretation"]
    assert semantic["exact_correct_of_20"] == 0
    assert semantic["semantic_quality_evidence"] is False
    evidence = amendment["development_evidence"]
    assert evidence["development_evidence_consumed"] is True
    assert evidence["cases_eligible_for_fresh_evaluation"] is False
    assert evidence["cases_eligible_for_final_acceptance"] is False


def test_provider_boundary_finding_is_not_overclaimed(amendment: dict[str, Any]) -> None:
    finding = amendment["provider_boundary_finding"]
    assert finding["groq_sdk_defaults"]["max_retries"] == 2
    assert finding["groq_sdk_defaults"]["request_timeout_seconds"] == 60
    assert finding["causal_mechanism_established"] is False
    assert finding["outer_asyncio_timeout_bounded_wall_clock_in_observed_timeouts"] is False


def test_amended_verifier_configuration_is_exact(amendment: dict[str, Any]) -> None:
    configuration = amendment["amended_verifier_configuration"]
    assert configuration["model"] == "openai/gpt-oss-20b"
    assert configuration["max_completion_tokens"] == 256
    assert configuration["supersedes_max_completion_tokens"] == 64
    assert configuration["reasoning_effort"] == "low"
    assert configuration["sdk_max_retries"] == 0
    assert configuration["sdk_timeout_seconds"] == 2.0
    assert configuration["outer_verifier_timeout_seconds"] == 2.0 == VERIFIER_TIMEOUT_SECONDS
    assert configuration["tool_choice"] == "auto"
    assert configuration["parallel_tool_calls"] is False
    assert configuration["temperature"] == 0
    assert configuration["application_retries"] == 0
    assert configuration["verifier_specific"] is True


def test_tool_choice_stays_auto_and_no_larger_budget_fallback(amendment: dict[str, Any]) -> None:
    assert amendment["tool_choice_rule"]["change_to_required_now"] is False
    experiment = amendment["next_development_experiment"]
    assert experiment["run_in_this_step"] is False
    assert experiment["automatic_larger_budget_fallback"] is False
    assert experiment["configuration"]["tool_choice"] == "auto"
    assert experiment["configuration"]["max_completion_tokens"] == 256
    assert experiment["evidence_role"] == "development_regression_comparison_only"
    assert experiment["may_become_fresh_acceptance_evidence"] is False


def test_unchanged_guarantees_and_thresholds(amendment: dict[str, Any]) -> None:
    unchanged = amendment["unchanged"]
    assert unchanged["decision_enum"] == [
        "EXPLICIT_CURRENT_ACTION",
        "AMBIGUOUS_OR_INFORMATIONAL",
        "NOT_REQUESTED",
    ]
    assert unchanged["classifier_runtime_integration"] is False
    assert unchanged["new_agent_service_or_infrastructure"] is False
    assert unchanged["conversational_provider_budget_and_reasoning"] is True
    thresholds = {gate["metric"]: gate["threshold"] for gate in unchanged["fresh_semantic_acceptance_thresholds"]}
    assert thresholds == {
        "protected_semantic_false_positive_rate": 0.01,
        "explicit_protected_request_recall": 0.8,
    }


def test_status_fresh_evaluation_holdout_and_step29i(amendment: dict[str, Any]) -> None:
    status = amendment["amendment_status"]
    assert status["development_evidence_consumed"] is True
    for field in (
        "amended_runtime_implemented",
        "amended_experiment_run",
        "additional_groq_calls_in_this_step",
        "tool_choice_changed",
        "fresh_fallback_evaluation_started",
        "fresh_fallback_evaluation_authored",
        "final_acceptance_claimed",
        "production_ready_claimed",
        "step29i_authorized",
        "final_holdout_accessed",
    ):
        assert status[field] is False, field
    assert amendment["final_holdout_policy"]["access_permitted"] is False
    assert amendment["step29i_policy"] == {"blocked": True, "authorized_by_this_amendment": False}
    assert amendment["next_required"] == (
        "v2c6_routing_fallback_amended_verifier_implementation_and_development_validation"
    )
