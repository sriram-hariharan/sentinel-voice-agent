from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from backend.app.agent.protected_action_verifier import (
    DECISION_TOOL_NAME,
    VERIFIER_TIMEOUT_SECONDS,
    LLMProtectedActionSemanticVerifier,
)
from backend.app.config.settings import Settings
from backend.app.providers.groq_llm import GroqLLMProvider, LLMProviderError
from backend.app.providers.llm import LLMResponse, LLMToolCall, LLMUsage
from scripts import run_v2c6_routing_fallback_verifier_budget_validation as runner

ROOT = Path(__file__).resolve().parents[2]
CASES_PATH = ROOT / runner.CASES_RELATIVE_PATH
CONTRACT_PATH = ROOT / runner.CONTRACT_RELATIVE_PATH


@pytest.fixture(scope="module")
def cases_artifact() -> dict[str, Any]:
    return json.loads(CASES_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def contract() -> dict[str, Any]:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _settings() -> Settings:
    return Settings(groq_api_key="test-key", llm_model="openai/gpt-oss-20b")


def _record(**overrides: Any) -> dict[str, Any]:
    record = {
        "case_id": "case",
        "proposed_action": "freeze_card",
        "expected_decision": "EXPLICIT_CURRENT_ACTION",
        "observed_decision": "EXPLICIT_CURRENT_ACTION",
        "failure_category": None,
        "finish_reason": "tool_calls",
        "provider_call_count": 1,
        "structured_tool_call_count": 1,
        "provider_error_type": None,
        "provider_error_code": None,
        "provider_error_status": None,
        "provider_error_mentions_token_limit": False,
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "total_tokens": 120,
        "latency_ms": 300.0,
    }
    record.update(overrides)
    return record


class ScriptedInnerProvider:
    """Stands in for the real Groq provider; one scripted outcome per call."""

    provider = "groq"
    model = "openai/gpt-oss-20b"

    def __init__(self, outcome) -> None:
        self.outcome = outcome
        self.calls = 0

    async def generate(self, *, messages, tools=None) -> LLMResponse:
        self.calls += 1
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome(messages)


def _decision_response(messages) -> LLMResponse:
    payload = json.loads(messages[1]["content"])
    utterance = payload["customer_utterance"].casefold()
    explicit = not any(
        marker in utterance
        for marker in ("what happens", "if ", "should", "don't", "worried", "how long",
                       "worth", "no need", "instead")
    )
    decision = "EXPLICIT_CURRENT_ACTION" if explicit else "AMBIGUOUS_OR_INFORMATIONAL"
    return LLMResponse(
        tool_calls=[LLMToolCall(id="c", name=DECISION_TOOL_NAME, arguments={"decision": decision})],
        model="openai/gpt-oss-20b",
        finish_reason="tool_calls",
        usage=LLMUsage(prompt_tokens=180, completion_tokens=30, total_tokens=210),
    )


def _patch_outputs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(runner, "RESULTS_PATH", tmp_path / "results.json")
    monkeypatch.setattr(runner, "MANIFEST_PATH", tmp_path / "results.manifest.json")


def _patch_historical_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    contract = json.loads(CONTRACT_PATH.read_text())
    cases_bytes = CASES_PATH.read_bytes()
    cases = runner.validate_cases(json.loads(cases_bytes))
    monkeypatch.setattr(
        runner,
        "load_and_validate",
        lambda: (contract, cases, cases_bytes),
    )

    def build_historical(_settings: Settings) -> LLMProtectedActionSemanticVerifier:
        client = SimpleNamespace()
        provider = GroqLLMProvider(
            api_key="test-key",
            client=client,
            max_completion_tokens=64,
        )
        return LLMProtectedActionSemanticVerifier(llm=provider)

    monkeypatch.setattr(
        runner,
        "build_protected_action_verifier",
        build_historical,
    )


def _run_with(monkeypatch, tmp_path, outcome) -> tuple[dict[str, Any], ScriptedInnerProvider]:
    _patch_outputs(monkeypatch, tmp_path)
    _patch_historical_runtime(monkeypatch)
    inner = ScriptedInnerProvider(outcome)
    aggregate = runner.run(settings=_settings(), inner_provider_override=inner)
    return aggregate, inner


# Frozen artifacts.


def test_exact_twenty_case_composition_and_development_only(
    cases_artifact: dict[str, Any],
) -> None:
    cases = runner.validate_cases(cases_artifact)
    assert len(cases) == 20
    assert cases_artifact["development_only"] is True
    assert cases_artifact["eligible_for_future_fresh_evaluation"] is False
    assert cases_artifact["eligible_for_final_acceptance"] is False
    counts: dict[tuple[str, str], int] = {}
    for case in cases:
        key = (case["proposed_action"], case["case_kind"])
        counts[key] = counts.get(key, 0) + 1
    assert counts == {
        ("freeze_card", "explicit"): 5,
        ("freeze_card", "boundary"): 5,
        ("create_dispute", "explicit"): 5,
        ("create_dispute", "boundary"): 5,
    }
    categories = {case["boundary_category"] for case in cases if case["case_kind"] == "boundary"}
    assert {"informational", "hypothetical", "advice", "negation"} <= categories


def test_contract_freezes_budget_timeout_model_and_bindings(contract: dict[str, Any]) -> None:
    assert hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest() == (
        runner.EXPECTED_CONTRACT_SHA256
    )
    configuration = contract["frozen_run_configuration"]
    assert configuration["max_completion_tokens"] == 64
    assert configuration["timeout_seconds"] == 2.0 == VERIFIER_TIMEOUT_SECONDS
    assert configuration["automatic_retries"] == 0
    assert configuration["cases"] == 20
    assert configuration["fresh_evaluation"] is False
    assert configuration["alternate_budgets_tested"] is False
    for name, binding in contract["source_bindings"].items():
        if name in {"protected_action_verifier", "dependencies", "groq_provider"}:
            continue
        assert hashlib.sha256((ROOT / binding["path"]).read_bytes()).hexdigest() == (
            binding["sha256"]
        )
    assert contract["result_statuses"]["semantic_errors_imply_token_exhaustion"] is False
    assert contract["step29i_policy"]["blocked"] is True


def test_final_holdout_access_is_blocked() -> None:
    with pytest.raises(PermissionError):
        runner.read_bytes(ROOT / "data/evals/v2/ml/v2c5_final_holdout.json")


# Preflight and check-results never call a provider.


def test_preflight_makes_no_provider_call_and_writes_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_outputs(monkeypatch, tmp_path)

    async def forbidden(*_args, **_kwargs):
        pytest.fail("preflight called the provider")

    monkeypatch.setattr(GroqLLMProvider, "generate", forbidden)
    with pytest.raises(ValueError, match="bound source changed"):
        runner.preflight(_settings)

    assert list(tmp_path.iterdir()) == []


def test_check_results_makes_no_provider_call_and_recomputes_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    aggregate, _ = _run_with(monkeypatch, tmp_path, _decision_response)

    async def forbidden(*_args, **_kwargs):
        pytest.fail("check-results called the provider")

    monkeypatch.setattr(GroqLLMProvider, "generate", forbidden)
    before = sorted(path.name for path in tmp_path.iterdir())
    report = runner.check_results()

    assert report["results_valid"] is True
    assert report["result_status"] == aggregate["result_status"] == runner.KEEP_64
    assert report["provider_calls_performed"] is False
    assert sorted(path.name for path in tmp_path.iterdir()) == before


def test_check_results_rejects_tampered_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _run_with(monkeypatch, tmp_path, _decision_response)
    payload = json.loads(runner.RESULTS_PATH.read_text())
    payload["cases"][0]["finish_reason"] = "length"
    runner.RESULTS_PATH.write_bytes(runner.stable_json_bytes(payload))

    with pytest.raises(ValueError):
        runner.check_results()


# Classification.


def test_twenty_structural_successes_keep_64() -> None:
    assert runner.classify([_record() for _ in range(20)]) == runner.KEEP_64


def test_wrong_semantics_alone_still_keep_64() -> None:
    records = [_record(observed_decision="NOT_REQUESTED") for _ in range(20)]
    assert runner.classify(records) == runner.KEEP_64


@pytest.mark.parametrize(
    "evidence",
    [
        {"finish_reason": "length", "observed_decision": None,
         "failure_category": "zero_structured_calls", "structured_tool_call_count": 0},
        {"failure_category": "provider_error", "observed_decision": None,
         "provider_error_mentions_token_limit": True, "structured_tool_call_count": 0},
        {"finish_reason": "length"},
    ],
)
def test_token_exhaustion_requires_budget_amendment(evidence: dict[str, Any]) -> None:
    records = [_record() for _ in range(19)] + [_record(**evidence)]
    assert runner.classify(records) == runner.TOKEN_BUDGET_AMENDMENT_REQUIRED


@pytest.mark.parametrize(
    "evidence",
    [
        {"failure_category": "zero_structured_calls", "finish_reason": "stop",
         "structured_tool_call_count": 0, "observed_decision": None},
        {"failure_category": "provider_error", "provider_error_code": "tool_use_failed",
         "observed_decision": None, "structured_tool_call_count": 0, "finish_reason": None},
    ],
)
def test_text_only_or_tool_use_failure_is_structured_output_issue(
    evidence: dict[str, Any],
) -> None:
    records = [_record() for _ in range(19)] + [_record(**evidence)]
    assert runner.classify(records) == runner.STRUCTURED_OUTPUT_POLICY_ISSUE


@pytest.mark.parametrize("category", ["timeout", "provider_error"])
def test_provider_only_failure_is_inconclusive(category: str) -> None:
    records = [_record() for _ in range(19)] + [
        _record(failure_category=category, observed_decision=None,
                structured_tool_call_count=0, finish_reason=None)
    ]
    assert runner.classify(records) == runner.INCONCLUSIVE_PROVIDER_FAILURE


# End-to-end runner behavior with fake providers.


def test_run_uses_one_call_per_case_without_retry_and_no_raw_text(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cases_artifact: dict[str, Any]
) -> None:
    aggregate, inner = _run_with(monkeypatch, tmp_path, _decision_response)

    assert inner.calls == 20
    assert aggregate["total_provider_calls"] == 20
    assert aggregate["retries"] == 0
    assert aggregate["structural_success_count"] == 20
    content = runner.RESULTS_PATH.read_text()
    for case in cases_artifact["cases"]:
        assert case["customer_utterance"] not in content
    payload = json.loads(content)
    assert payload["governance"]["development_only"] is True
    assert payload["governance"]["final_acceptance_evidence"] is False
    assert payload["governance"]["fresh_fallback_evaluation_started"] is False
    assert payload["governance"]["step29i_authorized"] is False
    assert payload["governance"]["final_holdout_accessed"] is False


def test_run_text_only_responses_report_structured_output_issue(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def text_only(_messages) -> LLMResponse:
        return LLMResponse(content="EXPLICIT", model="m", finish_reason="stop")

    aggregate, inner = _run_with(monkeypatch, tmp_path, text_only)

    assert aggregate["result_status"] == runner.STRUCTURED_OUTPUT_POLICY_ISSUE
    assert inner.calls == 20


def test_run_length_truncation_reports_budget_amendment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def truncated(_messages) -> LLMResponse:
        return LLMResponse(content="", model="m", finish_reason="length")

    aggregate, _ = _run_with(monkeypatch, tmp_path, truncated)

    assert aggregate["result_status"] == runner.TOKEN_BUDGET_AMENDMENT_REQUIRED


def test_run_transport_failure_is_inconclusive_without_retry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    aggregate, inner = _run_with(monkeypatch, tmp_path, LLMProviderError("connection reset"))

    assert aggregate["result_status"] == runner.INCONCLUSIVE_PROVIDER_FAILURE
    assert inner.calls == 20


def test_provider_error_details_capture_code_without_message_text() -> None:
    class FakeBadRequest(Exception):
        status_code = 400

    cause = FakeBadRequest("Error code: 400")
    cause.body = {"error": {"code": "tool_use_failed", "message": "Failed to call a function."}}
    error = LLMProviderError("Groq LLM request failed")
    error.__cause__ = cause
    details = runner.describe_provider_error(error)

    assert details == {
        "provider_error_type": "FakeBadRequest",
        "provider_error_code": "tool_use_failed",
        "provider_error_status": 400,
        "provider_error_mentions_token_limit": False,
    }


def test_results_are_create_once(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _run_with(monkeypatch, tmp_path, _decision_response)

    with pytest.raises(FileExistsError):
        runner.run(settings=_settings(), inner_provider_override=ScriptedInnerProvider(None))
