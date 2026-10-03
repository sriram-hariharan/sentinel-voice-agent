from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import pytest

from backend.app.agent.protected_action_verifier import (
    DECISION_TOOL_NAME,
    VERIFIER_MAX_COMPLETION_TOKENS,
    VERIFIER_TIMEOUT_SECONDS,
)
from backend.app.config.settings import Settings
from backend.app.providers.groq_llm import GroqLLMProvider
from backend.app.providers.llm import LLMProvider, LLMResponse, LLMToolCall, LLMUsage
from scripts import run_v2c6_routing_fallback_verifier_budget_256_validation as runner

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / runner.CONTRACT_RELATIVE_PATH
CASES_PATH = ROOT / runner.CASES_RELATIVE_PATH
HISTORICAL_RESULTS_PATH = (
    ROOT
    / "data/evals/v2/ml/v2c6_routing_fallback_verifier_budget_validation_results.json"
)
HISTORICAL_MANIFEST_PATH = ROOT / (
    "data/evals/v2/ml/"
    "v2c6_routing_fallback_verifier_budget_validation_results.manifest.json"
)


def _settings() -> Settings:
    return Settings(groq_api_key="test-key", llm_model="openai/gpt-oss-20b")


def _record(**overrides: Any) -> dict[str, Any]:
    record = {
        "case_id": "case",
        "proposed_action": "freeze_card",
        "expected_decision": "EXPLICIT_CURRENT_ACTION",
        "observed_decision": "EXPLICIT_CURRENT_ACTION",
        "failure_category": None,
        "provider": "groq",
        "model": "openai/gpt-oss-20b",
        "finish_reason": "tool_calls",
        "provider_call_count": 1,
        "structured_tool_call_count": 1,
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "total_tokens": 120,
        "provider_call_latency_ms": 1.0,
        "verifier_wall_latency_ms": 1.2,
        "provider_error_type": None,
        "provider_error_code": None,
        "provider_error_status": None,
        "token_limit_evidence": False,
    }
    record.update(overrides)
    return record


class FakeProvider:
    provider = "groq"
    model = "openai/gpt-oss-20b"

    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, *, messages, tools=None) -> LLMResponse:
        self.calls += 1
        return LLMResponse(
            tool_calls=[
                LLMToolCall(
                    id=f"call-{self.calls}",
                    name=DECISION_TOOL_NAME,
                    arguments={"decision": "EXPLICIT_CURRENT_ACTION"},
                )
            ],
            model=self.model,
            finish_reason="tool_calls",
            usage=LLMUsage(
                prompt_tokens=100,
                completion_tokens=20,
                total_tokens=120,
            ),
        )


def _patch_outputs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(runner, "RESULTS_PATH", tmp_path / "results.json")
    monkeypatch.setattr(runner, "MANIFEST_PATH", tmp_path / "results.manifest.json")


def test_contract_binds_exact_consumed_sources_and_amended_runtime() -> None:
    contract_bytes = CONTRACT_PATH.read_bytes()
    contract = json.loads(contract_bytes)
    assert hashlib.sha256(contract_bytes).hexdigest() == runner.EXPECTED_CONTRACT_SHA256
    assert contract["development_only"] is True
    assert contract["reuses_consumed_development_cases"] is True
    assert contract["fresh_acceptance_evidence"] is False
    assert contract["final_acceptance_evidence"] is False
    assert contract["fresh_evaluation_started"] is False
    configuration = contract["frozen_run_configuration"]
    assert configuration == {
        "alternate_budget_fallback": False,
        "application_retries": 0,
        "case_count": 20,
        "execution": "sequential",
        "max_completion_tokens": 256,
        "model": "openai/gpt-oss-20b",
        "outer_timeout_seconds": 2.0,
        "parallel_tool_calls": False,
        "provider_calls_per_case": 1,
        "reasoning_effort": "low",
        "sdk_max_retries": 0,
        "sdk_timeout_seconds": 2.0,
        "temperature": 0,
        "tool_choice": "auto",
    }
    for binding in contract["source_bindings"].values():
        assert "final_holdout" not in binding["path"]
        assert hashlib.sha256((ROOT / binding["path"]).read_bytes()).hexdigest() == (
            binding["sha256"]
        )


def test_exact_same_twenty_consumed_cases_are_reused() -> None:
    contract, cases, cases_bytes = runner.load_and_validate()
    binding = contract["source_bindings"]["budget_development_cases"]
    assert len(cases) == 20
    assert binding["path"] == runner.CASES_RELATIVE_PATH
    assert hashlib.sha256(cases_bytes).hexdigest() == binding["sha256"]
    assert [case["case_id"] for case in cases] == [
        case["case_id"] for case in json.loads(CASES_PATH.read_bytes())["cases"]
    ]


def test_production_configuration_is_exact_and_conversational_protocol_unchanged() -> None:
    verifier = runner.verify_production_construction(_settings())
    provider = verifier._llm
    assert VERIFIER_MAX_COMPLETION_TOKENS == provider.max_completion_tokens == 256
    assert VERIFIER_TIMEOUT_SECONDS == verifier._timeout_seconds == 2.0
    assert provider.reasoning_effort == "low"
    assert provider._client.max_retries == 0
    assert provider._client.timeout == 2.0
    assert provider.model == "openai/gpt-oss-20b"
    assert list(inspect.signature(LLMProvider.generate).parameters) == [
        "self",
        "messages",
        "tools",
    ]


def test_preflight_has_no_provider_call_or_write(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_outputs(monkeypatch, tmp_path)

    async def forbidden(*_args, **_kwargs):
        pytest.fail("preflight called a provider")

    monkeypatch.setattr(GroqLLMProvider, "generate", forbidden)
    report = runner.preflight(_settings)

    assert report["status"] == "READY"
    assert report["production_construction_verified"] is True
    assert report["verifier_max_completion_tokens"] == 256
    assert report["reasoning_effort"] == "low"
    assert report["sdk_max_retries"] == 0
    assert report["sdk_timeout_seconds"] == 2.0
    assert report["tool_choice"] == "auto"
    assert report["fresh_evaluation_started"] is False
    assert report["fresh_evaluation_data_used"] is False
    assert report["provider_calls_performed"] is False
    assert list(tmp_path.iterdir()) == []


def test_status_classification_precedence_and_semantic_independence() -> None:
    successes = [_record() for _ in range(20)]
    assert runner.classify(successes) == runner.KEEP_256
    semantic_errors = [
        _record(observed_decision="NOT_REQUESTED") for _ in range(20)
    ]
    assert runner.classify(semantic_errors) == runner.KEEP_256

    token_failure = successes[:-1] + [
        _record(
            finish_reason="length",
            failure_category="zero_structured_calls",
            observed_decision=None,
            structured_tool_call_count=0,
        )
    ]
    assert runner.classify(token_failure) == runner.TOKEN_BUDGET_STILL_INSUFFICIENT

    structural_failure = successes[:-1] + [
        _record(
            finish_reason="stop",
            failure_category="zero_structured_calls",
            observed_decision=None,
            structured_tool_call_count=0,
        )
    ]
    assert runner.classify(structural_failure) == runner.STRUCTURED_OUTPUT_POLICY_ISSUE

    provider_failure = successes[:-1] + [
        _record(
            finish_reason=None,
            failure_category="timeout",
            observed_decision=None,
            structured_tool_call_count=0,
        )
    ]
    assert runner.classify(provider_failure) == runner.INCONCLUSIVE_PROVIDER_FAILURE


def test_run_uses_one_call_per_case_and_persists_no_raw_utterance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_outputs(monkeypatch, tmp_path)
    provider = FakeProvider()

    aggregate = runner.run(settings=_settings(), inner_provider_override=provider)

    assert provider.calls == 20
    assert aggregate["result_status"] == runner.KEEP_256
    assert aggregate["total_provider_calls"] == 20
    assert aggregate["retries"] == 0
    assert aggregate["latency_ms"]["provider_call"]["max"] is not None
    assert aggregate["latency_ms"]["verifier_wall"]["max"] is not None
    content = runner.RESULTS_PATH.read_text()
    cases = json.loads(CASES_PATH.read_text())["cases"]
    assert all(case["customer_utterance"] not in content for case in cases)
    payload = json.loads(content)
    assert all(case["provider_call_count"] == 1 for case in payload["cases"])
    assert all("provider_call_latency_ms" in case for case in payload["cases"])
    assert all("verifier_wall_latency_ms" in case for case in payload["cases"])


def test_check_results_has_no_provider_call_or_write(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_outputs(monkeypatch, tmp_path)
    runner.run(settings=_settings(), inner_provider_override=FakeProvider())
    before = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in tmp_path.iterdir()
    }

    async def forbidden(*_args, **_kwargs):
        pytest.fail("check-results called a provider")

    monkeypatch.setattr(GroqLLMProvider, "generate", forbidden)
    report = runner.check_results()
    after = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in tmp_path.iterdir()
    }
    assert report["results_valid"] is True
    assert report["provider_calls_performed"] is False
    assert before == after


def test_historical_results_remain_immutable_and_step29i_blocked() -> None:
    assert hashlib.sha256(HISTORICAL_RESULTS_PATH.read_bytes()).hexdigest() == (
        "fcc21b6faf947d83b4f1024054dc823299fa0b29318290207f8dbce7a675336b"
    )
    assert hashlib.sha256(HISTORICAL_MANIFEST_PATH.read_bytes()).hexdigest() == (
        "2da6bf3abee0e1c8e802a8cffb5a3663dab46ae93b2a0bb513cafa589bbe7c71"
    )
    contract = json.loads(CONTRACT_PATH.read_text())
    assert contract["step29i_policy"] == {"authorized": False, "blocked": True}


def test_prohibited_holdout_is_rejected_without_access() -> None:
    with pytest.raises(PermissionError):
        runner.read_bytes(ROOT / "data/evals/v2/ml/v2c5_final_holdout.json")
