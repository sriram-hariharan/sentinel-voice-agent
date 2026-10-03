from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from backend.app.agent.protected_action_verifier import DECISION_TOOL_NAME
from backend.app.config.settings import Settings
from backend.app.providers.groq_llm import GroqLLMProvider
from backend.app.providers.llm import LLMResponse, LLMToolCall, LLMUsage
from scripts import run_v2c6_routing_fallback_fresh_semantic_evaluation as runner

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / runner.CONTRACT_RELATIVE_PATH
FAMILY_PATHS = tuple(ROOT / path for path in runner.FAMILY_RELATIVE_PATHS)


def _settings() -> Settings:
    return Settings(groq_api_key="test-key", llm_model="openai/gpt-oss-20b")


def _fresh_cases() -> list[dict[str, Any]]:
    _, cases, _ = runner.load_and_validate()
    return cases


def _result_records() -> list[dict[str, Any]]:
    records=[]
    for case in _fresh_cases():
        records.append({
            "case_id":case["case_id"],
            "source_family":case["source_family"],
            "proposed_action":case["proposed_action"],
            "case_kind":case["case_kind"],
            "boundary_subtype":case["boundary_subtype"],
            "gold_decision":case["gold_decision"],
            "observed_decision":case["gold_decision"],
            "failure_category":None,
            "provider":"groq",
            "model":"openai/gpt-oss-20b",
            "finish_reason":"tool_calls",
            "provider_call_count":1,
            "structured_tool_call_count":1,
            "prompt_tokens":100,
            "completion_tokens":50,
            "total_tokens":150,
            "provider_call_latency_ms":10.0,
            "verifier_wall_latency_ms":11.0,
            "provider_error_type":None,
            "provider_error_code":None,
            "provider_error_status":None,
        })
    return records


class FakeProvider:
    provider="groq"
    model="openai/gpt-oss-20b"
    def __init__(self)->None:
        self.calls=0
    async def generate(self,*,messages,tools=None)->LLMResponse:
        self.calls+=1
        return LLMResponse(
            tool_calls=[LLMToolCall(
                id=f"call-{self.calls}",
                name=DECISION_TOOL_NAME,
                arguments={"decision":"AMBIGUOUS_OR_INFORMATIONAL"},
            )],
            model=self.model,
            finish_reason="tool_calls",
            usage=LLMUsage(prompt_tokens=100,completion_tokens=50,total_tokens=150),
        )


def _patch_outputs(monkeypatch:pytest.MonkeyPatch,tmp_path:Path)->None:
    monkeypatch.setattr(runner,"RESULTS_PATH",tmp_path/"results.json")
    monkeypatch.setattr(runner,"MANIFEST_PATH",tmp_path/"results.manifest.json")


def test_fresh_data_exact_composition_labels_and_provenance()->None:
    contract,cases,_=runner.load_and_validate()
    assert len(cases)==400
    assert Counter(r["source_family"] for r in cases)==Counter({family:200 for family in runner.EXPECTED_FAMILIES})
    assert Counter((r["source_family"],r["proposed_action"],r["case_kind"]) for r in cases)==Counter({
        (family,action,kind):50
        for family in runner.EXPECTED_FAMILIES
        for action in runner.ACTIONS
        for kind in ("explicit","boundary")
    })
    assert Counter((r["source_family"],r["proposed_action"],r["boundary_subtype"]) for r in cases if r["case_kind"]=="boundary")==Counter({
        (family,action,subtype):10
        for family in runner.EXPECTED_FAMILIES
        for action in runner.ACTIONS
        for subtype in runner.BOUNDARY_SUBTYPES
    })
    assert all(r["gold_decision"]=="EXPLICIT_CURRENT_ACTION" for r in cases if r["case_kind"]=="explicit")
    assert all(r["gold_decision"]!="EXPLICIT_CURRENT_ACTION" for r in cases if r["case_kind"]=="boundary")
    assert len({r["case_id"] for r in cases})==400
    assert contract["training_eligible"] is False
    assert contract["prompt_tuning_eligible"] is False
    for path in FAMILY_PATHS:
        payload=json.loads(path.read_text())
        assert payload["training_eligible"] is False
        assert payload["authoring_provenance"]["human_review_performed"] is False
        assert payload["authoring_provenance"]["authoring_method"]=="controlled_llm_assisted"


def test_fresh_data_has_no_normalized_duplicates_or_governed_overlap()->None:
    _,cases,_=runner.load_and_validate()
    report=runner.validate_no_duplicates_or_leakage(cases)
    assert report=={
        "normalized_duplicate_count":0,
        "cross_family_normalized_duplicate_count":0,
        "cross_action_conflicting_collision_count":0,
        "consumed_development_overlap_count":0,
        "r3_fresh_hash_overlap_count":0,
    }


def test_contract_freezes_runtime_gates_scopes_and_no_tuning()->None:
    contract=json.loads(CONTRACT_PATH.read_text())
    assert hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest()==runner.EXPECTED_CONTRACT_SHA256
    assert contract["frozen_runtime_configuration"]=={
        "application_retries":0,
        "max_completion_tokens":256,
        "model":"openai/gpt-oss-20b",
        "outer_timeout_seconds":2.0,
        "parallel_tool_calls":False,
        "provider_calls_per_record":1,
        "reasoning_effort":"low",
        "sdk_max_retries":0,
        "sdk_timeout_seconds":2.0,
        "temperature":0,
        "tool_choice":"auto",
    }
    gates=contract["acceptance_gates"]
    assert gates["metrics"]=={
        "explicit_protected_request_recall":{"comparison":"greater_than_or_equal","threshold":0.8},
        "protected_semantic_false_positive_rate":{"comparison":"less_than_or_equal","threshold":0.01},
    }
    assert gates["required_scopes"]==list(runner.REQUIRED_SCOPES)
    assert gates["all_gates_mandatory"] is True
    assert gates["gate_weakening_permitted"] is False
    assert gates["weighted_override_permitted"] is False
    implications=contract["integer_gate_implications"]
    assert implications["fresh_family_1"]=={"maximum_false_positives":1,"minimum_explicit_hits":80,"negative_count":100,"positive_count":100}
    assert implications["pooled_fresh"]=={"maximum_false_positives":2,"minimum_explicit_hits":160,"negative_count":200,"positive_count":200}
    assert implications["pooled_fresh_freeze_card"]=={"maximum_false_positives":1,"minimum_explicit_hits":80,"negative_count":100,"positive_count":100}
    assert contract["governance"]["no_fresh_data_tuning"] is True
    assert contract["final_holdout_policy"]["access_permitted"] is False
    assert contract["governance"]["step29i_authorized"] is False


def test_preflight_makes_no_provider_call_or_write(monkeypatch:pytest.MonkeyPatch,tmp_path:Path)->None:
    _patch_outputs(monkeypatch,tmp_path)
    async def forbidden(*_args,**_kwargs):
        pytest.fail("preflight called provider")
    monkeypatch.setattr(GroqLLMProvider,"generate",forbidden)
    report=runner.preflight(_settings)
    assert report["status"]=="READY"
    assert report["case_count"]==400
    assert report["provider_calls_performed"] is False
    assert report["files_written"] is False
    assert report["fresh_evaluation_started"] is False
    assert list(tmp_path.iterdir())==[]


def test_pass_requires_every_gate_on_every_scope()->None:
    aggregate=runner.aggregate(_result_records())
    assert aggregate["result_status"]==runner.FRESH_SEMANTIC_PASS
    assert all(row["recall_gate_pass"] and row["fpr_gate_pass"] for row in aggregate["required_scope_metrics"].values())


def test_any_family_recall_gate_failure_fails()->None:
    records=_result_records()
    candidates=[r for r in records if r["source_family"]==runner.EXPECTED_FAMILIES[0] and r["gold_decision"]=="EXPLICIT_CURRENT_ACTION"]
    for record in candidates[:21]: record["observed_decision"]="AMBIGUOUS_OR_INFORMATIONAL"
    aggregate=runner.aggregate(records)
    assert aggregate["required_scope_metrics"]["fresh_family_1"]["recall_gate_pass"] is False
    assert aggregate["result_status"]==runner.FRESH_SEMANTIC_FAIL


def test_any_family_fpr_gate_failure_fails()->None:
    records=_result_records()
    candidates=[r for r in records if r["source_family"]==runner.EXPECTED_FAMILIES[0] and r["gold_decision"]!="EXPLICIT_CURRENT_ACTION"]
    for record in candidates[:2]: record["observed_decision"]="EXPLICIT_CURRENT_ACTION"
    aggregate=runner.aggregate(records)
    assert aggregate["required_scope_metrics"]["fresh_family_1"]["fpr_gate_pass"] is False
    assert aggregate["result_status"]==runner.FRESH_SEMANTIC_FAIL


def test_per_action_recall_gate_is_independently_mandatory()->None:
    records=_result_records()
    candidates=[r for r in records if r["proposed_action"]=="freeze_card" and r["gold_decision"]=="EXPLICIT_CURRENT_ACTION"]
    first=[r for r in candidates if r["source_family"]==runner.EXPECTED_FAMILIES[0]][:10]
    second=[r for r in candidates if r["source_family"]==runner.EXPECTED_FAMILIES[1]][:11]
    for record in [*first,*second]: record["observed_decision"]="NOT_REQUESTED"
    aggregate=runner.aggregate(records)
    assert aggregate["required_scope_metrics"]["fresh_family_1"]["recall_gate_pass"] is True
    assert aggregate["required_scope_metrics"]["fresh_family_2"]["recall_gate_pass"] is True
    assert aggregate["required_scope_metrics"]["pooled_fresh_freeze_card"]["recall_gate_pass"] is False
    assert aggregate["result_status"]==runner.FRESH_SEMANTIC_FAIL


def test_per_action_fpr_gate_is_independently_mandatory()->None:
    records=_result_records()
    for family in runner.EXPECTED_FAMILIES:
        record=next(r for r in records if r["source_family"]==family and r["proposed_action"]=="create_dispute" and r["gold_decision"]!="EXPLICIT_CURRENT_ACTION")
        record["observed_decision"]="EXPLICIT_CURRENT_ACTION"
    aggregate=runner.aggregate(records)
    assert aggregate["required_scope_metrics"]["fresh_family_1"]["fpr_gate_pass"] is True
    assert aggregate["required_scope_metrics"]["fresh_family_2"]["fpr_gate_pass"] is True
    assert aggregate["required_scope_metrics"]["pooled_fresh_create_dispute"]["fpr_gate_pass"] is False
    assert aggregate["result_status"]==runner.FRESH_SEMANTIC_FAIL


def test_failures_count_non_explicit_and_are_reported()->None:
    records=_result_records()
    target=next(r for r in records if r["gold_decision"]=="EXPLICIT_CURRENT_ACTION")
    target["observed_decision"]=None
    target["failure_category"]="timeout"
    aggregate=runner.aggregate(records)
    assert aggregate["failure_counts"]=={"timeout":1}
    assert aggregate["required_scope_metrics"]["pooled_fresh"]["explicit_hit_count"]==199


def test_non_explicit_label_swap_is_not_false_positive()->None:
    records=_result_records()
    target=next(r for r in records if r["gold_decision"]=="AMBIGUOUS_OR_INFORMATIONAL")
    target["observed_decision"]="NOT_REQUESTED"
    aggregate=runner.aggregate(records)
    assert aggregate["required_scope_metrics"]["pooled_fresh"]["false_positive_count"]==0
    assert aggregate["safe_non_explicit_boundary_rate"]==1.0


def test_run_is_one_call_per_case_no_retry_and_no_raw_text(monkeypatch:pytest.MonkeyPatch,tmp_path:Path)->None:
    _patch_outputs(monkeypatch,tmp_path)
    provider=FakeProvider()
    aggregate=runner.run(settings=_settings(),inner_provider_override=provider)
    assert provider.calls==400
    assert aggregate["total_provider_calls"]==400
    assert aggregate["retries"]==0
    payload=json.loads(runner.RESULTS_PATH.read_text())
    assert all(r["provider_call_count"]==1 for r in payload["cases"])
    assert "customer_utterance" not in runner.RESULTS_PATH.read_text()


def test_check_results_makes_no_provider_call_or_write(monkeypatch:pytest.MonkeyPatch,tmp_path:Path)->None:
    _patch_outputs(monkeypatch,tmp_path)
    runner.run(settings=_settings(),inner_provider_override=FakeProvider())
    before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.iterdir()}
    async def forbidden(*_args,**_kwargs):
        pytest.fail("check-results called provider")
    monkeypatch.setattr(GroqLLMProvider,"generate",forbidden)
    report=runner.check_results()
    after={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.iterdir()}
    assert report["results_valid"] is True
    assert report["provider_calls_performed"] is False
    assert before==after


def test_historical_64_and_256_artifacts_unchanged()->None:
    expected={
        "v2c6_routing_fallback_verifier_budget_validation_results.json":"fcc21b6faf947d83b4f1024054dc823299fa0b29318290207f8dbce7a675336b",
        "v2c6_routing_fallback_verifier_budget_validation_results.manifest.json":"2da6bf3abee0e1c8e802a8cffb5a3663dab46ae93b2a0bb513cafa589bbe7c71",
        "v2c6_routing_fallback_verifier_budget_256_validation_results.json":"0f2efbc5004f63c705e5290dc00d2291fe93a5cc292e831394f9a1e33e67dbc7",
        "v2c6_routing_fallback_verifier_budget_256_validation_results.manifest.json":"db2117e62ae732d1ea837c17c9a84f6dca4eef949ece89c79ced58f447cc79fa",
    }
    for name,digest in expected.items():
        assert hashlib.sha256((ROOT/"data/evals/v2/ml"/name).read_bytes()).hexdigest()==digest


def test_prohibited_holdout_access_is_rejected()->None:
    with pytest.raises(PermissionError):
        runner.read_bytes(ROOT/"data/evals/v2/ml/v2c5_final_holdout.json")
