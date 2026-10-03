from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
ML_PATH = ROOT / "data/evals/v2/ml"
CONTRACT_PATH = (
    ML_PATH / "v2c6_routing_fallback_verifier_budget_256_validation_contract.json"
)
CASES_PATH = ML_PATH / "v2c6_routing_fallback_verifier_budget_development_cases.json"
RESULTS_PATH = (
    ML_PATH / "v2c6_routing_fallback_verifier_budget_256_validation_results.json"
)
MANIFEST_PATH = ML_PATH / (
    "v2c6_routing_fallback_verifier_budget_256_validation_results.manifest.json"
)

CONTRACT_SHA256 = "7b6c2e937218044978a99bf550c4b69d3ac846ab87fad60d040783a1bc8dfc58"
CASES_SHA256 = "46c0da1502767a6999f0e2fab8fa3c3ae56ba08d573ba1a1a78cac168de19343"
RESULTS_SHA256 = "0f2efbc5004f63c705e5290dc00d2291fe93a5cc292e831394f9a1e33e67dbc7"
MANIFEST_SHA256 = "db2117e62ae732d1ea837c17c9a84f6dca4eef949ece89c79ced58f447cc79fa"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_256_result_artifacts_and_lineage_are_hash_pinned() -> None:
    assert _sha256(CONTRACT_PATH) == CONTRACT_SHA256
    assert _sha256(CASES_PATH) == CASES_SHA256
    assert _sha256(RESULTS_PATH) == RESULTS_SHA256
    assert _sha256(MANIFEST_PATH) == MANIFEST_SHA256

    results = _load(RESULTS_PATH)
    manifest = _load(MANIFEST_PATH)
    assert results["contract"]["sha256"] == manifest["contract_sha256"] == (
        CONTRACT_SHA256
    )
    assert results["development_cases"]["sha256"] == (
        manifest["development_cases_sha256"]
    ) == CASES_SHA256
    assert manifest["results"]["sha256"] == RESULTS_SHA256
    assert manifest["result_status"] == "KEEP_256"


def test_256_observed_result_is_frozen_exactly() -> None:
    results = _load(RESULTS_PATH)
    aggregate = results["aggregate"]
    assert aggregate["result_status"] == "KEEP_256"
    assert aggregate["case_count"] == 20
    assert aggregate["structural_success_count"] == 20
    assert aggregate["failure_counts"] == {}
    assert aggregate["finish_reason_counts"] == {"tool_calls": 20}
    assert aggregate["total_provider_calls"] == 20
    assert aggregate["retries"] == 0
    assert aggregate["tokens"]["completion"] == {
        "count": 20,
        "max": 86,
        "median": 73,
        "min": 49,
        "sum": 1434,
    }
    assert aggregate["latency_ms"]["provider_call"] == {
        "max": 598.43,
        "p50": 177.675,
        "p90": 315.795,
        "p95": 333.607,
    }
    assert aggregate["latency_ms"]["verifier_wall"] == {
        "max": 598.709,
        "p50": 177.776,
        "p90": 315.972,
        "p95": 333.765,
    }
    assert aggregate["semantic_diagnostics"] == {
        "boundary_exact_correct_of_10": 8,
        "boundary_safe_non_explicit_of_10": 10,
        "decision_confusion_counts": {
            "AMBIGUOUS_OR_INFORMATIONAL->AMBIGUOUS_OR_INFORMATIONAL": 5,
            "AMBIGUOUS_OR_INFORMATIONAL->NOT_REQUESTED": 2,
            "EXPLICIT_CURRENT_ACTION->EXPLICIT_CURRENT_ACTION": 10,
            "NOT_REQUESTED->NOT_REQUESTED": 3,
        },
        "exact_correct_of_20": 18,
        "explicit_correct_of_10": 10,
        "role": "diagnostic_only_not_acceptance",
    }


def test_256_configuration_is_retained_without_acceptance_overclaim() -> None:
    results = _load(RESULTS_PATH)
    assert results["run_configuration"] == {
        "application_retries": 0,
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
    governance = results["governance"]
    assert governance["development_only"] is True
    assert governance["reuses_consumed_development_cases"] is True
    assert governance["fresh_acceptance_evidence"] is False
    assert governance["final_acceptance_evidence"] is False
    assert governance["fresh_evaluation_started"] is False
    assert governance["step29i_authorized"] is False
    assert governance["tool_choice_changed"] is False
    assert governance["alternate_budget_fallback"] is False
    assert governance["final_holdout_accessed"] is False
    assert len(results["cases"]) == 20
    assert all(case["provider_call_count"] == 1 for case in results["cases"])
    assert "customer_utterance" not in json.dumps(results)
