from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CLOSURE_PATH = ROOT / "data/evals/v2/ml/v2c6_final_closure_decision.json"
CLOSURE_SHA256 = "7945b8aa5b70df0976bbc53320ca9705683cf00853175e568289482e3bf50adb"

FROZEN_CONFIGURATION = {
    "model": "openai/gpt-oss-20b",
    "max_completion_tokens": 256,
    "reasoning_effort": "low",
    "sdk_max_retries": 0,
    "sdk_timeout_seconds": 2.0,
    "outer_timeout_seconds": 2.0,
    "application_retries": 0,
    "tool_choice": "auto",
    "temperature": 0,
    "parallel_tool_calls": False,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_v2c6_final_closure_decision_and_evidence_are_hash_pinned() -> None:
    assert _sha256(CLOSURE_PATH) == CLOSURE_SHA256
    closure = _load(CLOSURE_PATH)
    assert closure["schema_version"] == "v2c6-final-closure-decision.v1"
    assert closure["phase"] == "V2-C6"
    assert closure["status"] == "CLOSED"

    for binding in closure["evidence_bindings"].values():
        assert _sha256(ROOT / binding["path"]) == binding["sha256"]


def test_classifier_path_closes_without_a_selected_candidate() -> None:
    closure = _load(CLOSURE_PATH)
    classifier = closure["classifier_path"]
    assert classifier == {
        "selection_status": "NO_ACCEPTABLE_CANDIDATE",
        "selected_candidate": None,
        "step29i_authorized": False,
        "runtime_classifier_promoted": False,
    }

    selection_path = ROOT / closure["evidence_bindings"][
        "r3_classifier_selection_result"
    ]["path"]
    selection = _load(selection_path)["selection"]
    assert selection["selection_status"] == "NO_ACCEPTABLE_CANDIDATE"
    assert selection["selected_candidate_id"] is None
    assert selection["step29i_authorized"] is False


def test_fallback_path_and_verifier_configuration_are_exact() -> None:
    closure = _load(CLOSURE_PATH)
    assert closure["fallback_path"] == {
        "architecture": "CLARIFICATION_GATED_STRUCTURED_LLM_PROTECTED_ROUTING",
        "development_verifier_status": "KEEP_256",
        "fresh_semantic_status": "FRESH_SEMANTIC_PASS",
        "deterministic_runtime_status": "RUNTIME_SAFETY_PASS",
        "runtime_fallback_accepted": True,
    }
    assert closure["frozen_verifier_configuration"] == FROZEN_CONFIGURATION

    bindings = closure["evidence_bindings"]
    decision = _load(ROOT / bindings["routing_fallback_decision"]["path"])
    development = _load(ROOT / bindings["verifier_budget_256_result"]["path"])
    fresh = _load(ROOT / bindings["fresh_semantic_result"]["path"])
    runtime = _load(ROOT / bindings["deterministic_runtime_result"]["path"])
    assert decision["decision_id"] == (
        "CLARIFICATION_GATED_STRUCTURED_LLM_PROTECTED_ROUTING"
    )
    assert development["aggregate"]["result_status"] == "KEEP_256"
    assert fresh["aggregate"]["result_status"] == "FRESH_SEMANTIC_PASS"
    assert runtime["final_status"] == "RUNTIME_SAFETY_PASS"
    assert development["run_configuration"] == {
        **FROZEN_CONFIGURATION,
        "execution": "sequential",
        "provider_calls_per_case": 1,
    }
    assert fresh["run_configuration"] == {
        **FROZEN_CONFIGURATION,
        "execution": "sequential",
        "provider_calls_per_record": 1,
    }


def test_closure_governance_and_v2d_handoff_fail_closed() -> None:
    closure = _load(CLOSURE_PATH)
    governance = closure["governance"]
    assert governance["semantic_tuning_authorized"] is False
    assert governance["fresh_evidence_consumed"] is True
    assert governance["fresh_evidence_tuning_permitted"] is False
    assert governance["final_holdout_accessed"] is False
    assert governance["prohibited_v2c5_holdout_remains_prohibited"] is True
    assert governance["step29i_authorized"] is False
    assert governance["backend_runtime_change_authorized"] is False
    assert governance["sentinelvoice_product_production_ready_claimed"] is False

    handoff = closure["v2d_handoff"]
    assert handoff["entry_conditions_satisfied"] is True
    assert handoff["classifier_step29i_included"] is False
    assert handoff["scope_defined_in_current_readme"] is False
    assert handoff["implementation_started"] is False
    assert handoff["separate_frozen_scope_contract_required_before_implementation"] is (
        True
    )
    assert closure["next_phase"] == "V2-D"


def test_bound_backend_runtime_sources_remain_exact() -> None:
    closure = _load(CLOSURE_PATH)
    bindings = closure["evidence_bindings"]
    for name in ("protected_action_verifier", "dependencies", "groq_provider"):
        binding = bindings[name]
        assert _sha256(ROOT / binding["path"]) == binding["sha256"]
