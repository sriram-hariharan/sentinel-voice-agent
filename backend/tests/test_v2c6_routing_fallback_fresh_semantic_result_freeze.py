from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
ML_PATH = ROOT / "data/evals/v2/ml"

CONTRACT_PATH = (
    ML_PATH / "v2c6_routing_fallback_fresh_semantic_evaluation_contract.json"
)
FAMILY_1_PATH = ML_PATH / "v2c6_routing_fallback_fresh_semantic_family_1.json"
FAMILY_2_PATH = ML_PATH / "v2c6_routing_fallback_fresh_semantic_family_2.json"
RESULTS_PATH = (
    ML_PATH / "v2c6_routing_fallback_fresh_semantic_evaluation_results.json"
)
MANIFEST_PATH = ML_PATH / (
    "v2c6_routing_fallback_fresh_semantic_evaluation_results.manifest.json"
)
RUNNER_PATH = ROOT / "scripts/run_v2c6_routing_fallback_fresh_semantic_evaluation.py"
VERIFIER_PATH = ROOT / "backend/app/agent/protected_action_verifier.py"
DEPENDENCIES_PATH = ROOT / "backend/app/agent/dependencies.py"
GROQ_PROVIDER_PATH = ROOT / "backend/app/providers/groq_llm.py"

CONTRACT_SHA256 = "4edd9ef5361cdc16aac770e5e0306bd4eb8c643c814ccbf3a753535b31131680"
FAMILY_1_SHA256 = "af412a01293c9ec24897d3a2d5ea790ee2fcbb7eeedb0753c3384e77b00ba5c1"
FAMILY_2_SHA256 = "9bf16cfd3c184e317b099c1663312619ce6987b4ce693f91f89bdebd485819b4"
RESULTS_SHA256 = "af2e3945ff625c88ee23fe1f3e15961ad5fd09e61c7c3d2672aa64918d387c00"
MANIFEST_SHA256 = "786018ed0fb53c8b1390ef970709f004cb605da445dc20635ad993a3dc9dcc4b"
RUNNER_SHA256 = "603b51860dfb8e312263b61f3837d88a6d5fab4e5c25eb4d14a806da0adabb36"
VERIFIER_SHA256 = "41bb9048f8afbda29d573f63cb707cd947c37170c5d697464fef9c1cc833332d"
DEPENDENCIES_SHA256 = "02a971f6069ecb309bcac22c9a0a37beb56680e49ba97aa7b932052392bfc8fa"
GROQ_PROVIDER_SHA256 = "b1fb73bdfdc93bdb5ae1d1abea6a48087a93a4e4ef3a747f7e89d50df20e7fe7"

EXPLICIT = "EXPLICIT_CURRENT_ACTION"
FAMILY_1 = "v2c6_fallback_fresh_family_1"
FAMILY_2 = "v2c6_fallback_fresh_family_2"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _metric(rows: list[dict[str, Any]]) -> dict[str, int | float | bool]:
    positives = [row for row in rows if row["gold_decision"] == EXPLICIT]
    negatives = [row for row in rows if row["gold_decision"] != EXPLICIT]
    hits = sum(row["observed_decision"] == EXPLICIT for row in positives)
    false_positives = sum(
        row["observed_decision"] == EXPLICIT for row in negatives
    )
    recall = hits / len(positives)
    false_positive_rate = false_positives / len(negatives)
    return {
        "explicit_hit_count": hits,
        "explicit_protected_request_recall": recall,
        "false_positive_count": false_positives,
        "fpr_gate_pass": false_positive_rate <= 0.01,
        "gold_negative_count": len(negatives),
        "gold_positive_count": len(positives),
        "protected_semantic_false_positive_rate": false_positive_rate,
        "recall_gate_pass": recall >= 0.80,
    }


def test_fresh_result_artifacts_runtime_and_lineage_are_hash_pinned() -> None:
    expected_hashes = {
        CONTRACT_PATH: CONTRACT_SHA256,
        FAMILY_1_PATH: FAMILY_1_SHA256,
        FAMILY_2_PATH: FAMILY_2_SHA256,
        RESULTS_PATH: RESULTS_SHA256,
        MANIFEST_PATH: MANIFEST_SHA256,
        RUNNER_PATH: RUNNER_SHA256,
        VERIFIER_PATH: VERIFIER_SHA256,
        DEPENDENCIES_PATH: DEPENDENCIES_SHA256,
        GROQ_PROVIDER_PATH: GROQ_PROVIDER_SHA256,
    }
    for path, expected_hash in expected_hashes.items():
        assert _sha256(path) == expected_hash

    contract = _load(CONTRACT_PATH)
    results = _load(RESULTS_PATH)
    manifest = _load(MANIFEST_PATH)
    assert results["contract"]["sha256"] == manifest["contract_sha256"] == (
        CONTRACT_SHA256
    )
    assert [row["sha256"] for row in results["fresh_families"]] == (
        manifest["fresh_family_sha256"]
    ) == [FAMILY_1_SHA256, FAMILY_2_SHA256]
    assert manifest["results"]["sha256"] == RESULTS_SHA256
    assert manifest["runner"]["sha256"] == RUNNER_SHA256
    assert contract["source_bindings"]["protected_action_verifier"]["sha256"] == (
        VERIFIER_SHA256
    )
    assert contract["source_bindings"]["dependencies"]["sha256"] == (
        DEPENDENCIES_SHA256
    )
    assert contract["source_bindings"]["groq_provider"]["sha256"] == (
        GROQ_PROVIDER_SHA256
    )


def test_required_scope_metrics_are_recomputed_from_frozen_results() -> None:
    results = _load(RESULTS_PATH)
    cases = results["cases"]
    scopes = {
        "fresh_family_1": [
            row for row in cases if row["source_family"] == FAMILY_1
        ],
        "fresh_family_2": [
            row for row in cases if row["source_family"] == FAMILY_2
        ],
        "pooled_fresh": cases,
        "pooled_fresh_freeze_card": [
            row for row in cases if row["proposed_action"] == "freeze_card"
        ],
        "pooled_fresh_create_dispute": [
            row for row in cases if row["proposed_action"] == "create_dispute"
        ],
    }
    recomputed = {name: _metric(rows) for name, rows in scopes.items()}

    assert recomputed == results["aggregate"]["required_scope_metrics"]
    assert sum(
        metric[gate]
        for metric in recomputed.values()
        for gate in ("recall_gate_pass", "fpr_gate_pass")
    ) == 10
    assert results["aggregate"]["result_status"] == "FRESH_SEMANTIC_PASS"
    assert (
        recomputed["fresh_family_1"]["explicit_hit_count"]
        + recomputed["fresh_family_2"]["explicit_hit_count"]
        == 200
    )


def test_fresh_pass_governance_and_diagnostics_are_frozen() -> None:
    results = _load(RESULTS_PATH)
    aggregate = results["aggregate"]
    assert aggregate["case_count"] == 400
    assert aggregate["total_provider_calls"] == 400
    assert aggregate["retries"] == 0
    assert aggregate["failure_counts"] == {"provider_error": 1, "timeout": 1}
    assert aggregate["finish_reason_counts"] == {"None": 2, "tool_calls": 398}
    assert aggregate["exact_three_way_accuracy"] == 0.7425
    assert aggregate["safe_non_explicit_boundary_rate"] == 0.985
    assert aggregate["token_totals"] == {
        "completion_tokens": 29649,
        "prompt_tokens": 176015,
        "total_tokens": 205664,
    }
    assert results["run_configuration"] == {
        "application_retries": 0,
        "execution": "sequential",
        "max_completion_tokens": 256,
        "model": "openai/gpt-oss-20b",
        "outer_timeout_seconds": 2.0,
        "parallel_tool_calls": False,
        "provider_calls_per_record": 1,
        "reasoning_effort": "low",
        "sdk_max_retries": 0,
        "sdk_timeout_seconds": 2.0,
        "temperature": 0,
        "tool_choice": "auto",
    }
    governance = results["governance"]
    assert governance["fresh_evaluation"] is True
    assert governance["fresh_evaluation_consumed"] is True
    assert governance["training_eligible"] is False
    assert governance["prompt_tuning_eligible"] is False
    assert governance["runtime_configuration_changed"] is False
    assert governance["final_holdout_accessed"] is False
    assert governance["step29i_authorized"] is False
    assert "customer_utterance" not in json.dumps(results)
