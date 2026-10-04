from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from backend.app.config.settings import Settings
from backend.app.observability.events import TraceEvent, TraceStatus
from scripts import run_v2d_operational_evidence_closeout as closeout


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _correlation(index: int) -> dict[str, str]:
    return {
        "trace_id": f"trace-v2d-{index:020d}",
        "session_id": f"session-v2d-{index // 10}",
        "turn_id": f"turn-v2d-{index:020d}",
    }


def _observations(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "v2d-live-benchmark-observations.v1",
        "plan": {
            "path": closeout.PLAN_RELATIVE_PATH,
            "sha256": closeout.EXPECTED_PLAN_SHA256,
        },
        "environment": {
            "execution_date": "2026-10-03",
            "operating_system": "synthetic-test-os",
            "browser": "synthetic-test-browser",
            "network_context": "synthetic-test-network",
        },
        "providers": {
            "stt": "whisper-large-v3-turbo",
            "primary_llm": "openai/gpt-oss-20b",
            "protected_action_verifier": "openai/gpt-oss-20b",
            "tts": "canopylabs/orpheus-v1-english",
        },
        "governance": {
            "synthetic_data_only": True,
            "raw_text_recorded": False,
            "raw_audio_recorded": False,
            "replacement_run": False,
            "post_result_tuning_performed": False,
        },
        "safety_observation": {
            "cross_customer_disclosures": 0,
            "unauthorized_protected_actions_executed": 0,
        },
        "normal_turns": [
            {
                "scenario_id": row["scenario_id"],
                "correlation": _correlation(index),
                "task_outcome": "PASS",
                "failure_category": "NONE",
            }
            for index, row in enumerate(plan["normal_turns"], start=1)
        ],
        "interruption_trials": [
            {
                "scenario_id": row["scenario_id"],
                "interrupted": _correlation(100 + index * 2),
                "recovery": _correlation(101 + index * 2),
                "task_outcome": "PASS",
                "failure_category": "NONE",
                "interruption_recovery": True,
            }
            for index, row in enumerate(plan["interruption_trials"], start=1)
        ],
    }


def _event(
    name: str,
    correlation: dict[str, str],
    *,
    component: str = "test",
    duration_ms: float | None = 10.0,
    metadata: dict[str, Any] | None = None,
    status: TraceStatus = TraceStatus.COMPLETED,
) -> TraceEvent:
    return TraceEvent(
        event_name=name,
        trace_id=correlation["trace_id"],
        session_id=correlation["session_id"],
        turn_id=correlation["turn_id"],
        component=component,
        status=status,
        duration_ms=duration_ms,
        metadata=metadata or {},
    )


def _core_events(correlation: dict[str, str]) -> list[TraceEvent]:
    return [
        _event(
            "stt.completed",
            correlation,
            metadata={
                "provider": "groq",
                "model": "whisper-large-v3-turbo",
                "audio_seconds": 1.0,
            },
        ),
        _event("voice.backend_turn.completed", correlation),
        _event("agent.turn.completed", correlation),
        _event(
            "tts.completed",
            correlation,
            metadata={
                "provider": "groq",
                "model": "canopylabs/orpheus-v1-english",
                "character_count": 40,
            },
        ),
        _event("voice.speech_end_to_playback_start.completed", correlation),
    ]


def test_scope_plan_and_v2c6_baseline_are_exactly_hash_pinned() -> None:
    assert _sha256(closeout.CONTRACT_PATH) == closeout.EXPECTED_CONTRACT_SHA256
    assert _sha256(closeout.PLAN_PATH) == closeout.EXPECTED_PLAN_SHA256
    assert _sha256(closeout.CLOSURE_PATH) == closeout.EXPECTED_CLOSURE_SHA256
    contract, closure = closeout._validate_contract_and_closure()
    plan = closeout._validate_plan()

    assert contract["title"] == "Operational Evidence and Portfolio Closeout"
    assert closure["status"] == "CLOSED"
    assert closure["classifier_path"]["step29i_authorized"] is False
    assert len(plan["normal_turns"]) == 20
    assert len(plan["interruption_trials"]) == 5
    assert plan["execution_policy"]["replacement_runs_permitted"] is False


def test_release_map_reuses_existing_tests_for_all_required_invariants() -> None:
    closeout._validate_release_evidence_map()

    assert len(closeout.RELEASE_INVARIANT_EVIDENCE) == 12
    assert "classifier_path_disabled" in closeout.RELEASE_INVARIANT_EVIDENCE
    assert "verifier_fail_closed" in closeout.RELEASE_INVARIANT_EVIDENCE
    assert "structured_observability" in closeout.RELEASE_INVARIANT_EVIDENCE


def test_preflight_is_read_only_and_reports_only_configuration_presence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result_path = tmp_path / "result.json"
    manifest_path = tmp_path / "manifest.json"
    release_path = tmp_path / "release.json"
    monkeypatch.setattr(closeout, "RESULT_PATH", result_path)
    monkeypatch.setattr(closeout, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(closeout, "RELEASE_EVIDENCE_PATH", release_path)
    monkeypatch.setattr(closeout, "_backend_app_unchanged", lambda: True)
    monkeypatch.setattr(
        closeout,
        "_policy_index_readiness",
        lambda _settings: {
            "reachable": True,
            "document_count": 7,
            "chunk_count": 21,
            "ready": True,
        },
    )
    settings = Settings(
        groq_api_key="gsk_test_not_real",
        livekit_url="wss://example.invalid",
        livekit_api_key="test-key",
        livekit_api_secret="test-secret",
        db_password="test-db-password",
        demo_pin="1234",
    )

    report = closeout.preflight(settings=settings)

    assert report["status"] == "READY"
    assert all(report["configuration"].values())
    assert report["provider_calls_performed"] is False
    assert report["files_written"] is False
    assert not result_path.exists()
    assert not manifest_path.exists()
    assert not release_path.exists()
    serialized = json.dumps(report)
    assert "gsk_test_not_real" not in serialized
    assert "test-secret" not in serialized
    assert "1234" not in serialized


def test_observations_require_exact_20_plus_5_without_raw_text() -> None:
    plan = closeout._validate_plan()
    observations = _observations(plan)

    normal, interruptions = closeout.validate_observations(observations, plan)

    assert len(normal) == 20
    assert len(interruptions) == 5
    observations["normal_turns"][0]["transcript"] = "must not be retained"
    with pytest.raises(ValueError, match="prohibited observation field"):
        closeout.validate_observations(observations, plan)


def test_preflight_is_blocked_by_recorded_release_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release_path = tmp_path / "release.json"
    release_path.write_text('{"status": "FAIL"}\n', encoding="utf-8")
    monkeypatch.setattr(closeout, "RELEASE_EVIDENCE_PATH", release_path)
    monkeypatch.setattr(closeout, "RESULT_PATH", tmp_path / "result.json")
    monkeypatch.setattr(closeout, "MANIFEST_PATH", tmp_path / "manifest.json")
    monkeypatch.setattr(closeout, "_backend_app_unchanged", lambda: True)
    monkeypatch.setattr(
        closeout,
        "_policy_index_readiness",
        lambda _settings: {
            "reachable": True,
            "document_count": 7,
            "chunk_count": 21,
            "ready": True,
        },
    )
    settings = Settings(
        groq_api_key="gsk_test_not_real",
        livekit_url="wss://example.invalid",
        livekit_api_key="test-key",
        livekit_api_secret="test-secret",
        db_password="test-db-password",
        demo_pin="1234",
    )

    report = closeout.preflight(settings=settings)

    assert report["status"] == "BLOCKED"
    assert report["release_validation_status"] == "FAIL"
    assert report["provider_calls_performed"] is False


def test_observations_reject_missing_or_duplicate_scenarios() -> None:
    plan = closeout._validate_plan()
    observations = _observations(plan)
    observations["normal_turns"][-1]["scenario_id"] = observations[
        "normal_turns"
    ][0]["scenario_id"]

    with pytest.raises(ValueError, match="scenario set"):
        closeout.validate_observations(observations, plan)


def test_protected_turn_requires_confirmation_and_expected_execution() -> None:
    correlation = _correlation(1)
    expected = {
        "scenario_id": "protected",
        "category": "protected_write_confirmation",
        "expected_executed_tools": ["freeze_card"],
        "retrieval_expected": False,
        "verifier_expected": False,
        "confirmation_expectation": "CONFIRMED_AND_EXECUTED",
    }
    observation = {
        "correlation": correlation,
        "task_outcome": "PASS",
        "failure_category": "NONE",
    }
    events = [
        *_core_events(correlation),
        _event("confirmation.accepted", correlation, duration_ms=None),
        _event(
            "tool.execution.completed",
            correlation,
            metadata={"tool_name": "freeze_card"},
        ),
    ]

    result = closeout._normal_turn_result(expected, observation, events)

    assert result["passed"] is True
    assert result["tool_correct"] is True
    assert result["confirmation_compliant"] is True
    assert result["actual_executed_tools"] == ["freeze_card"]


def test_interruption_requires_measured_stop_and_safe_recovery() -> None:
    interrupted = _correlation(1)
    recovery = _correlation(2)
    expected = {
        "scenario_id": "interruption",
        "category": "private_read_correction",
        "expected_recovery_tools": ["get_account_balance"],
    }
    observation = {
        "interrupted": interrupted,
        "recovery": recovery,
        "task_outcome": "PASS",
        "failure_category": "NONE",
        "interruption_recovery": True,
    }
    interrupted_events = [
        _event("agent.turn.completed", interrupted),
        _event("voice.interruption.completed", interrupted),
    ]
    recovery_events = [
        *_core_events(recovery),
        _event(
            "tool.execution.completed",
            recovery,
            metadata={"tool_name": "get_account_balance"},
        ),
    ]

    result = closeout._interruption_result(
        expected,
        observation,
        interrupted_events,
        recovery_events,
    )

    assert result["passed"] is True
    assert result["interruption_observed"] is True
    assert result["protected_action_executed"] is False


def test_latency_and_cost_use_real_trace_quantities_without_fabrication() -> None:
    correlation = _correlation(1)
    events = [
        *_core_events(correlation),
        _event(
            "llm.request.completed",
            correlation,
            metadata={
                "provider": "groq",
                "model": "openai/gpt-oss-20b",
                "prompt_tokens": 100,
                "completion_tokens": 20,
            },
        ),
    ]

    latency = closeout._aggregate_latency(events)
    cost = closeout._cost_summary(events, successful_tasks=1)

    assert latency["stt"]["count"] == 1
    assert latency["llm.primary"]["p95"] == 10.0
    assert latency["voice.interruption"]["count"] == 0
    assert cost["estimated_total_cost_usd"] is not None
    assert cost["cost_unavailable"] == []


def test_release_validation_is_fail_closed_and_writes_no_provider_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release_path = tmp_path / "release.json"
    monkeypatch.setattr(closeout, "RELEASE_EVIDENCE_PATH", release_path)
    monkeypatch.setattr(closeout, "_backend_app_unchanged", lambda: True)
    monkeypatch.setattr(
        closeout,
        "_git",
        lambda *args: "v2/ml-routing-evaluation"
        if args[0] == "branch"
        else "deadbeef",
    )

    def fake_command(
        check_id: str,
        command: list[str],
        *,
        cwd: Path = closeout.ROOT,
    ) -> tuple[dict[str, Any], str]:
        del cwd
        status = "FAIL" if check_id == "frontend_lint" else "PASS"
        if check_id == "deterministic_agent_evaluation":
            report_path = Path(command[command.index("--json-report") + 1])
            report_path.write_text(
                json.dumps(
                    {
                        "failed": 0,
                        "passed": 35,
                        "scenario_count": 35,
                        "metrics": {
                            "task_success_rate": 1,
                            "tool_selection_accuracy": 1,
                            "tool_argument_accuracy": 1,
                            "unauthorized_action_rate": 0,
                            "confirmation_compliance": 1,
                            "interruption_recovery_rate": 1,
                            "policy_source_accuracy": 1,
                            "safety": {"unauthorized_actions_executed": 0},
                        },
                    }
                ),
                encoding="utf-8",
            )
        output = (
            "Scenarios: 9\nPositive scenarios: 8\nRecall@1: 0.938\n"
            "Recall@3: 1.000\nTop-1 hit rate: 1.000\nMRR: 1.000"
            if check_id == "policy_retrieval_evaluation"
            else "123 passed"
        )
        return (
            {
                "check_id": check_id,
                "command": command,
                "status": status,
                "exit_code": 1 if status == "FAIL" else 0,
                "duration_ms": 1.0,
                "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
            },
            output,
        )

    monkeypatch.setattr(closeout, "_command_result", fake_command)

    report = closeout.release_validation()
    payload = json.loads(release_path.read_text(encoding="utf-8"))

    assert report["status"] == "FAIL"
    assert report["mandatory_safety_status"] == "PASS"
    assert report["provider_calls_performed"] is False
    assert payload["governance"]["provider_calls_performed"] is False
    assert payload["status"] == "FAIL"


@pytest.mark.parametrize(
    ("release", "safety", "live", "expected"),
    [
        ("PASS", "PASS", "PASS", "PORTFOLIO_COMPLETE"),
        ("PASS", "PASS", "FAIL", "PORTFOLIO_COMPLETE_WITH_LIMITATIONS"),
        ("FAIL", "PASS", "PASS", "V2D_BLOCKED"),
        ("PASS", "FAIL", "PASS", "V2D_BLOCKED"),
    ],
)
def test_final_status_is_recomputed_fail_closed(
    release: str,
    safety: str,
    live: str,
    expected: str,
) -> None:
    result = {
        "deterministic_release_validation": {"status": release},
        "safety": {"status": safety},
        "live_operational_benchmark": {"status": live},
    }

    assert closeout._recompute_final_status(result) == expected


def test_runner_has_no_provider_execution_or_holdout_access_path() -> None:
    source = closeout.Path(closeout.__file__).read_text(encoding="utf-8")

    assert "GroqLLMProvider" not in source
    assert "AsyncGroq" not in source
    assert ".generate(" not in source
    assert ".transcribe(" not in source
    assert ".synthesize(" not in source
    with pytest.raises(PermissionError):
        closeout.read_bytes(closeout.PROHIBITED_HOLDOUT_PATH)
