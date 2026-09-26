import json
from pathlib import Path

from backend.app.observability.events import TraceEvent, TraceStatus
from scripts.summarize_voice_latency import (
    main,
    read_trace_events,
    render_latency_table,
    summarize_voice_latency,
)


def _event(
    event_name: str,
    *,
    duration_ms: float | None,
    status: TraceStatus = TraceStatus.COMPLETED,
    session_id: str = "session-1",
) -> TraceEvent:
    return TraceEvent(
        event_name=event_name,
        trace_id="trace_1234567890123456",
        session_id=session_id,
        turn_id="turn_12345678901234567",
        component="test",
        status=status,
        duration_ms=duration_ms,
    )


def _write_lines(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_reads_trace_json_and_skips_unrelated_log_lines(tmp_path: Path) -> None:
    log_path = tmp_path / "worker.log"
    trace = _event(
        "voice.speech_end_to_playback_start.completed",
        duration_ms=420,
    )
    _write_lines(
        log_path,
        [
            "INFO: LiveKit worker starting",
            json.dumps({"level": "info", "message": "not a trace"}),
            trace.model_dump_json(),
            "{invalid-json",
        ],
    )

    assert read_trace_events([log_path]) == [trace]


def test_reads_trace_event_embedded_in_real_runtime_log_line(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "runtime.log"
    trace = _event("stt.completed", duration_ms=75)
    runtime_line = (
        "2026-09-25 20:00:00,000 - INFO sentinelvoice.trace - "
        + trace.model_dump_json()
        + ' {"pid":123,"job_id":"job-1","room":"demo"}'
    )
    _write_lines(log_path, [runtime_line])

    assert read_trace_events([log_path]) == [trace]


def test_ignores_embedded_trace_shaped_json_from_unrelated_logger(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "other.log"
    trace = _event("stt.completed", duration_ms=75)
    unrelated_line = (
        "2026-09-25 20:00:00,000 - INFO some.other.logger - "
        + trace.model_dump_json()
    )
    _write_lines(log_path, [unrelated_line])

    assert read_trace_events([log_path]) == []


def test_multiple_logs_aggregate_new_and_existing_nearest_rank_stages(
    tmp_path: Path,
) -> None:
    backend_log = tmp_path / "backend.log"
    worker_log = tmp_path / "worker.log"
    backend_events = [
        _event("llm.request.completed", duration_ms=value)
        for value in range(1, 11)
    ] + [
        _event("rag.retrieval.completed", duration_ms=12),
        _event(
            "tool.execution.failed",
            duration_ms=999,
            status=TraceStatus.FAILED,
        ),
        _event(
            "tool.execution.completed",
            duration_ms=999,
            status=TraceStatus.FAILED,
        ),
    ]
    worker_events = [
        _event(
            "voice.speech_end_to_final_transcript.completed",
            duration_ms=80,
        ),
        _event(
            "voice.final_transcript_to_playback_start.completed",
            duration_ms=320,
        ),
        _event(
            "voice.speech_end_to_playback_start.completed",
            duration_ms=400,
        ),
        _event("stt.completed", duration_ms=70),
        _event("voice.backend_turn.completed", duration_ms=250),
        _event("tts.first_audio", duration_ms=90),
        _event("tts.completed", duration_ms=110),
        _event("voice.interruption.completed", duration_ms=45),
        _event("tts.completed", duration_ms=None),
    ]
    _write_lines(backend_log, [event.model_dump_json() for event in backend_events])
    _write_lines(worker_log, [event.model_dump_json() for event in worker_events])

    summaries = summarize_voice_latency(
        read_trace_events([backend_log, worker_log])
    )

    assert summaries["llm.request"].count == 10
    assert summaries["llm.request"].p50 == 5
    assert summaries["llm.request"].p90 == 9
    assert summaries["llm.request"].p95 == 10
    assert summaries["voice.speech_end_to_playback_start"].p50 == 400
    assert summaries["tts"].count == 1
    assert "tool.execution" not in summaries


def test_rendered_summary_contains_only_labels_and_numeric_samples() -> None:
    secret = "never-print-this-transcript"
    event = _event("tts.first_audio", duration_ms=123.45).model_copy(
        update={"metadata": {"transcript": secret}}
    )

    output = render_latency_table(summarize_voice_latency([event]))

    assert "TTS first emitted PCM" in output
    assert "123.5 ms" in output
    assert secret not in output
    assert "transcript" not in output


def test_cli_reports_missing_file_and_logs_without_latency(
    tmp_path: Path,
    capsys,
) -> None:
    missing = tmp_path / "missing.log"

    assert main([str(missing)]) == 1
    assert "log file does not exist" in capsys.readouterr().err

    unrelated = tmp_path / "unrelated.log"
    _write_lines(unrelated, [json.dumps({"message": "server ready"})])

    assert main([str(unrelated)]) == 1
    assert "no SentinelVoice trace events" in capsys.readouterr().err
