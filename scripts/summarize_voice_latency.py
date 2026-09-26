"""Summarize SentinelVoice latency samples from structured trace logs."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

from pydantic import ValidationError

from backend.app.observability.events import TraceEvent, TraceStatus
from backend.app.observability.metrics import (
    NumericSummary,
    latency_by_stage,
    summarize_numeric,
)

STAGE_LABELS = {
    "voice.speech_end_to_final_transcript": "Speech end → final transcript",
    "voice.final_transcript_to_playback_start": (
        "Final transcript → playback start"
    ),
    "voice.speech_end_to_playback_start": "Speech end → playback start",
    "stt": "STT provider",
    "voice.backend_turn": "Voice backend turn",
    "agent.turn": "Agent turn",
    "llm.request": "LLM request",
    "rag.retrieval": "Policy retrieval",
    "tool.execution": "Tool execution",
    "tts.first_audio": "TTS first emitted PCM",
    "tts": "TTS total",
    "voice.interruption": "Interruption stop",
}
_JSON_DECODER = json.JSONDecoder()
_TRACE_LOG_MARKER = "sentinelvoice.trace - "


def _trace_payload_from_line(line: str) -> object | None:
    stripped = line.strip()
    if stripped.startswith("{"):
        candidate = stripped
    else:
        marker_index = line.find(_TRACE_LOG_MARKER)
        if marker_index < 0:
            return None
        candidate = line[
            marker_index + len(_TRACE_LOG_MARKER) :
        ].lstrip()

    try:
        payload, _ = _JSON_DECODER.raw_decode(candidate)
    except (json.JSONDecodeError, TypeError):
        return None
    return payload


def _trace_event_from_line(line: str) -> TraceEvent | None:
    payload = _trace_payload_from_line(line)
    if not isinstance(payload, dict):
        return None
    try:
        return TraceEvent.model_validate(payload)
    except ValidationError:
        return None


def read_trace_events(paths: Iterable[Path]) -> list[TraceEvent]:
    events: list[TraceEvent] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"log file does not exist: {path}")
        with path.open(encoding="utf-8", errors="replace") as log_file:
            for line in log_file:
                event = _trace_event_from_line(line)
                if event is not None:
                    events.append(event)
    return events


def summarize_voice_latency(
    events: Iterable[TraceEvent],
) -> dict[str, NumericSummary]:
    collected = [
        event for event in events if event.status == TraceStatus.COMPLETED
    ]
    summaries = latency_by_stage(collected)
    first_audio = [
        event.duration_ms
        for event in collected
        if event.event_name == "tts.first_audio"
        and event.duration_ms is not None
    ]
    if first_audio:
        summaries["tts.first_audio"] = summarize_numeric(first_audio)
    return {
        stage: summaries[stage]
        for stage in STAGE_LABELS
        if stage in summaries
    }


def _format_duration(value: float | None) -> str:
    return "—" if value is None else f"{value:.1f} ms"


def render_latency_table(summaries: dict[str, NumericSummary]) -> str:
    label_width = max(
        [len("Stage"), *(len(STAGE_LABELS[stage]) for stage in summaries)]
    )
    header = (
        f"{'Stage':<{label_width}}  {'Count':>5}  "
        f"{'P50':>10}  {'P90':>10}  {'P95':>10}"
    )
    rows = [header, "-" * len(header)]
    for stage, summary in summaries.items():
        rows.append(
            f"{STAGE_LABELS[stage]:<{label_width}}  {summary.count:>5}  "
            f"{_format_duration(summary.p50):>10}  "
            f"{_format_duration(summary.p90):>10}  "
            f"{_format_duration(summary.p95):>10}"
        )
    return "\n".join(rows)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize live SentinelVoice latency from one or more structured "
            "trace log files."
        )
    )
    parser.add_argument(
        "log_files",
        nargs="+",
        type=Path,
        help="FastAPI and/or LiveKit worker log files",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        events = read_trace_events(args.log_files)
    except (OSError, UnicodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if not events:
        print("error: no SentinelVoice trace events were found", file=sys.stderr)
        return 1

    summaries = summarize_voice_latency(events)
    if not summaries:
        print(
            "error: no supported completed latency samples were found",
            file=sys.stderr,
        )
        return 1

    print(render_latency_table(summaries))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
