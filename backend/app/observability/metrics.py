import math
from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict, Field

from backend.app.observability.events import TraceEvent


class NumericSummary(BaseModel):
    count: int = Field(ge=0)
    minimum: float | None = None
    maximum: float | None = None
    mean: float | None = None
    p50: float | None = None
    p90: float | None = None
    p95: float | None = None

    model_config = ConfigDict(frozen=True)


def nearest_rank_percentile(values: Iterable[float], percentile: float) -> float | None:
    """Use the deterministic nearest-rank method: ceil(p * N) - 1."""
    if not 0 <= percentile <= 100:
        raise ValueError("percentile must be between 0 and 100")
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    if percentile == 0:
        return ordered[0]
    rank = math.ceil((percentile / 100) * len(ordered))
    return ordered[max(0, rank - 1)]


def summarize_numeric(values: Iterable[float]) -> NumericSummary:
    collected = [float(value) for value in values]
    if not collected:
        return NumericSummary(count=0)
    return NumericSummary(
        count=len(collected),
        minimum=min(collected),
        maximum=max(collected),
        mean=sum(collected) / len(collected),
        p50=nearest_rank_percentile(collected, 50),
        p90=nearest_rank_percentile(collected, 90),
        p95=nearest_rank_percentile(collected, 95),
    )


def latency_by_stage(events: Iterable[TraceEvent]) -> dict[str, NumericSummary]:
    grouped: dict[str, list[float]] = {}
    for event in events:
        if event.duration_ms is None or not event.event_name.endswith(".completed"):
            continue
        stage = event.event_name.removesuffix(".completed")
        grouped.setdefault(stage, []).append(event.duration_ms)
    return {
        stage: summarize_numeric(values)
        for stage, values in sorted(grouped.items())
    }
