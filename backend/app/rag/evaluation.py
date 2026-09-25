import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class RetrievalScenario(BaseModel):
    scenario_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    relevant_policy_ids: tuple[str, ...] = ()
    expected_no_results: bool = False

    model_config = ConfigDict(frozen=True, extra="forbid")


class RetrievalMetrics(BaseModel):
    scenario_count: int = Field(ge=0)
    positive_scenario_count: int = Field(ge=0)
    recall_at_1: float = Field(ge=0, le=1)
    recall_at_3: float = Field(ge=0, le=1)
    top_1_hit_rate: float = Field(ge=0, le=1)
    mrr: float = Field(ge=0, le=1)

    model_config = ConfigDict(frozen=True)


def load_retrieval_scenarios(path: Path) -> list[RetrievalScenario]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError("Retrieval evaluation data must be a list")
    scenarios = [RetrievalScenario.model_validate(item) for item in payload]
    if len({scenario.scenario_id for scenario in scenarios}) != len(
        scenarios
    ):
        raise ValueError("Retrieval scenario IDs must be unique")
    return scenarios


def calculate_retrieval_metrics(
    scenarios: list[RetrievalScenario],
    ranked_policy_ids: dict[str, list[str]],
) -> RetrievalMetrics:
    positive = [scenario for scenario in scenarios if scenario.relevant_policy_ids]
    recall_1_total = 0.0
    recall_3_total = 0.0
    reciprocal_rank_total = 0.0
    top_1_hits = 0

    for scenario in scenarios:
        ranked = ranked_policy_ids.get(scenario.scenario_id, [])
        relevant = set(scenario.relevant_policy_ids)

        if relevant:
            recall_1_total += len(relevant.intersection(ranked[:1])) / len(
                relevant
            )
            recall_3_total += len(relevant.intersection(ranked[:3])) / len(
                relevant
            )
            if ranked and ranked[0] in relevant:
                top_1_hits += 1
            first_rank = next(
                (
                    index
                    for index, policy_id in enumerate(ranked, start=1)
                    if policy_id in relevant
                ),
                None,
            )
            if first_rank is not None:
                reciprocal_rank_total += 1.0 / first_rank
        elif scenario.expected_no_results and not ranked:
            top_1_hits += 1

    positive_count = len(positive)
    scenario_count = len(scenarios)
    return RetrievalMetrics(
        scenario_count=scenario_count,
        positive_scenario_count=positive_count,
        recall_at_1=(recall_1_total / positive_count if positive_count else 0),
        recall_at_3=(recall_3_total / positive_count if positive_count else 0),
        top_1_hit_rate=(top_1_hits / scenario_count if scenario_count else 0),
        mrr=(
            reciprocal_rank_total / positive_count
            if positive_count
            else 0
        ),
    )
