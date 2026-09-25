from backend.app.rag.evaluation import (
    RetrievalScenario,
    calculate_retrieval_metrics,
)


def test_retrieval_metrics_calculate_recall_top1_and_mrr() -> None:
    scenarios = [
        RetrievalScenario(
            scenario_id="first",
            query="first query",
            relevant_policy_ids=("a",),
        ),
        RetrievalScenario(
            scenario_id="second",
            query="second query",
            relevant_policy_ids=("b", "c"),
        ),
        RetrievalScenario(
            scenario_id="miss",
            query="unsupported",
            expected_no_results=True,
        ),
    ]
    ranked = {
        "first": ["a", "x"],
        "second": ["x", "b", "c"],
        "miss": [],
    }

    metrics = calculate_retrieval_metrics(scenarios, ranked)

    assert metrics.scenario_count == 3
    assert metrics.positive_scenario_count == 2
    assert metrics.recall_at_1 == 0.5
    assert metrics.recall_at_3 == 1.0
    assert metrics.top_1_hit_rate == 2 / 3
    assert metrics.mrr == 0.75
