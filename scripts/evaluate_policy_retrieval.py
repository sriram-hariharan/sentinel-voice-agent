import argparse
import asyncio
from pathlib import Path

from backend.app.config.settings import get_settings
from backend.app.db.session import get_session_factory
from backend.app.providers.embeddings import FastEmbedProvider
from backend.app.rag.evaluation import (
    calculate_retrieval_metrics,
    load_retrieval_scenarios,
)
from backend.app.rag.retrieval import PolicyRetriever, PostgresPolicySearch

EVALUATION_PATH = Path("data/evals/policy_retrieval.json")


async def evaluate(*, details: bool = False) -> None:
    settings = get_settings()
    scenarios = load_retrieval_scenarios(EVALUATION_PATH)
    provider = FastEmbedProvider(
        model_name=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
        cache_dir=settings.fastembed_cache_dir,
    )
    retriever = PolicyRetriever(
        embedding_provider=provider,
        min_vector_similarity=settings.policy_retrieval_min_similarity,
    )
    ranked: dict[str, list[str]] = {}
    session_factory = get_session_factory()

    async with session_factory() as db:
        search = PostgresPolicySearch(db)
        for scenario in scenarios:
            results = await retriever.retrieve(
                query=scenario.query,
                search=search,
                top_k=3,
            )
            ranked[scenario.scenario_id] = [
                result.chunk.policy_id for result in results
            ]
            if details:
                formatted = ", ".join(
                    (
                        f"{result.chunk.policy_id}/{result.chunk.section} "
                        f"(keyword={result.keyword_rank}, "
                        f"vector={result.vector_rank}, "
                        f"similarity={result.vector_similarity})"
                    )
                    for result in results
                ) or "no results"
                print(f"{scenario.scenario_id}: {formatted}")

    metrics = calculate_retrieval_metrics(scenarios, ranked)
    print(f"Scenarios: {metrics.scenario_count}")
    print(f"Positive scenarios: {metrics.positive_scenario_count}")
    print(f"Recall@1: {metrics.recall_at_1:.3f}")
    print(f"Recall@3: {metrics.recall_at_3:.3f}")
    print(f"Top-1 hit rate: {metrics.top_1_hit_rate:.3f}")
    print(f"MRR: {metrics.mrr:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--details",
        action="store_true",
        help="Print per-scenario ranked policy IDs and channel scores.",
    )
    args = parser.parse_args()
    asyncio.run(evaluate(details=args.details))


if __name__ == "__main__":
    main()
