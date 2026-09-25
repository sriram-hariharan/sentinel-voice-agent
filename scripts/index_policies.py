import argparse
import asyncio
from pathlib import Path

from backend.app.config.settings import get_settings
from backend.app.db.session import get_session_factory
from backend.app.providers.embeddings import FastEmbedProvider
from backend.app.rag.chunking import load_policy_documents
from backend.app.rag.indexing import index_policy_documents

POLICY_DIRECTORY = Path("data/policies")


async def index_policies(*, reset: bool) -> None:
    settings = get_settings()
    documents = load_policy_documents(POLICY_DIRECTORY)
    provider = FastEmbedProvider(
        model_name=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
        cache_dir=settings.fastembed_cache_dir,
    )
    session_factory = get_session_factory()
    async with session_factory() as db:
        report = await index_policy_documents(
            db=db,
            documents=documents,
            embedding_provider=provider,
            reset=reset,
        )

    print(
        "Policy indexing complete: "
        f"{report.documents_seen} documents seen, "
        f"{report.documents_indexed} indexed, "
        f"{report.documents_unchanged} unchanged, "
        f"{report.documents_removed} removed, "
        f"{report.chunks_indexed} chunks indexed."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the existing policy index before rebuilding it.",
    )
    args = parser.parse_args()
    asyncio.run(index_policies(reset=args.reset))


if __name__ == "__main__":
    main()
