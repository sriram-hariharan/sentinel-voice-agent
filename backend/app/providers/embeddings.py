import asyncio
import os
from collections.abc import Sequence
from threading import Lock
from typing import Protocol


class EmbeddingProviderError(RuntimeError):
    """Raised when an embedding implementation cannot produce vectors."""


class EmbeddingProvider(Protocol):
    dimensions: int

    async def embed_documents(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...


class FastEmbedProvider:
    """Lazy FastEmbed adapter; model loading never occurs at API startup."""

    def __init__(
        self,
        *,
        model_name: str,
        dimensions: int = 384,
        cache_dir: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.dimensions = dimensions
        self.cache_dir = cache_dir
        self._model = None
        self._model_lock = Lock()

    def _get_model(self):
        if self._model is not None:
            return self._model

        with self._model_lock:
            if self._model is None:
                os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
                try:
                    from fastembed import TextEmbedding

                    self._model = TextEmbedding(
                        model_name=self.model_name,
                        cache_dir=self.cache_dir,
                    )
                except Exception as exc:
                    raise EmbeddingProviderError(
                        "FastEmbed model initialization failed"
                    ) from exc

        return self._model

    def _validate_vector(self, vector) -> list[float]:
        values = [float(value) for value in vector]
        if len(values) != self.dimensions:
            raise EmbeddingProviderError(
                f"Expected {self.dimensions} embedding dimensions, "
                f"received {len(values)}"
            )
        return values

    def _embed_documents_sync(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        try:
            vectors = self._get_model().passage_embed(list(texts))
            return [self._validate_vector(vector) for vector in vectors]
        except EmbeddingProviderError:
            raise
        except Exception as exc:
            raise EmbeddingProviderError(
                "Document embedding failed"
            ) from exc

    def _embed_query_sync(self, text: str) -> list[float]:
        try:
            vector = next(iter(self._get_model().query_embed(text)))
            return self._validate_vector(vector)
        except EmbeddingProviderError:
            raise
        except Exception as exc:
            raise EmbeddingProviderError("Query embedding failed") from exc

    async def embed_documents(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        if not texts:
            return []
        return await asyncio.to_thread(self._embed_documents_sync, texts)

    async def embed_query(self, text: str) -> list[float]:
        if not text.strip():
            raise EmbeddingProviderError("Embedding query cannot be empty")
        return await asyncio.to_thread(self._embed_query_sync, text)
