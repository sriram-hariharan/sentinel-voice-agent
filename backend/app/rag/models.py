from datetime import date
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class RetrievalMode(StrEnum):
    KEYWORD = "keyword"
    VECTOR = "vector"
    HYBRID = "hybrid"


class PolicySection(BaseModel):
    heading: str = Field(min_length=1, max_length=255)
    content: str = Field(min_length=1)

    model_config = ConfigDict(frozen=True)


class PolicyDocument(BaseModel):
    policy_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{1,99}$")
    title: str = Field(min_length=1, max_length=255)
    version: str = Field(min_length=1, max_length=32)
    effective_date: date
    source_path: str = Field(min_length=1, max_length=500)
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    sections: tuple[PolicySection, ...] = Field(min_length=1)

    model_config = ConfigDict(frozen=True)


class PolicyChunk(BaseModel):
    chunk_id: str = Field(min_length=1, max_length=180)
    policy_id: str = Field(min_length=1, max_length=100)
    title: str = Field(min_length=1, max_length=255)
    version: str = Field(min_length=1, max_length=32)
    effective_date: date
    section: str = Field(min_length=1, max_length=255)
    chunk_index: int = Field(ge=0)
    content: str = Field(min_length=1)

    model_config = ConfigDict(frozen=True)

    @property
    def source_label(self) -> str:
        return f"{self.title} · {self.section} · Version {self.version}"


class PolicySearchHit(BaseModel):
    chunk: PolicyChunk
    keyword_rank: int | None = Field(default=None, ge=1)
    vector_rank: int | None = Field(default=None, ge=1)
    keyword_score: float | None = None
    vector_similarity: float | None = None

    model_config = ConfigDict(frozen=True)


class RetrievedPolicyChunk(PolicySearchHit):
    rrf_score: float = Field(ge=0)


class PolicyIndexReport(BaseModel):
    documents_seen: int = Field(ge=0)
    documents_indexed: int = Field(ge=0)
    documents_unchanged: int = Field(ge=0)
    documents_removed: int = Field(ge=0)
    chunks_indexed: int = Field(ge=0)

    model_config = ConfigDict(frozen=True)
