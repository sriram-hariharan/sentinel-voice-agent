from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field


class LLMToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any]

    model_config = ConfigDict(frozen=True)


class LLMUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    model_config = ConfigDict(frozen=True)


class LLMResponse(BaseModel):
    content: str = ""
    tool_calls: list[LLMToolCall] = Field(default_factory=list)
    model: str
    finish_reason: str | None = None
    usage: LLMUsage = Field(default_factory=LLMUsage)

    model_config = ConfigDict(frozen=True)


class LLMProvider(Protocol):
    async def generate(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        """Generate one normalized model response."""
        ...
