import json
from typing import Any

from groq import AsyncGroq

from backend.app.providers.llm import (
    LLMResponse,
    LLMToolCall,
    LLMUsage,
)


class LLMProviderError(RuntimeError):
    """Raised when the external LLM provider call fails."""


class LLMProviderResponseError(LLMProviderError):
    """Raised when the provider returns an unusable response."""


class GroqLLMProvider:
    def __init__(
        self,
        *,
        api_key: str,
        model: str = "openai/gpt-oss-20b",
        max_completion_tokens: int = 1024,
        client: Any | None = None,
    ) -> None:
        if not api_key and client is None:
            raise ValueError("Groq API key is required")

        self.model = model
        self.max_completion_tokens = max_completion_tokens
        self._client = client or AsyncGroq(api_key=api_key)

    async def generate(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        request: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_completion_tokens": self.max_completion_tokens,
        }

        if tools:
            request.update(
                {
                    "tools": tools,
                    "tool_choice": "auto",
                    # GPT-OSS 20B does not support parallel tool use.
                    "parallel_tool_calls": False,
                }
            )

        try:
            response = await self._client.chat.completions.create(
                **request
            )
        except Exception as exc:
            raise LLMProviderError("Groq LLM request failed") from exc

        if not response.choices:
            raise LLMProviderResponseError(
                "Groq returned no completion choices"
            )

        choice = response.choices[0]
        message = choice.message

        normalized_tool_calls: list[LLMToolCall] = []

        for tool_call in message.tool_calls or []:
            raw_arguments = tool_call.function.arguments or "{}"

            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError as exc:
                raise LLMProviderResponseError(
                    f"Tool {tool_call.function.name} returned invalid JSON arguments"
                ) from exc

            if not isinstance(arguments, dict):
                raise LLMProviderResponseError(
                    f"Tool {tool_call.function.name} arguments must be a JSON object"
                )

            normalized_tool_calls.append(
                LLMToolCall(
                    id=tool_call.id,
                    name=tool_call.function.name,
                    arguments=arguments,
                )
            )

        provider_usage = getattr(response, "usage", None)

        usage = LLMUsage(
            prompt_tokens=getattr(
                provider_usage,
                "prompt_tokens",
                0,
            )
            or 0,
            completion_tokens=getattr(
                provider_usage,
                "completion_tokens",
                0,
            )
            or 0,
            total_tokens=getattr(
                provider_usage,
                "total_tokens",
                0,
            )
            or 0,
        )

        return LLMResponse(
            content=message.content or "",
            tool_calls=normalized_tool_calls,
            model=getattr(response, "model", None) or self.model,
            finish_reason=getattr(choice, "finish_reason", None),
            usage=usage,
        )
