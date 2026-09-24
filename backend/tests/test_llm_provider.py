from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from backend.app.providers.groq_llm import (
    GroqLLMProvider,
    LLMProviderResponseError,
)


def _client_for(response):
    create = AsyncMock(return_value=response)

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=create,
            )
        )
    )

    return client, create


@pytest.mark.asyncio
async def test_groq_provider_normalizes_direct_response() -> None:
    response = SimpleNamespace(
        model="openai/gpt-oss-20b",
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(
                    content="Your balance is available.",
                    tool_calls=None,
                ),
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=20,
            completion_tokens=8,
            total_tokens=28,
        ),
    )

    client, create = _client_for(response)

    provider = GroqLLMProvider(
        api_key="test-key",
        client=client,
    )

    result = await provider.generate(
        messages=[
            {
                "role": "user",
                "content": "Hello",
            }
        ]
    )

    assert result.content == "Your balance is available."
    assert result.tool_calls == []
    assert result.model == "openai/gpt-oss-20b"
    assert result.usage.total_tokens == 28

    request = create.await_args.kwargs

    assert request["model"] == "openai/gpt-oss-20b"
    assert "tools" not in request


@pytest.mark.asyncio
async def test_groq_provider_normalizes_tool_call() -> None:
    tool_call = SimpleNamespace(
        id="call-001",
        function=SimpleNamespace(
            name="get_account_balance",
            arguments='{"account_id":"aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaa1"}',
        ),
    )

    response = SimpleNamespace(
        model="openai/gpt-oss-20b",
        choices=[
            SimpleNamespace(
                finish_reason="tool_calls",
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[tool_call],
                ),
            )
        ],
        usage=None,
    )

    client, create = _client_for(response)

    provider = GroqLLMProvider(
        api_key="test-key",
        client=client,
    )

    result = await provider.generate(
        messages=[
            {
                "role": "user",
                "content": "What is my balance?",
            }
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_account_balance",
                    "description": "Get balance",
                    "parameters": {
                        "type": "object",
                    },
                },
            }
        ],
    )

    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "get_account_balance"
    assert (
        result.tool_calls[0].arguments["account_id"]
        == "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaa1"
    )

    request = create.await_args.kwargs

    assert request["tool_choice"] == "auto"
    assert request["parallel_tool_calls"] is False


@pytest.mark.asyncio
async def test_groq_provider_rejects_malformed_tool_arguments() -> None:
    tool_call = SimpleNamespace(
        id="call-001",
        function=SimpleNamespace(
            name="get_account_balance",
            arguments="{bad-json",
        ),
    )

    response = SimpleNamespace(
        model="openai/gpt-oss-20b",
        choices=[
            SimpleNamespace(
                finish_reason="tool_calls",
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[tool_call],
                ),
            )
        ],
        usage=None,
    )

    client, _ = _client_for(response)

    provider = GroqLLMProvider(
        api_key="test-key",
        client=client,
    )

    with pytest.raises(LLMProviderResponseError):
        await provider.generate(
            messages=[
                {
                    "role": "user",
                    "content": "What is my balance?",
                }
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_account_balance",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        )
