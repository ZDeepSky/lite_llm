import pytest

from entrypoint.logging import LOGGER_NAME


@pytest.mark.asyncio
async def test_completions_minimal(client):
    response = await client.post(
        "/v1/completions",
        json={"model": "qwen3", "prompt": "hello", "max_tokens": 16},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "text_completion"
    assert data["choices"][0]["text"] == ""


@pytest.mark.asyncio
async def test_completions_logs_prompt(client, caplog):
    caplog.set_level("INFO", logger=LOGGER_NAME)
    await client.post(
        "/v1/completions",
        json={"model": "qwen3", "prompt": "hello world", "max_tokens": 8},
    )
    assert "hello world" in caplog.text


@pytest.mark.asyncio
async def test_completions_empty_prompt_400(client):
    response = await client.post(
        "/v1/completions",
        json={"model": "qwen3", "prompt": ""},
    )
    assert response.status_code == 400
    body = response.json()
    assert body["error"]["message"] == "prompt must not be empty"


@pytest.mark.asyncio
async def test_completions_stream_501(client):
    response = await client.post(
        "/v1/completions",
        json={"model": "qwen3", "prompt": "hi", "stream": True},
    )
    assert response.status_code == 501
