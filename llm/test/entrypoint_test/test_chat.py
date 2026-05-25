import pytest

from entrypoint.logging import LOGGER_NAME


@pytest.mark.asyncio
async def test_chat_completions_minimal(client):
    response = await client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen3",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"
    assert data["model"] == "qwen3"
    assert data["choices"][0]["message"]["content"] == ""
    assert data["usage"]["total_tokens"] == 0


@pytest.mark.asyncio
async def test_chat_logs_messages(client, caplog):
    caplog.set_level("INFO", logger=LOGGER_NAME)
    await client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen3",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )
    assert "hello" in caplog.text


@pytest.mark.asyncio
async def test_chat_empty_messages_400(client):
    response = await client.post(
        "/v1/chat/completions",
        json={"model": "qwen3", "messages": []},
    )
    assert response.status_code == 400
    body = response.json()
    assert body["error"]["message"] == "messages must not be empty"
    assert body["error"]["type"] == "invalid_request_error"


@pytest.mark.asyncio
async def test_chat_missing_field_422(client):
    response = await client.post(
        "/v1/chat/completions",
        json={"model": "qwen3"},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_chat_stream_501(client):
    response = await client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen3",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    )
    assert response.status_code == 501
    body = response.json()
    assert body["error"]["message"] == "stream not implemented"
