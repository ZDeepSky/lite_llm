# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import logging
from typing import Any

LOGGER_NAME = "lite_llm.entrypoint"
logger = logging.getLogger(LOGGER_NAME)


def log_chat_request(
    *,
    request_id: str | None,
    model: str,
    messages: list[dict[str, Any]],
) -> None:
    payload = {
        "request_id": request_id,
        "model": model,
        "messages": messages,
    }
    logger.info(
        "chat_completions request: %s",
        json.dumps(payload, ensure_ascii=False),
    )


def log_completion_request(
    *,
    request_id: str | None,
    model: str,
    prompt: str,
    max_tokens: int | None,
    temperature: float | None,
) -> None:
    payload = {
        "request_id": request_id,
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    logger.info(
        "completions request: %s",
        json.dumps(payload, ensure_ascii=False),
    )


def log_access(
    *,
    request_id: str,
    method: str,
    path: str,
    status_code: int,
    duration_ms: float,
) -> None:
    payload = {
        "request_id": request_id,
        "method": method,
        "path": path,
        "status": status_code,
        "duration_ms": round(duration_ms, 2),
    }
    logger.info("access: %s", json.dumps(payload, ensure_ascii=False))
