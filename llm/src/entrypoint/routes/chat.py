# SPDX-License-Identifier: Apache-2.0
from fastapi import APIRouter, HTTPException, Request

from entrypoint.logging import log_chat_request
from entrypoint.schemas import ChatCompletionRequest, ChatCompletionResponse
from entrypoint.stub import build_chat_stub

router = APIRouter(tags=["chat"])


@router.post("/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    body: ChatCompletionRequest,
    request: Request,
) -> ChatCompletionResponse:
    if body.stream:
        raise HTTPException(status_code=501, detail="stream not implemented")
    if not body.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")

    log_chat_request(
        request_id=getattr(request.state, "request_id", None),
        model=body.model,
        messages=[m.model_dump() for m in body.messages],
    )
    return build_chat_stub(body)
