# SPDX-License-Identifier: Apache-2.0
from fastapi import APIRouter, HTTPException, Request

from entrypoint.logging import log_completion_request
from entrypoint.schemas import CompletionRequest, CompletionResponse
from entrypoint.stub import build_completion_stub

router = APIRouter(tags=["completions"])


@router.post("/completions", response_model=CompletionResponse)
async def completions(
    body: CompletionRequest,
    request: Request,
) -> CompletionResponse:
    if body.stream:
        raise HTTPException(status_code=501, detail="stream not implemented")
    if not body.prompt:
        raise HTTPException(status_code=400, detail="prompt must not be empty")

    log_completion_request(
        request_id=getattr(request.state, "request_id", None),
        model=body.model,
        prompt=body.prompt,
        max_tokens=body.max_tokens,
        temperature=body.temperature,
    )
    return build_completion_stub(body)
