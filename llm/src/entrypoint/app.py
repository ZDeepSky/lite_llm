import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from entrypoint.logging import log_access
from entrypoint.routes import chat_router, completions_router, health_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.engine = None  # phase2: AsyncLLM(...)
    yield
    # phase2: await app.state.engine.shutdown()


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(HTTPException)
    async def http_exception_handler(
        request: Request,
        exc: HTTPException,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "message": exc.detail,
                    "type": "invalid_request_error",
                    "code": None,
                }
            },
        )


def register_middleware(app: FastAPI) -> None:
    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        rid = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = rid
        start = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        duration_ms = (time.perf_counter() - start) * 1000
        log_access(
            request_id=rid,
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=duration_ms,
        )
        return response


def create_app() -> FastAPI:
    app = FastAPI(
        title="lite_llm",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(health_router)
    app.include_router(chat_router, prefix="/v1")
    app.include_router(completions_router, prefix="/v1")
    register_exception_handlers(app)
    register_middleware(app)
    return app
