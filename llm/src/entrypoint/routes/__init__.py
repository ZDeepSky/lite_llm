"""HTTP route modules."""

from entrypoint.routes.chat import router as chat_router
from entrypoint.routes.completions import router as completions_router
from entrypoint.routes.health import router as health_router

__all__ = ["chat_router", "completions_router", "health_router"]
