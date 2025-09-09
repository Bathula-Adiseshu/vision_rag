from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router as api_router
from app.core.config import get_settings
from app.utils.logging import configure_logging


def create_app() -> FastAPI:
    s = get_settings()
    configure_logging(s.log_level)
    app = FastAPI(title=s.app_name, version=s.app_version)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router, prefix=s.api_prefix)

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    return app


app = create_app()


