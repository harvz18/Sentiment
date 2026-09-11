"""Run with uvicorn app.main:app --reload."""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.config.settings import Settings, get_settings
from app.services.runtime import Runtime

STATIC = Path(__file__).parent / "static"


def create_app(settings: Settings | None = None, runtime: Runtime | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        instance = runtime or Runtime(settings or get_settings())
        if runtime is None:
            instance.load()
        application.state.runtime = instance
        yield

    application = FastAPI(title="Multivent Binary Sentiment Lab", version="0.2.0", lifespan=lifespan)
    application.include_router(router)
    application.mount("/static", StaticFiles(directory=STATIC), name="static")

    @application.get("/", include_in_schema=True)
    def root(request: Request):
        # Browser visits open the UI; REST clients receive application status.
        if "text/html" in request.headers.get("accept", ""):
            return FileResponse(STATIC / "index.html")
        return {"application": "Multivent Sentiment Lab", **request.app.state.runtime.health().model_dump(),
                "ui": "/ui", "docs": "/docs"}

    @application.get("/ui", include_in_schema=False)
    def ui():
        return FileResponse(STATIC / "index.html")

    return application


app = create_app()
