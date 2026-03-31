"""
FastAPI Application for Siru AI Labs Tamil Screenplay SLM.

Usage:
    python api/main.py
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env_load import load_project_env

load_project_env()

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from api.models.schemas import ApiResponse, ErrorDetails, HealthResponse
from api.routes.dialogue import rewrite_service, router as dialogue_router, set_rag_retriever
from api.routes.ideation import router as ideation_router
from api.services.exceptions import ServiceError
from api.services.health_service import HealthService
from api.services.logger import configure_logging, get_logger
from rag.retrieval import RAGRetriever

APP_VERSION = "0.1.0"

configure_logging(os.getenv("LOG_LEVEL", "INFO"))
logger = get_logger("siru.api")


def _allowed_origins() -> list[str]:
    raw_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
    origins = [origin.strip() for origin in raw_origins.split(",") if origin.strip()]
    return origins or ["http://localhost:3000"]

app = FastAPI(
    title="Siru AI Labs -- Tamil Screenplay SLM",
    description="Tamil dialogue rewrite engine with Mass, Emotion, and Subtext modes",
    version=APP_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(dialogue_router)
app.include_router(ideation_router)


@app.exception_handler(ServiceError)
async def service_error_handler(_: Request, exc: ServiceError):
    return JSONResponse(
        status_code=exc.status_code,
        content=ApiResponse[dict](
            success=False,
            data=None,
            error=ErrorDetails(code=exc.code, message=exc.message, details=exc.details),
        ).model_dump(),
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content=ApiResponse[dict](
            success=False,
            data=None,
            error=ErrorDetails(
                code="validation_error",
                message="Request validation failed.",
                details={"errors": exc.errors()},
            ),
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def unhandled_error_handler(_: Request, exc: Exception):
    logger.exception("Unhandled API error: %s", exc)
    return JSONResponse(
        status_code=500,
        content=ApiResponse[dict](
            success=False,
            data=None,
            error=ErrorDetails(
                code="internal_server_error",
                message="An unexpected error occurred.",
                details=None,
            ),
        ).model_dump(),
    )


@app.get("/health", response_model=ApiResponse[HealthResponse])
async def health_check():
    health_service = HealthService(
        model_router=rewrite_service.model_router,
        retrieval_service=rewrite_service.retrieval_service,
        version=APP_VERSION,
    )
    result = health_service.get_health()
    return ApiResponse(success=True, data=result, error=None)


@app.get("/")
async def root():
    return ApiResponse(
        success=True,
        data={
            "name": "Siru AI Labs -- Tamil Screenplay SLM",
            "version": APP_VERSION,
            "endpoints": {
                "POST /rewrite/dialogue": "General dialogue rewrite (specify mode in body)",
                "POST /rewrite/mass": "Mass style rewrite",
                "POST /rewrite/emotion": "Emotion style rewrite",
                "POST /rewrite/subtext": "Subtext style rewrite",
                "POST /ideate/scene": "70B base scene generation",
                "GET /health": "Health check",
            },
        },
        error=None,
    )


@app.on_event("startup")
async def startup_event():
    try:
        retriever = RAGRetriever()
        set_rag_retriever(retriever)
        logger.info("RAG retriever initialized.")
    except Exception as exc:
        logger.warning("RAG retriever not available: %s", exc)
        logger.warning("Running without RAG context injection.")


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run("api.main:app", host=host, port=port, reload=True)
