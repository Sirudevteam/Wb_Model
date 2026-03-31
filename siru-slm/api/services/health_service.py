"""Health checks for runtime dependencies."""

from __future__ import annotations

from api.models.schemas import HealthResponse
from api.services.model_router import ModelRouter
from api.services.retrieval_service import RetrievalService
from llm_client import has_remote_llm_credentials


class HealthService:
    def __init__(self, model_router: ModelRouter, retrieval_service: RetrievalService, version: str):
        self.model_router = model_router
        self.retrieval_service = retrieval_service
        self.version = version

    def get_health(self) -> HealthResponse:
        slm_status = "ready" if self.model_router.is_slm_configured() else "missing"
        rag_status = "ready" if self.retrieval_service.is_ready else "degraded"
        remote_status = "ready" if has_remote_llm_credentials() else "missing"

        overall_status = "healthy"
        if slm_status != "ready":
            overall_status = "degraded"

        if remote_status != "ready" and rag_status != "ready":
            overall_status = "degraded"

        return HealthResponse(
            status=overall_status,
            model_loaded=slm_status == "ready",
            version=self.version,
            checks={
                "slm": slm_status,
                "remote_llm": remote_status,
                "rag": rag_status,
            },
        )

