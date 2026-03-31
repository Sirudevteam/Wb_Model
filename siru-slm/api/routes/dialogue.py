"""Dialogue rewrite API routes for Siru AI Labs."""

from fastapi import APIRouter

from api.models.schemas import ApiResponse, RewriteRequest, RewriteResponse
from api.services.cache import TTLCache
from api.services.logger import EventLogger
from api.services.model_router import ModelRouter
from api.services.retrieval_service import RetrievalService
from api.services.rewrite_service import RewriteService

router = APIRouter(prefix="/rewrite", tags=["rewrite"])

rewrite_service = RewriteService(
    model_router=ModelRouter(),
    retrieval_service=RetrievalService(),
    cache=TTLCache(ttl_seconds=3600, max_items=3000),
    event_logger=EventLogger(),
)


def set_rag_retriever(retriever):
    """Called from main.py to inject the RAG retriever."""
    rewrite_service.retrieval_service.set_retriever(retriever)


@router.post("/dialogue", response_model=ApiResponse[RewriteResponse])
async def rewrite_dialogue(request: RewriteRequest):
    """General dialogue rewrite -- mode is specified in the request body."""
    result = await rewrite_service.rewrite(request)
    return ApiResponse(success=True, data=result, error=None)


@router.post("/mass", response_model=ApiResponse[RewriteResponse])
async def rewrite_mass(request: RewriteRequest):
    """Mass style rewrite -- overrides mode to 'mass'."""
    request.mode = "mass"
    result = await rewrite_service.rewrite(request)
    return ApiResponse(success=True, data=result, error=None)


@router.post("/emotion", response_model=ApiResponse[RewriteResponse])
async def rewrite_emotion(request: RewriteRequest):
    """Emotion style rewrite -- overrides mode to 'emotion'."""
    request.mode = "emotion"
    result = await rewrite_service.rewrite(request)
    return ApiResponse(success=True, data=result, error=None)


@router.post("/subtext", response_model=ApiResponse[RewriteResponse])
async def rewrite_subtext(request: RewriteRequest):
    """Subtext style rewrite -- overrides mode to 'subtext'."""
    request.mode = "subtext"
    result = await rewrite_service.rewrite(request)
    return ApiResponse(success=True, data=result, error=None)
