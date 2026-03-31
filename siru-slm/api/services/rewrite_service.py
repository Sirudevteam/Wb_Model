"""Business logic for dialogue rewriting."""

from __future__ import annotations

from api.models.schemas import RewriteRequest, RewriteResponse
from api.services.cache import TTLCache
from api.services.exceptions import ServiceError
from api.services.logger import EventLogger
from api.services.model_router import ModelRouter
from api.services.retrieval_service import RetrievalService
from inference.prompt_engine import build_chat_messages


class RewriteService:
    def __init__(
        self,
        model_router: ModelRouter,
        retrieval_service: RetrievalService,
        cache: TTLCache,
        event_logger: EventLogger,
    ):
        self.model_router = model_router
        self.retrieval_service = retrieval_service
        self.cache = cache
        self.event_logger = event_logger

    async def rewrite(self, request: RewriteRequest) -> RewriteResponse:
        cache_key = self.cache.key_for(
            {
                "text": request.text,
                "mode": request.mode,
                "context": request.context,
                "use_rag": request.use_rag,
            }
        )
        cached = self.cache.get(cache_key)
        if cached:
            self.event_logger.log("rewrite_cache_hit", {"mode": request.mode})
            return RewriteResponse(**cached)

        rag_context = await self.retrieval_service.get_context(
            text=request.text,
            mode=request.mode,
            enabled=request.use_rag,
        )
        messages = build_chat_messages(
            text=request.text,
            mode=request.mode,
            rag_context=rag_context,
        )

        try:
            response = self.model_router.slm_client.chat.completions.create(
                model=self.model_router.slm_model,
                messages=messages,
                temperature=0.7,
                max_tokens=256,
                top_p=0.9,
            )
            rewritten = (response.choices[0].message.content or "").strip()
        except Exception as exc:
            self.event_logger.log(
                "rewrite_failed",
                {
                    "mode": request.mode,
                    "input_length": len(request.text),
                    "error": str(exc),
                },
            )
            raise ServiceError(
                code="slm_inference_failed",
                message="SLM inference failed.",
                status_code=503,
                details={"mode": request.mode},
            ) from exc

        result = RewriteResponse(
            original=request.text,
            rewritten=rewritten,
            mode=request.mode,
            rag_context_used=rag_context,
        )
        self.cache.set(cache_key, result.model_dump())
        self.event_logger.log(
            "rewrite_completed",
            {
                "mode": request.mode,
                "input_length": len(request.text),
                "output_length": len(rewritten),
                "used_rag": bool(rag_context),
            },
        )
        return result

