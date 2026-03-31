"""Dialogue rewrite API routes for Siru AI Labs."""

from fastapi import APIRouter, HTTPException

from api.models.schemas import RewriteRequest, RewriteResponse
from api.services.cache import TTLCache
from api.services.logger import EventLogger
from api.services.model_router import ModelRouter
from inference.prompt_engine import build_chat_messages

router = APIRouter(prefix="/rewrite", tags=["rewrite"])

model_router = ModelRouter()
cache = TTLCache(ttl_seconds=3600, max_items=3000)
event_logger = EventLogger()
_rag_retriever = None


def set_rag_retriever(retriever):
    """Called from main.py to inject the RAG retriever."""
    global _rag_retriever
    _rag_retriever = retriever


async def _do_rewrite(request: RewriteRequest) -> RewriteResponse:
    cache_key = cache.key_for(
        {
            "text": request.text,
            "mode": request.mode,
            "context": request.context,
            "use_rag": request.use_rag,
        }
    )
    cached = cache.get(cache_key)
    if cached:
        event_logger.log("rewrite_cache_hit", {"mode": request.mode})
        return RewriteResponse(**cached)

    rag_context = None
    if request.use_rag and _rag_retriever:
        try:
            rag_context = await _rag_retriever.retrieve(request.text, request.mode)
        except Exception:
            pass

    messages = build_chat_messages(
        text=request.text,
        mode=request.mode,
        rag_context=rag_context,
    )

    try:
        response = model_router.slm_client.chat.completions.create(
            model=model_router.slm_model,
            messages=messages,
            temperature=0.7,
            max_tokens=256,
            top_p=0.9,
        )
        rewritten = response.choices[0].message.content.strip()
    except Exception as e:
        event_logger.log(
            "rewrite_failed",
            {
                "mode": request.mode,
                "input_length": len(request.text),
                "error": str(e),
            },
        )
        raise HTTPException(status_code=503, detail=f"SLM inference failed: {str(e)}")

    result = RewriteResponse(
        original=request.text,
        rewritten=rewritten,
        mode=request.mode,
        rag_context_used=rag_context,
    )
    cache.set(cache_key, result.model_dump())
    event_logger.log(
        "rewrite_completed",
        {
            "mode": request.mode,
            "input_length": len(request.text),
            "output_length": len(rewritten),
            "used_rag": bool(rag_context),
        },
    )
    return result


@router.post("/dialogue", response_model=RewriteResponse)
async def rewrite_dialogue(request: RewriteRequest):
    """General dialogue rewrite -- mode is specified in the request body."""
    return await _do_rewrite(request)


@router.post("/mass", response_model=RewriteResponse)
async def rewrite_mass(request: RewriteRequest):
    """Mass style rewrite -- overrides mode to 'mass'."""
    request.mode = "mass"
    return await _do_rewrite(request)


@router.post("/emotion", response_model=RewriteResponse)
async def rewrite_emotion(request: RewriteRequest):
    """Emotion style rewrite -- overrides mode to 'emotion'."""
    request.mode = "emotion"
    return await _do_rewrite(request)


@router.post("/subtext", response_model=RewriteResponse)
async def rewrite_subtext(request: RewriteRequest):
    """Subtext style rewrite -- overrides mode to 'subtext'."""
    request.mode = "subtext"
    return await _do_rewrite(request)
