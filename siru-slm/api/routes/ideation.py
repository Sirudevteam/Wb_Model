"""Remote LLM ideation routes for base scene generation."""

import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from llm_client import chat_completion, has_remote_llm_credentials

from api.models.schemas import IdeateRequest, IdeateResponse
from api.services.model_router import ModelRouter

router = APIRouter(prefix="/ideate", tags=["ideation"])
model_router = ModelRouter()


@router.post("/scene", response_model=IdeateResponse)
async def ideate_scene(request: IdeateRequest):
    if not has_remote_llm_credentials():
        raise HTTPException(
            status_code=503,
            detail="Configure LLM: set LLM_PROVIDER=replicate with REPLICATE_API_TOKEN, or OpenAI-compatible LLM_API_KEY",
        )

    genre = request.genre or "tamil cinema"
    characters = ", ".join(request.characters) if request.characters else "hero, heroine, supporting cast"

    prompt = (
        "You are an expert Tamil screenplay writer. Generate a base scene draft in Tamil that can later be improved "
        "by a rewrite model. Keep it cinematic, character-driven, and clear.\n\n"
        f"Genre: {genre}\n"
        f"Characters: {characters}\n"
        f"Scene brief: {request.scene_description}\n\n"
        "Output only the scene text with dialogue and action lines."
    )

    try:
        generated_scene = chat_completion(
            messages=[
                {"role": "system", "content": "You are a Tamil screenplay writer."},
                {"role": "user", "content": prompt},
            ],
            model=model_router.ideation_model,
            temperature=0.9,
            max_tokens=900,
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Ideation model failed: {e}")

    return IdeateResponse(
        scene_description=request.scene_description,
        generated_scene=generated_scene,
        model=model_router.ideation_model,
    )
