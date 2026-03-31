"""Business logic for remote scene ideation."""

from __future__ import annotations

from api.models.schemas import IdeateRequest, IdeateResponse
from api.services.exceptions import ServiceError
from api.services.model_router import ModelRouter
from llm_client import chat_completion, has_remote_llm_credentials


class IdeationService:
    def __init__(self, model_router: ModelRouter):
        self.model_router = model_router

    async def ideate_scene(self, request: IdeateRequest) -> IdeateResponse:
        if not has_remote_llm_credentials():
            raise ServiceError(
                code="remote_llm_not_configured",
                message="Remote LLM credentials are not configured.",
                status_code=503,
                details={
                    "hint": (
                        "Set LLM_PROVIDER=replicate with REPLICATE_API_TOKEN, "
                        "or configure an OpenAI-compatible LLM_API_KEY."
                    )
                },
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
                model=self.model_router.ideation_model,
                temperature=0.9,
                max_tokens=900,
            )
        except Exception as exc:
            raise ServiceError(
                code="ideation_failed",
                message="Ideation model failed.",
                status_code=503,
                details={"model": self.model_router.ideation_model},
            ) from exc

        return IdeateResponse(
            scene_description=request.scene_description,
            generated_scene=generated_scene,
            model=self.model_router.ideation_model,
        )

