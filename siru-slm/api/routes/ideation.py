"""Remote LLM ideation routes for base scene generation."""

from fastapi import APIRouter

from api.models.schemas import ApiResponse, IdeateRequest, IdeateResponse
from api.services.ideation_service import IdeationService
from api.services.model_router import ModelRouter

router = APIRouter(prefix="/ideate", tags=["ideation"])
ideation_service = IdeationService(model_router=ModelRouter())


@router.post("/scene", response_model=ApiResponse[IdeateResponse])
async def ideate_scene(request: IdeateRequest):
    result = await ideation_service.ideate_scene(request)
    return ApiResponse(success=True, data=result, error=None)
