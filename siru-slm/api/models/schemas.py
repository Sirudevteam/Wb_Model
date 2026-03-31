"""Pydantic models for the Siru AI Labs API."""

from typing import Any, Generic, Literal, Optional, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class ErrorDetails(BaseModel):
    code: str
    message: str
    details: Optional[dict[str, Any]] = None


class ApiResponse(BaseModel, Generic[T]):
    success: bool
    data: Optional[T] = None
    error: Optional[ErrorDetails] = None


class RewriteRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, description="Tamil dialogue text to rewrite")
    mode: Literal["mass", "emotion", "subtext"] = Field(..., description="Rewrite style mode")
    context: Optional[str] = Field(None, max_length=1000, description="Optional scene/character context")
    use_rag: bool = Field(True, description="Whether to use RAG context injection")


class RewriteResponse(BaseModel):
    original: str
    rewritten: str
    mode: str
    rag_context_used: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str
    checks: dict[str, str] = Field(default_factory=dict)


class IdeateRequest(BaseModel):
    scene_description: str = Field(..., min_length=1, max_length=5000)
    genre: Optional[str] = None
    characters: Optional[list[str]] = None


class IdeateResponse(BaseModel):
    scene_description: str
    generated_scene: str
    model: str


class HealthCheckResult(BaseModel):
    status: str
    checks: dict[str, str]
