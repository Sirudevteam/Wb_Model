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

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.models.schemas import HealthResponse
from api.routes.dialogue import router as dialogue_router
from api.routes.ideation import router as ideation_router

app = FastAPI(
    title="Siru AI Labs -- Tamil Screenplay SLM",
    description="Tamil dialogue rewrite engine with Mass, Emotion, and Subtext modes",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(dialogue_router)
app.include_router(ideation_router)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        version="0.1.0",
    )


@app.get("/")
async def root():
    return {
        "name": "Siru AI Labs -- Tamil Screenplay SLM",
        "version": "0.1.0",
        "endpoints": {
            "POST /rewrite/dialogue": "General dialogue rewrite (specify mode in body)",
            "POST /rewrite/mass": "Mass style rewrite",
            "POST /rewrite/emotion": "Emotion style rewrite",
            "POST /rewrite/subtext": "Subtext style rewrite",
            "POST /ideate/scene": "70B base scene generation",
            "GET /health": "Health check",
        },
    }


@app.on_event("startup")
async def startup_event():
    try:
        from rag.retrieval import RAGRetriever
        retriever = RAGRetriever()
        from api.routes.dialogue import set_rag_retriever
        set_rag_retriever(retriever)
        print("RAG retriever initialized.")
    except Exception as e:
        print(f"RAG retriever not available: {e}")
        print("Running without RAG context injection.")


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run("api.main:app", host=host, port=port, reload=True)
