"""Service wrapper around RAG retrieval dependencies."""

from __future__ import annotations


class RetrievalService:
    def __init__(self):
        self._retriever = None

    def set_retriever(self, retriever) -> None:
        self._retriever = retriever

    async def get_context(self, text: str, mode: str, enabled: bool) -> str | None:
        if not enabled or self._retriever is None:
            return None

        try:
            return await self._retriever.retrieve(text, mode)
        except Exception:
            return None

    @property
    def is_ready(self) -> bool:
        return self._retriever is not None

