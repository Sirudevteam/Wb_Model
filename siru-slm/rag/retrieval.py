"""
RAG Retrieval Pipeline for Siru AI Labs Tamil Screenplay SLM.

At inference time:
1. Embeds the user input
2. Searches top-k similar knowledge chunks from Supabase pgvector
3. Returns formatted context for prompt injection
"""

import json
import os
import sys
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from env_load import load_project_env

load_project_env()

EMBEDDING_MODEL = "text-embedding-3-small"
TABLE_NAME = "knowledge_embeddings"
TOP_K = 3


class RAGRetriever:
    def __init__(self):
        self.openai_client = None
        self.supabase_client = None
        self._local_fallback = None

        openai_key = (os.getenv("OPENAI_API_KEY") or "").strip()
        if openai_key:
            self.openai_client = OpenAI(api_key=openai_key)

        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")

        if supabase_url and supabase_key:
            try:
                from supabase import create_client
                self.supabase_client = create_client(supabase_url, supabase_key)
            except ImportError:
                pass

        if not self.supabase_client:
            self._load_local_fallback()

    def _load_local_fallback(self):
        """Load knowledge files directly for keyword-based fallback when Supabase is unavailable."""
        knowledge_dir = Path(__file__).parent / "knowledge"
        self._local_fallback = []

        for json_file in knowledge_dir.glob("*.json"):
            if json_file.name == "embeddings_preview.json":
                continue
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for item in data:
                    text_parts = []
                    for key in ["rule", "description", "example"]:
                        if key in item:
                            text_parts.append(item[key])
                    self._local_fallback.append({
                        "content": " ".join(text_parts),
                        "category": item.get("category", item.get("dialogue_style", item.get("archetype", "general"))),
                        "raw": item,
                    })
            except Exception:
                pass

    def _embed(self, text: str) -> list[float]:
        if not self.openai_client:
            return []
        response = self.openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=[text],
        )
        return response.data[0].embedding

    async def retrieve(self, query: str, mode: str, top_k: int = TOP_K) -> str | None:
        """Retrieve relevant knowledge context for prompt injection."""
        if self.supabase_client and self.openai_client:
            return await self._retrieve_vector(query, mode, top_k)
        elif self._local_fallback:
            return self._retrieve_keyword(query, mode, top_k)
        return None

    async def _retrieve_vector(self, query: str, mode: str, top_k: int) -> str | None:
        """Vector similarity search via Supabase pgvector."""
        try:
            embedding = self._embed(query)
            if not embedding:
                return None

            result = self.supabase_client.rpc(
                "match_knowledge",
                {
                    "query_embedding": embedding,
                    "match_count": top_k,
                    "filter_category": mode,
                },
            ).execute()

            if not result.data:
                return None

            context_parts = []
            for row in result.data:
                context_parts.append(row["content"])

            return "\n".join(context_parts)

        except Exception:
            return self._retrieve_keyword(query, mode, top_k)

    def _retrieve_keyword(self, query: str, mode: str, top_k: int) -> str | None:
        """Keyword-based fallback retrieval from local JSON files."""
        if not self._local_fallback:
            return None

        query_lower = query.lower()
        mode_lower = mode.lower()

        scored = []
        for item in self._local_fallback:
            score = 0.0
            content_lower = item["content"].lower()
            category_lower = str(item["category"]).lower()

            if mode_lower in category_lower or mode_lower in content_lower:
                score += 2.0

            query_words = query_lower.split()
            for word in query_words:
                if len(word) > 3 and word in content_lower:
                    score += 1.0

            if score > 0:
                scored.append((score, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_items = scored[:top_k]

        if not top_items:
            mode_items = [item for item in self._local_fallback if mode_lower in str(item["category"]).lower()]
            top_items = [(1.0, item) for item in mode_items[:top_k]]

        if not top_items:
            return None

        context_parts = [item["content"] for _, item in top_items]
        return "\n".join(context_parts)


SUPABASE_MATCH_FUNCTION_SQL = """
-- Run this in Supabase SQL Editor to create the match function:

CREATE OR REPLACE FUNCTION match_knowledge(
    query_embedding vector(1536),
    match_count int DEFAULT 3,
    filter_category text DEFAULT NULL
)
RETURNS TABLE (
    id text,
    content text,
    category text,
    source text,
    similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        ke.id,
        ke.content,
        ke.category,
        ke.source,
        1 - (ke.embedding <=> query_embedding) AS similarity
    FROM knowledge_embeddings ke
    WHERE
        filter_category IS NULL
        OR ke.category = filter_category
    ORDER BY ke.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;
"""
