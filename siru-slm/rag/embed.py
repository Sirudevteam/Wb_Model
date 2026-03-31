"""
Embedding Pipeline for Siru AI Labs RAG System.

Reads knowledge JSON files, generates embeddings using OpenAI's
text-embedding-3-small, and inserts them into Supabase pgvector.

Usage:
    python rag/embed.py [--reset]  # --reset drops and recreates the table
"""

import argparse
import json
import os
import sys
from pathlib import Path

from openai import OpenAI
from rich.console import Console
from rich.progress import Progress

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from env_load import load_project_env

load_project_env()

console = Console()

KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"
KNOWLEDGE_FILES = [
    "dialogue_rules.json",
    "scene_rules.json",
    "genre_rules.json",
    "character_archetypes.json",
]

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
TABLE_NAME = "knowledge_embeddings"


def load_knowledge() -> list[dict]:
    """Load all knowledge chunks from JSON files."""
    chunks = []
    for filename in KNOWLEDGE_FILES:
        path = KNOWLEDGE_DIR / filename
        if not path.exists():
            console.print(f"[yellow]Warning: {path} not found, skipping.[/yellow]")
            continue

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        source = filename.replace(".json", "")
        for item in data:
            text_parts = []
            for key in ["rule", "description", "example"]:
                if key in item:
                    text_parts.append(f"{key}: {item[key]}")

            content = " | ".join(text_parts)
            category = item.get("category", item.get("dialogue_style", item.get("archetype", "general")))

            chunks.append({
                "id": item["id"],
                "content": content,
                "category": category,
                "source": source,
                "raw": item,
            })

    return chunks


def get_embeddings(client: OpenAI, texts: list[str], batch_size: int = 50) -> list[list[float]]:
    """Generate embeddings in batches."""
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

    return all_embeddings


def setup_supabase_table(supabase_client, reset: bool = False):
    """Create the knowledge_embeddings table with pgvector."""
    if reset:
        console.print("[yellow]Dropping existing table...[/yellow]")
        supabase_client.rpc("exec_sql", {
            "query": f"DROP TABLE IF EXISTS {TABLE_NAME};"
        }).execute()

    create_sql = f"""
    CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
        id TEXT PRIMARY KEY,
        content TEXT NOT NULL,
        category TEXT,
        source TEXT,
        metadata JSONB,
        embedding vector({EMBEDDING_DIM})
    );

    CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_embedding
    ON {TABLE_NAME}
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 10);
    """

    try:
        supabase_client.rpc("exec_sql", {"query": create_sql}).execute()
        console.print("[green]Table created/verified.[/green]")
    except Exception as e:
        console.print(f"[yellow]Note: Table setup via RPC may need manual SQL execution: {e}[/yellow]")
        console.print(f"\n[bold]Run this SQL in Supabase SQL Editor:[/bold]\n")
        console.print(f"-- Enable pgvector extension")
        console.print(f"CREATE EXTENSION IF NOT EXISTS vector;\n")
        console.print(create_sql)


def insert_embeddings(supabase_client, chunks: list[dict], embeddings: list[list[float]]):
    """Insert knowledge chunks with embeddings into Supabase."""
    rows = []
    for chunk, embedding in zip(chunks, embeddings):
        rows.append({
            "id": chunk["id"],
            "content": chunk["content"],
            "category": chunk["category"],
            "source": chunk["source"],
            "metadata": json.dumps(chunk["raw"], ensure_ascii=False),
            "embedding": embedding,
        })

    batch_size = 50
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        try:
            supabase_client.table(TABLE_NAME).upsert(batch).execute()
        except Exception as e:
            console.print(f"[red]Error inserting batch {i//batch_size}: {e}[/red]")


def main():
    parser = argparse.ArgumentParser(description="Embed knowledge into Supabase pgvector")
    parser.add_argument("--reset", action="store_true", help="Drop and recreate table")
    parser.add_argument("--dry-run", action="store_true", help="Generate embeddings but don't insert")
    args = parser.parse_args()

    openai_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")

    if not openai_key:
        console.print("[red]Error: OPENAI_API_KEY not set in .env[/red]")
        sys.exit(1)

    openai_client = OpenAI(api_key=openai_key)

    console.print(f"\n[bold green]Siru AI Labs -- Knowledge Embedding Pipeline[/bold green]")

    chunks = load_knowledge()
    console.print(f"Knowledge chunks loaded: {len(chunks)}")

    texts = [chunk["content"] for chunk in chunks]

    console.print(f"Generating embeddings ({EMBEDDING_MODEL})...")
    with Progress() as progress:
        task = progress.add_task("Embedding...", total=1)
        embeddings = get_embeddings(openai_client, texts)
        progress.advance(task)

    console.print(f"Embeddings generated: {len(embeddings)}")

    if args.dry_run:
        console.print("[yellow]Dry run -- skipping Supabase insert.[/yellow]")
        console.print(f"Would insert {len(chunks)} rows.")
        return

    if not supabase_url or not supabase_key:
        console.print("[yellow]SUPABASE_URL/KEY not set. Saving embeddings locally instead.[/yellow]")
        output = []
        for chunk, emb in zip(chunks, embeddings):
            output.append({**chunk, "embedding": emb[:5]})  # truncated for readability
        output_path = KNOWLEDGE_DIR / "embeddings_preview.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        console.print(f"Preview saved to {output_path}")
        return

    from supabase import create_client
    supabase_client = create_client(supabase_url, supabase_key)

    setup_supabase_table(supabase_client, reset=args.reset)

    console.print("Inserting into Supabase...")
    insert_embeddings(supabase_client, chunks, embeddings)

    console.print(f"\n[bold green]Done![/bold green]")
    console.print(f"Inserted {len(chunks)} knowledge chunks into {TABLE_NAME}")


if __name__ == "__main__":
    main()
