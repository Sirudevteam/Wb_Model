"""
Dataset Augmentation for Siru AI Labs Tamil Screenplay SLM.

Takes the best samples from filtered.jsonl and generates further variations
via the 70B model to expand the training dataset.

Usage:
    python dataset/augment.py [--input dataset/filtered.jsonl] [--output dataset/final_train.jsonl]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from env_load import clean_env, load_project_env
from llm_client import chat_completion, get_llm_provider, openai_api_key, replicate_token

load_project_env()

console = Console()

AUGMENT_PROMPT = """You are an expert Tamil cinema dialogue writer.

Given this Tamil dialogue, create 3 variations that preserve the same emotion/style but use different words and structure. Each variation should feel like it could be from a different scene or character.

Category: {category}
Original: {text}

Rules:
- Stay in Tamil (Tanglish/transliterated is fine)
- Each variation must be a complete standalone dialogue line
- Maintain the {category_lower} style and tone
- Don't repeat the original

Output exactly 3 variations, one per line, numbered 1-3."""


def load_filtered(path: str) -> list[dict]:
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def parse_augmented(response_text: str) -> list[str]:
    import re
    lines = response_text.strip().split("\n")
    results = []
    for line in lines:
        line = line.strip()
        cleaned = re.sub(r"^\d+[\.\)]\s*", "", line)
        if cleaned and len(cleaned) > 5:
            results.append(cleaned)
    return results[:3]


def augment_sample(
    model: str,
    sample: dict,
    max_retries: int = 3,
) -> list[str]:
    prompt = AUGMENT_PROMPT.format(
        category=sample["category"],
        text=sample["variation"],
        category_lower=sample["category"].lower(),
    )

    for attempt in range(max_retries):
        try:
            text = chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert Tamil cinema dialogue writer. Respond only in Tamil.",
                    },
                    {"role": "user", "content": prompt},
                ],
                model=model,
                temperature=0.9,
                max_tokens=512,
            )
            return parse_augmented(text)
        except Exception as e:
            console.print(f"  [red]Attempt {attempt + 1} failed: {e}[/red]")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

    return []


def main():
    parser = argparse.ArgumentParser(description="Augment Tamil dialogue dataset")
    parser.add_argument("--input", default="dataset/filtered.jsonl", help="Filtered dataset")
    parser.add_argument("--output", default="dataset/final_train.jsonl", help="Output augmented dataset")
    parser.add_argument("--target", type=int, default=1000, help="Target total samples")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between API calls")
    args = parser.parse_args()

    if get_llm_provider() == "replicate":
        if not replicate_token():
            console.print("[red]Error: set REPLICATE_API_TOKEN or LLM_API_KEY for Replicate[/red]")
            sys.exit(1)
    elif not openai_api_key():
        console.print("[red]Error: LLM_API_KEY not set in .env[/red]")
        sys.exit(1)

    model = clean_env(os.getenv("LLM_MODEL")) or (
        "meta/meta-llama-3-70b-instruct" if get_llm_provider() == "replicate" else "kimi-k2.5"
    )
    samples = load_filtered(args.input)

    if not samples:
        console.print("[red]Error: No filtered samples found. Run filter.py first.[/red]")
        sys.exit(1)

    console.print(f"\n[bold green]Siru AI Labs -- Dataset Augmentor[/bold green]")
    console.print(f"Filtered samples: {len(samples)}")
    console.print(f"Target total: {args.target}")

    all_samples = list(samples)
    needed = max(0, args.target - len(all_samples))

    console.print(f"Need to generate: ~{needed} more samples\n")

    if needed == 0:
        console.print("[green]Already at target. No augmentation needed.[/green]")
    else:
        rounds = (needed // 3) + 1
        source_samples = samples * ((rounds // len(samples)) + 1)
        source_samples = source_samples[:rounds]

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
        ) as progress:
            task = progress.add_task("Augmenting...", total=rounds)

            for sample in source_samples:
                if len(all_samples) >= args.target:
                    break

                new_variations = augment_sample(model, sample)
                for var in new_variations:
                    if len(all_samples) >= args.target:
                        break
                    all_samples.append({
                        "category": sample["category"],
                        "original": sample.get("original", sample["variation"]),
                        "variation": var,
                    })

                progress.advance(task)
                time.sleep(args.delay)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in all_samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    category_counts = {}
    for item in all_samples:
        cat = item["category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1

    console.print(f"\n[bold green]Augmentation complete![/bold green]")
    console.print(f"Total samples: {len(all_samples)}")
    for cat, count in sorted(category_counts.items()):
        console.print(f"  {cat}: {count}")
    console.print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
