"""
Dataset Generator for Siru AI Labs Tamil Screenplay SLM.

Reads seed dialogues from seeds.txt, calls LLaMA 70B API to generate
5 variations per seed using category-specific prompts, and outputs raw_outputs.json.

Usage:
    python dataset/generate.py [--seeds dataset/seeds.txt] [--output dataset/raw_outputs.json]
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from env_load import clean_env, load_project_env
from llm_client import chat_completion, check_llm_auth, get_llm_provider, openai_api_key, replicate_token

_env_file = load_project_env()

console = Console()

PROMPT_DIR = Path(__file__).parent / "prompts"
PROMPT_FILES = {
    "MASS": PROMPT_DIR / "mass_prompt.txt",
    "EMOTION": PROMPT_DIR / "emotion_prompt.txt",
    "SUBTEXT": PROMPT_DIR / "subtext_prompt.txt",
}


def load_prompts() -> dict[str, str]:
    prompts = {}
    for category, path in PROMPT_FILES.items():
        with open(path, "r", encoding="utf-8") as f:
            prompts[category] = f.read().strip()
    return prompts


def parse_seeds(seeds_path: str) -> list[dict]:
    seeds = []
    pattern = re.compile(r"^\[(MASS|EMOTION|SUBTEXT)\]\s+(.+)$")
    with open(seeds_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            match = pattern.match(line)
            if match:
                seeds.append({
                    "category": match.group(1),
                    "text": match.group(2),
                })
    return seeds


def parse_variations(response_text: str) -> list[str]:
    """Extract numbered variations from the model response."""
    lines = response_text.strip().split("\n")
    variations = []
    for line in lines:
        line = line.strip()
        cleaned = re.sub(r"^\d+[\.\)]\s*", "", line)
        if cleaned and len(cleaned) > 5:
            variations.append(cleaned)
    return variations[:5]


def generate_variations(
    model: str,
    seed: dict,
    prompt_template: str,
    max_retries: int = 3,
) -> list[str]:
    prompt = prompt_template.replace("{seed}", seed["text"])

    for attempt in range(max_retries):
        try:
            text = chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert Tamil cinema dialogue writer. Respond only in Tamil (Tanglish/transliterated Tamil is acceptable).",
                    },
                    {"role": "user", "content": prompt},
                ],
                model=model,
                temperature=0.8,
                max_tokens=1024,
            )
            variations = parse_variations(text)
            if variations:
                return variations
        except Exception as e:
            console.print(f"  [red]Attempt {attempt + 1} failed: {e}[/red]")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

    return []


def _mask_key(key: str) -> str:
    if len(key) <= 8:
        return "(too short)"
    return f"{key[:4]}...{key[-4:]}"


def _probe_moonshot_models(api_key: str) -> dict[str, int]:
    """GET /v1/models on both Moonshot bases. Returns {base_url: http_status}."""
    import urllib.error
    import urllib.request

    bases = [
        "https://api.moonshot.ai/v1",
        "https://api.moonshot.cn/v1",
    ]
    statuses: dict[str, int] = {}
    console.print("\n[bold]Regional endpoint probe[/bold] (GET /v1/models):")
    for base in bases:
        url = f"{base.rstrip('/')}/models"
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
        try:
            with urllib.request.urlopen(req, timeout=25) as resp:
                code = resp.status
                statuses[base] = code
                console.print(f"  [green]HTTP {code}[/green]  {base}")
        except urllib.error.HTTPError as e:
            statuses[base] = e.code
            console.print(f"  [red]HTTP {e.code}[/red]  {base}")
        except Exception as e:
            console.print(f"  [red]{e}[/red]  {base}")
    if statuses and all(c == 401 for c in statuses.values()):
        console.print(
            "\n[bold yellow]Both regions returned 401 — this is not a .ai vs .cn mix-up.[/bold yellow]\n"
            "The key is not accepted by Moonshot at all. Typical causes:\n"
            "  • Key was copied from another provider (OpenRouter, OpenAI, etc.) — use a key from\n"
            "    https://platform.moonshot.ai/console/api-keys (or .cn)\n"
            "  • Key revoked, expired, or typo — create a new secret key and paste the full string\n"
            "  • Account has no credits / billing not enabled — check the Moonshot console\n"
            "  • Wrong variable — ensure you put the secret in LLM_API_KEY in the loaded .env\n"
        )
    else:
        console.print(
            "[dim]International keys (platform.moonshot.ai) use .ai; China keys use .cn. "
            "Set LLM_API_BASE to the URL that returns 200.[/dim]"
        )
    return statuses


def check_auth():
    """Verify remote LLM credentials (OpenAI-compatible or Replicate)."""
    console.print(f"[dim]Env file priority: parent .env wins over siru-slm/.env for duplicate keys[/dim]")
    if _env_file:
        console.print(f"[dim]Shallowest loaded first: {_env_file}[/dim]")
    console.print(f"[dim]LLM_PROVIDER:[/dim] {get_llm_provider()}")
    if get_llm_provider() == "replicate":
        tok = replicate_token()
        console.print(f"[dim]REPLICATE_API_TOKEN / LLM_API_KEY:[/dim] {_mask_key(tok)} (len={len(tok)})")
        console.print(f"[dim]LLM_MODEL:[/dim] {clean_env(os.getenv('LLM_MODEL')) or 'meta/meta-llama-3-70b-instruct'}")
        if not tok:
            console.print("[red]Set REPLICATE_API_TOKEN or LLM_API_KEY for Replicate[/red]")
            sys.exit(1)
    else:
        api_key = openai_api_key()
        api_base = clean_env(os.getenv("LLM_API_BASE")) or "https://api.moonshot.ai/v1"
        model = clean_env(os.getenv("LLM_MODEL")) or "kimi-k2.5"
        console.print(f"[dim]LLM_API_BASE:[/dim] {api_base}")
        console.print(f"[dim]LLM_MODEL:[/dim] {model}")
        console.print(f"[dim]LLM_API_KEY:[/dim] {_mask_key(api_key)} (len={len(api_key)})")
        if not api_key:
            console.print("[red]LLM_API_KEY is empty. Fix parent .env or siru-slm/.env[/red]")
            sys.exit(1)

    ok, msg = check_llm_auth()
    if ok:
        console.print(f"[green]Auth OK. Sample reply:[/green] {msg!r}")
        return
    console.print(f"[red]Auth failed:[/red] {msg}")
    if get_llm_provider() != "replicate":
        _probe_moonshot_models(openai_api_key())
        console.print(
            "\n[yellow]If exactly one endpoint returned 200 above: set LLM_API_BASE to that base URL and run again.[/yellow]"
        )
    else:
        console.print(
            "[yellow]Check replicate.com account, token (r8_...), and that LLM_MODEL exists on Replicate.[/yellow]"
        )
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Generate Tamil dialogue variations")
    parser.add_argument("--seeds", default="dataset/seeds.txt", help="Path to seeds file")
    parser.add_argument("--output", default="dataset/raw_outputs.json", help="Output path")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of seeds to process")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between API calls (seconds)")
    parser.add_argument("--check-auth", action="store_true", help="Verify LLM API key with one test call, then exit")
    args = parser.parse_args()

    if args.check_auth:
        check_auth()
        return

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
    prompts = load_prompts()
    seeds = parse_seeds(args.seeds)

    if args.limit:
        seeds = seeds[:args.limit]

    console.print(f"\n[bold green]Siru AI Labs -- Dataset Generator[/bold green]")
    if _env_file:
        console.print(f"[dim]Loaded env: {_env_file}[/dim]")
    console.print(f"Seeds loaded: {len(seeds)}")
    console.print(f"Provider: {get_llm_provider()} | Model: {model}")
    console.print(f"Expected output: ~{len(seeds) * 5} variations\n")

    results = []
    failed = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
    ) as progress:
        task = progress.add_task("Generating variations...", total=len(seeds))

        for i, seed in enumerate(seeds):
            category = seed["category"]
            prompt_template = prompts.get(category)

            if not prompt_template:
                console.print(f"[yellow]Skipping unknown category: {category}[/yellow]")
                progress.advance(task)
                continue

            variations = generate_variations(model, seed, prompt_template)

            if variations:
                results.append({
                    "seed_index": i,
                    "category": category,
                    "original": seed["text"],
                    "variations": variations,
                })
            else:
                failed += 1
                console.print(f"  [yellow]No variations for seed {i}[/yellow]")

            progress.advance(task)
            time.sleep(args.delay)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    total_variations = sum(len(r["variations"]) for r in results)
    console.print(f"\n[bold green]Done![/bold green]")
    console.print(f"Seeds processed: {len(results)}")
    console.print(f"Total variations: {total_variations}")
    console.print(f"Failed: {failed}")
    console.print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
