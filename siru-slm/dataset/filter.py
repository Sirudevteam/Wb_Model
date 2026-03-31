"""
Manual Filtering Tool for Siru AI Labs Tamil Screenplay SLM.

Loads raw_outputs.json and presents each variation for accept/reject.
Outputs filtered.jsonl with only the approved samples.

Usage:
    python dataset/filter.py [--input dataset/raw_outputs.json] [--output dataset/filtered.jsonl]
    python dataset/filter.py --auto  # Auto-filter using basic heuristics (no manual review)
"""

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

console = Console()


def auto_filter(variation: str) -> bool:
    """Basic heuristic filter -- reject obvious garbage."""
    if len(variation) < 10:
        return False
    if len(variation) > 500:
        return False
    if variation.count("...") > 5:
        return False

    english_chars = sum(1 for c in variation if c.isascii() and c.isalpha())
    total_alpha = sum(1 for c in variation if c.isalpha())
    if total_alpha > 0 and english_chars / total_alpha > 0.8:
        return False

    garbage_markers = ["sorry", "i can't", "as an ai", "here are", "note:", "explanation"]
    lower = variation.lower()
    if any(marker in lower for marker in garbage_markers):
        return False

    return True


def manual_filter(raw_data: list[dict]) -> list[dict]:
    """Interactive terminal-based filtering."""
    filtered = []
    total = sum(len(entry["variations"]) for entry in raw_data)
    current = 0

    console.print(Panel(
        f"[bold]Manual Filtering Mode[/bold]\n"
        f"Total variations to review: {total}\n\n"
        f"Commands:\n"
        f"  [green]y[/green] / Enter = Accept\n"
        f"  [red]n[/red] = Reject\n"
        f"  [yellow]a[/yellow] = Accept all from this seed\n"
        f"  [yellow]s[/yellow] = Skip all from this seed\n"
        f"  [blue]q[/blue] = Quit and save",
        title="Siru AI Labs -- Dataset Filter",
    ))

    for entry in raw_data:
        console.print(f"\n[bold cyan]── Seed ({entry['category']}) ──[/bold cyan]")
        console.print(f"[dim]Original:[/dim] {entry['original']}\n")

        skip_seed = False
        accept_all = False

        for var in entry["variations"]:
            current += 1

            if skip_seed:
                continue

            if accept_all:
                filtered.append({
                    "category": entry["category"],
                    "original": entry["original"],
                    "variation": var,
                })
                continue

            console.print(f"  [{current}/{total}] {var}")
            choice = Prompt.ask(
                "  Accept?",
                choices=["y", "n", "a", "s", "q"],
                default="y",
            )

            if choice == "q":
                console.print("[yellow]Saving and quitting...[/yellow]")
                return filtered
            elif choice == "s":
                skip_seed = True
                continue
            elif choice == "a":
                accept_all = True
                filtered.append({
                    "category": entry["category"],
                    "original": entry["original"],
                    "variation": var,
                })
            elif choice == "y":
                filtered.append({
                    "category": entry["category"],
                    "original": entry["original"],
                    "variation": var,
                })

    return filtered


def main():
    parser = argparse.ArgumentParser(description="Filter generated Tamil dialogue variations")
    parser.add_argument("--input", default="dataset/raw_outputs.json", help="Input raw outputs")
    parser.add_argument("--output", default="dataset/filtered.jsonl", help="Output filtered JSONL")
    parser.add_argument("--auto", action="store_true", help="Auto-filter using heuristics (skip manual review)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        console.print(f"[red]Error: {input_path} not found. Run generate.py first.[/red]")
        sys.exit(1)

    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    total_variations = sum(len(entry["variations"]) for entry in raw_data)
    console.print(f"\n[bold green]Siru AI Labs -- Dataset Filter[/bold green]")
    console.print(f"Loaded: {len(raw_data)} seeds, {total_variations} variations")

    if args.auto:
        console.print("[yellow]Running auto-filter (heuristic mode)...[/yellow]\n")
        filtered = []
        for entry in raw_data:
            for var in entry["variations"]:
                if auto_filter(var):
                    filtered.append({
                        "category": entry["category"],
                        "original": entry["original"],
                        "variation": var,
                    })
    else:
        filtered = manual_filter(raw_data)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in filtered:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    keep_rate = (len(filtered) / total_variations * 100) if total_variations > 0 else 0
    console.print(f"\n[bold green]Filtering complete![/bold green]")
    console.print(f"Accepted: {len(filtered)} / {total_variations} ({keep_rate:.1f}%)")
    console.print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
