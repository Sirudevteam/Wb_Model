"""
Dataset Formatter for Siru AI Labs Tamil Screenplay SLM.

Converts final_train.jsonl into instruction-tuning format suitable
for QLoRA fine-tuning with the SFTTrainer.

Usage:
    python training/format_dataset.py [--input dataset/final_train.jsonl] [--output training/train_formatted.jsonl]
"""

import argparse
import json
import random
from pathlib import Path

from rich.console import Console

console = Console()

INSTRUCTIONS = {
    "MASS": [
        "Rewrite this Tamil dialogue in mass style with punch and authority.",
        "Transform this dialogue into a powerful Tamil mass punch line.",
        "Make this Tamil dialogue hit harder -- add mass, pause, and impact.",
        "Rewrite as a Tamil hero introduction dialogue with full mass effect.",
        "Turn this into a whistle-worthy Tamil mass dialogue.",
    ],
    "EMOTION": [
        "Rewrite this Tamil dialogue with deep emotion and vulnerability.",
        "Transform this into a heartfelt Tamil emotional dialogue.",
        "Make this Tamil dialogue more emotionally powerful and authentic.",
        "Rewrite with the emotional depth of Tamil cinema's best moments.",
        "Turn this into a dialogue that makes the audience feel the pain.",
    ],
    "SUBTEXT": [
        "Rewrite this Tamil dialogue so it says one thing but means another.",
        "Add subtext to this Tamil dialogue -- hide the real meaning beneath the surface.",
        "Transform this into a layered Tamil dialogue with hidden emotional depth.",
        "Rewrite so the true meaning is felt, not heard directly.",
        "Make this Tamil dialogue carry a deeper truth beneath casual words.",
    ],
}


def format_as_instruction(sample: dict) -> dict:
    category = sample["category"]
    instruction = random.choice(INSTRUCTIONS.get(category, INSTRUCTIONS["MASS"]))

    return {
        "instruction": instruction,
        "input": sample["original"],
        "output": sample["variation"],
        "category": category,
    }


def format_as_chat(sample: dict) -> dict:
    """Alternative format for chat-style fine-tuning."""
    category = sample["category"]
    instruction = random.choice(INSTRUCTIONS.get(category, INSTRUCTIONS["MASS"]))

    return {
        "messages": [
            {
                "role": "system",
                "content": f"You are Siru, an expert Tamil screenplay dialogue writer specializing in {category.lower()} dialogues.",
            },
            {
                "role": "user",
                "content": f"{instruction}\n\nDialogue: {sample['original']}",
            },
            {
                "role": "assistant",
                "content": sample["variation"],
            },
        ],
        "category": category,
    }


def main():
    parser = argparse.ArgumentParser(description="Format dataset for instruction tuning")
    parser.add_argument("--input", default="dataset/final_train.jsonl", help="Input JSONL")
    parser.add_argument("--output", default="training/train_formatted.jsonl", help="Output formatted JSONL")
    parser.add_argument("--format", choices=["instruction", "chat"], default="instruction", help="Output format")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    input_path = Path(args.input)
    if not input_path.exists():
        console.print(f"[red]Error: {input_path} not found. Run augment.py first.[/red]")
        return

    samples = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    random.shuffle(samples)

    formatter = format_as_chat if args.format == "chat" else format_as_instruction
    formatted = [formatter(s) for s in samples]

    val_size = int(len(formatted) * args.val_split)
    train_data = formatted[val_size:]
    val_data = formatted[:val_size]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    val_path = output_path.with_name(output_path.stem + "_val" + output_path.suffix)
    with open(val_path, "w", encoding="utf-8") as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    category_counts = {}
    for item in train_data:
        cat = item.get("category", "unknown")
        category_counts[cat] = category_counts.get(cat, 0) + 1

    console.print(f"\n[bold green]Siru AI Labs -- Dataset Formatter[/bold green]")
    console.print(f"Format: {args.format}")
    console.print(f"Total samples: {len(formatted)}")
    console.print(f"Training: {len(train_data)}")
    console.print(f"Validation: {len(val_data)}")
    for cat, count in sorted(category_counts.items()):
        console.print(f"  {cat}: {count}")
    console.print(f"Output: {output_path}")
    console.print(f"Validation: {val_path}")


if __name__ == "__main__":
    main()
