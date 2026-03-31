"""
Evaluation Script for Siru AI Labs Tamil Screenplay SLM.

Tests the fine-tuned LoRA model against a set of prompts and optionally
compares outputs with the 70B baseline.

Usage:
    python training/evaluate.py [--config training/config.yaml]
    python training/evaluate.py --compare-70b  # Also generate 70B baseline outputs

Requires an NVIDIA GPU with CUDA for local model load; ~8GB VRAM matches default training config.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from rich.console import Console
from rich.table import Table
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from env_load import clean_env, load_project_env
from llm_client import chat_completion, get_llm_provider, has_remote_llm_credentials

load_project_env()

console = Console()

EVAL_PROMPTS = [
    {"category": "MASS", "input": "Naan indha ooru la pudhusu."},
    {"category": "MASS", "input": "Ennakku bayam illa."},
    {"category": "MASS", "input": "Naan yaaru kittayum kai yetha maaten."},
    {"category": "MASS", "input": "Avan enna ethirpaan nu theriyala."},
    {"category": "MASS", "input": "En vazhi thani vazhi."},
    {"category": "MASS", "input": "Naan pesura vaarthai thirumba varaadhu."},
    {"category": "MASS", "input": "Indha oorukku oru thalivan venum."},
    {"category": "MASS", "input": "Ennai thadukka yaaru varaporaanga."},
    {"category": "MASS", "input": "En kaiyil iruppavan safe."},
    {"category": "MASS", "input": "Oru vaarthai podhum."},
    {"category": "EMOTION", "input": "Nee illaamal enakku enna irukkudhu."},
    {"category": "EMOTION", "input": "Amma kitta poi sonnatha mannichidu."},
    {"category": "EMOTION", "input": "En vazhla nee mattum dhaan unmai."},
    {"category": "EMOTION", "input": "Kadhal nu theriyaamal kadhalichaen."},
    {"category": "EMOTION", "input": "Enna vittu pogaadha."},
    {"category": "EMOTION", "input": "Un ninaivula dhaan uyirodirukkaen."},
    {"category": "EMOTION", "input": "Oru vaarthai sollirundha naan poidirukka maaten."},
    {"category": "EMOTION", "input": "En kanneer unnakku theriyaadhu."},
    {"category": "EMOTION", "input": "Nee pogumbodhu en paathi poachu."},
    {"category": "EMOTION", "input": "Amma kaila adichaalum valikaadhu."},
    {"category": "SUBTEXT", "input": "Nalla irukka?"},
    {"category": "SUBTEXT", "input": "Coffee kudikkalaama."},
    {"category": "SUBTEXT", "input": "Un veedu azhagaa irukkudhu."},
    {"category": "SUBTEXT", "input": "Nee nalla aalu nu solraanga."},
    {"category": "SUBTEXT", "input": "Paravaillai naanum pazhakiduvaen."},
    {"category": "SUBTEXT", "input": "Veliya mazhai peiyudhu."},
    {"category": "SUBTEXT", "input": "Unna namburaen."},
    {"category": "SUBTEXT", "input": "Nallaa thoongu."},
    {"category": "SUBTEXT", "input": "Ippo thirumba sollu."},
    {"category": "SUBTEXT", "input": "Vazhkai azhagaa irukkudhu."},
]

INSTRUCTION_MAP = {
    "MASS": "Rewrite this Tamil dialogue in mass style with punch and authority.",
    "EMOTION": "Rewrite this Tamil dialogue with deep emotion and vulnerability.",
    "SUBTEXT": "Rewrite this Tamil dialogue so it says one thing but means another.",
}


def load_slm(config: dict):
    """Load the fine-tuned SLM (base + LoRA)."""
    model_cfg = config["model"]
    quant_cfg = config["quantization"]

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant_cfg["load_in_4bit"],
        bnb_4bit_compute_dtype=getattr(torch, quant_cfg["bnb_4bit_compute_dtype"]),
        bnb_4bit_quant_type=quant_cfg["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=quant_cfg["bnb_4bit_use_double_quant"],
    )

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["base"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["base"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    lora_path = model_cfg["output_dir"]
    if Path(lora_path).exists():
        console.print(f"[green]Loading LoRA from {lora_path}[/green]")
        model = PeftModel.from_pretrained(model, lora_path)
    else:
        console.print(f"[yellow]Warning: LoRA path {lora_path} not found. Using base model.[/yellow]")

    model.eval()
    return model, tokenizer


def generate_slm_output(model, tokenizer, prompt: dict, max_new_tokens: int = 256) -> str:
    instruction = INSTRUCTION_MAP[prompt["category"]]
    full_prompt = (
        f"### Instruction:\n{instruction}\n\n"
        f"### Input:\n{prompt['input']}\n\n"
        f"### Response:\n"
    )

    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
        )

    generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return generated.strip()


def generate_70b_output(model: str, prompt: dict) -> str:
    instruction = INSTRUCTION_MAP[prompt["category"]]
    try:
        return chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert Tamil cinema dialogue writer. Respond only with the rewritten dialogue in Tamil.",
                },
                {
                    "role": "user",
                    "content": f"{instruction}\n\nDialogue: {prompt['input']}",
                },
            ],
            model=model,
            temperature=0.7,
            max_tokens=256,
        )
    except Exception as e:
        return f"[Error: {e}]"


def main():
    parser = argparse.ArgumentParser(description="Evaluate Siru Tamil Dialogue SLM")
    parser.add_argument("--config", default="training/config.yaml", help="Config file")
    parser.add_argument("--output", default="training/eval_results.json", help="Output results")
    parser.add_argument("--compare-70b", action="store_true", help="Also generate 70B outputs for comparison")
    parser.add_argument("--slm-only", action="store_true", help="Only run SLM evaluation (skip 70B)")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    console.print(f"\n[bold green]Siru AI Labs -- Model Evaluation[/bold green]")
    console.print(f"Prompts: {len(EVAL_PROMPTS)}")

    console.print("\n[bold]Loading SLM...[/bold]")
    model, tokenizer = load_slm(config)

    model_70b = None
    if args.compare_70b:
        if has_remote_llm_credentials():
            model_70b = clean_env(os.getenv("LLM_MODEL")) or (
                "meta/meta-llama-3-70b-instruct" if get_llm_provider() == "replicate" else "kimi-k2.5"
            )
            console.print(f"[green]Remote LLM comparison enabled: {model_70b}[/green]")
        else:
            console.print("[yellow]No LLM credentials -- skipping remote comparison[/yellow]")

    results = []

    for i, prompt in enumerate(EVAL_PROMPTS):
        console.print(f"\n[cyan]--- Prompt {i + 1}/{len(EVAL_PROMPTS)} ({prompt['category']}) ---[/cyan]")
        console.print(f"[dim]Input:[/dim] {prompt['input']}")

        slm_output = generate_slm_output(model, tokenizer, prompt)
        console.print(f"[green]SLM:[/green] {slm_output}")

        result = {
            "index": i,
            "category": prompt["category"],
            "input": prompt["input"],
            "slm_output": slm_output,
        }

        if model_70b:
            output_70b = generate_70b_output(model_70b, prompt)
            console.print(f"[blue]70B:[/blue] {output_70b}")
            result["output_70b"] = output_70b
            time.sleep(1)

        results.append(result)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    table = Table(title="Evaluation Summary")
    table.add_column("Category", style="cyan")
    table.add_column("Count", style="green")
    category_counts = {}
    for r in results:
        cat = r["category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1
    for cat, count in sorted(category_counts.items()):
        table.add_row(cat, str(count))
    console.print(table)

    console.print(f"\n[bold green]Evaluation complete![/bold green]")
    console.print(f"Results saved to: {output_path}")
    console.print(f"\n[dim]Review the outputs manually for:[/dim]")
    console.print(f"  - Tamil feel and authenticity")
    console.print(f"  - Punch factor (mass) / emotional depth / subtext quality")
    console.print(f"  - Non-generic, cinema-quality output")


if __name__ == "__main__":
    main()
