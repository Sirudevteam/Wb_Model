"""
Merge a PEFT LoRA adapter into a full Hugging Face model directory.

This is typically run once after training, on a GPU machine with enough VRAM/disk,
before converting the merged weights to GGUF (llama.cpp) for local runtimes like Ollama.

Usage:
  python training/merge_lora_to_hf.py \
    --base meta-llama/Llama-3.1-8B-Instruct \
    --adapter ./siru-dialogue-lora \
    --output ./merged-llama-3.1-8b-siru

Notes:
  - Requires HF access if the base model is gated (set HF_TOKEN in env / .env).
  - Merged export can be large; ensure you have enough free disk space.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from env_load import load_project_env


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge LoRA adapter into a full HF model directory")
    p.add_argument("--base", required=True, help="HF hub id or local path to the base model")
    p.add_argument("--adapter", required=True, help="Path to LoRA adapter folder (contains adapter_model.*)")
    p.add_argument("--output", required=True, help="Output directory for merged HF model")
    p.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
        help="Torch dtype for merged weights in RAM (float16 is a common default on GPU)",
    )
    return p.parse_args()


def torch_dtype(name: str):
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def main() -> None:
    load_project_env()
    args = parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    dtype = torch_dtype(args.dtype)

    print(f"Loading tokenizer: {args.base}")
    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)

    print(f"Loading base model: {args.base} ({args.dtype})")
    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Loading adapter: {args.adapter}")
    model = PeftModel.from_pretrained(model, args.adapter)

    print("Merging LoRA into base weights...")
    merged = model.merge_and_unload()

    print(f"Saving merged model to: {out_dir}")
    merged.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)

    print("Done.")


if __name__ == "__main__":
    main()
