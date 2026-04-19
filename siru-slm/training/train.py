"""
QLoRA Training Script for Siru AI Labs Tamil Screenplay SLM.

Fine-tunes LLaMA 3 8B with QLoRA on Tamil dialogue data.

Usage:
    python training/train.py [--config training/config.yaml]

Requirements:
    - NVIDIA GPU with CUDA (~8GB VRAM is enough for default config; see README / training/config.yaml)
    - pip install transformers peft datasets trl bitsandbytes accelerate torch
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from env_load import load_project_env

load_project_env()


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_jsonl(path: str) -> list[dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def format_instruction_prompt(sample: dict) -> str:
    """Convert a sample into the instruction-following format for training."""
    return (
        f"### Instruction:\n{sample['instruction']}\n\n"
        f"### Input:\n{sample['input']}\n\n"
        f"### Response:\n{sample['output']}"
    )


def format_chat_prompt(sample: dict, tokenizer) -> str:
    """Convert a chat-format sample using the tokenizer's chat template."""
    if "messages" in sample:
        return tokenizer.apply_chat_template(
            sample["messages"], tokenize=False, add_generation_prompt=False
        )
    return format_instruction_prompt(sample)


def main():
    parser = argparse.ArgumentParser(description="Train Siru Tamil Dialogue LoRA")
    parser.add_argument("--config", default="training/config.yaml", help="Config file path")
    args = parser.parse_args()

    config = load_config(args.config)
    model_cfg = config["model"]
    lora_cfg = config["lora"]
    train_cfg = config["training"]
    quant_cfg = config["quantization"]
    data_cfg = config["data"]

    print(f"\n{'='*60}")
    print(f"  Siru AI Labs -- Tamil Dialogue LoRA Training")
    print(f"{'='*60}")
    print(f"  Base model:  {model_cfg['base']}")
    print(f"  LoRA rank:   {lora_cfg['r']}")
    print(f"  Epochs:      {train_cfg['epochs']}")
    print(f"  LR:          {train_cfg['learning_rate']}")
    print(f"  Output:      {model_cfg['output_dir']}")
    print(f"{'='*60}\n")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant_cfg["load_in_4bit"],
        bnb_4bit_compute_dtype=getattr(torch, quant_cfg["bnb_4bit_compute_dtype"]),
        bnb_4bit_quant_type=quant_cfg["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=quant_cfg["bnb_4bit_use_double_quant"],
    )

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["base"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("Loading base model (4-bit quantized)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["base"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    trainable_params, total_params = model.get_nb_trainable_parameters()
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
          f"({100 * trainable_params / total_params:.2f}%)\n")

    print("Loading dataset...")
    train_data = load_jsonl(data_cfg["train_file"])
    val_data = load_jsonl(data_cfg["val_file"]) if Path(data_cfg["val_file"]).exists() else []

    has_messages = "messages" in train_data[0] if train_data else False

    def formatting_func(example: dict) -> str:
        if has_messages:
            return format_chat_prompt(example, tokenizer)
        return format_instruction_prompt(example)

    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data) if val_data else None

    print(f"Training samples: {len(train_dataset)}")
    if val_dataset:
        print(f"Validation samples: {len(val_dataset)}")

    output_dir = model_cfg["output_dir"]

    def pick_mixed_precision() -> tuple[bool, bool]:
        """
        Choose safe AMP settings for Transformers + Accelerate.

        On many modern GPUs (Ampere+), bfloat16 training is supported and avoids fp16 grad-scaler
        edge cases with bf16-native checkpoints.
        """
        dtype_mode = str(train_cfg.get("dtype", "auto")).lower()

        if dtype_mode in {"fp32", "float32", "full", "full_fp32"}:
            return False, False
        if dtype_mode in {"bf16", "bfloat16"}:
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                return True, False
            print(
                "[WARN] dtype=bf16 requested, but this GPU does not advertise bf16 support. "
                "Falling back to fp32 (set dtype=fp16 to force fp16 on older GPUs).\n"
            )
            return False, False
        if dtype_mode in {"fp16", "float16"}:
            if not torch.cuda.is_available():
                print("[WARN] dtype=fp16 requested but CUDA is not available. Falling back to fp32.\n")
                return False, False
            return False, True

        # auto
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return True, False
        if torch.cuda.is_available():
            return False, True
        return False, False

    bf16, fp16 = pick_mixed_precision()
    print(f"Mixed precision: bf16={bf16}, fp16={fp16}")

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=train_cfg["epochs"],
        per_device_train_batch_size=train_cfg["batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        warmup_ratio=train_cfg["warmup_ratio"],
        bf16=bf16,
        fp16=fp16,
        logging_steps=train_cfg.get("logging_steps", 10),
        save_steps=train_cfg.get("save_steps", 50),
        eval_steps=train_cfg.get("eval_steps", 50) if val_dataset else None,
        eval_strategy="steps" if val_dataset else "no",
        save_total_limit=3,
        report_to="none",
        remove_unused_columns=False,
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        max_length=train_cfg["max_seq_length"],
        dataloader_pin_memory=torch.cuda.is_available(),
    )

    if not torch.cuda.is_available():
        print(
            "WARNING: No CUDA GPU detected. QLoRA + bitsandbytes expects an NVIDIA GPU; "
            "CPU training will be extremely slow or may fail. Install CUDA PyTorch and use a GPU machine.\n"
        )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        formatting_func=formatting_func,
        processing_class=tokenizer,
    )

    print("\nStarting training...")
    trainer.train()

    print(f"\nSaving LoRA adapter to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  LoRA adapter saved to: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
