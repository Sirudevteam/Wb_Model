# Training (Siru SLM)

QLoRA fine-tuning scripts for Tamil dialogue. **Hardware:** NVIDIA GPU with CUDA; defaults in [`config.yaml`](config.yaml) target **~8GB VRAM** (adjust `batch_size`, `gradient_accumulation_steps`, and `max_seq_length` if you OOM or have more headroom).

| Script | Purpose |
| ------ | ------- |
| [`format_dataset.py`](format_dataset.py) | Build `train_formatted.jsonl` (+ val) from the dataset pipeline |
| [`train.py`](train.py) | QLoRA training; reads [`config.yaml`](config.yaml) |
| [`evaluate.py`](evaluate.py) | Load adapter and run eval prompts |
| [`merge_lora_to_hf.py`](merge_lora_to_hf.py) | Merge adapter + base into one HF folder (optional before Hub upload) |

Full instructions, Hugging Face gating, and troubleshooting: [main README § Training (8GB GPU)](../README.md#training-8gb-gpu).

**Publishing to Hugging Face Hub** (adapter or merged weights, Inference Endpoints, Spaces): [Hugging Face hosting guide](../ops/huggingface_hosting_guide.md).
