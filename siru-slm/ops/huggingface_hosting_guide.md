# Hosting Siru models on Hugging Face

This guide covers publishing **weights** to the [Hugging Face Hub](https://huggingface.co/) and optionally running **managed inference** there. It complements the local **vLLM** path in the main README and the **GGUF** path in [`gguf_conversion_runbook.md`](gguf_conversion_runbook.md).

## What you can host

| Artifact | Hub repo contents | Best for |
| -------- | ----------------- | -------- |
| **LoRA adapter only** | `adapter_model.safetensors`, `adapter_config.json`, tokenizer files (`tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`), model card | Small uploads; consumers load **base + adapter** with PEFT. Matches a trimmed [`siru-dialogue-lora`](../siru-dialogue-lora/) export. |
| **Merged full model** | Full weight shards (`model*.safetensors` or `.bin`), `config.json`, tokenizer files — produced by [`training/merge_lora_to_hf.py`](../training/merge_lora_to_hf.py) | Easiest for **`from_pretrained("user/repo")`**, **Inference Endpoints**, and tools that expect a single causal LM directory. |

**Base model:** Training in this repo targets **`meta-llama/Llama-3.1-8B-Instruct`** (see [`training/config.yaml`](../training/config.yaml)). Document that base on every **adapter-only** model card so users know which weights to pair with your LoRA.

## Prerequisites

1. **Hugging Face account** — [sign up](https://huggingface.co/join).
2. **Access to gated bases** — For Llama, open the [base model page](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct), accept the license, then use a token with **read** access to pull weights during merge/train/inference.
3. **Upload token** — [Settings → Access Tokens](https://huggingface.co/settings/tokens): create a token with **write** (and read) to push model repos. Prefer **`huggingface-cli login`** on the machine that uploads; avoid pasting tokens into shell history where possible.

**Environment (local or VM):**

- `HF_TOKEN` — used by `huggingface_hub`, `transformers`, and `huggingface-cli` when set (same token often covers read + write depending on scope you selected).

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli login
# paste token when prompted; or: huggingface-cli login --token hf_... --add-to-git-credential-helper
```

## Option A — Push the LoRA adapter only

Your adapter folder should look like a minimal PEFT export (see [`siru-dialogue-lora/README.md`](../siru-dialogue-lora/README.md) for load code):

- `adapter_model.safetensors`
- `adapter_config.json`
- `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` (recommended for inference parity)

### Create a Hub model repo

1. In the browser: [Create new model repository](https://huggingface.co/new) (e.g. `your-username/siru-dialogue-lora`).
2. Choose **private** if you do not want public weights.

### Upload from disk

From **inside** the adapter directory (so files land at the repo root):

```bash
cd siru-dialogue-lora   # or path to your adapter export
huggingface-cli upload your-username/siru-dialogue-lora . --repo-type model
```

Or upload a subset explicitly:

```bash
huggingface-cli upload your-username/siru-dialogue-lora \
  ./adapter_model.safetensors ./adapter_config.json ./tokenizer.json \
  ./tokenizer_config.json ./chat_template.jinja ./README.md \
  --repo-type model
```

**Model card:** Edit `README.md` on the Hub (or commit locally before upload) to state:

- Base model id: `meta-llama/Llama-3.1-8B-Instruct`
- Task: Tamil dialogue rewrite (SFT / LoRA)
- Minimal Python example using `PeftModel.from_pretrained` (see existing adapter README in the repo)

### Load from Hub in Python

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "meta-llama/Llama-3.1-8B-Instruct"
ADAPTER = "your-username/siru-dialogue-lora"

tokenizer = AutoTokenizer.from_pretrained(ADAPTER, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    BASE, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
)
model = PeftModel.from_pretrained(model, ADAPTER)
```

## Option B — Push a merged full model

1. On a GPU machine with enough RAM/VRAM for merge (see [`training/merge_lora_to_hf.py`](../training/merge_lora_to_hf.py)):

   ```bash
   cd siru-slm
   python training/merge_lora_to_hf.py \
     --base meta-llama/Llama-3.1-8B-Instruct \
     --adapter ./siru-dialogue-lora \
     --output ./merged-llama-3.1-8b-siru \
     --dtype float16
   ```

2. Confirm **`tokenizer.json`** exists under the output directory (required for many downstream tools).

3. Upload the **entire** merged directory:

   ```bash
   cd merged-llama-3.1-8b-siru
   huggingface-cli upload your-username/siru-dialogue-merged . --repo-type model
   ```

Merged repos are the most straightforward input for **Inference Endpoints** if the UI expects a standard `LlamaForCausalLM` tree.

## Managed inference on Hugging Face (no Ollama)

### Inference Endpoints

1. Open your **model** repo on the Hub → **Deploy** → **Inference Endpoints** (wording may vary slightly in the UI).
2. Select a **GPU** sufficient for your model size (8B fp16 needs substantial VRAM; **quantized** serving or a smaller deploy image may be required — follow Hub prompts for supported handlers).
3. Point the endpoint at **your merged model repo** (or a supported server image + model id, per current HF documentation).

**Gated base:** If the endpoint runtime loads **`meta-llama/...`** or your private adapter, configure **repository secrets** / **HF token** on the endpoint so the container can authenticate.

### Spaces (Gradio demo)

Spaces are useful for **demos**, not always for production 8B full-precision on free tiers.

1. Create a **Space** (Docker or Gradio SDK).
2. Install `transformers`, `accelerate`, `torch`, `peft` (and optionally `bitsandbytes` for 4-bit loading).
3. `from_pretrained` your **merged** repo, or **base + adapter** as above.
4. Set Space **secrets** for `HF_TOKEN` if pulling gated models.

Refer to official docs: [Inference Endpoints](https://huggingface.co/docs/inference-endpoints), [Spaces](https://huggingface.co/docs/hub/spaces).

## Hugging Face vs this repo’s FastAPI + vLLM

| Approach | Role of Hub |
| -------- | ------------ |
| **This repo (default)** | Hub may only supply **downloads** (`HF_TOKEN`); inference runs on **your** GPU with **vLLM** + FastAPI. |
| **Hub-centric** | Hub stores **canonical weights**; you run inference on HF Endpoints/Spaces **or** still self-host but install weights with `from_pretrained("user/repo")`. |

You can use **both**: publish weights to the Hub for backup and collaboration, and still point `MODEL_PATH` / `LORA_PATH` at local paths or at Hub ids where vLLM and Transformers support it.

## Security checklist

- Do **not** commit `HF_TOKEN` or upload tokens into Git. Use `.env` (gitignored) or CI secrets.
- Prefer **private** Hub repos for unreleased adapters until you intend a public release.
- **Rotate** any token that was ever pasted into chat logs or screenshots.

## Related docs

- [`README.md`](../README.md) — local API + vLLM quick start  
- [`training/merge_lora_to_hf.py`](../training/merge_lora_to_hf.py) — merge LoRA into a full HF folder  
- [`gguf_conversion_runbook.md`](gguf_conversion_runbook.md) — GGUF for llama.cpp (separate from Hub hosting)  
- [`vast_ai_end_to_end_guide.md`](vast_ai_end_to_end_guide.md) — cloud training workflow; upload adapter after download  
