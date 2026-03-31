# Siru AI Labs — Tamil Screenplay SLM

A Tamil dialogue rewrite engine: **QLoRA** on a **7B-class** instruction model (default **Qwen2.5-7B-Instruct**; optional **Meta Llama 3 8B**), plus **RAG** (Supabase pgvector) for context-aware output.

## Prerequisites

| Component | Notes |
| --------- | ----- |
| **Python** | 3.10+ recommended |
| **Training** | NVIDIA GPU, **CUDA** PyTorch, **~8GB VRAM** minimum for default QLoRA config |
| **Inference** | NVIDIA GPU recommended for local vLLM + LoRA (`inference/server.py`) |
| **Frontend** | Node.js 18+ for `frontend/` |

Install PyTorch with CUDA from [pytorch.org](https://pytorch.org) if you train or run inference locally. Verify: `python -c "import torch; print(torch.cuda.is_available())"`.

## Overview

Rewrites Tamil film dialogue in three modes:

- **Mass** — Short, punchy lines with authority and pauses  
- **Emotion** — Heartfelt, relationship-driven lines  
- **Subtext** — Surface meaning with deeper intent underneath  

## Architecture

```
Frontend (Next.js)
       │
  API (FastAPI)
       │
  Prompt Engine
       │
  RAG (Supabase pgvector)
       │
  SLM (7B-class base + LoRA)
       │
    Response
```

## Project structure

```
siru-slm/
  dataset/          Seeds, generation, filter/augment pipeline
  training/         QLoRA scripts, config — see training/README.md
  inference/        vLLM-compatible server + prompt engine
  api/              FastAPI rewrite + ideation routes
  rag/              Embeddings + Supabase retrieval
  frontend/         Next.js UI
  env_load.py       Loads workspace `.env` then `siru-slm/.env` (parent keys win)
  llm_client.py     Remote LLM abstraction (dataset + `/ideate/scene`)
```

## Environment variables

Copy [`.env.example`](.env.example) to **`Writers Block Model/.env`** (workspace root) and/or **`siru-slm/.env`**. Loading order: workspace `.env` first, then `siru-slm/.env`; duplicate keys keep the **workspace** value.

| Purpose | Variables |
| ------- | --------- |
| Remote LLM (dataset, augment, ideation) | `LLM_API_KEY`, `LLM_API_BASE`, `LLM_MODEL` — or `LLM_PROVIDER=replicate` + `REPLICATE_API_TOKEN` |
| Hugging Face (gated / downloads) | `HF_TOKEN` |
| Supabase RAG | `SUPABASE_URL`, `SUPABASE_KEY` |
| OpenAI (embeddings) | `OPENAI_API_KEY` |
| Inference + API wiring | `MODEL_PATH` (must match trained base), `LORA_PATH`, `SLM_BASE_URL`, ports — see `.env.example` |

## Remote LLM providers

Dataset scripts and `/ideate/scene` use [`llm_client.py`](llm_client.py):

| `LLM_PROVIDER` | Credentials | Notes |
| -------------- | ----------- | ----- |
| *(unset)* or `openai` | `LLM_API_KEY`, `LLM_API_BASE`, `LLM_MODEL` | Moonshot, OpenRouter, Together, etc. (OpenAI-compatible) |
| `replicate` | `REPLICATE_API_TOKEN` or `LLM_API_KEY`, `LLM_MODEL` | e.g. `meta/meta-llama-3-70b-instruct` — [Replicate](https://replicate.com) |

### Replicate example

```env
LLM_PROVIDER=replicate
REPLICATE_API_TOKEN=r8_...
LLM_MODEL=meta/meta-llama-3-70b-instruct
```

```bash
python dataset/generate.py --check-auth
```

## Quick start

Run shell commands from the **`siru-slm`** directory (or prefix paths with `siru-slm/` from the repo root).

### 1. Install dependencies

```bash
cd siru-slm
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env (and/or parent folder .env) with API keys — see Environment variables above
```

### 3. Generate dataset

```bash
python dataset/generate.py
python dataset/filter.py
python dataset/augment.py
```

### 4. Train the LoRA adapter (GPU)

Needs an **NVIDIA GPU + CUDA**. Optional until a GPU machine is available. See [Training (8GB GPU)](#training-8gb-gpu).

```bash
python training/format_dataset.py
python training/train.py
```

Evaluate the adapter (also GPU):

```bash
python training/evaluate.py
```

### 5. Inference + API

Terminal A — SLM server (vLLM-style OpenAI API):

```bash
python inference/server.py --mode openai --port 8001
```

Terminal B — FastAPI:

```bash
python api/main.py
```

Set `MODEL_PATH` to the **same base** you trained on (default in config: `Qwen/Qwen2.5-7B-Instruct`). Point `LORA_PATH` at `./siru-dialogue-lora` (or your `output_dir` from `training/config.yaml`).

### 6. Frontend

```bash
cd frontend
npm install
npm run dev
```

### 7. Smoke test

```bash
curl -X POST http://localhost:8000/rewrite/dialogue \
  -H "Content-Type: application/json" \
  -d '{"text": "Naan indha ooru la pudhusu", "mode": "mass"}'
```

## API endpoints

| Method | Path | Description |
| ------ | ---- | ----------- |
| POST | `/rewrite/dialogue` | General rewrite |
| POST | `/rewrite/mass` | Mass style |
| POST | `/rewrite/emotion` | Emotion style |
| POST | `/rewrite/subtext` | Subtext style |
| POST | `/ideate/scene` | Large remote model scene ideation |

## Product flow (dual model)

1. User enters a scene brief in the frontend.  
2. `/ideate/scene` calls the configured **remote** large model for a base scene.  
3. User uses **Add Mass** / **Add Emotion** / **Improve Dialogue**.  
4. `/rewrite/*` uses **Siru SLM** (LoRA on the local fine-tuned base).  
5. API merges **RAG** context, cache, and usage logging.  

## Testing and ops

```bash
python testing/internal_test.py
```

User testing: [`testing/user_testing_guide.md`](testing/user_testing_guide.md)

```bash
python ops/analyze_logs.py
```

Roadmap notes: [`ops/expansion_roadmap.md`](ops/expansion_roadmap.md)

## Troubleshooting: Moonshot / Kimi `401 Invalid Authentication`

Keys are **region-specific**. A key from [platform.moonshot.ai](https://platform.moonshot.ai) must use `LLM_API_BASE=https://api.moonshot.ai/v1`. A China-console key must use `https://api.moonshot.cn/v1`. Mixing key and base URL causes 401.

```bash
python dataset/generate.py --check-auth
```

On failure the script probes both `/v1/models` endpoints — set `LLM_API_BASE` to whichever returns HTTP 200. If **both** return 401, create a new key on the correct console, confirm billing/credits, and update `.env`.

## Training config

Full defaults: [`training/config.yaml`](training/config.yaml).

| Setting | Value |
| ------- | ----- |
| Base model | `Qwen/Qwen2.5-7B-Instruct` (optional: `meta-llama/Meta-Llama-3-8B` after HF access) |
| Method | QLoRA — 4-bit NF4, double quant |
| LoRA | r=16, α=32, targets `q/k/v/o_proj` |
| Schedule | 3 epochs, LR `2e-4`, cosine warmup |
| Batch (8GB) | `batch_size` 1, `gradient_accumulation_steps` 16 |
| Sequence | `max_seq_length` 512 (lower if OOM) |
| Data | `training/train_formatted.jsonl` + val split |

### Training (8GB GPU)

Defaults target **~8GB VRAM** (e.g. RTX 3060/4060 mobile, cloud T4). You still need **CUDA** PyTorch and an **NVIDIA** GPU.

- **OOM:** reduce `max_seq_length` first (e.g. `384` / `256`), then `batch_size` if it is ever above `1`.  
- **Larger GPUs:** raise `batch_size` and/or `max_seq_length` in `training/config.yaml`.  
- **Hugging Face:** gated models need `HF_TOKEN` in `.env` and approval on the model card.  
- **CPU:** not practical for this QLoRA recipe; use a GPU host.  

## License

Proprietary — Siru AI Labs
