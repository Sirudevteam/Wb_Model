## Vast.ai End-to-End Training Guide

This guide covers the full workflow for `siru-slm`:
- prepare the dataset locally
- upload the project to a Vast.ai GPU machine
- train the LoRA adapter in the cloud
- evaluate the adapter
- download the trained artifacts
- run and test the model locally

The workflow is designed for the current repo structure and scripts.

## 1. Recommended Hardware

For this project, use:
- Minimum: `1 x T4 16GB`
- Recommended: `1 x L4 24GB` or `1 x A10 24GB`

Do not use:
- CPU-only machines
- 6GB GPUs for the current 7B/8B QLoRA setup

## 2. What You Will Do Locally vs In The Cloud

### Local machine
Use your local machine for:
- editing config
- dataset generation
- filtering and augmentation
- formatting the training set
- downloading the final adapter
- local inference and testing after training

### Vast.ai machine
Use the cloud GPU only for:
- dependency install
- LoRA training
- evaluation
- packaging artifacts

This keeps GPU rental time low and cost-effective.

## 3. Local Preparation

Run all local commands from:

```powershell
cd "g:\Project Plans\Writers Block Model\siru-slm"
```

### 3.1 Install dependencies locally

```powershell
pip install -r requirements.txt
```

### 3.2 Configure environment

Create or update your workspace `.env` file.

Important values:

```env
HF_TOKEN=hf_your_token_here
LLM_API_KEY=your_remote_llm_key_if_needed
LLM_API_BASE=https://api.moonshot.ai/v1
LLM_MODEL=kimi-k2.5
MODEL_PATH=meta-llama/Llama-3.1-8B-Instruct
LORA_PATH=./siru-dialogue-lora
```

Notes:
- `HF_TOKEN` is required for gated Hugging Face models such as Llama.
- `MODEL_PATH` should match the exact base model used for training.
- If you are generating the dataset with a remote LLM, ensure the remote provider variables are valid.

### 3.3 Generate and prepare the dataset

Run:

```powershell
python dataset/generate.py
python dataset/filter.py
python dataset/augment.py
python training/format_dataset.py
```

Expected outputs:
- `dataset/raw_outputs.json`
- `dataset/filtered.jsonl`
- `dataset/final_train.jsonl`
- `training/train_formatted.jsonl`
- `training/train_formatted_val.jsonl`

### 3.4 Review the dataset before paying for cloud GPU

Check:
- outputs are Tamil or Tanglish as intended
- mode balance across `mass`, `emotion`, and `subtext`
- no empty or low-quality outputs
- no repeated junk rows

Minimum spot-check:
- open `training/train_formatted.jsonl`
- read at least 50 to 100 random samples

### 3.5 Set the base model in training config

Edit `training/config.yaml`.

Example for Llama 3.1 8B:

```yaml
model:
  base: "meta-llama/Llama-3.1-8B-Instruct"
  output_dir: "./siru-dialogue-lora"

training:
  epochs: 3
  learning_rate: 2.0e-4
  batch_size: 1
  gradient_accumulation_steps: 16
  warmup_ratio: 0.03
  max_seq_length: 384
  fp16: true
```

Recommended starting point:
- start with `max_seq_length: 384`
- only move to `512` if the GPU has comfortable headroom

## 4. Create A Minimal Upload Package

To reduce upload time, you do not need to send unnecessary files.

Keep these:
- `training/`
- `dataset/` outputs used by training
- `api/`, `inference/`, `rag/` if you want evaluation and local parity
- `.env.example`
- `.env` only if you are comfortable placing secrets on the cloud box
- `requirements.txt`
- `env_load.py`
- `llm_client.py`

At minimum for training, you need:
- `training/train.py`
- `training/evaluate.py`
- `training/format_dataset.py`
- `training/config.yaml`
- `training/train_formatted.jsonl`
- `training/train_formatted_val.jsonl`
- `requirements.txt`
- `env_load.py`

## 5. Vast.ai Setup

### 5.1 Create an account

1. Sign in to [Vast.ai](https://vast.ai/)
2. Add billing
3. Open the instance search page

### 5.2 Choose an instance

Search for:
- `CUDA` enabled image
- `PyTorch` or `NVIDIA CUDA` ready environment
- `16GB+ VRAM`
- `40GB+ disk`
- stable internet and decent reliability score

Recommended filters:
- GPU RAM: `>= 16 GB`
- Disk: `>= 40 GB`
- CUDA available
- SSH enabled

Good choices:
- `T4 16GB` for budget
- `L4 24GB` or `A10 24GB` for smoother runs

### 5.3 Start the instance

Use a standard PyTorch/CUDA image if available.

If the image allows Jupyter and SSH, SSH is usually easier for scripted runs.

## 6. Upload Project To Vast.ai

You have two common options.

### Option A: Push code to GitHub and clone on the VM

On the Vast.ai machine:

```bash
git clone https://github.com/Sirudevteam/Wb_Model.git
cd Wb_Model/siru-slm
```

Then upload or copy only the generated training files if they are not yet pushed:
- `training/train_formatted.jsonl`
- `training/train_formatted_val.jsonl`

### Option B: Zip locally and upload

From Windows PowerShell:

```powershell
Compress-Archive -Path .\* -DestinationPath ..\siru-slm-vast-upload.zip
```

Upload the zip through Vast.ai file transfer or SCP, then on the VM:

```bash
unzip siru-slm-vast-upload.zip
cd siru-slm
```

## 7. Cloud Machine Setup

SSH into the Vast.ai machine, then run:

```bash
cd siru-slm
python3 --version
python3 -m pip install --upgrade pip
pip install -r requirements.txt
```

If needed, verify GPU:

```bash
nvidia-smi
python3 -c "import torch; print(torch.cuda.is_available())"
```

Expected result:
- `nvidia-smi` shows your GPU
- Python prints `True`

## 8. Configure Secrets On The Cloud Machine

Create `.env` on the VM if needed:

```bash
cp .env.example .env
```

Set:

```env
HF_TOKEN=hf_your_token_here
MODEL_PATH=meta-llama/Llama-3.1-8B-Instruct
LORA_PATH=./siru-dialogue-lora
```

Only add remote LLM credentials if you still need them on the VM. For pure training from a finished dataset, they are usually unnecessary.

## 9. Run Training On Vast.ai

From the project root on the VM:

```bash
python3 training/train.py
```

If you want logs saved:

```bash
python3 training/train.py 2>&1 | tee training_run.log
```

What to monitor:
- GPU memory in `nvidia-smi`
- training loss in logs
- step progress
- checkpoint saves

## 10. If Training Fails

### Out of memory

Lower these in `training/config.yaml`:

1. `max_seq_length` from `512` to `384`
2. then to `256` if needed
3. keep `batch_size: 1`
4. optionally reduce checkpoint frequency

### Hugging Face access error

If you see `403` or gated model access errors:
- confirm the model page license is accepted
- confirm `HF_TOKEN` is valid on the VM
- confirm the model ID is exact

For Llama 3.1 8B instruct, use:

```text
meta-llama/Llama-3.1-8B-Instruct
```

### Slow training

This usually means:
- weak GPU
- too large sequence length
- using a crowded or slow instance

Move from `T4` to `L4` or `A10` if training speed is unacceptable.

## 11. Evaluate After Training

Run:

```bash
python3 training/evaluate.py
```

Also inspect:
- final training logs
- saved adapter directory
- a few manual generations

If you want a quick sanity check, compare:
- style adherence
- instruction following
- output fluency
- mode differences between `mass`, `emotion`, and `subtext`

## 12. Save Artifacts Before Stopping The VM

Important outputs usually include:
- `siru-dialogue-lora/`
- `training_run.log`
- copy of `training/config.yaml`
- any evaluation outputs

Recommended run structure:

```text
artifacts/
  run-001/
    siru-dialogue-lora/
    config.yaml
    training_run.log
    notes.md
```

Example:

```bash
mkdir -p artifacts/run-001
cp -r siru-dialogue-lora artifacts/run-001/
cp training/config.yaml artifacts/run-001/
cp training_run.log artifacts/run-001/ 2>/dev/null || true
```

## 13. Download Artifacts Back To Your Local Machine

Use SCP, SFTP, or Vast.ai file transfer.

For example, download:
- `artifacts/run-001/siru-dialogue-lora`

Place it locally under:

```text
g:\Project Plans\Writers Block Model\siru-slm\siru-dialogue-lora
```

## 14. Run The Trained Model Locally

Back on your Windows machine:

```powershell
cd "g:\Project Plans\Writers Block Model\siru-slm"
python inference/server.py --mode openai --port 8001 --model meta-llama/Llama-3.1-8B-Instruct --lora .\siru-dialogue-lora
```

In another terminal:

```powershell
python api/main.py
```

If you want the UI:

```powershell
cd frontend
npm install
npm run dev
```

## 15. Test The Trained Model

### API test

```powershell
curl -X POST http://localhost:8000/rewrite/dialogue `
  -H "Content-Type: application/json" `
  -d "{\"text\":\"Naan indha ooru la pudhusu\",\"mode\":\"mass\"}"
```

Expected shape:

```json
{
  "success": true,
  "data": {
    "original": "Naan indha ooru la pudhusu",
    "rewritten": "...",
    "mode": "mass",
    "rag_context_used": null
  },
  "error": null
}
```

### Internal evaluation script

You can also run:

```powershell
python testing/internal_test.py
```

This helps compare behavior across prompt sets.

## 16. Suggested First Training Strategy

To control cost and risk:

1. generate and format the full dataset locally
2. upload the project and formatted data
3. run a pilot training on a smaller subset first
4. inspect outputs and loss behavior
5. train on the full set once the config is stable

Good pilot size:
- `500` to `2,000` formatted examples

Then scale to the full dataset.

## 17. Recommended Versioning Discipline

For each cloud run, record:
- date
- base model
- GPU type
- dataset size
- epoch count
- sequence length
- output directory
- notes on quality

This is important because LoRA training is iterative and small config changes can materially affect quality.

## 18. Cost Optimization Tips

- do dataset generation locally, not on rented GPU
- upload only what you need
- start with one GPU
- run a pilot subset before the full run
- stop the VM immediately after downloading artifacts
- prefer `L4` or `A10` if they reduce total wall-clock time enough to offset hourly price

## 19. Recommended Exact Workflow For This Repo

For `siru-slm`, the most practical path is:

1. prepare dataset locally
2. format training set locally
3. use `meta-llama/Llama-3.1-8B-Instruct` or `Qwen/Qwen2.5-7B-Instruct`
4. rent `1 x 16GB` or `1 x 24GB` GPU on Vast.ai
5. train with `python3 training/train.py`
6. evaluate with `python3 training/evaluate.py`
7. download `siru-dialogue-lora`
8. run local inference server and API
9. test through API and UI
10. *(optional)* publish the adapter or a merged model to the **Hugging Face Hub** — see [`huggingface_hosting_guide.md`](huggingface_hosting_guide.md) (`huggingface-cli upload`, private vs public repos, gated Llama tokens).

## 20. Final Checklist

Before renting:
- dataset prepared
- formatted JSONL files exist
- config checked
- Hugging Face access confirmed

Before training:
- `nvidia-smi` works
- PyTorch sees CUDA
- `.env` is correct

Before stopping the VM:
- adapter saved
- logs downloaded
- config copied
- evaluation run completed

