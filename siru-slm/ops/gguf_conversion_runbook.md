## GGUF Conversion Runbook (Merged HF → GGUF)

This repo trains a **PEFT LoRA adapter** (for example `./siru-dialogue-lora`). Tools like **Ollama** and **llama.cpp** typically want a **GGUF** file, so you do:

> **Hosting on Hugging Face instead?** To upload LoRA or merged weights to the Hub (no GGUF), use [`huggingface_hosting_guide.md`](huggingface_hosting_guide.md).

1. **Merge** LoRA + base → full Hugging Face model directory
2. **Convert** merged HF → **GGUF** using `llama.cpp`
3. Copy the `.gguf` to your **Windows** machine and import into **Ollama** (or run `llama.cpp` directly)

The merge step is implemented as:

- [`training/merge_lora_to_hf.py`](../training/merge_lora_to_hf.py)

## 1) Merge on a Linux GPU machine (recommended: your Vast instance)

From `siru-slm/`:

```bash
python3 training/merge_lora_to_hf.py \
  --base meta-llama/Llama-3.1-8B-Instruct \
  --adapter ./siru-dialogue-lora \
  --output ./merged-llama-3.1-8b-siru \
  --dtype float16
```

Requirements:

- `transformers`, `peft`, `torch` installed (your training environment usually already has these)
- `HF_TOKEN` set if the base model is gated

Outputs:

- `./merged-llama-3.1-8b-siru/` (a normal HF model folder)

## 2) Convert merged HF → GGUF using llama.cpp

`llama.cpp` changes script names over time, so the most reliable approach is:

1. Clone `llama.cpp` on the same Linux machine:

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
```

2. Install its python requirements (see `requirements.txt` inside `llama.cpp`):

```bash
python3 -m pip install -r requirements.txt
```

**Use a dedicated venv for conversion** (do not reuse a broken training stack). Mixing an old `torch` / `torchvision` with a newer `transformers` often produces errors like `partially initialized module 'torchvision'` or `Could not import module 'LlamaConfig'` when the converter loads the tokenizer.

```bash
python3 -m venv .venv-gguf
source .venv-gguf/bin/activate   # Windows: .venv-gguf\Scripts\activate
python3 -m pip install -U pip
python3 -m pip install -r requirements.txt
```

3. Locate the HF→GGUF converter script in your checkout:

```bash
cd llama.cpp
ls -1 convert*.py || true
python3 convert_hf_to_gguf.py --help
```

Notes:

- In many `llama.cpp` revisions, `convert_hf_to_gguf.py` lives at the **repository root**, not in `tools/`.
- Your `tools/` directory may contain **compiled binaries** (like `quantize`, `server`) after you build `llama.cpp`, which is separate from the Python conversion script.

If `convert_hf_to_gguf.py` is not at the repo root, search for it:

```bash
find . -maxdepth 3 -name 'convert_hf_to_gguf.py' -print
```

4. Run conversion pointing at your merged HF folder:

```bash
python3 convert_hf_to_gguf.py /workspace/Wb_Model/siru-slm/merged-llama-3.1-8b-siru \
  --outfile /workspace/Wb_Model/siru-slm/artifacts/siru-llama3.1-8b-siru.gguf \
  --outtype f16
```

Then apply quantization as a second step if your `llama.cpp` version uses a separate quant tool, for example a common target for 8GB GPUs is **`Q4_K_M`**.

Because quant flags differ by `llama.cpp` version, use the help output from the quantizer in your checkout.

### Troubleshooting (Llama 3.1 merged models)

1. **Confirm tokenizer files exist in the merged folder** (Llama 3.x uses the Hugging Face layout, not necessarily `tokenizer.model`):

   ```bash
   ls -1 merged-llama-3.1-8b-siru/tokenizer.json merged-llama-3.1-8b-siru/tokenizer_config.json merged-llama-3.1-8b-siru/config.json
   ```

   If `tokenizer.json` is missing, re-run [`training/merge_lora_to_hf.py`](../training/merge_lora_to_hf.py) (it calls `tokenizer.save_pretrained(out_dir)`), or copy the tokenizer files from your base model directory into the merged output.

2. **`FileNotFoundError: ... tokenizer.model` then `TypeError: Llama 3 must be converted with BpeVocab`**: this is normal for current `convert_hf_to_gguf.py` logic: it tries SentencePiece first, then falls through to the **GPT-2 / `tokenizer.json` (BPE)** path for Llama 3. You only have a problem if the next failure is inside `_set_vocab_gpt2` / `transformers`.

3. **`torchvision` / `LlamaConfig` import errors**: fix the **Python environment**, not the merge folder. Prefer the dedicated venv above; optionally `pip uninstall torchvision` in that venv if you do not need it (some `transformers` versions still probe optional video deps).

4. **Stale `llama.cpp`**: `git pull` inside your `llama.cpp` clone so `convert_hf_to_gguf.py` and `gguf-py` match current Llama 3.1 support.

## 3) Download the GGUF to Windows

Download:

- `siru-llama3.1-8b-siru.gguf` (name whatever you chose)

## 4) Import into Ollama (Windows)

Create a `Modelfile` referencing the GGUF, then:

```text
FROM ./siru-llama3.1-8b-siru.gguf
```

Then:

```powershell
ollama create siru-dialogue -f Modelfile
ollama run siru-dialogue
```

## Practical guidance for 8GB GPUs

For **8GB VRAM**, start with a **4-bit class** GGUF quantization (commonly `Q4_K_M`) before trying larger formats.

## Product integration note

Your FastAPI rewrite path currently expects an **OpenAI-compatible** server (vLLM). Ollama is compatible-ish but not identical across all features.

If you want “one button” local mode, the next code change is to add an inference backend switch (vLLM vs Ollama) in the API layer.
