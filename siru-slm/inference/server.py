"""
vLLM Inference Server for Siru AI Labs Tamil Screenplay SLM.

Loads the base causal LM (e.g. Qwen2.5-7B-Instruct or Meta Llama 3 8B) with the fine-tuned LoRA adapter
and serves it via vLLM for high-throughput inference.

Usage:
    python inference/server.py [--host 0.0.0.0] [--port 8001]

Note: This is the raw model server. The FastAPI backend (api/main.py)
wraps this with the prompt engine and RAG for the full product API.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from env_load import load_project_env

load_project_env()


def start_vllm_server(
    model_path: str,
    lora_path: str | None,
    host: str = "0.0.0.0",
    port: int = 8001,
    max_model_len: int = 2048,
):
    """Start the vLLM OpenAI-compatible server with LoRA."""
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    lora_request = None
    enable_lora = False

    if lora_path and Path(lora_path).exists():
        enable_lora = True
        print(f"LoRA adapter found at: {lora_path}")
    else:
        print(f"No LoRA adapter found. Running base model only.")

    print(f"\nLoading model: {model_path}")
    print(f"LoRA enabled: {enable_lora}")
    print(f"Max model length: {max_model_len}")

    llm = LLM(
        model=model_path,
        enable_lora=enable_lora,
        max_lora_rank=16,
        max_model_len=max_model_len,
        trust_remote_code=True,
        dtype="half",
    )

    if enable_lora:
        lora_request = LoRARequest("siru-dialogue", 1, lora_path)

    return llm, lora_request


def generate(
    llm,
    prompt: str,
    lora_request=None,
    max_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
):
    """Generate a single response from the model."""
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        repetition_penalty=1.1,
    )

    outputs = llm.generate(
        [prompt],
        sampling_params,
        lora_request=lora_request,
    )

    return outputs[0].outputs[0].text.strip()


def start_openai_compatible_server(
    model_path: str,
    lora_path: str | None,
    host: str,
    port: int,
):
    """Start vLLM's built-in OpenAI-compatible API server."""
    import subprocess
    import sys

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--host", host,
        "--port", str(port),
        "--dtype", "half",
        "--max-model-len", "2048",
        "--trust-remote-code",
    ]

    if lora_path and Path(lora_path).exists():
        cmd.extend([
            "--enable-lora",
            "--max-lora-rank", "16",
            "--lora-modules", f"siru-dialogue={lora_path}",
        ])

    print(f"\nStarting vLLM OpenAI-compatible server...")
    print(f"Host: {host}:{port}")
    print(f"Model: {model_path}")
    if lora_path:
        print(f"LoRA: {lora_path}")
    print(f"\nServer URL: http://{host}:{port}/v1")
    print(f"Use model name 'siru-dialogue' in API calls.\n")

    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser(description="Siru SLM Inference Server")
    parser.add_argument("--host", default=os.getenv("INFERENCE_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("INFERENCE_PORT", "8001")))
    parser.add_argument(
        "--model",
        default=os.getenv("MODEL_PATH", "Qwen/Qwen2.5-7B-Instruct"),
    )
    parser.add_argument("--lora", default=os.getenv("LORA_PATH", "./siru-dialogue-lora"))
    parser.add_argument(
        "--mode",
        choices=["openai", "standalone"],
        default="openai",
        help="'openai' starts the vLLM OpenAI-compatible server; 'standalone' loads model for direct use",
    )
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Siru AI Labs -- Inference Server")
    print(f"{'='*60}")

    if args.mode == "openai":
        start_openai_compatible_server(args.model, args.lora, args.host, args.port)
    else:
        llm, lora_request = start_vllm_server(args.model, args.lora, args.host, args.port)
        print(f"\nModel loaded. Ready for inference.")
        print(f"Use generate(llm, prompt, lora_request) to generate responses.")


if __name__ == "__main__":
    main()
