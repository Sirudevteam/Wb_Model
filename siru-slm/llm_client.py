"""
Unified chat completion for OpenAI-compatible HTTP APIs and Replicate.

Set ``LLM_PROVIDER=replicate`` and ``REPLICATE_API_TOKEN`` (or ``LLM_API_KEY``) to use
Replicate models (default model: ``meta/meta-llama-3-70b-instruct``).

Leave ``LLM_PROVIDER`` unset or ``openai`` for Moonshot / OpenRouter / any OpenAI-compatible base.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from openai import OpenAI

from env_load import clean_env

DEFAULT_OPENAI_BASE = "https://api.moonshot.ai/v1"
DEFAULT_OPENAI_MODEL = "kimi-k2.5"
DEFAULT_REPLICATE_MODEL = "meta/meta-llama-3-70b-instruct"


def get_llm_provider() -> str:
    p = clean_env(os.getenv("LLM_PROVIDER")).lower()
    if p == "replicate":
        return "replicate"
    return "openai"


def replicate_token() -> str:
    return clean_env(os.getenv("REPLICATE_API_TOKEN") or os.getenv("LLM_API_KEY"))


def openai_api_key() -> str:
    return clean_env(os.getenv("LLM_API_KEY"))


def has_remote_llm_credentials() -> bool:
    if get_llm_provider() == "replicate":
        return bool(replicate_token())
    return bool(openai_api_key())


def _collect_replicate_output(out: Any) -> str:
    if out is None:
        return ""
    if isinstance(out, str):
        return out
    if isinstance(out, (list, tuple)):
        return "".join(str(x) for x in out)
    try:
        return "".join(str(x) for x in out)
    except TypeError:
        return str(out)


def _replicate_run(
    model_id: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
) -> str:
    import replicate

    token = replicate_token()
    if not token:
        raise ValueError("Set REPLICATE_API_TOKEN or LLM_API_KEY for LLM_PROVIDER=replicate")
    os.environ["REPLICATE_API_TOKEN"] = token

    inp = {
        "system_prompt": system_prompt or "You are a helpful assistant.",
        "prompt": user_prompt,
        "temperature": float(temperature),
        "max_tokens": max(1, min(int(max_tokens), 8192)),
    }
    out = replicate.run(model_id, input=inp)
    return _collect_replicate_output(out).strip()


def chat_completion(
    messages: list[dict],
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> str:
    """
    Run a chat-style request. For Replicate, system + user messages map to ``system_prompt`` + ``prompt``.
    """
    provider = get_llm_provider()
    if provider == "replicate":
        model_id = clean_env(model) or clean_env(os.getenv("LLM_MODEL")) or DEFAULT_REPLICATE_MODEL
        system_parts: list[str] = []
        user_parts: list[str] = []
        for m in messages:
            role = m.get("role", "")
            content = m.get("content", "")
            if role == "system":
                system_parts.append(content)
            elif role == "user":
                user_parts.append(content)
            elif role == "assistant":
                user_parts.append(f"[assistant]\n{content}")
        system_prompt = "\n\n".join(system_parts)
        user_prompt = "\n\n".join(user_parts)
        return _replicate_run(
            model_id,
            system_prompt,
            user_prompt,
            float(temperature),
            int(max_tokens),
        )

    api_key = openai_api_key()
    if not api_key:
        raise ValueError(
            "Set LLM_API_KEY for OpenAI-compatible APIs (or LLM_PROVIDER=replicate with REPLICATE_API_TOKEN)"
        )
    base = clean_env(os.getenv("LLM_API_BASE")) or DEFAULT_OPENAI_BASE
    model_name = clean_env(model) or clean_env(os.getenv("LLM_MODEL")) or DEFAULT_OPENAI_MODEL
    client = OpenAI(api_key=api_key, base_url=base)
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    content = response.choices[0].message.content
    return (content or "").strip()


def check_llm_auth() -> tuple[bool, str]:
    """Return (success, message or sample output)."""
    if get_llm_provider() == "replicate":
        if not replicate_token():
            return False, "Set REPLICATE_API_TOKEN or LLM_API_KEY for Replicate"
        try:
            text = chat_completion(
                messages=[{"role": "user", "content": "Reply with exactly: ok"}],
                temperature=0,
                max_tokens=16,
            )
            return True, text
        except Exception as e:
            return False, str(e)

    api_key = openai_api_key()
    if not api_key:
        return False, "LLM_API_KEY is empty"
    base = clean_env(os.getenv("LLM_API_BASE")) or DEFAULT_OPENAI_BASE
    model_name = clean_env(os.getenv("LLM_MODEL")) or DEFAULT_OPENAI_MODEL
    try:
        client = OpenAI(api_key=api_key, base_url=base)
        r = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Reply with exactly: ok"}],
            max_tokens=10,
            temperature=0,
        )
        text = (r.choices[0].message.content or "").strip()
        return True, text
    except Exception as e:
        return False, str(e)
