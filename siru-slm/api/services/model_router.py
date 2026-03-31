"""Routes requests to remote LLM (ideation) and local SLM (rewrite)."""

import os
import sys
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from env_load import clean_env
from llm_client import get_llm_provider


class ModelRouter:
    def __init__(self):
        if get_llm_provider() == "replicate":
            self.ideation_model = clean_env(os.getenv("LLM_MODEL")) or "meta/meta-llama-3-70b-instruct"
        else:
            self.ideation_model = clean_env(os.getenv("LLM_MODEL")) or "kimi-k2.5"
        self.slm_base_url = clean_env(os.getenv("SLM_BASE_URL")) or "http://localhost:8001/v1"
        self.slm_model = clean_env(os.getenv("SLM_MODEL_NAME")) or "siru-dialogue"

        self._slm_client = None

    @property
    def slm_client(self):
        if self._slm_client is None:
            self._slm_client = OpenAI(api_key="not-needed", base_url=self.slm_base_url)
        return self._slm_client
