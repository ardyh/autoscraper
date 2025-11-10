from __future__ import annotations

import os
import threading
from typing import Optional

import httpx


class OllamaClient:
    _instance: Optional["OllamaClient"] = None
    _lock = threading.Lock()

    def __init__(self, base_url: str, model_id: str, timeout: int = 600):
        self.base_url = base_url.rstrip("/")
        self.model_id = model_id
        # Increased timeout for RAG operations (multiple LLM calls)
        # Default: 600 seconds (10 minutes)
        # Configurable via OLLAMA_TIMEOUT env var
        timeout = int(os.getenv("OLLAMA_TIMEOUT", str(timeout)))
        self._client = httpx.Client(timeout=timeout)

    @classmethod
    def get(cls, model_id: Optional[str] = None, base_url: Optional[str] = None) -> "OllamaClient":
        with cls._lock:
            if cls._instance is None:
                mid = model_id or os.getenv("MODEL_ID", "llama3")
                burl = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
                cls._instance = OllamaClient(burl, mid)
            return cls._instance

    def generate(self, prompt: str, max_tokens: int = 7000, temperature: float = 0.0) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model_id,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        resp = self._client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "")


