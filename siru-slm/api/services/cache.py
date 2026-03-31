"""Simple in-memory cache for rewrite responses."""

import hashlib
import json
import time


class TTLCache:
    def __init__(self, ttl_seconds: int = 3600, max_items: int = 5000):
        self.ttl_seconds = ttl_seconds
        self.max_items = max_items
        self._store: dict[str, tuple[float, dict]] = {}

    def _cleanup(self):
        now = time.time()
        expired = [k for k, (expires_at, _) in self._store.items() if expires_at < now]
        for k in expired:
            del self._store[k]

        if len(self._store) > self.max_items:
            keys = list(self._store.keys())[: len(self._store) - self.max_items]
            for k in keys:
                del self._store[k]

    def key_for(self, payload: dict) -> str:
        normalized = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def get(self, key: str):
        self._cleanup()
        hit = self._store.get(key)
        if not hit:
            return None
        expires_at, value = hit
        if expires_at < time.time():
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: dict):
        self._cleanup()
        self._store[key] = (time.time() + self.ttl_seconds, value)
