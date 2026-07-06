"""
===============================================================================
Result Cache
===============================================================================
Short Description:
- Thin wrapper around cachetools.TTLCache.
- Normalizes keys (case/whitespace) to avoid duplicate entries.
- Encapsulated as an instance (no module-level global) so it can be
  constructed per-application, per-tenant, or swapped for a Redis-backed
  implementation without touching call sites.
===============================================================================
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from cachetools import TTLCache

from .logging_setup import get_trace_id


class ResultCache:
    """TTL-based cache for agent run results, keyed by normalized query."""

    def __init__(self, logger: logging.Logger, maxsize: int = 100, ttl: int = 300):
        self._cache: TTLCache = TTLCache(maxsize=maxsize, ttl=ttl)
        self._logger = logger

    @staticmethod
    def _normalize(key: str) -> str:
        """Normalize a cache key to avoid duplicates from case/whitespace.

        Example:
            "  Dubai Trip  " -> "dubai trip"
        """
        return key.lower().strip()

    def get(self, key: str) -> Optional[Any]:
        """Return the cached value for `key`, or None on a cache miss."""
        normalized_key = self._normalize(key)
        if normalized_key in self._cache:
            self._logger.info("Cache HIT", extra={"trace_id": get_trace_id(), "key": normalized_key})
            return self._cache[normalized_key]

        self._logger.info("Cache MISS", extra={"trace_id": get_trace_id(), "key": normalized_key})
        return None

    def set(self, key: str, value: Any) -> None:
        """Store `value` under the normalized `key`, subject to TTL/maxsize."""
        normalized_key = self._normalize(key)
        self._cache[normalized_key] = value
        self._logger.info("Saved to cache", extra={"trace_id": get_trace_id(), "key": normalized_key})
