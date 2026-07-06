"""
===============================================================================
Schemas
===============================================================================
Short Description:
- TravelOutput: structured schema every agent must return, validated with
  Pydantic so malformed/hallucinated output fails fast and loud.
- AppConfig: single source of truth for runtime configuration. Replaces the
  untyped `dict` that used to get passed around and silently mismatched the
  `AgentConfig` type hint on create_agent().
===============================================================================
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class TravelOutput(BaseModel):
    """Structured output contract every travel agent must satisfy."""

    destination: str = Field(..., description="Travel destination")
    duration: str = Field(..., description="Trip duration")
    summary: str = Field(..., description="Short trip summary")


class AppConfig(BaseModel):
    """Runtime configuration for the whole system.

    This is the single typed object threaded through agent creation,
    orchestration, and resilience layers — no more passing a bare dict that
    downstream functions assume (incorrectly) is a Pydantic model.
    """

    model: str = "gpt-4.1-mini"
    reasoning_effort: str = "medium"
    verbosity: str = "medium"

    retry_attempts: int = Field(3, ge=1)
    retry_delay: float = Field(1.0, ge=0)
    timeout_seconds: float = Field(30.0, gt=0)

    max_concurrency: int = Field(3, ge=1)
    max_requests: int = Field(10, ge=1)

    cache_ttl: int = Field(300, ge=0)
    cache_maxsize: int = Field(100, ge=1)

    circuit_failure_threshold: int = Field(3, ge=1)
    circuit_recovery_seconds: float = Field(30.0, ge=0)

    @classmethod
    def with_fallback(cls) -> "AppConfig":
        """Return a safe, fully-defaulted config for local/dev fallback."""
        return cls()
