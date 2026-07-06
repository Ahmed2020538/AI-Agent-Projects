"""
===============================================================================
Environment Loading
===============================================================================
Short Description:
- Loads secrets/env vars from .env via python-dotenv.
- Validates required variables and fails loudly if missing (no silent
  fallback for a missing API key — that should never happen quietly).
- Optional vars get typed defaults.
===============================================================================
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict

from dotenv import find_dotenv, load_dotenv

REQUIRED_ENV_VARS = ("OPENAI_API_KEY",)

OPTIONAL_ENV_VARS: Dict[str, Any] = {
    "TIMEOUT": 30,
    "CACHE_TTL": 300,
    "LOG_LEVEL": "INFO",
}


def load_environment(logger: logging.Logger) -> Dict[str, Any]:
    """Load and validate environment variables.

    Args:
        logger: Logger to report loading events to.

    Returns:
        Dict[str, Any]: Loaded environment values (required + optional).

    Raises:
        EnvironmentError: If a required variable is missing.
    """
    logger.info("Loading environment variables")
    load_dotenv(find_dotenv(usecwd=True))

    env_data: Dict[str, Any] = {}

    for var in REQUIRED_ENV_VARS:
        value = os.getenv(var)
        if not value:
            logger.error("Required environment variable missing", extra={"env_var": var})
            raise EnvironmentError(f"{var} is required but not set")
        env_data[var] = value

    for var, default in OPTIONAL_ENV_VARS.items():
        raw_value = os.getenv(var, default)
        if isinstance(default, int) and not isinstance(raw_value, int):
            try:
                raw_value = int(raw_value)
            except (TypeError, ValueError):
                logger.warning(
                    "Invalid int for env var, using default",
                    extra={"env_var": var, "provided": raw_value, "default": default},
                )
                raw_value = default
        env_data[var] = raw_value

    api_key = env_data.get("OPENAI_API_KEY", "")
    if api_key and not api_key.startswith("sk-"):
        logger.warning("OPENAI_API_KEY format looks unusual (expected 'sk-' prefix)")

    logger.info("Environment loaded successfully", extra={"loaded_vars": list(env_data.keys())})
    return env_data
