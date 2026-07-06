"""
===============================================================================
Config Loading
===============================================================================
Short Description:
- Loads runtime configuration from a JSON file into a typed AppConfig.
- Falls back to safe defaults for local/dev if the file is missing or
  malformed, but always logs loudly when that happens (never a silent swap).
===============================================================================
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import ValidationError

from .schemas import AppConfig


def load_config(logger: logging.Logger, path: str = "config.json") -> AppConfig:
    """Load configuration from a JSON file into a validated AppConfig.

    Args:
        logger: Logger to report loading events to.
        path: Path to the JSON config file.

    Returns:
        AppConfig: Validated configuration (fallback defaults on failure).
    """
    config_path = Path(path)
    logger.info("Loading config file", extra={"path": str(config_path)})

    if not config_path.exists():
        logger.warning("Config file not found, using fallback config", extra={"path": str(config_path)})
        return AppConfig.with_fallback()

    try:
        raw = json.loads(config_path.read_text())
        config = AppConfig(**raw)
        logger.info("Config loaded successfully", extra={"config": config.model_dump()})
        return config

    except json.JSONDecodeError as exc:
        logger.error("Invalid JSON in config file, using fallback", extra={"error": str(exc)})
    except ValidationError as exc:
        logger.error("Config failed schema validation, using fallback", extra={"error": exc.errors()})
    except Exception:
        logger.exception("Unexpected error loading config, using fallback")

    fallback = AppConfig.with_fallback()
    logger.warning("Using fallback config", extra={"fallback": fallback.model_dump()})
    return fallback
