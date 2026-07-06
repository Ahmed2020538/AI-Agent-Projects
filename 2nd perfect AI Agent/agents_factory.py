"""
===============================================================================
Agent Factory
===============================================================================
Short Description:
- Centralized, typed creation of AI agents.
- Fixes the original bug where create_agent() required an `output_type`
  argument that call sites never passed, and where `config` was type-hinted
  as a Pydantic model but actually received as a plain dict.
===============================================================================
"""

from __future__ import annotations

import logging
from typing import Type

from agents import Agent, ModelSettings
from pydantic import BaseModel

from .schemas import AppConfig


def create_agent(
    name: str,
    instructions: str,
    config: AppConfig,
    output_type: Type[BaseModel],
    logger: logging.Logger,
) -> Agent:
    """Create and configure an AI agent.

    Args:
        name: Agent name (used for logging/tracing).
        instructions: Task instructions for the agent.
        config: Validated application configuration.
        output_type: Pydantic model the agent must return structured output as.
        logger: Logger for lifecycle events.

    Returns:
        Agent: A configured agent instance.
    """
    agent = Agent(
        name=name,
        model=config.model,
        instructions=instructions,
        output_type=output_type,
        model_settings=ModelSettings(
            reasoning={"effort": config.reasoning_effort},
            extra_body={"text": {"verbosity": config.verbosity}},
        ),
    )

    logger.info("Agent created", extra={"agent_name": name, "output_type": output_type.__name__})
    return agent
