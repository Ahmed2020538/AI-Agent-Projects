"""
===============================================================================
Advanced Agentic AI System (Production-Ready)
===============================================================================
Short Description:
- Multi-Agent AI system for intelligent task execution (Travel Planning use case)
- Uses structured outputs with Pydantic for reliability
- Includes rate limiting, retry logic, and budget control
- Implements caching to reduce cost and improve performance
- Designed with scalable and production-level architecture

Key Features:
- Multi-Agent orchestration (Planner + Explorer)
- Async execution with controlled concurrency
- API cost management (budget + retries)
- Config-driven design (no hardcoding)
- Clean logging for observability

Author: Ahmad (Senior AI Developer)
===============================================================================
"""

import os
import asyncio
import json
import logging
from logging.handlers import RotatingFileHandler
from typing import Union, Dict
from dotenv import load_dotenv, find_dotenv
from agents import Agent, Runner, ModelSettings
from pydantic import BaseModel, ValidationError


# ---------------------------------------------------------------------------
# Logging
# Short Description:
# - Centralized logging system instead of print statements
# - Helps in debugging, monitoring, and production observability
# ---------------------------------------------------------------------------
handler = RotatingFileHandler("app.log", maxBytes=1_000_000, backupCount=3)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    handlers=[handler]
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Globals (Rate limit + Budget)
# Short Description:
# - Controls API usage to prevent over-consumption
# - Ensures system stability under load
# ---------------------------------------------------------------------------
SEMAPHORE = asyncio.Semaphore(1)
REQUEST_COUNT = 0
MAX_REQUESTS = 10

# ---------------------------------------------------------------------------
# Environment
# Short Description:
# - Loads API keys securely from .env file
# - Prevents hardcoding sensitive credentials
# ---------------------------------------------------------------------------
def load_environment():
    load_dotenv(find_dotenv())
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise EnvironmentError("❌ OPENAI_API_KEY is missing")

    logger.info(f"API KEY Loaded: {bool(api_key)}")


# ---------------------------------------------------------------------------
# Config
# Short Description:
# - Loads dynamic system configuration (model, retries, limits)
# - Enables flexibility without modifying code
# ---------------------------------------------------------------------------
def load_config(path: str = "config.json") -> Dict:
    with open(path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Schema
# Short Description:
# - Defines structured output format for AI responses
# - Ensures consistency and validation using Pydantic
# ---------------------------------------------------------------------------
class TravelOutput(BaseModel):
    destination: str
    duration: str
    summary: str


# ---------------------------------------------------------------------------
# Cache
# Short Description:
# - In-memory caching to reduce duplicate API calls
# - Improves performance and reduces cost
# - Can be replaced with Redis in production
# ---------------------------------------------------------------------------
CACHE = {}

def get_from_cache(query: str):
    return CACHE.get(query.lower().strip())

def save_to_cache(query: str, result):
    CACHE[query.lower().strip()] = result


# ---------------------------------------------------------------------------
# Agent Factory
# Short Description:
# - Centralized creation of AI agents
# - Supports scalability and modular design
# ---------------------------------------------------------------------------
def create_agent(name: str, instructions: str, config: Dict) -> Agent:
    return Agent(
        name=name,
        model=config["model"],
        instructions=instructions,
        output_type=TravelOutput,
        model_settings=ModelSettings(
            reasoning={"effort": config["reasoning_effort"]},
            extra_body={"text": {"verbosity": config["verbosity"]}}
        )
    )


# ---------------------------------------------------------------------------
# Retry + Budget + Rate Limit
# Short Description:
# - Handles API failures and retries intelligently
# - Implements exponential backoff
# - Stops execution if quota is exceeded
# - Enforces budget limits
# ---------------------------------------------------------------------------
async def run_with_retry(agent: Agent, query: str, config: Dict):
    global REQUEST_COUNT, MAX_REQUESTS

    attempts = config["retry_attempts"]
    delay = config["retry_delay"]

    for attempt in range(1, attempts + 1):
        try:
            if REQUEST_COUNT >= MAX_REQUESTS:
                raise Exception("❌ Budget exceeded (max API calls reached)")

            async with SEMAPHORE:
                REQUEST_COUNT += 1
                logger.info(f"[{REQUEST_COUNT}] Running {agent.name} (Attempt {attempt})")

                return await Runner.run(agent, query)

        except Exception as e:
            error_str = str(e)
            logger.warning(f"Attempt {attempt} failed: {e}")

            if "insufficient_quota" in error_str:
                raise Exception("❌ API quota exceeded — check billing.")

            if attempt == attempts:
                raise

            await asyncio.sleep(delay * attempt)


# ---------------------------------------------------------------------------
# Parsing
# Short Description:
# - Converts raw AI output into validated structured objects
# - Handles invalid JSON or schema mismatches safely
# ---------------------------------------------------------------------------
def parse_output(data: Union[str, TravelOutput]) -> Union[TravelOutput, None]:
    if isinstance(data, TravelOutput):
        return data

    try:
        return TravelOutput(**json.loads(data))
    except (json.JSONDecodeError, ValidationError):
        logger.error("❌ Failed to parse output")
        return None


# ---------------------------------------------------------------------------
# Multi-Agent Execution
# Short Description:
# - Orchestrates multiple AI agents
# - Executes them sequentially to reduce API pressure
# - Combines and validates outputs
# ---------------------------------------------------------------------------
async def run_multi_agents(query: str, config: Dict):
    cached = get_from_cache(query)
    if cached:
        logger.info("✅ Cache hit")
        return cached

    planner = create_agent(
        "Planner Agent",
        "Plan a trip and return structured output.",
        config
    )

    explorer = create_agent(
        "Explorer Agent",
        "Find hidden and uncommon travel destinations.",
        config
    )

    results = []

    for agent in [planner, explorer]:
        res = await run_with_retry(agent, query, config)
        results.append(res)

    outputs = []
    for res in results:
        parsed = parse_output(res.final_output)
        if parsed:
            outputs.append(parsed)

    save_to_cache(query, outputs)
    return outputs


# ---------------------------------------------------------------------------
# Display
# Short Description:
# - Formats and prints results in a readable structure
# - Uses logging for consistency with system design
# ---------------------------------------------------------------------------
def display(results):
    for i, r in enumerate(results, 1):
        logger.info(f"\n--- Result {i} ---")
        logger.info(f"Destination: {r.destination}")
        logger.info(f"Duration   : {r.duration}")
        logger.info(f"Summary    : {r.summary}")


# ---------------------------------------------------------------------------
# Main
# Short Description:
# - Entry point for the application
# - Initializes environment, config, and executes agents
# ---------------------------------------------------------------------------
async def main():
    global MAX_REQUESTS

    try:
        load_environment()
        config = load_config()

        MAX_REQUESTS = config.get("max_requests", 10)

        query = "Plan a 3-day trip to Dubai under $1500 with hidden gems."

        results = await run_multi_agents(query, config)
        display(results)

    except Exception as e:
        logger.exception(f"❌ Fatal error: {e}")


# ---------------------------------------------------------------------------
# Entry Point
# Short Description:
# - Ensures proper async execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    asyncio.run(main())