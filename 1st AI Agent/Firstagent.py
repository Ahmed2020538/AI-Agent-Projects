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
# =============================================================================
# Imports (Organized - Clean Architecture Style)
# =============================================================================

# ==============================
# Standard Library
# ==============================
import os
import sys
import uuid
import json
import asyncio
import logging
import random
import asyncio
from logging.handlers import RotatingFileHandler
from contextvars import ContextVar
from typing import Union, Dict

# ==============================
# Third-Party Libraries
# ==============================
from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel, ValidationError, Field
from cachetools import TTLCache

# ==============================
# AI / Agents Layer
# ==============================
# ⚠️ Choose ONE based on your implementation

# Option 1: OpenAI Agents SDK
from agents import Agent, Runner, ModelSettings

# Option 2: CrewAI (uncomment if used instead)
# from crewai import Agent

# Option 3: Local module (if you created agents.py)
# from .agents import Agent, Runner, ModelSettings


# =============================================================================
# Logging System (Production-Grade Observability)
# Short Description:
# - Structured JSON logging (ELK / Datadog ready)
# - Trace ID per request using contextvars
# - Console readable + File JSON logs
# - Prevent duplicate handlers
# - Supports LOG_LEVEL from environment
# =============================================================================
# -----------------------------
# Trace ID (Per Request)
# Short Description:
# - Unique ID for each request
# - Enables end-to-end tracing across logs
# -----------------------------
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

def set_request_id():
    request_id_var.set(str(uuid.uuid4()))

def get_request_id():
    return request_id_var.get()
# -----------------------------
# JSON Formatter (File Logs)
# Short Description:
# - Converts logs into structured JSON
# - Compatible with ELK / Grafana / Datadog
# -----------------------------
class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, "%Y-%m-%d %H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "trace_id": get_request_id(),
            "file": record.filename,
            "line": record.lineno,
            "function": record.funcName,
        }

        # Include extra fields if provided
        for key, value in record.__dict__.items():
            if key not in log_record and not key.startswith("_"):
                try:
                    json.dumps({key: value})
                    log_record[key] = value
                except:
                    pass

        return json.dumps(log_record)


# -----------------------------
# Log Colors
# Short Description:
# - Adds colors to console logs based on level
# - Improves readability during development
# -----------------------------
LOG_COLORS = {
    "DEBUG": "\033[94m",     # Blue
    "INFO": "\033[92m",      # Green
    "WARNING": "\033[93m",   # Yellow
    "ERROR": "\033[91m",     # Red
    "CRITICAL": "\033[95m",  # Magenta
}

RESET_COLOR = "\033[0m"


# -----------------------------
# Console Formatter (Readable, Colored)
# Short Description:
# - Clean and human-readable format
# - Useful for development and debugging
# - Human-readable + colored logs
# - Highlights log levels visually
# -----------------------------
class ConsoleFormatter(logging.Formatter):
    def format(self, record):
        color = LOG_COLORS.get(record.levelname, "")
        reset = RESET_COLOR

        return (
            f"{self.formatTime(record)} "
            f"{color}[{record.levelname}]{reset} "
            f"[{record.name}] "
            f"[trace={get_request_id()}] "
            f"{record.getMessage()}"
        )
# -----------------------------
# Logger Setup
# Short Description:
# - Initializes reusable logger
# - Adds file + console handlers
# - Prevents duplicate logs
# -----------------------------
def setup_logger(name: str = "app") -> logging.Logger:
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logger.setLevel(log_level)

    # File handler (JSON logs)
    file_handler = RotatingFileHandler(
        "app.log",
        maxBytes=5_000_000,
        backupCount=5
    )
    file_handler.setFormatter(JSONFormatter())

    # Console handler (readable logs)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(ConsoleFormatter())

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.propagate = False

    return logger


# Initialize logger (replacement for basicConfig)
logger = setup_logger()


# =============================================================================
# Globals (Rate Limit + Budget)
# Short Description:
# - Controls API usage
# - Prevents over-consumption
# - Controls API usage to prevent over-consumption
# - Ensures system stability under load
# =============================================================================
SEMAPHORE = asyncio.Semaphore(3)
REQUEST_COUNT = 0
MAX_REQUESTS = 10


# =============================================================================
# Environment
# Short Description:
# - Loads environment variables securely
# - Avoids hardcoding secrets
# - Loads API keys securely from .env file
# - Prevents hardcoding sensitive credentials
# =============================================================================
def load_environment():
    """
    Short Description:
    - Loads environment variables from .env
    - Validates required variables dynamically
    - Supports optional variables with defaults
    - Logs all events with trace_id
    """

    logger.info("Loading environment variables...")

    REQUIRED_ENV_VARS = ["OPENAI_API_KEY"]

    OPTIONAL_ENV_VARS = {
        "TIMEOUT": 30,
        "CACHE_TTL": 300
    }

    try:
        load_dotenv(find_dotenv())

        env_data = {}

        # -----------------------------
        # Required variables
        # -----------------------------
        for var in REQUIRED_ENV_VARS:
            value = os.getenv(var)

            if not value:
                logger.error(
                    f"{var} missing",
                    extra={"env_var": var}
                )
                raise EnvironmentError(f"{var} is missing")

            env_data[var] = value

        # -----------------------------
        # Optional variables (with defaults)
        # -----------------------------
        for var, default in OPTIONAL_ENV_VARS.items():
            value = os.getenv(var, default)

            # Convert numeric values from string → int
            if isinstance(default, int):
                try:
                    value = int(value)
                except ValueError:
                    logger.warning(
                        f"{var} should be int, using default",
                        extra={"provided": value, "default": default}
                    )
                    value = default

            env_data[var] = value

        # -----------------------------
        # Optional validation
        # -----------------------------
        api_key = env_data.get("OPENAI_API_KEY")

        if api_key and not api_key.startswith("sk-"):
            logger.warning(
                "OPENAI_API_KEY format looks unusual",
                extra={"hint": "Key should start with 'sk-'"}
            )

        logger.info(
            "Environment loaded successfully",
            extra={"loaded_vars": list(env_data.keys())}
        )

        return env_data

    except Exception as e:
        logger.exception(
            "Failed to load environment",
            extra={"error": str(e)}
        )
        raise

# =============================================================================
# Config
# Short Description:
# - Loads dynamic configuration from JSON
# - Loads dynamic system configuration (model, retries, limits)
# - Enables flexibility without modifying code
# =============================================================================
def load_config(path: str = "config.json") -> Dict:
    """
    Short Description:
    - Loads configuration from a JSON file
    - Logs all events with trace_id
    - Uses smart fallback based on expected schema
    """

    logger.info(f"Loading config file: {path}")

    try:
        with open(path, "r") as f:
            config = json.load(f)

            logger.info(
                "Config loaded successfully",
                extra={"config": config}
            )
            return config

    except FileNotFoundError as e:
        logger.error(
            f"Config file not found: {path}",
            extra={"error": str(e)}
        )

    except json.JSONDecodeError as e:
        logger.error(
            f"Invalid JSON format in config file: {path}",
            extra={"error": str(e)}
        )

    except Exception as e:
        logger.exception(
            "Unexpected error while loading config",
            extra={"error": str(e)}
        )

    # -----------------------------
    # Smart Fallback (matches your schema)
    # -----------------------------
    fallback_config = {
        "model": "gpt-4.1-mini",
        "reasoning_effort": "medium",
        "verbosity": "medium",

        "retry_attempts": 3,
        "retry_delay": 1,
        "timeout": 30,

        "max_requests": 5,
        "cache_ttl": 300   
    }

    logger.warning(
        "Using fallback config",
        extra={"fallback": fallback_config}
    )

    return fallback_config

# =============================================================================
# Schema (Pydantic)
# Short Description:
# - Defines structured output schema
# - Ensures validation and consistency
# - Defines structured output format for AI responses
# - Ensures consistency and validation using Pydantic
# =============================================================================
class TravelOutput(BaseModel):
    destination: str = Field(..., description="Travel destination")
    duration: str = Field(..., description="Trip duration")
    summary: str = Field(..., description="Short trip summary")


# =============================================================================
# Cache
# Short Description:
# - Simple in-memory cache
# - Reduces redundant API calls
# - In-memory caching to reduce duplicate API calls
# - Improves performance and reduces cost
# - Can be replaced with Redis in production
# =============================================================================


# =============================================================================
# Cache (TTLCache + Normalization)
# Short Description:
# - Uses TTLCache for automatic expiration
# - Normalizes keys to avoid duplicates (case / spaces)
# - Logs cache activity (HIT / MISS / SAVE)
# =============================================================================

# Initialize cache
cache = TTLCache(maxsize=100, ttl=300)


def _normalize(key: str) -> str:
    """
    Short Description:
    - Normalizes cache key
    - Prevents duplicates بسبب اختلاف case أو spaces

    Example:
        "  Dubai Trip  " → "dubai trip"
    """
    return key.lower().strip()


def get_from_cache(key: str):
    """
    Short Description:
    - Retrieves value from cache if exists
    - Logs HIT or MISS

    Args:
        key (str): Raw cache key

    Returns:
        Cached value or None
    """

    normalized_key = _normalize(key)

    if normalized_key in cache:
        logger.info(
            "Cache HIT",
            extra={"key": normalized_key}
        )
        return cache[normalized_key]

    logger.info(
        "Cache MISS",
        extra={"key": normalized_key}
    )
    return None


def save_to_cache(key: str, value):
    """
    Short Description:
    - Saves value to cache after normalizing key
    - Automatically expires after TTL

    Args:
        key (str): Raw cache key
        value: Data to cache
    """

    normalized_key = _normalize(key)

    cache[normalized_key] = value

    logger.info(
        "Saved to cache",
        extra={"key": normalized_key}
    )

# =============================================================================
# Agent Factory
# Short Description:
# - Centralized agent creation
# - Enables modular and scalable design
# - Centralized creation of AI agents
# - Supports scalability and modular design
# =============================================================================
# =============================================================================
# Agent Factory
# Short Description:
# - Centralized creation of AI agents
# - Improves modularity and scalability
# =============================================================================

class AgentConfig(BaseModel):
    model: str
    reasoning_effort: str
    verbosity: str


def create_agent(
    name: str,
    instructions: str,
    config: AgentConfig,
    output_type
) -> Agent:
    """
    Creates and configures an AI agent.

    Args:
        name (str): Agent name
        instructions (str): Task instructions
        config (AgentConfig): Configuration object
        output_type: Expected structured output

    Returns:
        Agent: Configured AI agent instance
    """

    agent = Agent(
        name=name,
        model=config.model,
        instructions=instructions,
        output_type=output_type,
        model_settings=ModelSettings(
            reasoning={"effort": config.reasoning_effort},
            extra_body={"text": {"verbosity": config.verbosity}}
        )
    )

    logger.info("Agent created", extra={"name": name})

    return agent

# =============================================================================
# Retry + Rate Limit + Budget
# Short Description:
# - Handles retries with backoff
# - Enforces API usage limits
# - Handles API failures and retries intelligently
# - Implements exponential backoff
# - Stops execution if quota is exceeded
# - Enforces budget limits
# =============================================================================
# =============================================================================
# Retry + Rate Limit + Budget Control
# Short Description:
# - Handles retries with exponential backoff + jitter
# - Enforces rate limiting and budget constraints
# =============================================================================

# =============================================================================
# Retry + Circuit Breaker + Rate Limit + Observability
# =============================================================================

import asyncio
import random
import time
import uuid
from contextvars import ContextVar

# ==============================
# Tracing ID (per request)
# ==============================
trace_id_var: ContextVar[str] = ContextVar("trace_id", default="-")


def set_trace_id():
    trace_id_var.set(str(uuid.uuid4()))


def get_trace_id():
    return trace_id_var.get()


# ==============================
# Circuit Breaker
# ==============================
class CircuitBreaker:
    """
    Short Description:
    - Prevents repeated calls to failing services
    - Opens after failure threshold
    - Recovers after cooldown
    """

    def __init__(self, failure_threshold=3, recovery_time=30):
        self.failure_threshold = failure_threshold
        self.recovery_time = recovery_time
        self.failures = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED → OPEN → HALF_OPEN

    def can_execute(self):
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_time:
                self.state = "HALF_OPEN"
                return True
            return False
        return True

    def record_success(self):
        self.failures = 0
        self.state = "CLOSED"

    def record_failure(self):
        self.failures += 1
        self.last_failure_time = time.time()

        if self.failures >= self.failure_threshold:
            self.state = "OPEN"


# ==============================
# Global Components
# ==============================
circuit_breaker = CircuitBreaker()
lock = asyncio.Lock()


# ==============================
# Main Execution with Protection
# ==============================
async def run_with_retry(agent: Agent, query: str, config: Dict):
    attempts = config.get("retry_attempts", 3)
    delay = config.get("retry_delay", 1)

    # Set trace ID لكل request
    set_trace_id()
    trace_id = get_trace_id()

    for attempt in range(1, attempts + 1):
        try:
            # 🔴 Circuit Breaker Check
            if not circuit_breaker.can_execute():
                raise Exception("Circuit is OPEN - skipping execution")

            # 🔒 Rate limit protection
            async with SEMAPHORE:
                async with lock:
                    global REQUEST_COUNT
                    if REQUEST_COUNT >= MAX_REQUESTS:
                        raise Exception("Budget exceeded")

                    REQUEST_COUNT += 1

                logger.info(
                    f"Running {agent.name}",
                    extra={
                        "trace_id": trace_id,
                        "attempt": attempt,
                        "request_count": REQUEST_COUNT,
                        "circuit_state": circuit_breaker.state
                    }
                )

                result = await asyncio.wait_for(
                    Runner.run(agent, query),
                    timeout=30
                )

                # ✅ Success
                circuit_breaker.record_success()

                logger.info(
                    "Request succeeded",
                    extra={"trace_id": trace_id}
                )

                return result

        except (TimeoutError, ConnectionError, Exception) as e:
            circuit_breaker.record_failure()

            logger.warning(
                f"Retry {attempt} failed",
                extra={
                    "trace_id": trace_id,
                    "error": str(e),
                    "circuit_state": circuit_breaker.state
                }
            )

            if attempt == attempts:
                logger.exception(
                    "Max retries reached",
                    extra={"trace_id": trace_id}
                )
                raise

            # ⏱ Exponential backoff + jitter
            sleep_time = delay * attempt + random.uniform(0, 1)
            await asyncio.sleep(sleep_time)
# =============================================================================
# Parsing
# Short Description:
# - Converts raw output into validated objects
# - Converts raw AI output into validated structured objects
# - Handles invalid JSON or schema mismatches safely
# =============================================================================
def parse_output(data: Union[str, TravelOutput]) -> Union[TravelOutput, None]:
    """
    Short Description:
    - Parses raw agent output into TravelOutput
    - Supports both string JSON and direct object
    - Adds validation, logging, and traceability
    """

    trace_id = get_trace_id()

    # ✅ Already parsed
    if isinstance(data, TravelOutput):
        logger.info(
            "Output already parsed",
            extra={"trace_id": trace_id}
        )
        return data

    # ❌ Empty input
    if not data:
        logger.error(
            "Empty output received",
            extra={"trace_id": trace_id}
        )
        return None

    try:
        parsed = TravelOutput(**json.loads(data))

        logger.info(
            "Output parsed successfully",
            extra={"trace_id": trace_id}
        )

        return parsed

    except json.JSONDecodeError as e:
        logger.error(
            "JSON parsing failed",
            extra={
                "trace_id": trace_id,
                "error": str(e),
                "raw_data": str(data)[:200]  # prevent huge logs
            }
        )

    except ValidationError as e:
        logger.error(
            "Schema validation failed",
            extra={
                "trace_id": trace_id,
                "error": e.errors()
            }
        )

    return None


# =============================================================================
# Multi-Agent Execution
# Short Description:
# - Orchestrates multiple agents
# - Adds trace_id per request
# - Orchestrates multiple AI agents
# - Executes them sequentially to reduce API pressure
# - Combines and validates outputs
# =============================================================================
async def run_multi_agents(query: str, config: Dict):
    """
    Short Description:
    - Orchestrates multiple AI agents with observability
    - Uses caching to optimize performance
    - Adds trace_id and timing for monitoring
    """

    set_request_id()
    trace_id = get_request_id()

    start_time = asyncio.get_event_loop().time()

    # ==============================
    # Cache Check
    # ==============================
    cached = get_from_cache(query)
    if cached:
        logger.info(
            "Cache hit",
            extra={
                "trace_id": trace_id,
                "query": query
            }
        )
        return cached

    # ==============================
    # Create Agents
    # ==============================
    planner = create_agent("Planner Agent", "Plan a trip.", config)
    explorer = create_agent("Explorer Agent", "Find hidden gems.", config)

    agents = [planner, explorer]

    results = []

    # ==============================
    # Execute Agents (Isolated)
    # ==============================
    for agent in agents:
        try:
            logger.info(
                f"Running {agent.name}",
                extra={"trace_id": trace_id}
            )

            res = await run_with_retry(agent, query, config)
            results.append(res)

        except Exception as e:
            logger.error(
                f"{agent.name} failed",
                extra={
                    "trace_id": trace_id,
                    "error": str(e)
                }
            )

    # ==============================
    # Parse Outputs
    # ==============================
    outputs = []

    for res in results:
        parsed = parse_output(res.final_output)

        if parsed:
            outputs.append(parsed)

    # ==============================
    # Save Cache
    # ==============================
    save_to_cache(query, outputs)

    # ==============================
    # Metrics
    # ==============================
    duration = asyncio.get_event_loop().time() - start_time

    logger.info(
        "Multi-agent execution completed",
        extra={
            "trace_id": trace_id,
            "agents_count": len(agents),
            "success_count": len(outputs),
            "duration_sec": round(duration, 2)
        }
    )

    return outputs
# =============================================================================
# Display
# Short Description:
# - Displays results using logging
# - Formats and prints results in a readable structure
# - Uses logging for consistency with system design
# =============================================================================
# =============================================================================
# Display Layer (Hybrid - Readable + Structured + Tracing)
# Short Description:
# - Combines human-readable logs + structured logging
# - Includes trace_id for observability
# - Supports monitoring & debugging
# - Safe handling of missing fields
# =============================================================================
def display(results):
    trace_id = get_request_id()

    if not results:
        logger.warning(
            "No results to display",
            extra={"trace_id": trace_id}
        )
        return

    # Summary Log (Monitoring)
    logger.info(
        "Displaying results",
        extra={
            "trace_id": trace_id,
            "results_count": len(results)
        }
    )

    # Detailed Logs (Debug + Readability)
    for i, r in enumerate(results, 1):
        logger.info(
            f"Result {i}: {getattr(r, 'destination', 'N/A')} "
            f"({getattr(r, 'duration', 'N/A')})"
        )

        # Structured Log (for systems)
        logger.info(
            "Travel Result",
            extra={
                "trace_id": trace_id,
                "result_index": i,
                "destination": getattr(r, "destination", "N/A"),
                "duration": getattr(r, "duration", "N/A"),
                "summary": getattr(r, "summary", "N/A"),
            }
        )

# =============================================================================
# Main
# Short Description:
# - Application entry point
# - Entry point for the application
# - Initializes environment, config, and executes agents
# =============================================================================
# =============================================================================
# Main Entry Point (Production-Ready)
# Short Description:
# - Application entry point
# - Initializes environment and configuration
# - Adds tracing (trace_id)
# - Handles full request lifecycle logging
# - Measures execution time
# =============================================================================
async def main(query: str):
    global MAX_REQUESTS

    set_request_id()
    trace_id = get_request_id()

    start_time = asyncio.get_event_loop().time()

    try:
        logger.info(
            "Application started",
            extra={"trace_id": trace_id}
        )

        load_environment()
        config = load_config()

        MAX_REQUESTS = config.get("max_requests", 10)

        logger.info(
            "Processing request",
            extra={
                "trace_id": trace_id,
                "query": query
            }
        )

        results = await run_multi_agents(query, config)

        display(results)

        duration = round(asyncio.get_event_loop().time() - start_time, 2)

        logger.info(
            "Request completed",
            extra={
                "trace_id": trace_id,
                "duration_sec": duration,
                "results_count": len(results)
            }
        )

    except Exception as e:
        logger.exception(
            "Fatal error",
            extra={
                "trace_id": trace_id,
                "error": str(e)
            }
        )

# =============================================================================
# Entry Point
# Short Description:
# - Runs async application
# - Ensures proper async execution
# =============================================================================
if __name__ == "__main__":
    asyncio.run(main())