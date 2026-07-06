"""
===============================================================================
Multi-Agent Orchestration
===============================================================================
Short Description:
- Owns everything needed to run one logical "request" through multiple
  agents: cache, circuit breaker, budget, concurrency semaphore, config.
- No module-level global state - one AgentOrchestrator instance = one
  isolated set of resilience/rate-limit resources, safe to construct
  per-test, per-tenant, or as a single app-lifetime singleton.
- Runs agents CONCURRENTLY (bounded by config.max_concurrency) instead of
  the original sequential for-loop, while still respecting the shared
  budget and circuit breaker.
- Tracks partial failures explicitly instead of silently dropping them.
===============================================================================
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import List, Optional, Union

from agents import Agent, Runner
from pydantic import ValidationError

from .cache import ResultCache
from .logging_setup import get_trace_id, new_trace_id
from .resilience import BudgetManager, CircuitBreaker, run_with_retry
from .schemas import AppConfig, TravelOutput


@dataclass
class AgentRunOutcome:
    """Result of a single agent's run, success or failure, for observability."""

    agent_name: str
    succeeded: bool
    output: Optional[TravelOutput] = None
    error: Optional[str] = None


def parse_output(data: Union[str, TravelOutput], logger: logging.Logger) -> Optional[TravelOutput]:
    """Parse raw agent output into a validated TravelOutput.

    Args:
        data: Either an already-parsed TravelOutput, or a raw JSON string.
        logger: Logger for validation/parsing events.

    Returns:
        TravelOutput if parsing/validation succeeds, otherwise None.
    """
    trace_id = get_trace_id()

    if isinstance(data, TravelOutput):
        return data

    if not data:
        logger.error("Empty output received from agent", extra={"trace_id": trace_id})
        return None

    try:
        return TravelOutput(**json.loads(data))
    except json.JSONDecodeError as exc:
        logger.error(
            "JSON parsing failed for agent output",
            extra={"trace_id": trace_id, "error": str(exc), "raw_data": str(data)[:200]},
        )
    except ValidationError as exc:
        logger.error(
            "Schema validation failed for agent output",
            extra={"trace_id": trace_id, "error": exc.errors()},
        )
    return None


class AgentOrchestrator:
    """Coordinates multi-agent execution with caching, resilience, and tracing."""

    def __init__(self, config: AppConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.cache = ResultCache(logger, maxsize=config.cache_maxsize, ttl=config.cache_ttl)
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=config.circuit_failure_threshold,
            recovery_time=config.circuit_recovery_seconds,
        )
        self.budget = BudgetManager(max_requests=config.max_requests)
        self.semaphore = asyncio.Semaphore(config.max_concurrency)

    async def _run_agent(self, agent: Agent, query: str) -> AgentRunOutcome:
        """Run a single agent with full resilience protection, never raising."""
        try:
            result = await run_with_retry(
                operation=lambda: Runner.run(agent, query),
                operation_name=agent.name,
                circuit_breaker=self.circuit_breaker,
                budget=self.budget,
                semaphore=self.semaphore,
                attempts=self.config.retry_attempts,
                base_delay=self.config.retry_delay,
                timeout_seconds=self.config.timeout_seconds,
                logger=self.logger,
            )
            parsed = parse_output(result.final_output, self.logger)
            if parsed is None:
                return AgentRunOutcome(agent.name, succeeded=False, error="output_parse_failed")
            return AgentRunOutcome(agent.name, succeeded=True, output=parsed)

        except Exception as exc:
            self.logger.error(
                f"{agent.name} failed permanently",
                extra={"trace_id": get_trace_id(), "error": str(exc), "error_type": type(exc).__name__},
            )
            return AgentRunOutcome(agent.name, succeeded=False, error=str(exc))

    async def run_multi_agents(self, query: str, agents: List[Agent]) -> List[AgentRunOutcome]:
        """Run all `agents` against `query` concurrently (bounded), with caching.

        Args:
            query: The user's request/query.
            agents: Agents to run against this query.

        Returns:
            List[AgentRunOutcome]: One outcome per agent, success or failure —
            failures are reported, not silently dropped.
        """
        new_trace_id()
        trace_id = get_trace_id()
        start = time.monotonic()

        cached = self.cache.get(query)
        if cached is not None:
            self.logger.info("Cache hit, skipping agent execution", extra={"trace_id": trace_id, "query": query})
            return cached

        self.logger.info(
            "Starting multi-agent run",
            extra={"trace_id": trace_id, "agent_count": len(agents), "query": query},
        )

        outcomes = await asyncio.gather(*(self._run_agent(agent, query) for agent in agents))

        success_count = sum(1 for o in outcomes if o.succeeded)
        duration = round(time.monotonic() - start, 2)

        self.logger.info(
            "Multi-agent run completed",
            extra={
                "trace_id": trace_id,
                "agents_count": len(agents),
                "success_count": success_count,
                "failure_count": len(agents) - success_count,
                "duration_sec": duration,
            },
        )

        # Only cache when at least one agent produced a usable result - an
        # all-failed run should not be remembered as a valid cached answer.
        if success_count > 0:
            self.cache.set(query, outcomes)

        return outcomes


def display(outcomes: List[AgentRunOutcome], logger: logging.Logger) -> None:
    """Log a human- and machine-readable summary of agent outcomes.

    Args:
        outcomes: Results returned by AgentOrchestrator.run_multi_agents.
        logger: Logger to write the summary to.
    """
    trace_id = get_trace_id()

    if not outcomes:
        logger.warning("No results to display", extra={"trace_id": trace_id})
        return

    logger.info("Displaying results", extra={"trace_id": trace_id, "results_count": len(outcomes)})

    for i, outcome in enumerate(outcomes, start=1):
        if not outcome.succeeded or outcome.output is None:
            logger.warning(
                f"Result {i} ({outcome.agent_name}): FAILED - {outcome.error}",
                extra={"trace_id": trace_id, "result_index": i, "agent_name": outcome.agent_name},
            )
            continue

        result = outcome.output
        logger.info(
            f"Result {i} ({outcome.agent_name}): {result.destination} ({result.duration})",
            extra={
                "trace_id": trace_id,
                "result_index": i,
                "agent_name": outcome.agent_name,
                "destination": result.destination,
                "duration": result.duration,
                "summary": result.summary,
            },
        )
