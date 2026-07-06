"""
===============================================================================
Application Entry Point
===============================================================================
Short Description:
- Wires together env loading, config loading, agent creation, and the
  orchestrator.
- Accepts the query via CLI argument (the original code called
  `asyncio.run(main())` with no argument even though `main()` required one -
  that's fixed here with argparse + a sensible default for quick testing).
- Full request-lifecycle logging with duration and trace_id.
===============================================================================
"""

from __future__ import annotations

import argparse
import asyncio
import time

from .agents_factory import create_agent
from .config_loader import load_config
from .env_config import load_environment
from .logging_setup import get_trace_id, new_trace_id, setup_logger
from .orchestrator import AgentOrchestrator, display
from .schemas import TravelOutput

logger = setup_logger()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-agent travel planning system")
    parser.add_argument(
        "query",
        nargs="?",
        default="Plan a 5-day trip to Tokyo for a couple who loves food and design.",
        help="Travel planning query to send to the agents.",
    )
    parser.add_argument("--config", default="config.json", help="Path to config.json")
    return parser.parse_args()


async def run(query: str, config_path: str) -> None:
    """Full application lifecycle for a single query.

    Args:
        query: The travel planning query to run.
        config_path: Path to the JSON config file.
    """
    new_trace_id()
    trace_id = get_trace_id()
    start = time.monotonic()

    try:
        logger.info("Application started", extra={"trace_id": trace_id})

        load_environment(logger)
        config = load_config(logger, path=config_path)

        planner = create_agent("Planner Agent", "Plan a trip.", config, TravelOutput, logger)
        explorer = create_agent("Explorer Agent", "Find hidden gems.", config, TravelOutput, logger)

        orchestrator = AgentOrchestrator(config, logger)

        logger.info("Processing request", extra={"trace_id": trace_id, "query": query})
        outcomes = await orchestrator.run_multi_agents(query, [planner, explorer])

        display(outcomes, logger)

        duration = round(time.monotonic() - start, 2)
        success_count = sum(1 for o in outcomes if o.succeeded)
        logger.info(
            "Request completed",
            extra={
                "trace_id": trace_id,
                "duration_sec": duration,
                "results_count": len(outcomes),
                "success_count": success_count,
            },
        )

    except Exception:
        logger.exception("Fatal error", extra={"trace_id": trace_id})
        raise


def main() -> None:
    args = parse_args()
    asyncio.run(run(args.query, args.config))


if __name__ == "__main__":
    main()
