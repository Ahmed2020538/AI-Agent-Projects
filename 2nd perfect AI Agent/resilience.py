"""
===============================================================================
Resilience Layer
===============================================================================
Short Description:
- CircuitBreaker: async-safe (single lock guards state transitions AND the
  can_execute check, closing the race window the original version had).
- BudgetManager: replaces the module-level REQUEST_COUNT / MAX_REQUESTS
  globals with an encapsulated, lock-protected counter.
- run_with_retry: generic retry-with-exponential-backoff-and-jitter wrapper,
  parameterized over any async callable, honoring the circuit breaker,
  a concurrency semaphore, a request budget, and a timeout — all injected,
  none global.
===============================================================================
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, TypeVar

from .logging_setup import get_trace_id

T = TypeVar("T")


class CircuitOpenError(RuntimeError):
    """Raised when a call is attempted while the circuit breaker is OPEN."""


class BudgetExceededError(RuntimeError):
    """Raised when the configured request budget has been exhausted."""


class CircuitBreaker:
    """Async-safe circuit breaker (CLOSED -> OPEN -> HALF_OPEN -> CLOSED).

    Prevents hammering a failing downstream dependency: after
    `failure_threshold` consecutive failures the circuit opens and rejects
    calls immediately until `recovery_time` seconds have elapsed, at which
    point a single trial call is allowed through (HALF_OPEN).
    """

    def __init__(self, failure_threshold: int = 3, recovery_time: float = 30.0):
        self.failure_threshold = failure_threshold
        self.recovery_time = recovery_time
        self._failures = 0
        self._last_failure_time: float = 0.0
        self._state = "CLOSED"
        self._lock = asyncio.Lock()

    @property
    def state(self) -> str:
        return self._state

    async def guard(self) -> None:
        """Raise CircuitOpenError if the circuit is OPEN and hasn't cooled down."""
        async with self._lock:
            if self._state == "OPEN":
                if time.monotonic() - self._last_failure_time > self.recovery_time:
                    self._state = "HALF_OPEN"
                else:
                    raise CircuitOpenError("Circuit is OPEN - skipping execution")

    async def record_success(self) -> None:
        async with self._lock:
            self._failures = 0
            self._state = "CLOSED"

    async def record_failure(self) -> None:
        async with self._lock:
            self._failures += 1
            self._last_failure_time = time.monotonic()
            if self._failures >= self.failure_threshold:
                self._state = "OPEN"


@dataclass
class BudgetManager:
    """Encapsulated request budget (replaces global REQUEST_COUNT/MAX_REQUESTS)."""

    max_requests: int
    _count: int = field(default=0, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    async def consume(self) -> int:
        """Consume one unit of budget; raises BudgetExceededError if exhausted."""
        async with self._lock:
            if self._count >= self.max_requests:
                raise BudgetExceededError("Request budget exceeded")
            self._count += 1
            return self._count

    @property
    def used(self) -> int:
        return self._count


async def run_with_retry(
    operation: Callable[[], Awaitable[T]],
    *,
    operation_name: str,
    circuit_breaker: CircuitBreaker,
    budget: BudgetManager,
    semaphore: asyncio.Semaphore,
    attempts: int,
    base_delay: float,
    timeout_seconds: float,
    logger: logging.Logger,
) -> T:
    """Execute `operation` with retries, exponential backoff+jitter, a circuit
    breaker, a concurrency semaphore, a request budget, and a timeout.

    Args:
        operation: Zero-arg async callable to execute (wrap your real call in
            a lambda/closure so this function stays generic).
        operation_name: Human-readable name for logging.
        circuit_breaker: Shared circuit breaker instance.
        budget: Shared request budget instance.
        semaphore: Concurrency limiter.
        attempts: Max attempts before giving up.
        base_delay: Base delay (seconds) for exponential backoff.
        timeout_seconds: Per-attempt timeout.
        logger: Logger for lifecycle/observability events.

    Returns:
        T: The operation's result.

    Raises:
        Exception: Re-raises the last error once all attempts are exhausted.
    """
    trace_id = get_trace_id()

    for attempt in range(1, attempts + 1):
        try:
            await circuit_breaker.guard()

            async with semaphore:
                request_number = await budget.consume()

                logger.info(
                    f"Running {operation_name}",
                    extra={
                        "trace_id": trace_id,
                        "attempt": attempt,
                        "request_count": request_number,
                        "circuit_state": circuit_breaker.state,
                    },
                )

                result = await asyncio.wait_for(operation(), timeout=timeout_seconds)

            await circuit_breaker.record_success()
            logger.info(f"{operation_name} succeeded", extra={"trace_id": trace_id, "attempt": attempt})
            return result

        except Exception as exc:  # noqa: BLE001 - intentional: any failure counts against the breaker
            await circuit_breaker.record_failure()

            logger.warning(
                f"{operation_name} attempt {attempt} failed",
                extra={
                    "trace_id": trace_id,
                    "attempt": attempt,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "circuit_state": circuit_breaker.state,
                },
            )

            if attempt == attempts:
                logger.error(
                    f"{operation_name} exhausted all retries",
                    extra={"trace_id": trace_id, "attempts": attempts},
                )
                raise

            sleep_time = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 1)
            await asyncio.sleep(sleep_time)

    # Unreachable, but keeps type checkers happy.
    raise RuntimeError("run_with_retry exited loop without returning or raising")
