# Travel Agent System

A production-grade, multi-agent AI orchestration system for travel planning, built on the OpenAI Agents SDK with structured outputs, caching, retries, circuit breaking, and full observability.

## 📌 Project Overview

The system takes a natural-language travel query (e.g. *"Plan a 5-day trip to Tokyo"*) and runs it concurrently through two specialized agents — a **Planner** and an **Explorer** — each returning a validated, structured `TravelOutput` (destination, duration, summary). Results are cached, retried on failure with exponential backoff, and protected by a circuit breaker so a failing upstream API doesn't get hammered.

This is a refactor of an original single-file prototype. The refactor fixed three runtime-breaking bugs (entry point called `main()` with a missing required argument, `create_agent()` called with a missing argument, and a config type mismatch — a `dict` passed where a Pydantic model was expected) and restructured the code into a modular package.

## 🏗 Architecture

```
travel_agent_system/
├── main.py            # CLI entry point, wires everything together
├── env_config.py       # .env loading & validation
├── config_loader.py     # config.json loading -> typed AppConfig
├── schemas.py           # Pydantic models: AppConfig, TravelOutput
├── logging_setup.py      # Structured JSON + console logging, trace_id
├── cache.py              # TTL result cache (normalized keys)
├── resilience.py          # CircuitBreaker, BudgetManager, retry-with-backoff
├── agents_factory.py       # Agent construction
└── orchestrator.py         # AgentOrchestrator: runs agents, parses output
```

**Key design decision — no global mutable state.** The original version used module-level globals (`REQUEST_COUNT`, `SEMAPHORE`, a bare `cache` dict, a bare `circuit_breaker` instance) mutated via the `global` keyword. Every piece of shared state is now encapsulated inside `AgentOrchestrator`, constructed once at app startup. This makes the system testable in isolation and safe to run multiple independent instances (e.g. per-tenant) in the same process.

**Data flow:**
1. `main.py` loads environment + config, builds two agents via `agents_factory`.
2. `AgentOrchestrator.run_multi_agents()` checks the cache; on a miss, it runs every agent **concurrently** (bounded by `max_concurrency`), each call wrapped by `resilience.run_with_retry()`.
3. Each agent's raw output is validated into a `TravelOutput` by `orchestrator.parse_output()`.
4. Outcomes (success or failure, per agent) are returned and logged by `orchestrator.display()`.

## ⚙️ Features

- **Multi-agent orchestration** — Planner + Explorer agents run concurrently, not sequentially, bounded by a semaphore.
- **Structured, validated output** — every agent response is parsed and validated against a Pydantic schema; malformed output is rejected, logged, and reported as a failure rather than silently producing bad data.
- **Retry with exponential backoff + jitter** — configurable attempts and base delay.
- **Circuit breaker** — CLOSED → OPEN → HALF_OPEN, async-safe (a single lock guards both the trip-check and the state transition, closing a race window present in the original implementation).
- **Request budget** — a hard ceiling on total API calls per orchestrator instance, enforced under lock.
- **TTL caching** — results are cached by normalized query string; an all-failed run is never cached as if it were valid.
- **Structured logging & tracing** — every log line (console and file) carries a single `trace_id` per logical request, generated once and propagated via `contextvars` — no more disconnected trace IDs across the call stack.
- **Config-driven** — every tunable (timeouts, retries, concurrency, cache TTL, circuit breaker thresholds) lives in `config.json` / `AppConfig`, not hardcoded.

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set environment variables

```bash
cp .env.example .env
# edit .env and set OPENAI_API_KEY
```

### 3. Configure the app (optional — sensible defaults apply if skipped)

```bash
cp config.example.json config.json
# edit config.json to taste
```

### 4. Run

```bash
python -m travel_agent_system.main "Plan a 5-day trip to Tokyo for a couple who loves food and design."
```

If no query is passed, a default example query is used.

## 📊 Observability

Every log line, in both the console and `app.log`, includes a `trace_id` that is generated once per request (in `main.run()` / `AgentOrchestrator.run_multi_agents()`) and is stable across every agent call, retry attempt, and cache lookup for that request — making it possible to grep a single request's full lifecycle out of the logs.

**Console** (human-readable, colorized):
```
2026-07-05 10:00:01 [INFO    ] [travel_agent_system] [trace=3f2a1c9e-...] Running Planner Agent
```

**File** (`app.log`, JSON — ELK/Datadog/Grafana Loki ready):
```json
{"timestamp": "2026-07-05 10:00:01", "level": "INFO", "logger": "travel_agent_system", "message": "Running Planner Agent", "trace_id": "3f2a1c9e-...", "attempt": 1, "request_count": 1, "circuit_state": "CLOSED"}
```

Lifecycle events logged: application start, environment/config load, per-agent start/success/failure with attempt number and circuit state, cache hit/miss, multi-agent run completion (success/failure counts + duration), request completion, and any fatal error with full traceback.

## 🔮 Future Improvements

- **Distributed rate limiting / budget** — the current `BudgetManager` and cache are in-process; for multi-instance deployment, back them with Redis (e.g. `redis-py` + a token-bucket script) so budget and cache are shared across replicas.
- **OpenTelemetry integration** — swap the custom `trace_id` contextvar for OTel spans to get distributed tracing across service boundaries for free.
- **Pluggable agent registry** — agents are currently hardcoded (Planner, Explorer) in `main.py`; a registry/config-driven agent list would let new agents be added without code changes.
- **Structured partial-failure reporting** — `AgentRunOutcome` already tracks per-agent failure reasons; a follow-up could surface these to the caller/API response rather than just logging them.
- **Async-native file I/O** — `config_loader.py` uses synchronous file reads; fine at startup, but would want `aiofiles` if config reload becomes a runtime feature.

## ⚠️ Notes

- All original functionality (multi-agent execution, caching, retries, circuit breaking, budget limiting, structured output validation) is preserved and extended — nothing was removed, only fixed and reorganized.
- Three runtime-breaking bugs from the original were fixed: the missing `query` argument to `main()`, the missing `output_type` argument to `create_agent()`, and the `dict`/`AgentConfig` type mismatch.
