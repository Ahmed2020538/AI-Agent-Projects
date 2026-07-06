"""
===============================================================================
Logging & Tracing
===============================================================================
Short Description:
- Provides a single, unified trace_id (per logical request) propagated via
  contextvars across every coroutine spawned from that request.
- Structured JSON logs to a rotating file (ELK / Datadog / Grafana Loki ready).
- Human-readable, colorized console logs for local development.
- Idempotent logger setup (safe to call setup_logger() multiple times without
  duplicating handlers).
===============================================================================
"""

from __future__ import annotations

import json
import logging
import os
import sys
import uuid
from contextvars import ContextVar
from logging.handlers import RotatingFileHandler

# -----------------------------------------------------------------------------
# Trace ID (per logical request)
# -----------------------------------------------------------------------------
# A single ContextVar is used across the whole system. Previous versions of
# this code had two disconnected trace-id systems (request_id / trace_id) —
# that has been consolidated into one, so every log line for a single
# request/query carries the SAME trace_id end-to-end.
# -----------------------------------------------------------------------------
_trace_id_var: ContextVar[str] = ContextVar("trace_id", default="-")


def new_trace_id() -> str:
    """Generate and bind a new trace_id to the current async context.

    Returns:
        str: The newly generated trace_id.
    """
    trace_id = str(uuid.uuid4())
    _trace_id_var.set(trace_id)
    return trace_id


def get_trace_id() -> str:
    """Return the trace_id bound to the current async context ('-' if unset)."""
    return _trace_id_var.get()


class JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON for machine ingestion."""

    def format(self, record: logging.LogRecord) -> str:
        log_record = {
            "timestamp": self.formatTime(record, "%Y-%m-%d %H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "trace_id": get_trace_id(),
            "file": record.filename,
            "line": record.lineno,
            "function": record.funcName,
        }

        # Merge any `extra={...}` fields passed to the logging call, skipping
        # anything that isn't JSON-serializable so a bad field never crashes
        # logging itself.
        for key, value in record.__dict__.items():
            if key in log_record or key.startswith("_"):
                continue
            if key in _STANDARD_LOG_RECORD_ATTRS:
                continue
            try:
                json.dumps(value)
            except (TypeError, ValueError):
                log_record.setdefault("_dropped_extra_fields", []).append(key)
                continue
            log_record[key] = value

        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_record, default=str)


# Standard attributes every LogRecord has - used to distinguish genuine
# `extra=` fields from logging internals when building JSON output.
_STANDARD_LOG_RECORD_ATTRS = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "message", "taskName",
}

_LOG_COLORS = {
    "DEBUG": "\033[94m",
    "INFO": "\033[92m",
    "WARNING": "\033[93m",
    "ERROR": "\033[91m",
    "CRITICAL": "\033[95m",
}
_RESET_COLOR = "\033[0m"


class ConsoleFormatter(logging.Formatter):
    """Human-readable, colorized console formatter for local development."""

    def format(self, record: logging.LogRecord) -> str:
        color = _LOG_COLORS.get(record.levelname, "")
        base = (
            f"{self.formatTime(record)} "
            f"{color}[{record.levelname:<8}]{_RESET_COLOR} "
            f"[{record.name}] "
            f"[trace={get_trace_id()}] "
            f"{record.getMessage()}"
        )
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


def setup_logger(name: str = "travel_agent_system", log_file: str = "app.log") -> logging.Logger:
    """Create (or fetch) an idempotently-configured logger.

    Args:
        name: Logger name.
        log_file: Path to the rotating JSON log file.

    Returns:
        logging.Logger: A logger with console + rotating-file handlers attached
        exactly once, regardless of how many times this is called.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logger.setLevel(log_level)

    file_handler = RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=5)
    file_handler.setFormatter(JSONFormatter())

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(ConsoleFormatter())

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.propagate = False

    return logger
