"""
Structured logging to stdout (Railway captures) and JSONL file (local persistence).
"""

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import orjson

from src.config import LOG_LEVEL, LOG_FILE_PATH


# Log level ordering
_LEVEL_ORDER = {"DEBUG": 0, "INFO": 1, "WARN": 2, "ERROR": 3}
_CURRENT_LEVEL = _LEVEL_ORDER.get(LOG_LEVEL.upper(), 1)

# Ensure log directory exists
_log_path = Path(LOG_FILE_PATH)
_log_path.parent.mkdir(parents=True, exist_ok=True)


def _should_log(level: str) -> bool:
    """Check if message at given level should be logged."""
    return _LEVEL_ORDER.get(level.upper(), 1) >= _CURRENT_LEVEL


def _format_record(level: str, event: str, **fields: Any) -> bytes:
    """Format log record as JSON bytes."""
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "level": level.upper(),
        "event": event,
        **fields,
    }
    return orjson.dumps(record)


def _emit(level: str, event: str, **fields: Any) -> None:
    """Emit log record to stdout and file."""
    if not _should_log(level):
        return

    record_bytes = _format_record(level, event, **fields)
    record_str = record_bytes.decode("utf-8")

    # stdout (Railway captures this)
    print(record_str, file=sys.stdout, flush=True)

    # JSONL file (local persistence)
    with open(_log_path, "a") as f:
        f.write(record_str + "\n")


def debug(event: str, **fields: Any) -> None:
    """Log debug message."""
    _emit("DEBUG", event, **fields)


def info(event: str, **fields: Any) -> None:
    """Log info message."""
    _emit("INFO", event, **fields)


def warn(event: str, **fields: Any) -> None:
    """Log warning message."""
    _emit("WARN", event, **fields)


def error(event: str, **fields: Any) -> None:
    """Log error message."""
    _emit("ERROR", event, **fields)


def signal(event_type: str, **fields: Any) -> None:
    """
    Log a trading signal. Always logged regardless of level.
    Use this for threshold crossings, paper trades, etc.
    """
    record_bytes = _format_record("SIGNAL", event_type, **fields)
    record_str = record_bytes.decode("utf-8")

    print(record_str, file=sys.stdout, flush=True)
    with open(_log_path, "a") as f:
        f.write(record_str + "\n")