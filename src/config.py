"""
Configuration management via environment variables.
All config flows through here — no magic strings scattered in code.
"""

import os
from dotenv import load_dotenv

# Load .env file if present (local dev); Railway injects env vars directly
load_dotenv()


def _get_env(key: str, default: str | None = None, required: bool = False) -> str:
    """Get environment variable with optional default and required check."""
    value = os.getenv(key, default)
    if required and value is None:
        raise ValueError(f"Required environment variable {key} is not set")
    return value


def _get_env_float(key: str, default: float) -> float:
    """Get environment variable as float."""
    return float(os.getenv(key, str(default)))


def _get_env_int(key: str, default: int) -> int:
    """Get environment variable as int."""
    return int(os.getenv(key, str(default)))


# Execution mode
MODE = _get_env("MODE", "paper")  # "paper" or "live"
IS_PAPER_MODE = MODE == "paper"

# Polymarket API credentials
POLYMARKET_API_KEY = _get_env("POLYMARKET_API_KEY", "")
POLYMARKET_API_SECRET = _get_env("POLYMARKET_API_SECRET", "")
POLYMARKET_WALLET_ADDRESS = _get_env("POLYMARKET_WALLET_ADDRESS", "")
POLYMARKET_PRIVATE_KEY = _get_env("POLYMARKET_PRIVATE_KEY", "")

# Strategy parameters
THRESHOLD_ENTRY = _get_env_float("THRESHOLD_ENTRY", 0.51)
DIP_THRESHOLD_POINTS = _get_env_int("DIP_THRESHOLD_POINTS", 10)
POSITION_SIZE_USD = _get_env_float("POSITION_SIZE_USD", 100.0)

# Time windows (hours to resolution)
WINDOW_OUTER_HOURS = _get_env_int("WINDOW_OUTER_HOURS", 48)
WINDOW_INNER_HOURS = _get_env_int("WINDOW_INNER_HOURS", 12)

# Logging
LOG_LEVEL = _get_env("LOG_LEVEL", "INFO")
LOG_FILE_PATH = _get_env("LOG_FILE_PATH", "data/signals.jsonl")

# Health check
HEALTH_CHECK_PORT = _get_env_int("HEALTH_CHECK_PORT", 8080)

# Polymarket endpoints (not configurable, but centralized)
POLYMARKET_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
POLYMARKET_REST_URL = "https://clob.polymarket.com"
POLYMARKET_GAMMA_URL = "https://gamma-api.polymarket.com"