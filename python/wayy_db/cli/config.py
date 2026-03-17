"""Configuration management for the WayyDB CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


CONFIG_DIR = Path.home() / ".wayy"
CONFIG_FILE = CONFIG_DIR / "config.json"

DEFAULTS: dict[str, Any] = {
    "server_url": "http://localhost:8080",
    "format": "table",
    "db_name": "default",
}


def load_config() -> dict[str, Any]:
    """Load config from ~/.wayy/config.json, creating defaults if missing."""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            return {**DEFAULTS, **json.load(f)}
    return dict(DEFAULTS)


def save_config(config: dict[str, Any]) -> None:
    """Save config to ~/.wayy/config.json."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)


def get_server_url() -> str:
    """Get the configured server URL."""
    return load_config()["server_url"]


def get_db_name() -> str:
    """Get the configured database name."""
    return load_config()["db_name"]
