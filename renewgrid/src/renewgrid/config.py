"""Configuration and region presets for RenewGrid."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class RegionPreset:
    """Region-level defaults used by data connectors."""

    name: str
    eia_respondent: str
    latitude: float
    longitude: float


REGION_PRESETS: dict[str, RegionPreset] = {
    "CAISO": RegionPreset("CAISO", "CISO", 36.7783, -119.4179),
    "ERCOT": RegionPreset("ERCOT", "ERCO", 31.9686, -99.9018),
}


def load_environment(env_path: str | Path = ".env") -> None:
    """Load environment variables from a local .env file if present."""
    load_dotenv(Path(env_path), override=False)
