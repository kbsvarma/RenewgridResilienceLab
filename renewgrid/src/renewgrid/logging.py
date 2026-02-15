"""Project logging helpers."""

from __future__ import annotations

import logging


def setup_logging(level: int = logging.INFO) -> None:
    """Initialize package-level logging format and level."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
