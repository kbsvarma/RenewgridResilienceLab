"""Backward-compatible parquet helpers.

Use ``renewgrid.utils.parquet`` as the canonical import path.
"""

from __future__ import annotations

from renewgrid.utils.parquet import has_parquet_engine, require_parquet_engine

__all__ = ["has_parquet_engine", "require_parquet_engine"]
