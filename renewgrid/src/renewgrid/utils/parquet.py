"""Parquet engine availability checks."""

from __future__ import annotations

import importlib


def _importable(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
    except ImportError:
        return False
    return True


def has_parquet_engine() -> bool:
    """Return True when either pyarrow or fastparquet is importable."""
    return _importable("pyarrow") or _importable("fastparquet")


def require_parquet_engine() -> None:
    """Raise a clear error when parquet support is unavailable."""
    if not has_parquet_engine():
        raise RuntimeError(
            "Parquet output requires pyarrow (preferred) or fastparquet. "
            "Install dependencies with `uv sync --extra dev` or "
            '`pip install -e ".[dev]"`.'
        )

