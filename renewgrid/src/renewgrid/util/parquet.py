"""Parquet engine availability checks."""

from __future__ import annotations


def has_parquet_engine() -> bool:
    """Return True when pyarrow is importable."""
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        return False
    return True


def require_parquet_engine() -> None:
    """Raise a clear error when parquet support is unavailable."""
    if not has_parquet_engine():
        raise RuntimeError(
            "Parquet output requires pyarrow. Run `uv sync --extra dev` or "
            '`pip install -e ".[dev]"` and try again.'
        )
