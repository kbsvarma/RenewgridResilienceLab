"""Page wrapper for monitor map component."""

from __future__ import annotations

from pathlib import Path

from renewgrid.app.components.monitor_map import render_monitor_map as render_monitor_map_component


def render_monitor_map(base_dir: Path) -> None:
    """Render monitor map tab using shared component."""
    render_monitor_map_component(base_dir=base_dir, key_prefix="monitor_tab")

