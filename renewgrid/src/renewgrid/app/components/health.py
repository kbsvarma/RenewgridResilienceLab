"""Health checklist renderer."""

from __future__ import annotations

import streamlit as st


def render_health_checklist(flags: dict[str, bool], messages: dict[str, str]) -> None:
    """Render pipeline health checks as pass/fail bullets."""
    st.markdown("### Health Checklist")
    ordered = [
        "dataset_loaded",
        "schema_valid",
        "units_valid",
        "eval_ran",
        "snapshot_saved",
    ]
    for key in ordered:
        icon = "✅" if flags.get(key, False) else "⚠️"
        st.markdown(f"- {icon} **{key}**: {messages.get(key, 'not run')} ")
