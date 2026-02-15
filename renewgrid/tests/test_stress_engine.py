"""Tests for stress engine."""

from __future__ import annotations

import pandas as pd

from renewgrid.stress.stress_engine import apply_outage


def test_apply_outage_reduces_served_and_updates_unserved() -> None:
    """Outage should reduce served load and increase unserved load."""
    frame = pd.DataFrame({"demand": [100.0], "served": [90.0], "unserved": [10.0]})

    stressed = apply_outage(frame, outage_fraction=0.1)

    assert stressed["served"].iloc[0] == 81.0
    assert stressed["unserved"].iloc[0] == 19.0
