"""Tests for NASA POWER sentinel cleanup."""

from __future__ import annotations

import pandas as pd

from renewgrid.data.nasa_power import clean_power_values


def test_clean_power_values_replaces_known_sentinels() -> None:
    """Sentinel weather values should be converted to NaN."""
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "weather_t2m": [-999.0, 18.2],
            "weather_ws10m": [-999, 5.1],
            "weather_allsky_sfc_sw_dwn": [3.2, -901.0],
        }
    )
    cleaned = clean_power_values(frame)
    assert pd.isna(cleaned.loc[0, "weather_t2m"])
    assert pd.isna(cleaned.loc[0, "weather_ws10m"])
    assert pd.isna(cleaned.loc[1, "weather_allsky_sfc_sw_dwn"])
    assert cleaned.loc[1, "weather_t2m"] == 18.2
