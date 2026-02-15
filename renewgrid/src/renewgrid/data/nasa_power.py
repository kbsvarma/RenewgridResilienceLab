"""NASA POWER daily endpoint client."""

from __future__ import annotations

from datetime import date

import pandas as pd
import requests

NASA_POWER_DAILY_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"


def fetch_daily_solar(
    latitude: float,
    longitude: float,
    start_date: date,
    end_date: date,
    parameter: str = "ALLSKY_SFC_SW_DWN",
) -> pd.DataFrame:
    """Fetch a daily NASA POWER time series for one location."""
    params = {
        "parameters": parameter,
        "community": "RE",
        "longitude": longitude,
        "latitude": latitude,
        "start": start_date.strftime("%Y%m%d"),
        "end": end_date.strftime("%Y%m%d"),
        "format": "JSON",
    }
    response = requests.get(NASA_POWER_DAILY_URL, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()

    values = payload["properties"]["parameter"][parameter]
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(list(values.keys()), format="%Y%m%d"),
            "value": list(values.values()),
            "source": "nasa_power",
            "latitude": latitude,
            "longitude": longitude,
        }
    )
    return frame
