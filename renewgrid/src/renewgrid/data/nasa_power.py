"""NASA POWER daily endpoint client."""

from __future__ import annotations

from datetime import date
from typing import Iterable

import pandas as pd
import requests

NASA_POWER_DAILY_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"


def fetch_daily_weather(
    latitude: float,
    longitude: float,
    start_date: date,
    end_date: date,
    parameters: Iterable[str] = ("T2M", "WS10M", "ALLSKY_SFC_SW_DWN"),
) -> pd.DataFrame:
    """Fetch NASA POWER daily weather parameters for one location."""
    parameter_list = list(parameters)
    params = {
        "parameters": ",".join(parameter_list),
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

    values_by_param = payload["properties"]["parameter"]
    index = pd.to_datetime(list(values_by_param[parameter_list[0]].keys()), format="%Y%m%d")
    frame = pd.DataFrame({"date": index})
    for parameter in parameter_list:
        frame[f"weather_{parameter.lower()}"] = list(values_by_param[parameter].values())
    frame["source"] = "nasa_power"
    frame["latitude"] = latitude
    frame["longitude"] = longitude
    return frame


def fetch_daily_solar(
    latitude: float,
    longitude: float,
    start_date: date,
    end_date: date,
    parameter: str = "ALLSKY_SFC_SW_DWN",
) -> pd.DataFrame:
    """Fetch a daily NASA POWER time series for one location."""
    weather = fetch_daily_weather(
        latitude=latitude,
        longitude=longitude,
        start_date=start_date,
        end_date=end_date,
        parameters=(parameter,),
    )
    value_col = f"weather_{parameter.lower()}"
    frame = weather.rename(columns={value_col: "value"})[
        ["date", "value", "source", "latitude", "longitude"]
    ]
    return frame
