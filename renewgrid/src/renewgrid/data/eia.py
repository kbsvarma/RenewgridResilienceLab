"""EIA Open Data API client for RTO/BA demand/generation."""

from __future__ import annotations

from datetime import date
from os import getenv

import pandas as pd
import requests

EIA_RTO_REGION_DATA_URL = "https://api.eia.gov/v2/electricity/rto/region-data/data/"


def fetch_rto_daily(
    respondent: str,
    start_date: date,
    end_date: date,
    value_type: str = "D",
    api_key: str | None = None,
) -> pd.DataFrame:
    """Fetch EIA RTO/BA region data at hourly frequency for a day-scale window."""
    token = api_key or getenv("EIA_KEY")
    params = {
        "api_key": token,
        "frequency": "hourly",
        "data[0]": "value",
        "facets[respondent][]": respondent,
        "facets[type][]": value_type,
        "start": start_date.isoformat(),
        "end": end_date.isoformat(),
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "offset": 0,
        "length": 5000,
    }

    response = requests.get(EIA_RTO_REGION_DATA_URL, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()
    records = payload.get("response", {}).get("data", [])

    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        return pd.DataFrame(columns=["date", "value", "source", "region"])

    frame = frame.rename(columns={"period": "date"})
    frame["date"] = pd.to_datetime(frame["date"])  # type: ignore[assignment]
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame["source"] = "eia"
    frame["region"] = respondent
    return frame[["date", "value", "source", "region"]]
