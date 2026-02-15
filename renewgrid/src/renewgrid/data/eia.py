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
    token = (api_key or getenv("EIA_KEY") or "").strip()
    if not token:
        raise ValueError(
            "EIA_KEY is not set. Add it to renewgrid/.env or export EIA_KEY in your shell."
        )

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
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        if response.status_code == 403:
            raise PermissionError(
                "EIA API rejected the request (403). Check that EIA_KEY is valid and active."
            ) from exc
        raise
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
