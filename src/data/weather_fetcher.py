"""
Weather and solar irradiance data fetcher using Open-Meteo (free, no API key).

Fetches hourly forecasts and historical ERA5 reanalysis for representative
California grid locations. Solar irradiance is a primary feature for LMP
forecasting because solar over-generation drives negative prices in CAISO.
"""

from __future__ import annotations

import logging
import time
from datetime import date

import httpx
import pandas as pd

log = logging.getLogger(__name__)

# Representative lat/lon for each CAISO hub region
_HUB_COORDINATES: dict[str, tuple[float, float]] = {
    "TH_NP15_GEN-APND": (37.35, -121.90),  # San Jose area
    "TH_SP15_GEN-APND": (34.05, -118.25),  # Los Angeles area
    "TH_ZP26_GEN-APND": (36.75, -119.80),  # Fresno area
}

_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
_HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"

_HOURLY_VARS: list[str] = [
    "shortwave_radiation",
    "direct_radiation",
    "temperature_2m",
    "wind_speed_10m",
    "cloud_cover",
]

_OUTPUT_COLS: list[str] = [
    "time", "node", "shortwave_radiation", "direct_radiation",
    "temperature_2m", "wind_speed_10m", "cloud_cover",
]


class WeatherFetcher:
    """Fetches solar and meteorological data from Open-Meteo.

    No authentication required. Rate limit: ~10,000 calls/day.
    Use get_solar_history() for training data; get_solar_forecast() for live inference.
    """

    def __init__(self, timeout_s: float = 30.0) -> None:
        """Initialize with an httpx client.

        Args:
            timeout_s: HTTP request timeout in seconds.
        """
        self._client = httpx.Client(timeout=timeout_s)

    def get_solar_history(self, node: str, start: date, end: date) -> pd.DataFrame:
        """Fetch historical hourly solar/weather reanalysis (ERA5) for model training.

        Args:
            node: CAISO hub node string.
            start: Inclusive start date.
            end: Inclusive end date.

        Returns:
            DataFrame with columns [time, node, shortwave_radiation, direct_radiation,
            temperature_2m, wind_speed_10m, cloud_cover].
            `time` is UTC timezone-aware, hourly.
        """
        lat, lon = self._get_coords(node)
        log.info("Fetching weather history %s → %s for %s", start, end, node)
        resp = self._client.get(
            _HISTORICAL_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "start_date": start.strftime("%Y-%m-%d"),
                "end_date": end.strftime("%Y-%m-%d"),
                "hourly": ",".join(_HOURLY_VARS),
                "timezone": "UTC",
            },
        )
        resp.raise_for_status()
        return self._parse_response(resp.json(), node)

    def get_solar_forecast(self, node: str, days_ahead: int = 7) -> pd.DataFrame:
        """Fetch hourly solar irradiance + weather forecast for a hub node.

        Args:
            node: CAISO hub node string.
            days_ahead: Number of forecast days (1–16 supported).

        Returns:
            DataFrame with columns matching get_solar_history, UTC timezone.
        """
        lat, lon = self._get_coords(node)
        log.info("Fetching weather forecast %d days for %s", days_ahead, node)
        resp = self._client.get(
            _FORECAST_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "hourly": ",".join(_HOURLY_VARS),
                "forecast_days": days_ahead,
                "timezone": "UTC",
            },
        )
        resp.raise_for_status()
        return self._parse_response(resp.json(), node)

    def get_solar_history_all_nodes(
        self,
        nodes: list[str],
        start: date,
        end: date,
        sleep_s: float = 1.0,
    ) -> pd.DataFrame:
        """Fetch historical weather for multiple nodes, concatenated.

        Args:
            nodes: List of CAISO hub nodes.
            start: Inclusive start date.
            end: Inclusive end date.
            sleep_s: Sleep between requests to respect rate limits.

        Returns:
            Concatenated DataFrame for all nodes.
        """
        frames: list[pd.DataFrame] = []
        for node in nodes:
            df = self.get_solar_history(node, start, end)
            frames.append(df)
            if node != nodes[-1]:
                time.sleep(sleep_s)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=_OUTPUT_COLS)

    @staticmethod
    def _parse_response(data: dict, node: str) -> pd.DataFrame:
        """Parse Open-Meteo JSON response into a DataFrame.

        Args:
            data: Parsed JSON response dict with 'hourly' key.
            node: Hub node string (added as a column).

        Returns:
            DataFrame with columns [time, node, ...weather vars...].
        """
        hourly = data.get("hourly", {})
        times_raw = hourly.get("time", [])
        if not times_raw:
            return pd.DataFrame(columns=_OUTPUT_COLS)

        df = pd.DataFrame({"time": pd.to_datetime(times_raw, utc=True)})
        df["node"] = node
        for var in _HOURLY_VARS:
            df[var] = hourly.get(var, None)

        return df[_OUTPUT_COLS].copy()

    @staticmethod
    def _get_coords(node: str) -> tuple[float, float]:
        """Look up lat/lon for a hub node; raise if unknown.

        Args:
            node: CAISO PNode identifier.

        Returns:
            (latitude, longitude) tuple.
        """
        if node not in _HUB_COORDINATES:
            raise ValueError(
                f"Unknown node '{node}'. Add coordinates to _HUB_COORDINATES. "
                f"Known nodes: {list(_HUB_COORDINATES.keys())}"
            )
        return _HUB_COORDINATES[node]

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def __enter__(self) -> "WeatherFetcher":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
