"""
CAISO data fetcher built on top of gridstatus.

Column mapping from gridstatus (verified against live API):
  LMP:        Time→time, Location→node, Market→market, LMP→lmp,
              Energy→energy, Congestion→congestion, Loss→loss
  Load:       Time→time, Load→load_mw
  Fuel mix:   Time→time; wide fuel cols melted → fuel_type, generation_mw
  Load fcst:  Time→time, Load Forecast→forecast_mw, Publish Time→publish_time
  Storage SOC: Interval Start→time, SOC→soc_mwh, Schedule→schedule
  Curtailment: Interval Start→time, Fuel Type→fuel_type,
               Curtailment MW→curtailment_mw, Curtailment Type→curtailment_type,
               Curtailment Reason→reason

Storage SOC reports have a ~5-day data lag and return 404 for recent dates.
"""

from __future__ import annotations

import time
import functools
import logging
from datetime import date, timedelta
from typing import Callable, Iterator, TypeVar

import pandas as pd

import gridstatus  # type: ignore[import]

log = logging.getLogger(__name__)

_REQUEST_SLEEP_S: float = 5.0
_CHUNK_DAYS: int = 30
_MAX_RETRIES: int = 3

# Fuel type columns returned by gridstatus get_fuel_mix() (wide format)
_FUEL_COLS: list[str] = [
    "Solar", "Wind", "Geothermal", "Biomass", "Biogas",
    "Small Hydro", "Coal", "Nuclear", "Natural Gas",
    "Large Hydro", "Batteries", "Imports", "Other",
]

F = TypeVar("F", bound=Callable)


def _with_retry(func: F) -> F:
    """Decorator: retry up to _MAX_RETRIES times with linear backoff."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                last_exc = exc
                if attempt < _MAX_RETRIES - 1:
                    sleep_s = _REQUEST_SLEEP_S * (attempt + 1)
                    log.warning("Retry %d/%d after error: %s (sleeping %.0fs)",
                                attempt + 1, _MAX_RETRIES, exc, sleep_s)
                    time.sleep(sleep_s)
        raise last_exc  # type: ignore[misc]
    return wrapper  # type: ignore[return-value]


class CAISOFetcher:
    """Retry-aware wrapper around gridstatus.CAISO."""

    def __init__(self, api_key: str = "") -> None:
        """Initialize the fetcher.

        Args:
            api_key: Optional GridStatus API key for higher rate limits.
        """
        self._iso = gridstatus.CAISO()

    # ------------------------------------------------------------------
    # LMP
    # ------------------------------------------------------------------

    def get_lmp_history(
        self,
        start: date,
        end: date,
        market: str,
        nodes: list[str],
    ) -> pd.DataFrame:
        """Fetch historical LMPs for a date range, chunking automatically.

        Args:
            start: Inclusive start date.
            end: Inclusive end date.
            market: 'DAY_AHEAD_HOURLY', 'REAL_TIME_5_MIN', or 'REAL_TIME_15_MIN'.
            nodes: List of CAISO PNode identifiers.

        Returns:
            DataFrame with columns [time, node, market, lmp, energy, congestion, loss].
        """
        frames: list[pd.DataFrame] = []
        for chunk_start, chunk_end in self._date_chunks(start, end, _CHUNK_DAYS):
            log.info("Fetching LMP %s %s → %s", market, chunk_start, chunk_end)
            df = self._fetch_lmp_chunk(chunk_start, chunk_end, market, nodes)
            if not df.empty:
                frames.append(df)
            time.sleep(_REQUEST_SLEEP_S)

        if not frames:
            return pd.DataFrame(columns=["time", "node", "market", "lmp", "energy", "congestion", "loss"])

        result = pd.concat(frames, ignore_index=True)
        result = result.drop_duplicates(subset=["time", "node", "market"])
        return result.sort_values(["node", "time"]).reset_index(drop=True)

    @_with_retry
    def _fetch_lmp_chunk(
        self, start: date, end: date, market: str, nodes: list[str]
    ) -> pd.DataFrame:
        raw = self._iso.get_lmp(
            date=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            market=market,
            locations=nodes,
            verbose=False,
        )
        return self._normalize_lmp(raw)

    def get_lmp_latest(self, market: str, nodes: list[str]) -> pd.DataFrame:
        """Fetch the most recent LMP snapshot for live paper-trading.

        Args:
            market: Market resolution string.
            nodes: PNode identifiers.

        Returns:
            Normalized LMP DataFrame.
        """
        raw = self._iso.get_lmp(
            date="latest",
            market=market,
            locations=nodes,
            verbose=False,
        )
        return self._normalize_lmp(raw)

    @staticmethod
    def _normalize_lmp(df: pd.DataFrame) -> pd.DataFrame:
        """Rename and select LMP columns to match our schema."""
        if df.empty:
            return pd.DataFrame(columns=["time", "node", "market", "lmp", "energy", "congestion", "loss"])
        df = df.rename(columns={
            "Time": "time",
            "Location": "node",
            "Market": "market",
            "LMP": "lmp",
            "Energy": "energy",
            "Congestion": "congestion",
            "Loss": "loss",
        })
        return df[["time", "node", "market", "lmp", "energy", "congestion", "loss"]].copy()

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def get_load_history(self, start: date, end: date) -> pd.DataFrame:
        """Fetch CAISO system load (actual, 5-min) for a date range.

        Args:
            start: Inclusive start date.
            end: Inclusive end date.

        Returns:
            DataFrame with columns [time, load_mw].
        """
        frames: list[pd.DataFrame] = []
        for chunk_start, chunk_end in self._date_chunks(start, end, _CHUNK_DAYS):
            log.info("Fetching load %s → %s", chunk_start, chunk_end)
            df = self._fetch_load_chunk(chunk_start, chunk_end)
            if not df.empty:
                frames.append(df)
            time.sleep(_REQUEST_SLEEP_S)

        if not frames:
            return pd.DataFrame(columns=["time", "load_mw"])
        result = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["time"])
        return result.sort_values("time").reset_index(drop=True)

    @_with_retry
    def _fetch_load_chunk(self, start: date, end: date) -> pd.DataFrame:
        raw = self._iso.get_load(
            date=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            verbose=False,
        )
        if raw.empty:
            return pd.DataFrame(columns=["time", "load_mw"])
        df = raw.rename(columns={"Time": "time", "Load": "load_mw"})
        return df[["time", "load_mw"]].copy()

    def get_load_forecast_latest(self) -> pd.DataFrame:
        """Fetch the current day-ahead load forecast.

        Returns:
            DataFrame with columns [time, forecast_mw, publish_time].
        """
        raw = self._iso.get_load_forecast(date="latest", verbose=False)
        return self._normalize_load_forecast(raw)

    @staticmethod
    def _normalize_load_forecast(df: pd.DataFrame) -> pd.DataFrame:
        """Rename load forecast columns to match our schema."""
        if df.empty:
            return pd.DataFrame(columns=["time", "forecast_mw", "publish_time"])
        df = df.rename(columns={
            "Time": "time",
            "Load Forecast": "forecast_mw",
            "Publish Time": "publish_time",
        })
        return df[["time", "forecast_mw", "publish_time"]].copy()

    # ------------------------------------------------------------------
    # Fuel mix
    # ------------------------------------------------------------------

    def get_fuel_mix_history(self, start: date, end: date) -> pd.DataFrame:
        """Fetch hourly fuel mix (melted to long format) for a date range.

        Args:
            start: Inclusive start date.
            end: Inclusive end date.

        Returns:
            DataFrame with columns [time, fuel_type, generation_mw].
        """
        frames: list[pd.DataFrame] = []
        for chunk_start, chunk_end in self._date_chunks(start, end, _CHUNK_DAYS):
            log.info("Fetching fuel mix %s → %s", chunk_start, chunk_end)
            df = self._fetch_fuel_mix_chunk(chunk_start, chunk_end)
            if not df.empty:
                frames.append(df)
            time.sleep(_REQUEST_SLEEP_S)

        if not frames:
            return pd.DataFrame(columns=["time", "fuel_type", "generation_mw"])
        result = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["time", "fuel_type"])
        return result.sort_values(["time", "fuel_type"]).reset_index(drop=True)

    @_with_retry
    def _fetch_fuel_mix_chunk(self, start: date, end: date) -> pd.DataFrame:
        raw = self._iso.get_fuel_mix(
            date=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            verbose=False,
        )
        if raw.empty:
            return pd.DataFrame(columns=["time", "fuel_type", "generation_mw"])
        raw = raw.rename(columns={"Time": "time"})
        # Present fuel cols that actually exist (varies by date range)
        available_fuel_cols = [c for c in _FUEL_COLS if c in raw.columns]
        melted = raw[["time"] + available_fuel_cols].melt(
            id_vars="time", var_name="fuel_type", value_name="generation_mw"
        )
        return melted.copy()

    # ------------------------------------------------------------------
    # Storage SOC
    # ------------------------------------------------------------------

    def get_storage_soc_history(self, start: date, end: date) -> pd.DataFrame:
        """Fetch fleet-wide battery storage SOC (hourly, IFM schedule).

        CAISO storage reports have a ~5-day data lag. Dates within the lag
        window are skipped with a warning rather than raising an error.

        Args:
            start: Inclusive start date.
            end: Inclusive end date.

        Returns:
            DataFrame with columns [time, soc_mwh, schedule].
        """
        frames: list[pd.DataFrame] = []
        # Fetch day-by-day since the SOC report is per-day
        current = start
        while current <= end:
            df = self._fetch_soc_day(current)
            if df is not None and not df.empty:
                frames.append(df)
            current += timedelta(days=1)
            time.sleep(2.0)  # lighter sleep for per-day fetches

        if not frames:
            return pd.DataFrame(columns=["time", "soc_mwh", "schedule"])
        result = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["time", "schedule"])
        return result.sort_values("time").reset_index(drop=True)

    def _fetch_soc_day(self, day: date) -> pd.DataFrame | None:
        """Fetch storage SOC for a single day; return None on 404."""
        try:
            raw = self._iso.get_storage_soc_hourly(
                date=day.strftime("%Y-%m-%d"), verbose=False
            )
        except Exception as exc:
            if "404" in str(exc) or "No Daily Energy Storage" in str(exc):
                log.debug("Storage SOC unavailable for %s (lag window)", day)
                return None
            raise
        if raw.empty:
            return None
        df = raw.rename(columns={"Interval Start": "time", "SOC": "soc_mwh", "Schedule": "schedule"})
        return df[["time", "soc_mwh", "schedule"]].copy()

    # ------------------------------------------------------------------
    # Curtailment
    # ------------------------------------------------------------------

    def get_curtailment_history(self, start: date, end: date) -> pd.DataFrame:
        """Fetch renewable curtailment events.

        Args:
            start: Inclusive start date.
            end: Inclusive end date.

        Returns:
            DataFrame with columns [time, fuel_type, curtailment_mw, curtailment_type, reason].
        """
        frames: list[pd.DataFrame] = []
        for chunk_start, chunk_end in self._date_chunks(start, end, _CHUNK_DAYS):
            log.info("Fetching curtailment %s → %s", chunk_start, chunk_end)
            df = self._fetch_curtailment_chunk(chunk_start, chunk_end)
            if not df.empty:
                frames.append(df)
            time.sleep(_REQUEST_SLEEP_S)

        if not frames:
            return pd.DataFrame(columns=["time", "fuel_type", "curtailment_mw", "curtailment_type", "reason"])
        result = pd.concat(frames, ignore_index=True).drop_duplicates(
            subset=["time", "fuel_type", "curtailment_type", "reason"]
        )
        return result.sort_values(["time", "fuel_type"]).reset_index(drop=True)

    @_with_retry
    def _fetch_curtailment_chunk(self, start: date, end: date) -> pd.DataFrame:
        raw = self._iso.get_curtailment(
            date=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            verbose=False,
        )
        if raw.empty:
            return pd.DataFrame(columns=["time", "fuel_type", "curtailment_mw", "curtailment_type", "reason"])
        df = raw.rename(columns={
            "Interval Start": "time",
            "Fuel Type": "fuel_type",
            "Curtailment MW": "curtailment_mw",
            "Curtailment Type": "curtailment_type",
            "Curtailment Reason": "reason",
        })
        df["curtailment_type"] = df["curtailment_type"].fillna("Unknown")
        df["reason"] = df["reason"].fillna("Unknown")
        return df[["time", "fuel_type", "curtailment_type", "reason", "curtailment_mw"]].copy()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _date_chunks(start: date, end: date, chunk_days: int) -> Iterator[tuple[date, date]]:
        """Yield (chunk_start, chunk_end) pairs covering [start, end].

        Args:
            start: Range start (inclusive).
            end: Range end (inclusive).
            chunk_days: Maximum days per chunk.

        Yields:
            (chunk_start, chunk_end) tuples.
        """
        current = start
        while current <= end:
            chunk_end = min(current + timedelta(days=chunk_days - 1), end)
            yield current, chunk_end
            current = chunk_end + timedelta(days=1)
