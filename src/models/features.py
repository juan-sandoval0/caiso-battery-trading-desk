"""
Feature engineering for LMP price forecasting models.

Transforms raw LMP + load + weather DataFrames into model-ready feature matrices.
All features are computed from publicly observable data available before the
forecast horizon, preventing look-ahead bias.

Feature groups:
    Temporal   — hour-of-day, day-of-week, month, holiday flag, is_weekend
    LMP lags   — t-1h, t-2h, t-24h, t-48h, t-168h (1 week) LMP values
    Rolling    — 4h, 24h, 168h rolling mean and std of LMP
    Load       — current load, 24h-ahead load forecast
    Solar      — shortwave irradiance (current + 1h ahead), cloud cover
    Wind       — wind speed proxy
    Spread     — DA LMP minus previous-day RT LMP (arbitrage signal)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Lag offsets in hours for RT LMP features
_LMP_LAG_HOURS: list[int] = [1, 2, 3, 4, 24, 48, 168]

# Rolling window sizes in hours
_ROLLING_WINDOWS_H: list[int] = [4, 24, 168]

_WEATHER_COLS: list[str] = [
    "shortwave_radiation", "direct_radiation", "temperature_2m",
    "wind_speed_10m", "cloud_cover",
]

try:
    import holidays as _holidays_lib
    _US_HOLIDAYS: set = set(_holidays_lib.US(years=range(2020, 2030)).keys())

    def _is_holiday(ts: pd.Timestamp) -> int:
        return int(ts.date() in _US_HOLIDAYS)
except ImportError:
    def _is_holiday(ts: pd.Timestamp) -> int:  # type: ignore[misc]
        return 0


def build_feature_matrix(
    lmp_df: pd.DataFrame,
    load_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    target_col: str = "lmp",
    forecast_horizon_h: int = 24,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build a supervised feature matrix (X) and target vector (y) for model training.

    Args:
        lmp_df: Historical LMP DataFrame with columns [time, lmp, ...].
                May already contain load and weather columns if pre-joined.
        load_df: Load DataFrame with columns [time, load_mw]. Pass empty DF if
                 load is already in lmp_df.
        weather_df: Weather DataFrame with columns [time, shortwave_radiation, ...].
                    Pass empty DF if weather is already in lmp_df.
        target_col: Name of the column to use as the prediction target.
        forecast_horizon_h: Hours ahead to predict; labels are shifted by this amount.

    Returns:
        (X, y) where X is the feature DataFrame and y is the target Series,
        both aligned on the same DatetimeIndex with NaN rows dropped.
    """
    df = _merge_inputs(lmp_df, load_df, weather_df)
    df = df.sort_values("time").reset_index(drop=True)

    df = _add_temporal_features(df)
    df = _add_lmp_lags(df, lmp_col=target_col)
    df = _add_rolling_stats(df, lmp_col=target_col)

    df["_target"] = df[target_col].shift(-forecast_horizon_h)

    df = df.set_index("time")
    df.index = pd.to_datetime(df.index)

    feature_cols = get_feature_names(forecast_horizon_h)
    available_cols = [c for c in feature_cols if c in df.columns]

    df_clean = df[available_cols + ["_target"]].dropna()
    X = df_clean[available_cols]
    y = df_clean["_target"].rename(target_col)

    return X, y


def build_inference_features(
    recent_lmp: pd.DataFrame,
    load_forecast: pd.DataFrame,
    weather_forecast: pd.DataFrame,
) -> pd.DataFrame:
    """Build feature vectors for live inference (no target needed).

    Args:
        recent_lmp: Last ~200 hours of LMP data for lag computation.
        load_forecast: Forward-looking load forecast DataFrame.
        weather_forecast: Forward-looking solar/weather forecast DataFrame.

    Returns:
        DataFrame of feature rows with the most recent fully-populated rows
        (NaN-dropped). Each row can be passed to PriceModel.predict().
    """
    df = _merge_inputs(recent_lmp, load_forecast, weather_forecast)
    df = df.sort_values("time").reset_index(drop=True)

    df = _add_temporal_features(df)
    df = _add_lmp_lags(df)
    df = _add_rolling_stats(df)

    df = df.set_index("time")
    df.index = pd.to_datetime(df.index)

    feature_cols = get_feature_names()
    available_cols = [c for c in feature_cols if c in df.columns]

    return df[available_cols].dropna()


def _merge_inputs(
    lmp_df: pd.DataFrame,
    load_df: pd.DataFrame,
    weather_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge the three input DataFrames on the 'time' column.

    Skips a merge if the target columns are already present in lmp_df or the
    source DataFrame is empty.
    """
    df = lmp_df.copy()

    if not load_df.empty and "load_mw" not in df.columns:
        load_cols = [c for c in ["time", "load_mw"] if c in load_df.columns]
        df = df.merge(load_df[load_cols], on="time", how="left")

    missing_weather = [c for c in _WEATHER_COLS if c not in df.columns]
    if missing_weather and not weather_df.empty:
        wx_cols = ["time"] + [c for c in _WEATHER_COLS if c in weather_df.columns]
        df = df.merge(weather_df[wx_cols], on="time", how="left")

    return df


def _add_temporal_features(df: pd.DataFrame, time_col: str = "time") -> pd.DataFrame:
    """Add hour, day-of-week, month, weekend flag, holiday flag, and cyclical encodings.

    Args:
        df: DataFrame with a datetime column.
        time_col: Name of the datetime column.

    Returns:
        DataFrame with new temporal feature columns appended.
    """
    df = df.copy()
    t = pd.to_datetime(df[time_col])

    df["hour"] = t.dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * t.dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * t.dt.hour / 24)

    df["dow"] = t.dt.dayofweek
    df["dow_sin"] = np.sin(2 * np.pi * t.dt.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * t.dt.dayofweek / 7)

    df["month"] = t.dt.month
    df["month_sin"] = np.sin(2 * np.pi * t.dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * t.dt.month / 12)

    df["is_weekend"] = t.dt.dayofweek.isin([5, 6]).astype(int)
    df["is_holiday"] = t.apply(_is_holiday)

    return df


def _add_lmp_lags(df: pd.DataFrame, lmp_col: str = "lmp") -> pd.DataFrame:
    """Add lagged LMP values at _LMP_LAG_HOURS offsets (assuming hourly data).

    Args:
        df: DataFrame sorted by time with an LMP column.
        lmp_col: Name of the LMP column.

    Returns:
        DataFrame with lag columns appended: lmp_lag_1h, lmp_lag_24h, etc.
    """
    df = df.copy()
    for lag in _LMP_LAG_HOURS:
        df[f"lmp_lag_{lag}h"] = df[lmp_col].shift(lag)
    return df


def _add_rolling_stats(df: pd.DataFrame, lmp_col: str = "lmp") -> pd.DataFrame:
    """Add rolling mean and std of LMP over _ROLLING_WINDOWS_H window sizes.

    Args:
        df: DataFrame sorted by time with an LMP column.
        lmp_col: Name of the LMP column.

    Returns:
        DataFrame with rolling stat columns appended.
    """
    df = df.copy()
    for w in _ROLLING_WINDOWS_H:
        df[f"lmp_roll_mean_{w}h"] = df[lmp_col].rolling(w).mean()
        df[f"lmp_roll_std_{w}h"] = df[lmp_col].rolling(w).std()
    return df


def get_feature_names(forecast_horizon_h: int = 24) -> list[str]:
    """Return the ordered list of feature column names produced by build_feature_matrix.

    Args:
        forecast_horizon_h: Unused; kept for API consistency with callers that
                            pass it for documentation purposes.

    Returns:
        List of feature column name strings in canonical order.
    """
    temporal = [
        "hour", "hour_sin", "hour_cos",
        "dow", "dow_sin", "dow_cos",
        "month", "month_sin", "month_cos",
        "is_weekend", "is_holiday",
    ]
    lags = [f"lmp_lag_{h}h" for h in _LMP_LAG_HOURS]
    rolling = [
        f"lmp_roll_{stat}_{w}h"
        for w in _ROLLING_WINDOWS_H
        for stat in ["mean", "std"]
    ]
    load = ["load_mw"]
    weather = list(_WEATHER_COLS)
    return temporal + lags + rolling + load + weather
