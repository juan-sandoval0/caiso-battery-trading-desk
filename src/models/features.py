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

from typing import Sequence

import numpy as np
import pandas as pd


# Lag offsets in hours for RT LMP features
_LMP_LAG_HOURS: list[int] = [1, 2, 3, 4, 24, 48, 168]

# Rolling window sizes in hours
_ROLLING_WINDOWS_H: list[int] = [4, 24, 168]


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
        load_df: Load DataFrame with columns [time, load_mw, forecast_mw].
        weather_df: Weather DataFrame with columns [time, shortwave_radiation, ...].
        target_col: Name of the column to use as the prediction target.
        forecast_horizon_h: Hours ahead to predict; labels are shifted by this amount.

    Returns:
        (X, y) where X is the feature DataFrame and y is the target Series,
        both aligned on the same index with NaN rows dropped.
    """
    raise NotImplementedError(
        # 1. Merge lmp_df, load_df, weather_df on time (outer join).
        # 2. Add temporal features via _add_temporal_features().
        # 3. Add LMP lag features via _add_lmp_lags().
        # 4. Add rolling statistics via _add_rolling_stats().
        # 5. Shift target column by -forecast_horizon_h to create labels.
        # 6. Drop rows with any NaN (caused by lags / shift at boundaries).
        # 7. Return X (all feature cols) and y (target col).
    )


def build_inference_features(
    recent_lmp: pd.DataFrame,
    load_forecast: pd.DataFrame,
    weather_forecast: pd.DataFrame,
) -> pd.DataFrame:
    """Build a single-row feature vector for live inference (no target needed).

    Args:
        recent_lmp: Last ~200 hours of LMP data for lag computation.
        load_forecast: Forward-looking load forecast DataFrame.
        weather_forecast: Forward-looking solar/weather forecast DataFrame.

    Returns:
        Single-row DataFrame with all feature columns (no target column).
    """
    raise NotImplementedError(
        # Same pipeline as build_feature_matrix but without the target shift.
        # Return only the most recent fully-populated row.
    )


def _add_temporal_features(df: pd.DataFrame, time_col: str = "time") -> pd.DataFrame:
    """Add hour, day-of-week, month, weekend flag, and US holiday flag.

    Args:
        df: DataFrame with a timezone-aware datetime column.
        time_col: Name of the datetime column.

    Returns:
        DataFrame with new temporal feature columns appended.
    """
    raise NotImplementedError(
        # df['hour'] = df[time_col].dt.hour
        # df['dow'] = df[time_col].dt.dayofweek
        # df['month'] = df[time_col].dt.month
        # df['is_weekend'] = df['dow'].isin([5, 6]).astype(int)
        # Use pandas_market_calendars or hardcoded US holidays for holiday flag.
    )


def _add_lmp_lags(df: pd.DataFrame, lmp_col: str = "lmp") -> pd.DataFrame:
    """Add lagged LMP values at _LMP_LAG_HOURS offsets (assuming hourly data).

    Args:
        df: DataFrame sorted by time with an LMP column.
        lmp_col: Name of the LMP column.

    Returns:
        DataFrame with lag columns appended: lmp_lag_1h, lmp_lag_24h, etc.
    """
    raise NotImplementedError(
        # for lag in _LMP_LAG_HOURS:
        #     df[f'lmp_lag_{lag}h'] = df[lmp_col].shift(lag)
    )


def _add_rolling_stats(df: pd.DataFrame, lmp_col: str = "lmp") -> pd.DataFrame:
    """Add rolling mean and std of LMP over _ROLLING_WINDOWS_H window sizes.

    Args:
        df: DataFrame sorted by time with an LMP column.
        lmp_col: Name of the LMP column.

    Returns:
        DataFrame with rolling stat columns appended.
    """
    raise NotImplementedError(
        # for w in _ROLLING_WINDOWS_H:
        #     df[f'lmp_roll_mean_{w}h'] = df[lmp_col].rolling(w).mean()
        #     df[f'lmp_roll_std_{w}h'] = df[lmp_col].rolling(w).std()
    )


def get_feature_names(forecast_horizon_h: int = 24) -> list[str]:
    """Return the ordered list of feature column names produced by build_feature_matrix.

    Args:
        forecast_horizon_h: Must match the value used during training.

    Returns:
        List of feature column name strings.
    """
    raise NotImplementedError(
        # Return the canonical feature list so inference code can validate columns.
    )
