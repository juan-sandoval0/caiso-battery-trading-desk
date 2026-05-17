"""
Tests for PriceModel (XGBoost / LightGBM LMP forecasters).

Uses synthetic price data so no database or CAISO access is needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.price_model import PriceModel


def _make_synthetic_data(n: int = 200) -> tuple[pd.DataFrame, pd.Series]:
    """Generate synthetic feature matrix and target for testing.

    Args:
        n: Number of samples.

    Returns:
        (X, y) tuple with n rows.
    """
    raise NotImplementedError(
        # rng = np.random.default_rng(42)
        # X = pd.DataFrame({'hour': rng.integers(0, 24, n), 'lmp_lag_24h': rng.normal(50, 20, n)})
        # y = pd.Series(rng.normal(50, 20, n))
        # return X, y
    )


class TestPriceModelDA:
    """Tests for the DA XGBoost model."""

    def test_train_returns_rmse(self) -> None:
        """train() returns a dict containing 'train_rmse'."""
        raise NotImplementedError()

    def test_predict_returns_correct_shape(self) -> None:
        """predict() returns array with length matching number of input rows."""
        raise NotImplementedError()

    def test_evaluate_returns_expected_keys(self) -> None:
        """evaluate() returns dict with 'rmse', 'mae', 'r2' keys."""
        raise NotImplementedError()

    def test_save_and_load_roundtrip(self, tmp_path: "Path") -> None:
        """Model saved to disk can be loaded and produces identical predictions."""
        raise NotImplementedError()

    def test_feature_importance_returns_sorted_df(self) -> None:
        """feature_importance() returns DataFrame sorted by importance descending."""
        raise NotImplementedError()


class TestPriceModelRT:
    """Tests for the RT LightGBM model."""

    def test_rt_model_trains_without_error(self) -> None:
        """LightGBM RT model trains on synthetic data without raising."""
        raise NotImplementedError()
