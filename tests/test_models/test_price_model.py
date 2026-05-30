"""
Tests for PriceModel (XGBoost / LightGBM LMP forecasters).

Uses synthetic price data so no database or CAISO access is needed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.price_model import PriceModel


def _make_synthetic_data(n: int = 200) -> tuple[pd.DataFrame, pd.Series]:
    """Generate a synthetic feature matrix and target with a learnable signal.

    Args:
        n: Number of samples.

    Returns:
        (X, y) tuple with n rows. The target depends on the features so the
        model has real signal to fit (otherwise tree models just learn noise).
    """
    rng = np.random.default_rng(42)
    hour = rng.integers(0, 24, n)
    lmp_lag_24h = rng.normal(50, 20, n)
    lmp_lag_1h = rng.normal(50, 20, n)
    X = pd.DataFrame(
        {"hour": hour, "lmp_lag_24h": lmp_lag_24h, "lmp_lag_1h": lmp_lag_1h}
    )
    # Target: a deterministic function of the features plus small noise.
    y = pd.Series(
        0.6 * lmp_lag_1h + 0.3 * lmp_lag_24h + 5.0 * np.sin(hour / 24 * 2 * np.pi)
        + rng.normal(0, 2, n)
    )
    return X, y


class TestPriceModelDA:
    """Tests for the DA XGBoost model."""

    def test_train_returns_rmse(self) -> None:
        """train() returns a dict containing 'train_rmse'."""
        X, y = _make_synthetic_data()
        model = PriceModel(model_type="da")
        metrics = model.train(X, y)
        assert "train_rmse" in metrics
        assert metrics["train_rmse"] >= 0.0

    def test_predict_returns_correct_shape(self) -> None:
        """predict() returns array with length matching number of input rows."""
        X, y = _make_synthetic_data()
        model = PriceModel(model_type="da")
        model.train(X, y)
        preds = model.predict(X)
        assert isinstance(preds, np.ndarray)
        assert preds.shape[0] == len(X)

    def test_evaluate_returns_expected_keys(self) -> None:
        """evaluate() returns dict with 'rmse', 'mae', 'r2' keys."""
        X, y = _make_synthetic_data()
        model = PriceModel(model_type="da")
        model.train(X, y)
        metrics = model.evaluate(X, y)
        assert set(metrics) >= {"rmse", "mae", "r2"}

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        """Model saved to disk can be loaded and produces identical predictions."""
        X, y = _make_synthetic_data()
        model = PriceModel(model_type="da")
        model.train(X, y)
        before = model.predict(X)

        path = tmp_path / "da_test.joblib"
        model.save(path)
        loaded = PriceModel.load(path, model_type="da")
        after = loaded.predict(X)

        assert loaded.model_type == "da"
        np.testing.assert_allclose(before, after, rtol=1e-6)

    def test_feature_importance_returns_sorted_df(self) -> None:
        """feature_importance() returns DataFrame sorted by importance descending."""
        X, y = _make_synthetic_data()
        model = PriceModel(model_type="da")
        model.train(X, y)
        fi = model.feature_importance(top_n=3)
        assert list(fi.columns) == ["feature", "importance"]
        assert fi["importance"].is_monotonic_decreasing


class TestPriceModelRT:
    """Tests for the RT LightGBM model."""

    def test_rt_model_trains_without_error(self) -> None:
        """LightGBM RT model trains on synthetic data and predicts the right shape."""
        X, y = _make_synthetic_data()
        model = PriceModel(model_type="rt")
        metrics = model.train(X, y)
        assert "train_rmse" in metrics
        preds = model.predict(X)
        assert preds.shape[0] == len(X)
