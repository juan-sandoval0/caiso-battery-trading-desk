"""
Tests for PriceForecasterAgent.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.agents.forecaster import PriceForecast, PriceForecasterAgent
from src.models.price_model import PriceModel


def _train_tiny_da_model(path) -> None:
    """Train and save a minimal DA model artifact to ``path``."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=50), "b": rng.normal(size=50)})
    y = pd.Series(rng.normal(size=50))
    model = PriceModel(model_type="da")
    model.train(X, y)
    model.save(path)


class TestLoadModels:
    """Tests for model loading."""

    def test_load_da_model_from_valid_path(self, tmp_path) -> None:
        """load_models() correctly loads a DA model from disk."""
        path = tmp_path / "da_np15.joblib"
        _train_tiny_da_model(path)

        agent = PriceForecasterAgent(node="TH_NP15_GEN-APND")
        agent.load_models(da_path=path)
        assert agent._da_model is not None
        assert agent._da_model_path == path

    def test_run_without_model_raises(self) -> None:
        """Calling run() before load_models() raises a meaningful error."""
        agent = PriceForecasterAgent(node="TH_NP15_GEN-APND")
        with pytest.raises(RuntimeError):
            agent.run(db=MagicMock(), weather_fetcher=None, caiso_fetcher=None)


class TestRunForecast:
    """Tests for the run() method (models + features mocked)."""

    def _build_agent_with_mocks(self, horizon: int) -> tuple[PriceForecasterAgent, MagicMock]:
        agent = PriceForecasterAgent(node="TH_NP15_GEN-APND")
        # Mock a loaded DA model whose predict returns one value per input row.
        model = MagicMock()
        model.predict.side_effect = lambda X: np.zeros(len(X))
        agent._da_model = model
        agent._da_model_path = "models/artifacts/da_NP15.joblib"

        db = MagicMock()
        db.query_features.return_value = pd.DataFrame({"lmp": [1.0, 2.0]})  # non-empty
        return agent, db

    def test_returns_price_forecast_dataclass(self) -> None:
        """run() returns a PriceForecast with the correct node and horizon."""
        horizon = 24
        agent, db = self._build_agent_with_mocks(horizon)

        idx = pd.date_range("2024-07-01", periods=48, freq="h")
        X = pd.DataFrame({"a": np.arange(48), "b": np.arange(48)}, index=idx)
        with patch("src.models.features.build_feature_matrix", return_value=(X, pd.Series(np.zeros(48)))):
            forecast = agent.run(db=db, weather_fetcher=None, caiso_fetcher=None, horizon_h=horizon)

        assert isinstance(forecast, PriceForecast)
        assert forecast.node == "TH_NP15_GEN-APND"
        assert len(forecast.da_lmp_forecast) == horizon

    def test_forecast_length_matches_horizon(self) -> None:
        """da_lmp_forecast array length equals the requested horizon_h."""
        horizon = 12
        agent, db = self._build_agent_with_mocks(horizon)

        idx = pd.date_range("2024-07-01", periods=48, freq="h")
        X = pd.DataFrame({"a": np.arange(48), "b": np.arange(48)}, index=idx)
        with patch("src.models.features.build_feature_matrix", return_value=(X, pd.Series(np.zeros(48)))):
            forecast = agent.run(db=db, weather_fetcher=None, caiso_fetcher=None, horizon_h=horizon)

        assert len(forecast.da_lmp_forecast) == horizon
        assert len(forecast.times) == horizon
