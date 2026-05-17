"""
Tests for PriceForecasterAgent.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.agents.forecaster import PriceForecasterAgent


class TestLoadModels:
    """Tests for model loading."""

    def test_load_da_model_from_valid_path(self, tmp_path) -> None:
        """load_models() correctly loads a DA model from disk."""
        raise NotImplementedError()

    def test_run_without_model_raises(self) -> None:
        """Calling run() before load_models() raises a meaningful error."""
        raise NotImplementedError()


class TestRunForecast:
    """Tests for the run() method (models mocked)."""

    def test_returns_price_forecast_dataclass(self) -> None:
        """run() returns a PriceForecast with the correct node and horizon."""
        raise NotImplementedError()

    def test_forecast_length_matches_horizon(self) -> None:
        """da_lmp_forecast array length equals the requested horizon_h."""
        raise NotImplementedError()
