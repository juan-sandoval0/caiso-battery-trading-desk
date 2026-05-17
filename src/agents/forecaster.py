"""
PriceForecasterAgent — LMP price forecasting agent.

Wraps the trained XGBoost (DA) and LightGBM (RT) models and exposes a clean
agent interface consumed by the LangGraph orchestrator.

Responsibilities:
    - Load pre-trained model artifacts from disk
    - Pull the latest feature data from MarketDB and live CAISO / weather feeds
    - Produce a price forecast DataFrame for the next N hours
    - Write the forecast to SharedMarketState
    - Expose LangChain tool definitions for tool-calling within the graph

This agent is ADVISORY: it provides price signals to the DispatchOptimizerAgent
but does not directly issue dispatch commands.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config.battery import BatteryConfig
from src.config.nodes import DEFAULT_HUB_NODE
from src.models.price_model import PriceModel


@dataclass
class PriceForecast:
    """Output of the PriceForecasterAgent for one forecast run.

    Attributes:
        node: CAISO PNode the forecast applies to.
        times: DatetimeIndex of forecast intervals.
        da_lmp_forecast: Day-ahead LMP forecast ($/MWh), hourly.
        rt_lmp_forecast: Real-time LMP forecast ($/MWh), 5-min if available.
        confidence_low: Lower bound of 80% prediction interval.
        confidence_high: Upper bound of 80% prediction interval.
        model_version: Artifact path or version tag for traceability.
    """

    node: str
    times: pd.DatetimeIndex
    da_lmp_forecast: np.ndarray
    rt_lmp_forecast: np.ndarray | None
    confidence_low: np.ndarray | None
    confidence_high: np.ndarray | None
    model_version: str


class PriceForecasterAgent:
    """Forecasts CAISO LMPs using trained ML models.

    This agent is stateless between ticks — it loads the latest data,
    generates a forecast, and returns it. The orchestrator stores the result.

    Usage::

        agent = PriceForecasterAgent(node="TH_NP15_GEN-APND")
        agent.load_models(da_path="models/artifacts/da_np15.joblib")
        forecast = agent.run(db=market_db, weather_fetcher=wf, caiso_fetcher=cf)
    """

    def __init__(self, node: str = DEFAULT_HUB_NODE) -> None:
        """Initialize the agent for a specific CAISO hub node.

        Args:
            node: CAISO PNode identifier.
        """
        raise NotImplementedError(
            # Store self._node = node.
            # Initialize self._da_model: PriceModel | None = None.
            # Initialize self._rt_model: PriceModel | None = None.
        )

    def load_models(
        self,
        da_path: Path | str | None = None,
        rt_path: Path | str | None = None,
    ) -> None:
        """Load pre-trained model artifacts from disk.

        Args:
            da_path: Path to the day-ahead XGBoost joblib artifact.
            rt_path: Path to the real-time LightGBM joblib artifact.
        """
        raise NotImplementedError(
            # self._da_model = PriceModel.load(da_path, model_type='da') if da_path else None
            # self._rt_model = PriceModel.load(rt_path, model_type='rt') if rt_path else None
        )

    def run(
        self,
        db: Any,
        weather_fetcher: Any,
        caiso_fetcher: Any,
        horizon_h: int = 24,
    ) -> PriceForecast:
        """Execute one forecast tick.

        Args:
            db: MarketDB instance for historical feature retrieval.
            weather_fetcher: WeatherFetcher for solar forecast.
            caiso_fetcher: CAISOFetcher for latest LMP/load data.
            horizon_h: Forecast horizon in hours.

        Returns:
            PriceForecast dataclass with predicted prices and metadata.
        """
        raise NotImplementedError(
            # 1. Fetch recent LMP, load forecast, and weather forecast.
            # 2. Call build_inference_features() from src.models.features.
            # 3. Call self._da_model.predict(X) for DA forecast.
            # 4. Optionally call self._rt_model.predict(X) for RT forecast.
            # 5. Build and return PriceForecast.
        )

    async def arun(
        self,
        db: Any,
        weather_fetcher: Any,
        caiso_fetcher: Any,
        horizon_h: int = 24,
    ) -> PriceForecast:
        """Async version of run() for use within asyncio event loops.

        Args:
            db: MarketDB instance.
            weather_fetcher: WeatherFetcher instance.
            caiso_fetcher: CAISOFetcher instance.
            horizon_h: Forecast horizon in hours.

        Returns:
            PriceForecast dataclass.
        """
        raise NotImplementedError(
            # Wrap run() in asyncio.get_event_loop().run_in_executor() for
            # CPU-bound model inference, or call async fetchers directly.
        )

    def get_langchain_tools(self) -> list[Any]:
        """Return LangChain Tool definitions for use within the LangGraph graph.

        Returns:
            List of langchain_core.tools.Tool objects that the orchestrator
            can invoke via tool-calling.
        """
        raise NotImplementedError(
            # Define and return tools:
            #   'forecast_da_lmp' — runs DA forecast and returns JSON summary
            #   'get_forecast_confidence' — returns confidence interval for a given hour
        )
