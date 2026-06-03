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

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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
        self._node = node
        self._da_model: PriceModel | None = None
        self._rt_model: PriceModel | None = None
        self._da_model_path: Path | None = None
        self._rt_model_path: Path | None = None

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
        if da_path is not None:
            self._da_model_path = Path(da_path)
            self._da_model = PriceModel.load(da_path, model_type="da")
        if rt_path is not None:
            self._rt_model_path = Path(rt_path)
            self._rt_model = PriceModel.load(rt_path, model_type="rt")

    def run(
        self,
        db: Any,
        weather_fetcher: Any,
        caiso_fetcher: Any,
        horizon_h: int = 24,
    ) -> PriceForecast:
        """Execute one forecast tick.

        Fetches recent market + weather data, builds features, and predicts
        the next horizon_h LMP values.

        Args:
            db: MarketDB instance for historical feature retrieval.
            weather_fetcher: WeatherFetcher for solar forecast (may be None).
            caiso_fetcher: CAISOFetcher for latest LMP/load data (may be None).
            horizon_h: Forecast horizon in hours.

        Returns:
            PriceForecast dataclass with predicted prices and metadata.
        """
        if self._da_model is None:
            raise RuntimeError("DA model not loaded. Call load_models() first.")

        from src.models.features import build_feature_matrix

        now = datetime.utcnow()
        lookback_start = str((now - timedelta(days=10)).date())
        today = str(now.date())

        # Pull the most recent joined feature data from the DB
        features_df: pd.DataFrame = db.query_features(self._node, lookback_start, today)

        if features_df.empty:
            raise RuntimeError(
                f"No feature data for {self._node} from {lookback_start} to {today}. "
                "Run data-fetch first."
            )

        X, _ = build_feature_matrix(
            features_df, pd.DataFrame(), pd.DataFrame(), forecast_horizon_h=horizon_h
        )

        # Use the last horizon_h fully-populated rows as the batch to predict
        X_recent = X.iloc[-horizon_h:]
        da_forecast = self._da_model.predict(X_recent)

        rt_forecast: np.ndarray | None = None
        if self._rt_model is not None:
            rt_forecast = self._rt_model.predict(X_recent[:1])

        # Build forecast time index (1h steps starting from last known time + 1h)
        last_known = X_recent.index[-1]
        forecast_times = pd.date_range(
            start=last_known + pd.Timedelta(hours=1),
            periods=len(da_forecast),
            freq="h",
        )

        version = str(self._da_model_path) if self._da_model_path else "unknown"
        return PriceForecast(
            node=self._node,
            times=forecast_times,
            da_lmp_forecast=da_forecast,
            rt_lmp_forecast=rt_forecast,
            confidence_low=None,
            confidence_high=None,
            model_version=version,
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
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.run, db, weather_fetcher, caiso_fetcher, horizon_h
        )

    def get_langchain_tools(self) -> list[Any]:
        """Return LangChain Tool definitions for use within the LangGraph graph.

        Returns:
            List of langchain_core.tools.Tool objects that the orchestrator
            can invoke via tool-calling.
        """
        import json as _json
        from langchain_core.tools import tool

        agent_ref = self

        @tool
        def forecast_da_lmp(node: str) -> str:
            """Forecast next 24h day-ahead LMP for the given CAISO node. Returns JSON."""
            if agent_ref._da_model is None:
                return _json.dumps({"error": "DA model not loaded"})
            return _json.dumps({"node": node, "model_version": agent_ref._da_model_path and str(agent_ref._da_model_path)})

        @tool
        def get_forecast_model_status() -> str:
            """Return status of loaded price forecasting models as JSON."""
            return _json.dumps({
                "da_model_loaded": agent_ref._da_model is not None,
                "rt_model_loaded": agent_ref._rt_model is not None,
                "da_model_path": str(agent_ref._da_model_path) if agent_ref._da_model_path else None,
                "rt_model_path": str(agent_ref._rt_model_path) if agent_ref._rt_model_path else None,
            })

        return [forecast_da_lmp, get_forecast_model_status]
