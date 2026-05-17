"""
LMP price forecasting model: training, evaluation, and inference.

Two models are trained and kept:
    da_model — XGBoost forecasting day-ahead LMP (24 h horizon, hourly)
    rt_model — LightGBM forecasting real-time LMP (1 h horizon, 5-min)

Model artifacts are persisted as joblib files so they can be loaded without
retraining on each paper-trading run.

Evaluation metric: RMSE in $/MWh; target ≤ $15/MWh out-of-sample (NP15).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd


# Default artifact directory
_MODEL_DIR = Path("models/artifacts")

ModelType = Literal["da", "rt"]


class PriceModel:
    """Wrapper around XGBoost (DA) or LightGBM (RT) LMP forecasters.

    Usage::

        model = PriceModel(model_type="da")
        model.train(X_train, y_train)
        preds = model.predict(X_live)
        model.save("models/artifacts/da_np15.joblib")
    """

    def __init__(self, model_type: ModelType = "da") -> None:
        """Initialize the model container.

        Args:
            model_type: 'da' for XGBoost day-ahead model, 'rt' for LightGBM RT model.
        """
        raise NotImplementedError(
            # Set self.model_type = model_type.
            # Initialize self._model = None (populated by train() or load()).
            # XGBoost for 'da': xgb.XGBRegressor(n_estimators=500, learning_rate=0.05,
            #   max_depth=6, subsample=0.8, colsample_bytree=0.8, tree_method='hist').
            # LightGBM for 'rt': lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05,
            #   num_leaves=31, subsample=0.8).
        )

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> dict[str, float]:
        """Fit the model on training data with optional early stopping on validation.

        Args:
            X_train: Feature matrix for training.
            y_train: Target LMP values for training.
            X_val: Optional validation feature matrix for early stopping.
            y_val: Optional validation target values.

        Returns:
            Dict with 'train_rmse' and optionally 'val_rmse'.
        """
        raise NotImplementedError(
            # self._model.fit(X_train, y_train, eval_set=[(X_val, y_val)], ...)
            # Compute and return RMSE on train (and val if provided).
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate LMP forecasts for the given feature matrix.

        Args:
            X: Feature matrix with columns matching training schema.

        Returns:
            1D array of predicted LMP values in $/MWh.
        """
        raise NotImplementedError(
            # Validate columns match training schema.
            # return self._model.predict(X)
        )

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
        """Compute evaluation metrics on held-out test data.

        Args:
            X_test: Test feature matrix.
            y_test: True LMP values.

        Returns:
            Dict with 'rmse', 'mae', 'r2' metrics.
        """
        raise NotImplementedError(
            # preds = self.predict(X_test)
            # rmse = np.sqrt(mean_squared_error(y_test, preds))
            # mae = mean_absolute_error(y_test, preds)
            # r2 = r2_score(y_test, preds)
        )

    def save(self, path: Path | str) -> None:
        """Persist the trained model artifact to disk.

        Args:
            path: Destination .joblib file path.
        """
        raise NotImplementedError(
            # joblib.dump(self._model, path)
        )

    @classmethod
    def load(cls, path: Path | str, model_type: ModelType = "da") -> "PriceModel":
        """Load a trained model artifact from disk.

        Args:
            path: Path to the .joblib file.
            model_type: Must match the type used during training.

        Returns:
            PriceModel instance with self._model populated.
        """
        raise NotImplementedError(
            # instance = cls(model_type=model_type)
            # instance._model = joblib.load(path)
            # return instance
        )

    def feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Return the top-N most important features by gain.

        Args:
            top_n: Number of features to return.

        Returns:
            DataFrame with columns ['feature', 'importance'] sorted descending.
        """
        raise NotImplementedError(
            # Extract feature_importances_ from self._model.
            # Return as sorted DataFrame.
        )


def train_pipeline(
    node: str,
    start_train: str,
    end_train: str,
    start_val: str,
    end_val: str,
    model_type: ModelType = "da",
    db_path: str = "data/market.duckdb",
) -> PriceModel:
    """End-to-end training pipeline: load data → build features → train → evaluate.

    Args:
        node: CAISO PNode identifier.
        start_train: Training set start date (ISO string).
        end_train: Training set end date (ISO string).
        start_val: Validation set start date (ISO string).
        end_val: Validation set end date (ISO string).
        model_type: 'da' or 'rt'.
        db_path: Path to the DuckDB database.

    Returns:
        Trained PriceModel instance.
    """
    raise NotImplementedError(
        # 1. Open MarketDB, call query_features() for train and val ranges.
        # 2. Call build_feature_matrix() for each split.
        # 3. Instantiate PriceModel(model_type=model_type).
        # 4. Call model.train(X_train, y_train, X_val, y_val).
        # 5. Log eval metrics via structlog.
        # 6. Save artifact to _MODEL_DIR / f"{model_type}_{node_short_name}.joblib".
        # 7. Return the trained model.
    )
