"""
LMP price forecasting model: training, evaluation, and inference.

Two models are trained and kept:
    da_model — XGBoost forecasting day-ahead LMP (24 h horizon, hourly)
    rt_model — LightGBM forecasting real-time LMP (1 h horizon, hourly)

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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Default artifact directory (relative to project root)
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
        self.model_type = model_type
        self._model = None

        if model_type == "da":
            import xgboost as xgb
            self._model = xgb.XGBRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method="hist",
                verbosity=0,
            )
        else:
            import lightgbm as lgb
            self._model = lgb.LGBMRegressor(
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                verbose=-1,
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
        if self.model_type == "da":
            if X_val is not None:
                self._model.set_params(early_stopping_rounds=50)
                self._model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False,
                )
            else:
                self._model.fit(X_train, y_train)
        else:
            import lightgbm as lgb
            fit_kwargs: dict = {}
            if X_val is not None:
                fit_kwargs["eval_set"] = [(X_val, y_val)]
                fit_kwargs["callbacks"] = [
                    lgb.early_stopping(stopping_rounds=50, verbose=False),
                    lgb.log_evaluation(period=-1),
                ]
            self._model.fit(X_train, y_train, **fit_kwargs)

        train_preds = self._model.predict(X_train)
        metrics: dict[str, float] = {
            "train_rmse": float(np.sqrt(mean_squared_error(y_train, train_preds)))
        }
        if X_val is not None and y_val is not None:
            val_preds = self._model.predict(X_val)
            metrics["val_rmse"] = float(np.sqrt(mean_squared_error(y_val, val_preds)))

        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate LMP forecasts for the given feature matrix.

        Args:
            X: Feature matrix with columns matching training schema.

        Returns:
            1D array of predicted LMP values in $/MWh.
        """
        if self._model is None:
            raise RuntimeError("Model not trained. Call train() or load() first.")

        # Align columns to training schema, filling missing with 0
        if hasattr(self._model, "feature_names_in_"):
            expected = list(self._model.feature_names_in_)
            X = X.copy()
            for col in set(expected) - set(X.columns):
                X[col] = 0.0
            X = X[expected]

        return self._model.predict(X)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
        """Compute evaluation metrics on held-out test data.

        Args:
            X_test: Test feature matrix.
            y_test: True LMP values.

        Returns:
            Dict with 'rmse', 'mae', 'r2' metrics.
        """
        preds = self.predict(X_test)
        return {
            "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
            "mae": float(mean_absolute_error(y_test, preds)),
            "r2": float(r2_score(y_test, preds)),
        }

    def save(self, path: Path | str) -> None:
        """Persist the trained model artifact to disk.

        Args:
            path: Destination .joblib file path.
        """
        if self._model is None:
            raise RuntimeError("Cannot save: model has not been trained yet.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self._model, "model_type": self.model_type}, path)

    @classmethod
    def load(cls, path: Path | str, model_type: ModelType = "da") -> "PriceModel":
        """Load a trained model artifact from disk.

        Args:
            path: Path to the .joblib file.
            model_type: Must match the type used during training.

        Returns:
            PriceModel instance with self._model populated.
        """
        payload = joblib.load(path)
        instance = cls.__new__(cls)
        instance.model_type = payload.get("model_type", model_type)
        instance._model = payload["model"]
        return instance

    def feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Return the top-N most important features by gain.

        Args:
            top_n: Number of features to return.

        Returns:
            DataFrame with columns ['feature', 'importance'] sorted descending.
        """
        if self._model is None or not hasattr(self._model, "feature_importances_"):
            raise RuntimeError("Model not trained or does not expose feature_importances_.")

        importances = self._model.feature_importances_
        names: list[str]
        if hasattr(self._model, "feature_names_in_"):
            names = list(self._model.feature_names_in_)
        elif hasattr(self._model, "feature_name_"):
            names = list(self._model.feature_name_)
        else:
            names = [f"f{i}" for i in range(len(importances))]

        df = pd.DataFrame({"feature": names, "importance": importances})
        return df.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)


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
        Trained PriceModel instance (artifact also saved to _MODEL_DIR).
    """
    import structlog

    from src.config.nodes import node_short_name
    from src.data.db import MarketDB
    from src.models.features import build_feature_matrix

    log = structlog.get_logger()

    # Horizon: 24h for DA, 1h for RT (using same hourly DA data as proxy)
    horizon = 24 if model_type == "da" else 1

    with MarketDB(db_path) as db:
        train_df = db.query_features(node, start_train, end_train)
        val_df = db.query_features(node, start_val, end_val)

    if train_df.empty:
        raise ValueError(
            f"No training data for {node} from {start_train} to {end_train}. "
            "Run: python main.py data-fetch first."
        )

    log.info("data_loaded", node=node, n_train_rows=len(train_df), n_val_rows=len(val_df))

    X_train, y_train = build_feature_matrix(
        train_df, pd.DataFrame(), pd.DataFrame(), forecast_horizon_h=horizon
    )
    log.info("features_built", n_train=len(X_train), n_features=X_train.shape[1])

    X_val: pd.DataFrame | None = None
    y_val: pd.Series | None = None
    if not val_df.empty:
        X_val, y_val = build_feature_matrix(
            val_df, pd.DataFrame(), pd.DataFrame(), forecast_horizon_h=horizon
        )
        log.info("val_features_built", n_val=len(X_val))

    model = PriceModel(model_type=model_type)
    metrics = model.train(X_train, y_train, X_val, y_val)
    log.info("model_trained", **metrics)

    if X_val is not None and y_val is not None:
        eval_metrics = model.evaluate(X_val, y_val)
        log.info("model_evaluated", **eval_metrics)
        target_rmse = 15.0
        if eval_metrics["rmse"] > target_rmse:
            log.warning(
                "rmse_above_target",
                rmse=eval_metrics["rmse"],
                target=target_rmse,
            )

    short = node_short_name(node)
    artifact_path = _MODEL_DIR / f"{model_type}_{short}.joblib"
    model.save(artifact_path)
    log.info("model_saved", path=str(artifact_path))

    return model
