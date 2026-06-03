"""
DuckDB interface for local market data storage.

Schema note: `load_actual` stores 5-min real-time load readings;
`load_forecast` stores the day-ahead hourly load forecast with publish_time.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd


_SCHEMA_SQL: str = """
CREATE TABLE IF NOT EXISTS lmp (
    time        TIMESTAMPTZ NOT NULL,
    node        VARCHAR     NOT NULL,
    market      VARCHAR     NOT NULL,
    lmp         DOUBLE,
    energy      DOUBLE,
    congestion  DOUBLE,
    loss        DOUBLE,
    PRIMARY KEY (time, node, market)
);

CREATE TABLE IF NOT EXISTS load_actual (
    time        TIMESTAMPTZ NOT NULL PRIMARY KEY,
    load_mw     DOUBLE
);

CREATE TABLE IF NOT EXISTS load_forecast (
    time         TIMESTAMPTZ NOT NULL,
    forecast_mw  DOUBLE,
    publish_time TIMESTAMPTZ,
    PRIMARY KEY (time, publish_time)
);

CREATE TABLE IF NOT EXISTS fuel_mix (
    time            TIMESTAMPTZ NOT NULL,
    fuel_type       VARCHAR     NOT NULL,
    generation_mw   DOUBLE,
    PRIMARY KEY (time, fuel_type)
);

CREATE TABLE IF NOT EXISTS storage_soc (
    time         TIMESTAMPTZ NOT NULL PRIMARY KEY,
    soc_mwh      DOUBLE,
    schedule     VARCHAR
);

CREATE TABLE IF NOT EXISTS curtailment (
    time             TIMESTAMPTZ NOT NULL,
    fuel_type        VARCHAR     NOT NULL,
    curtailment_type VARCHAR     NOT NULL,
    reason           VARCHAR     NOT NULL,
    curtailment_mw   DOUBLE,
    PRIMARY KEY (time, fuel_type, curtailment_type, reason)
);

CREATE TABLE IF NOT EXISTS weather (
    time                TIMESTAMPTZ NOT NULL,
    node                VARCHAR     NOT NULL,
    shortwave_radiation DOUBLE,
    direct_radiation    DOUBLE,
    temperature_2m      DOUBLE,
    wind_speed_10m      DOUBLE,
    cloud_cover         DOUBLE,
    PRIMARY KEY (time, node)
);

CREATE TABLE IF NOT EXISTS agent_decisions (
    decision_id  VARCHAR     PRIMARY KEY,
    time         TIMESTAMPTZ NOT NULL,
    agent        VARCHAR     NOT NULL,
    action       VARCHAR,
    rationale    TEXT,
    metadata     JSON
);
"""


class MarketDB:
    """DuckDB connection wrapper with typed upsert and query helpers."""

    def __init__(self, path: Path | str = "data/market.duckdb") -> None:
        """Open (or create) the DuckDB database and initialize the schema.

        Args:
            path: File path to the DuckDB database file. Use ':memory:' for tests.
        """
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = duckdb.connect(str(path))
        self._conn.execute(_SCHEMA_SQL)

    # ------------------------------------------------------------------
    # Upsert methods
    # ------------------------------------------------------------------

    def upsert_lmp(self, df: pd.DataFrame) -> int:
        """Insert or replace LMP rows.

        Args:
            df: DataFrame with columns [time, node, market, lmp, energy, congestion, loss].

        Returns:
            Number of rows in the table after upsert (delta not guaranteed with REPLACE).
        """
        if df.empty:
            return 0
        cols = ["time", "node", "market", "lmp", "energy", "congestion", "loss"]
        insert_df = df[cols].copy()
        self._conn.register("_lmp_staging", insert_df)
        before = self._conn.execute("SELECT COUNT(*) FROM lmp").fetchone()[0]
        self._conn.execute(
            "INSERT OR REPLACE INTO lmp SELECT time, node, market, lmp, energy, congestion, loss "
            "FROM _lmp_staging"
        )
        after = self._conn.execute("SELECT COUNT(*) FROM lmp").fetchone()[0]
        self._conn.unregister("_lmp_staging")
        return after - before

    def upsert_load_actual(self, df: pd.DataFrame) -> int:
        """Insert or replace actual load rows.

        Args:
            df: DataFrame with columns [time, load_mw].

        Returns:
            Row count delta.
        """
        if df.empty:
            return 0
        insert_df = df[["time", "load_mw"]].copy()
        self._conn.register("_load_staging", insert_df)
        before = self._conn.execute("SELECT COUNT(*) FROM load_actual").fetchone()[0]
        self._conn.execute(
            "INSERT OR REPLACE INTO load_actual SELECT time, load_mw FROM _load_staging"
        )
        after = self._conn.execute("SELECT COUNT(*) FROM load_actual").fetchone()[0]
        self._conn.unregister("_load_staging")
        return after - before

    def upsert_load_forecast(self, df: pd.DataFrame) -> int:
        """Insert or replace load forecast rows.

        Args:
            df: DataFrame with columns [time, forecast_mw, publish_time].

        Returns:
            Row count delta.
        """
        if df.empty:
            return 0
        insert_df = df[["time", "forecast_mw", "publish_time"]].copy()
        self._conn.register("_lf_staging", insert_df)
        before = self._conn.execute("SELECT COUNT(*) FROM load_forecast").fetchone()[0]
        self._conn.execute(
            "INSERT OR REPLACE INTO load_forecast "
            "SELECT time, forecast_mw, publish_time FROM _lf_staging"
        )
        after = self._conn.execute("SELECT COUNT(*) FROM load_forecast").fetchone()[0]
        self._conn.unregister("_lf_staging")
        return after - before

    def upsert_fuel_mix(self, df: pd.DataFrame) -> int:
        """Insert or replace fuel mix rows (long format: time, fuel_type, generation_mw).

        Args:
            df: DataFrame with columns [time, fuel_type, generation_mw].

        Returns:
            Row count delta.
        """
        if df.empty:
            return 0
        insert_df = df[["time", "fuel_type", "generation_mw"]].copy()
        self._conn.register("_fm_staging", insert_df)
        before = self._conn.execute("SELECT COUNT(*) FROM fuel_mix").fetchone()[0]
        self._conn.execute(
            "INSERT OR REPLACE INTO fuel_mix "
            "SELECT time, fuel_type, generation_mw FROM _fm_staging"
        )
        after = self._conn.execute("SELECT COUNT(*) FROM fuel_mix").fetchone()[0]
        self._conn.unregister("_fm_staging")
        return after - before

    def upsert_storage_soc(self, df: pd.DataFrame) -> int:
        """Insert or replace storage SOC rows.

        Args:
            df: DataFrame with columns [time, soc_mwh, schedule].

        Returns:
            Row count delta.
        """
        if df.empty:
            return 0
        insert_df = df[["time", "soc_mwh", "schedule"]].copy()
        self._conn.register("_soc_staging", insert_df)
        before = self._conn.execute("SELECT COUNT(*) FROM storage_soc").fetchone()[0]
        self._conn.execute(
            "INSERT OR REPLACE INTO storage_soc "
            "SELECT time, soc_mwh, schedule FROM _soc_staging"
        )
        after = self._conn.execute("SELECT COUNT(*) FROM storage_soc").fetchone()[0]
        self._conn.unregister("_soc_staging")
        return after - before

    def upsert_curtailment(self, df: pd.DataFrame) -> int:
        """Insert or replace curtailment rows.

        Args:
            df: DataFrame with columns [time, fuel_type, curtailment_type, reason, curtailment_mw].

        Returns:
            Row count delta.
        """
        if df.empty:
            return 0
        insert_df = df[["time", "fuel_type", "curtailment_type", "reason", "curtailment_mw"]].copy()
        self._conn.register("_curt_staging", insert_df)
        before = self._conn.execute("SELECT COUNT(*) FROM curtailment").fetchone()[0]
        self._conn.execute(
            "INSERT OR REPLACE INTO curtailment "
            "SELECT time, fuel_type, curtailment_type, reason, curtailment_mw FROM _curt_staging"
        )
        after = self._conn.execute("SELECT COUNT(*) FROM curtailment").fetchone()[0]
        self._conn.unregister("_curt_staging")
        return after - before

    def upsert_weather(self, df: pd.DataFrame) -> int:
        """Insert or replace weather rows.

        Args:
            df: DataFrame with columns [time, node, shortwave_radiation, direct_radiation,
                temperature_2m, wind_speed_10m, cloud_cover].

        Returns:
            Row count delta.
        """
        if df.empty:
            return 0
        cols = ["time", "node", "shortwave_radiation", "direct_radiation",
                "temperature_2m", "wind_speed_10m", "cloud_cover"]
        insert_df = df[cols].copy()
        self._conn.register("_wx_staging", insert_df)
        before = self._conn.execute("SELECT COUNT(*) FROM weather").fetchone()[0]
        self._conn.execute(
            "INSERT OR REPLACE INTO weather "
            "SELECT time, node, shortwave_radiation, direct_radiation, "
            "temperature_2m, wind_speed_10m, cloud_cover FROM _wx_staging"
        )
        after = self._conn.execute("SELECT COUNT(*) FROM weather").fetchone()[0]
        self._conn.unregister("_wx_staging")
        return after - before

    def log_agent_decision(
        self,
        agent: str,
        action: str,
        rationale: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Persist an agent decision for audit and dashboard display.

        Args:
            agent: Agent name (e.g., 'RiskMonitor').
            action: Short action string (e.g., 'HALT_DISCHARGE').
            rationale: Human-readable explanation.
            metadata: Optional JSON-serializable dict.
        """
        import json
        decision_id = str(uuid.uuid4())
        self._conn.execute(
            "INSERT INTO agent_decisions (decision_id, time, agent, action, rationale, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [decision_id, datetime.utcnow(), agent, action, rationale,
             json.dumps(metadata or {})],
        )

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def query_lmp(
        self,
        node: str,
        market: str,
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """Query LMP data for a node/market over a date range.

        Args:
            node: CAISO PNode identifier.
            market: Market resolution string.
            start: ISO date string (inclusive), e.g. '2024-01-01'.
            end: ISO date string (inclusive).

        Returns:
            DataFrame sorted by time ascending.
        """
        return self._conn.execute(
            "SELECT * FROM lmp WHERE node = ? AND market = ? "
            "AND time >= ? AND time < (? ::DATE + INTERVAL 1 DAY) "
            "ORDER BY time",
            [node, market, start, end],
        ).df()

    def query_fuel_mix(self, fuel_type: str, start: str, end: str) -> pd.DataFrame:
        """Query fuel mix for a specific fuel type over a date range.

        Args:
            fuel_type: E.g. 'Solar', 'Wind'.
            start: ISO date string (inclusive).
            end: ISO date string (inclusive).

        Returns:
            DataFrame with columns [time, fuel_type, generation_mw].
        """
        return self._conn.execute(
            "SELECT * FROM fuel_mix WHERE fuel_type = ? "
            "AND time >= ? AND time < (? ::DATE + INTERVAL 1 DAY) "
            "ORDER BY time",
            [fuel_type, start, end],
        ).df()

    def query_features(self, node: str, start: str, end: str) -> pd.DataFrame:
        """Join LMP + load + weather into a feature DataFrame for model training.

        Args:
            node: CAISO PNode identifier.
            start: ISO date string (inclusive).
            end: ISO date string (inclusive).

        Returns:
            Wide DataFrame aligned on time (hourly, inner join).
        """
        return self._conn.execute(
            """
            SELECT
                l.time,
                l.lmp,
                l.energy,
                l.congestion,
                l.loss,
                la.load_mw,
                w.shortwave_radiation,
                w.direct_radiation,
                w.temperature_2m,
                w.wind_speed_10m,
                w.cloud_cover
            FROM lmp l
            -- load_actual is 5-min; aggregate to hourly so the hourly DA LMP
            -- join stays 1:1 (otherwise each LMP hour fans out to ~12 rows).
            LEFT JOIN (
                SELECT date_trunc('hour', time) AS h, AVG(load_mw) AS load_mw
                FROM load_actual
                GROUP BY 1
            ) la
                ON date_trunc('hour', l.time) = la.h
            LEFT JOIN weather w
                ON date_trunc('hour', l.time) = date_trunc('hour', w.time)
                AND w.node = l.node
            WHERE l.node = ?
                AND l.market = 'DAY_AHEAD_HOURLY'
                AND l.time >= ?
                AND l.time < (? ::DATE + INTERVAL 1 DAY)
            ORDER BY l.time
            """,
            [node, start, end],
        ).df()

    def get_latest_data_date(self, table: str) -> str | None:
        """Return the most recent date in a table (for incremental fetches).

        Args:
            table: Table name (e.g., 'lmp').

        Returns:
            ISO date string 'YYYY-MM-DD' of the latest row, or None if empty.
        """
        result = self._conn.execute(f"SELECT MAX(time) FROM {table}").fetchone()[0]  # noqa: S608
        if result is None:
            return None
        return str(result)[:10]

    def row_counts(self) -> dict[str, int]:
        """Return row counts for all data tables.

        Returns:
            Dict mapping table name to row count.
        """
        tables = ["lmp", "load_actual", "load_forecast", "fuel_mix",
                  "storage_soc", "curtailment", "weather", "agent_decisions"]
        return {
            t: self._conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]  # noqa: S608
            for t in tables
        }

    def execute(self, sql: str, params: list[Any] | None = None) -> pd.DataFrame:
        """Execute arbitrary SQL and return results as a DataFrame.

        Args:
            sql: SQL query string.
            params: Optional list of positional parameters.

        Returns:
            Query result as a pandas DataFrame.
        """
        return self._conn.execute(sql, params or []).df()

    def close(self) -> None:
        """Close the DuckDB connection."""
        self._conn.close()

    def __enter__(self) -> "MarketDB":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
