"""
Tests for MarketDB (DuckDB interface).

All tests use an in-memory DuckDB (':memory:') so no files are created.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.db import MarketDB


@pytest.fixture
def db() -> MarketDB:
    """Provide a fresh in-memory MarketDB for each test."""
    database = MarketDB(":memory:")
    yield database
    try:
        database.close()
    except Exception:
        pass


def _lmp_df(
    times: list[str],
    node: str = "TH_NP15_GEN-APND",
    market: str = "DAY_AHEAD_HOURLY",
) -> pd.DataFrame:
    """Build an LMP DataFrame matching the upsert_lmp schema."""
    n = len(times)
    return pd.DataFrame(
        {
            "time": pd.to_datetime(times, utc=True),
            "node": [node] * n,
            "market": [market] * n,
            "lmp": [50.0 + i for i in range(n)],
            "energy": [48.0 + i for i in range(n)],
            "congestion": [1.0] * n,
            "loss": [1.0] * n,
        }
    )


class TestSchemaInit:
    """Tests that the schema is correctly created on init."""

    def test_all_tables_exist(self, db: MarketDB) -> None:
        """All expected tables exist in a fresh database."""
        expected = {
            "lmp",
            "load_actual",
            "load_forecast",
            "fuel_mix",
            "storage_soc",
            "curtailment",
            "weather",
            "agent_decisions",
        }
        assert expected <= set(db.row_counts())


class TestUpsertLmp:
    """Tests for upsert_lmp()."""

    def test_insert_new_rows(self, db: MarketDB) -> None:
        """Inserting new LMP rows increases the table count."""
        db.upsert_lmp(_lmp_df(["2024-01-01 00:00", "2024-01-01 01:00"]))
        assert db.row_counts()["lmp"] == 2

    def test_upsert_deduplicates_on_primary_key(self, db: MarketDB) -> None:
        """Upserting the same (time, node, market) twice does not create duplicate rows."""
        df = _lmp_df(["2024-01-01 00:00"])
        db.upsert_lmp(df)
        db.upsert_lmp(df)
        assert db.row_counts()["lmp"] == 1

    def test_returns_correct_row_count(self, db: MarketDB) -> None:
        """upsert_lmp() returns the number of new rows written."""
        delta = db.upsert_lmp(_lmp_df(["2024-01-01 00:00", "2024-01-01 01:00"]))
        assert delta == 2

    def test_empty_df_returns_zero(self, db: MarketDB) -> None:
        """Upserting an empty DataFrame is a no-op returning 0."""
        assert db.upsert_lmp(pd.DataFrame()) == 0


class TestQueryLmp:
    """Tests for query_lmp()."""

    def test_returns_correct_date_range(self, db: MarketDB) -> None:
        """query_lmp() returns only rows within the requested date range."""
        # Noon UTC keeps each timestamp on the same calendar day in local time.
        db.upsert_lmp(
            _lmp_df(["2024-01-01 12:00", "2024-01-02 12:00", "2024-01-05 12:00"])
        )
        out = db.query_lmp("TH_NP15_GEN-APND", "DAY_AHEAD_HOURLY", "2024-01-01", "2024-01-02")
        assert len(out) == 2  # 01-01 and 01-02 inclusive, 01-05 excluded

    def test_filters_by_node_and_market(self, db: MarketDB) -> None:
        """Only rows matching the requested node and market are returned."""
        db.upsert_lmp(_lmp_df(["2024-01-01 12:00"], node="TH_NP15_GEN-APND"))
        db.upsert_lmp(_lmp_df(["2024-01-01 12:00"], node="TH_SP15_GEN-APND"))
        db.upsert_lmp(
            _lmp_df(["2024-01-01 12:00"], node="TH_NP15_GEN-APND", market="REAL_TIME_5_MIN")
        )
        out = db.query_lmp("TH_NP15_GEN-APND", "DAY_AHEAD_HOURLY", "2024-01-01", "2024-01-01")
        assert len(out) == 1
        assert out.iloc[0]["node"] == "TH_NP15_GEN-APND"
        assert out.iloc[0]["market"] == "DAY_AHEAD_HOURLY"

    def test_empty_result_when_no_data(self, db: MarketDB) -> None:
        """Returns empty DataFrame (not raises) when no matching rows exist."""
        out = db.query_lmp("TH_NP15_GEN-APND", "DAY_AHEAD_HOURLY", "2024-01-01", "2024-01-31")
        assert out.empty


class TestLogAgentDecision:
    """Tests for agent decision logging."""

    def test_decision_persisted_and_retrievable(self, db: MarketDB) -> None:
        """A logged agent decision can be read back from the database."""
        db.log_agent_decision(
            agent="RiskMonitor",
            action="HALT",
            rationale="EEA3 detected",
            metadata={"level": "halt"},
        )
        rows = db._conn.execute(
            "SELECT agent, action, rationale FROM agent_decisions"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "RiskMonitor"
        assert rows[0][1] == "HALT"
        assert rows[0][2] == "EEA3 detected"
