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
    raise NotImplementedError(
        # return MarketDB(':memory:')
    )


class TestSchemaInit:
    """Tests that the schema is correctly created on init."""

    def test_all_tables_exist(self, db: MarketDB) -> None:
        """All expected tables exist in a fresh database."""
        raise NotImplementedError()


class TestUpsertLmp:
    """Tests for upsert_lmp()."""

    def test_insert_new_rows(self, db: MarketDB) -> None:
        """Inserting new LMP rows increases the table count."""
        raise NotImplementedError()

    def test_upsert_deduplicates_on_primary_key(self, db: MarketDB) -> None:
        """Upserting the same (time, node, market) twice does not create duplicate rows."""
        raise NotImplementedError()

    def test_returns_correct_row_count(self, db: MarketDB) -> None:
        """upsert_lmp() returns the number of new rows written."""
        raise NotImplementedError()


class TestQueryLmp:
    """Tests for query_lmp()."""

    def test_returns_correct_date_range(self, db: MarketDB) -> None:
        """query_lmp() returns only rows within the requested date range."""
        raise NotImplementedError()

    def test_filters_by_node_and_market(self, db: MarketDB) -> None:
        """Only rows matching the requested node and market are returned."""
        raise NotImplementedError()

    def test_empty_result_when_no_data(self, db: MarketDB) -> None:
        """Returns empty DataFrame (not raises) when no matching rows exist."""
        raise NotImplementedError()


class TestLogAgentDecision:
    """Tests for agent decision logging."""

    def test_decision_persisted_and_retrievable(self, db: MarketDB) -> None:
        """A logged agent decision can be read back from the database."""
        raise NotImplementedError()
