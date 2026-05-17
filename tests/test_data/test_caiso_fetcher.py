"""
Tests for CAISOFetcher.

Integration tests (marked with @pytest.mark.integration) hit the real CAISO API.
Unit tests mock gridstatus.CAISO to run offline.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.caiso_fetcher import CAISOFetcher


class TestDateChunks:
    """Tests for the _date_chunks static helper."""

    def test_single_chunk_when_range_fits(self) -> None:
        """Date range shorter than chunk size yields one chunk."""
        raise NotImplementedError()

    def test_multiple_chunks_for_long_range(self) -> None:
        """A 90-day range with 30-day chunks yields exactly 3 chunks."""
        raise NotImplementedError()

    def test_last_chunk_clamped_to_end_date(self) -> None:
        """The last chunk's end date never exceeds the requested end date."""
        raise NotImplementedError()


class TestNormalizeColumns:
    """Tests for column normalization."""

    def test_column_names_lowercased(self) -> None:
        """Column names are converted to lowercase snake_case."""
        raise NotImplementedError()

    def test_time_column_is_tz_aware(self) -> None:
        """The 'time' column is timezone-aware US/Pacific after normalization."""
        raise NotImplementedError()


class TestGetLmpHistory:
    """Unit tests for get_lmp_history (gridstatus mocked)."""

    @patch("src.data.caiso_fetcher.gridstatus.CAISO")
    def test_returns_dataframe_with_expected_columns(self, mock_caiso: MagicMock) -> None:
        """get_lmp_history returns DataFrame with [time, node, lmp, ...] columns."""
        raise NotImplementedError()

    @patch("src.data.caiso_fetcher.gridstatus.CAISO")
    def test_deduplicates_overlapping_chunks(self, mock_caiso: MagicMock) -> None:
        """Duplicate rows from overlapping fetches are removed."""
        raise NotImplementedError()

    @patch("src.data.caiso_fetcher.gridstatus.CAISO")
    def test_handles_empty_response_gracefully(self, mock_caiso: MagicMock) -> None:
        """Returns empty DataFrame (not raises) when CAISO returns no data."""
        raise NotImplementedError()


@pytest.mark.integration
class TestCAISOIntegration:
    """Integration tests — require live CAISO API access."""

    def test_fetch_7_days_rt_lmp_np15(self) -> None:
        """Can fetch 7 days of RT 5-min LMP for NP15 without error."""
        raise NotImplementedError()

    def test_storage_soc_history_returns_data(self) -> None:
        """get_storage_soc_history returns non-empty DataFrame."""
        raise NotImplementedError()
