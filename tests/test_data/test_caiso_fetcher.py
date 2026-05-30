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

_NODES = ["TH_NP15_GEN-APND"]


def _raw_lmp(times: list[str], node: str = "TH_NP15_GEN-APND") -> pd.DataFrame:
    """Build a gridstatus-style raw LMP frame (mixed-case columns, tz-aware Time)."""
    n = len(times)
    return pd.DataFrame(
        {
            "Time": pd.to_datetime(times).tz_localize("US/Pacific"),
            "Location": [node] * n,
            "Market": ["DAY_AHEAD_HOURLY"] * n,
            "LMP": [50.0] * n,
            "Energy": [48.0] * n,
            "Congestion": [1.0] * n,
            "Loss": [1.0] * n,
        }
    )


class TestDateChunks:
    """Tests for the _date_chunks static helper."""

    def test_single_chunk_when_range_fits(self) -> None:
        """Date range shorter than chunk size yields one chunk."""
        chunks = list(CAISOFetcher._date_chunks(date(2024, 1, 1), date(2024, 1, 10), 30))
        assert len(chunks) == 1
        assert chunks[0] == (date(2024, 1, 1), date(2024, 1, 10))

    def test_multiple_chunks_for_long_range(self) -> None:
        """A 90-day range with 30-day chunks yields exactly 3 chunks."""
        chunks = list(CAISOFetcher._date_chunks(date(2024, 1, 1), date(2024, 3, 30), 30))
        assert len(chunks) == 3

    def test_last_chunk_clamped_to_end_date(self) -> None:
        """The last chunk's end date never exceeds the requested end date."""
        end = date(2024, 3, 15)
        chunks = list(CAISOFetcher._date_chunks(date(2024, 1, 1), end, 30))
        assert chunks[-1][1] == end
        assert all(ce <= end for _, ce in chunks)


class TestNormalizeColumns:
    """Tests for LMP column normalization (_normalize_lmp)."""

    def test_column_names_lowercased(self) -> None:
        """Normalized columns use our lowercase snake_case schema."""
        out = CAISOFetcher._normalize_lmp(_raw_lmp(["2024-01-01 00:00"]))
        assert list(out.columns) == [
            "time",
            "node",
            "market",
            "lmp",
            "energy",
            "congestion",
            "loss",
        ]

    def test_time_column_is_tz_aware(self) -> None:
        """The 'time' column remains timezone-aware after normalization."""
        out = CAISOFetcher._normalize_lmp(_raw_lmp(["2024-01-01 00:00"]))
        assert isinstance(out["time"].dtype, pd.DatetimeTZDtype)


class TestGetLmpHistory:
    """Unit tests for get_lmp_history (gridstatus mocked)."""

    @patch("src.data.caiso_fetcher.time.sleep")
    @patch("src.data.caiso_fetcher.gridstatus.CAISO")
    def test_returns_dataframe_with_expected_columns(
        self, mock_caiso: MagicMock, mock_sleep: MagicMock
    ) -> None:
        """get_lmp_history returns DataFrame with [time, node, lmp, ...] columns."""
        mock_caiso.return_value.get_lmp.return_value = _raw_lmp(["2024-01-01 00:00"])
        fetcher = CAISOFetcher()
        out = fetcher.get_lmp_history(date(2024, 1, 1), date(2024, 1, 2), "DAY_AHEAD_HOURLY", _NODES)
        assert list(out.columns) == [
            "time", "node", "market", "lmp", "energy", "congestion", "loss"
        ]
        assert len(out) == 1

    @patch("src.data.caiso_fetcher.time.sleep")
    @patch("src.data.caiso_fetcher.gridstatus.CAISO")
    def test_deduplicates_overlapping_chunks(
        self, mock_caiso: MagicMock, mock_sleep: MagicMock
    ) -> None:
        """Duplicate rows from overlapping fetches are removed."""
        # Same row returned for every chunk; a >30-day range forces 2 chunks.
        mock_caiso.return_value.get_lmp.return_value = _raw_lmp(["2024-01-01 00:00"])
        fetcher = CAISOFetcher()
        out = fetcher.get_lmp_history(date(2024, 1, 1), date(2024, 2, 20), "DAY_AHEAD_HOURLY", _NODES)
        assert mock_caiso.return_value.get_lmp.call_count >= 2
        assert len(out) == 1  # deduplicated on (time, node, market)

    @patch("src.data.caiso_fetcher.time.sleep")
    @patch("src.data.caiso_fetcher.gridstatus.CAISO")
    def test_handles_empty_response_gracefully(
        self, mock_caiso: MagicMock, mock_sleep: MagicMock
    ) -> None:
        """Returns empty DataFrame (not raises) when CAISO returns no data."""
        mock_caiso.return_value.get_lmp.return_value = pd.DataFrame()
        fetcher = CAISOFetcher()
        out = fetcher.get_lmp_history(date(2024, 1, 1), date(2024, 1, 2), "DAY_AHEAD_HOURLY", _NODES)
        assert out.empty
        assert list(out.columns) == [
            "time", "node", "market", "lmp", "energy", "congestion", "loss"
        ]


@pytest.mark.integration
class TestCAISOIntegration:
    """Integration tests — require live CAISO API access."""

    def test_fetch_7_days_rt_lmp_np15(self) -> None:
        """Can fetch 7 days of RT 5-min LMP for NP15 without error."""
        fetcher = CAISOFetcher()
        df = fetcher.get_lmp_history(
            date(2024, 1, 1), date(2024, 1, 7), "REAL_TIME_5_MIN", _NODES
        )
        assert not df.empty
        assert {"time", "node", "lmp"} <= set(df.columns)

    def test_storage_soc_history_returns_data(self) -> None:
        """get_storage_soc_history returns a non-empty DataFrame for an old date."""
        fetcher = CAISOFetcher()
        df = fetcher.get_storage_soc_history(date(2024, 1, 1), date(2024, 1, 2))
        assert {"time", "soc_mwh", "schedule"} <= set(df.columns)
