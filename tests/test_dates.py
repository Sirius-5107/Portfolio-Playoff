"""
Tests for rebalance schedule generation.
"""

import pytest
import pandas as pd
from config.research_config import Config
from src.utils.dates import generate_rebalance_schedule


def test_rebalance_schedule_research_count():
    trading_days = pd.date_range("2015-01-01", "2023-12-31", freq="B")
    res_dates = generate_rebalance_schedule(
        trading_days, Config.RESEARCH_START, Config.RESEARCH_END
    )
    assert abs(len(res_dates) - Config.REQUIRED_RESEARCH_REBALANCES) <= 5, (
        f"Research rebalances: expected ~{Config.REQUIRED_RESEARCH_REBALANCES}, got {len(res_dates)}"
    )


def test_rebalance_schedule_validation_count():
    trading_days = pd.date_range("2024-01-01", "2026-09-04", freq="B")
    val_dates = generate_rebalance_schedule(
        trading_days, Config.VALIDATION_START, Config.VALIDATION_END
    )
    assert len(val_dates) > 0, "Validation schedule should not be empty"


def test_rebalance_schedule_no_overlap():
    trading_days = pd.date_range("2015-01-01", "2026-09-04", freq="B")
    res_dates = generate_rebalance_schedule(
        trading_days, Config.RESEARCH_START, Config.RESEARCH_END
    )
    val_dates = generate_rebalance_schedule(
        trading_days, Config.VALIDATION_START, Config.VALIDATION_END
    )
    overlap = set(res_dates) & set(val_dates)
    assert not overlap, f"Research and validation schedules overlap: {sorted(overlap)[:5]}"


def test_rebalance_schedule_all_dates_in_range():
    trading_days = pd.date_range("2015-01-01", "2023-12-31", freq="B")
    res_dates = generate_rebalance_schedule(
        trading_days, Config.RESEARCH_START, Config.RESEARCH_END
    )
    assert all(Config.RESEARCH_START <= d <= Config.RESEARCH_END for d in res_dates)


def test_rebalance_schedule_empty_for_empty_input():
    empty = pd.DatetimeIndex([])
    result = generate_rebalance_schedule(
        empty, Config.RESEARCH_START, Config.RESEARCH_END
    )
    assert len(result) == 0
