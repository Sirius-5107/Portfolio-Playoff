"""
Point-In-Time (PIT) lookahead detection tests.

These tests verify that the PIT merge infrastructure cannot inadvertently
introduce future information into rebalance-date feature vectors.
"""

import pytest
import pandas as pd
import numpy as np
from src.data.pit_processor import build_pit_fundamental_timeline, merge_pit_features_to_grid


def _make_trading_days(start="2020-01-01", end="2023-12-31"):
    return pd.DatetimeIndex(pd.date_range(start, end, freq="B"))


def _make_fund_ttm(n=8):
    """8 quarters of quarterly fundamentals for one ticker."""
    dates = pd.date_range("2020-01-01", periods=n, freq="QE")
    return pd.DataFrame({
        "ticker": ["TCS"] * n,
        "quarter_end": dates,
        "net_income_ttm": np.arange(n, dtype=float) * 100,
        "revenue_ttm": np.arange(n, dtype=float) * 500,
        "eps": np.arange(n, dtype=float) + 1.0,
    })


def _make_filings(fund_df, lag_days=45):
    """Each filing is lag_days after quarter_end (simulating actual delay)."""
    df = fund_df[["ticker", "quarter_end"]].copy()
    df["filing_date"] = pd.to_datetime(df["quarter_end"]) + pd.Timedelta(days=lag_days)
    return df


def test_usable_date_strictly_after_filing():
    """usable_date must always be strictly after filing_date."""
    trading_days = _make_trading_days()
    fund = _make_fund_ttm()
    filings = _make_filings(fund, lag_days=45)

    pit = build_pit_fundamental_timeline(fund, filings, trading_days)
    assert (pit["usable_date"] > pit["actual_filing_date"]).all(), (
        "usable_date must be strictly after filing_date — no same-day usage"
    )


def test_pit_merge_no_future_leakage():
    """
    After merging, the fundamental data available on any rebalance date
    must have a usable_date ≤ that rebalance date.
    """
    trading_days = _make_trading_days()
    fund = _make_fund_ttm()
    filings = _make_filings(fund, lag_days=45)

    pit = build_pit_fundamental_timeline(fund, filings, trading_days)

    # Create a grid of monthly rebalance dates
    grid_dates = pd.date_range("2021-01-01", "2023-12-31", freq="ME")
    grid = pd.DataFrame({
        "date": grid_dates,
        "ticker": "TCS",
        "forward_20d": 0.0,
    })

    merged = merge_pit_features_to_grid(grid, pit)
    valid = merged.dropna(subset=["usable_date"])

    leakage = valid[valid["usable_date"] > valid["date"]]
    assert len(leakage) == 0, (
        f"Future data leaked into {len(leakage)} rebalance rows:\n{leakage[['date','usable_date']].head()}"
    )


def test_fallback_used_when_filing_date_missing():
    """When filing_date is absent, the fallback (quarter_end + 45 days) must be used."""
    trading_days = _make_trading_days()
    fund = _make_fund_ttm(n=4)
    # No filing dates provided
    filings = pd.DataFrame({"ticker": [], "quarter_end": [], "filing_date": []})

    pit = build_pit_fundamental_timeline(fund, filings, trading_days)
    assert len(pit) > 0, "PIT should still produce rows using fallback dates"
    assert pit["is_fallback_date"].all(), "All rows should be marked as fallback"
