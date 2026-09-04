"""
Reproducibility guard tests.

These tests verify determinism and internal consistency — they do NOT
require external data files and must pass on a fresh clone.
"""

import pytest
import pandas as pd
import numpy as np
from src.features.common import zscore_cross_section
from src.research.statistics import cross_sectional_rank_ic


def test_zscore_is_deterministic():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    z1 = zscore_cross_section(s)
    z2 = zscore_cross_section(s)
    pd.testing.assert_series_equal(z1, z2)


def test_zscore_mean_near_zero():
    np.random.seed(42)
    s = pd.Series(np.random.randn(100))
    z = zscore_cross_section(s, clip_lower=None, clip_upper=None)
    assert abs(z.mean()) < 1e-10


def test_zscore_std_near_one_without_clipping():
    np.random.seed(42)
    s = pd.Series(np.random.randn(50))
    z = zscore_cross_section(s, clip_lower=None, clip_upper=None)
    np.testing.assert_allclose(z.std(), 1.0, atol=1e-10)


def test_ic_between_minus_one_and_one():
    """Spearman IC must be in [-1, 1]."""
    np.random.seed(7)
    dates = pd.date_range("2020-01-01", periods=20, freq="ME")
    rows = []
    for d in dates:
        for i in range(10):
            rows.append({
                "date": d,
                "ticker": f"T{i}",
                "factor": np.random.randn(),
                "forward_20d": np.random.randn(),
            })
    df = pd.DataFrame(rows)
    ic = cross_sectional_rank_ic(df, "factor", "forward_20d")
    valid = ic.dropna()
    assert (valid >= -1.0).all() and (valid <= 1.0).all()


def test_no_lookahead_in_forward_returns():
    """forward_20d at time t must be NaN for the last 20 rows per ticker."""
    from src.data.prices import process_prices
    dates = pd.date_range("2020-01-01", periods=50, freq="B")
    rows = [{"date": d, "ticker": "TCS", "adj_close": 100.0 + i} for i, d in enumerate(dates)]
    df = pd.DataFrame(rows)
    out = process_prices(df)
    # Last 20 rows for TCS should have NaN forward_20d
    tcs_tail = out[out["ticker"] == "TCS"].tail(20)
    assert tcs_tail["forward_20d"].isna().all(), (
        "forward_20d must be NaN for the last 20 rows (no future data available)"
    )
