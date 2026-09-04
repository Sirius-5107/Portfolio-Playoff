"""
Tests for centralized statistical functions.
"""

import pytest
import pandas as pd
import numpy as np
from src.research.statistics import (
    residualize_cross_section,
    calculate_hac_tstat,
    cross_sectional_rank_ic,
    winsorize_cross_section,
)
from src.features.common import zscore_cross_section


class TestZscoreCrossSection:
    def test_deterministic_output(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        z1 = zscore_cross_section(s)
        z2 = zscore_cross_section(s)
        pd.testing.assert_series_equal(z1, z2)

    def test_clipping_applied(self):
        s = pd.Series([-10.0, 0.0, 0.0, 0.0, 10.0])
        z = zscore_cross_section(s, clip_lower=-3.0, clip_upper=3.0)
        assert z.max() <= 3.0
        assert z.min() >= -3.0

    def test_zero_std_returns_zeros(self):
        s = pd.Series([5.0, 5.0, 5.0])
        z = zscore_cross_section(s)
        assert (z == 0.0).all()


class TestResidualizeCrossSection:
    def test_residuals_orthogonal_to_regressor(self):
        np.random.seed(42)
        df = pd.DataFrame({
            "date": ["2023-01-01"] * 30,
            "ticker": [f"T_{i}" for i in range(30)],
            "z_mom": np.random.randn(30),
            "z_eps": np.random.randn(30),
        })
        res = residualize_cross_section(df, "z_mom", ["z_eps"])
        df["res"] = res
        corr = df[["res", "z_eps"]].corr().iloc[0, 1]
        np.testing.assert_allclose(corr, 0.0, atol=1e-7)

    def test_residuals_same_length(self):
        np.random.seed(0)
        df = pd.DataFrame({
            "date": ["2023-01-01"] * 20,
            "ticker": [f"T_{i}" for i in range(20)],
            "y": np.random.randn(20),
            "x": np.random.randn(20),
        })
        res = residualize_cross_section(df, "y", ["x"])
        assert len(res) == 20


class TestHacTstat:
    def test_positive_series(self):
        s = pd.Series([0.05] * 50)
        t, p = calculate_hac_tstat(s, max_lag=3)
        assert t > 0
        assert 0 <= p <= 1

    def test_too_short_returns_nan(self):
        s = pd.Series([0.01, 0.02])
        t, p = calculate_hac_tstat(s, max_lag=3)
        assert np.isnan(t)
        assert np.isnan(p)


class TestCrossSectionalRankIC:
    def test_positive_signal(self):
        """A factor that perfectly predicts rank should give IC ≈ 1.0."""
        dates = pd.date_range("2020-01-01", periods=10, freq="ME")
        rows = []
        for d in dates:
            vals = np.arange(1, 11, dtype=float)
            for i, v in enumerate(vals):
                rows.append({"date": d, "ticker": f"T{i}", "factor": v, "forward_20d": v + np.random.randn() * 0.01})
        df = pd.DataFrame(rows)
        ic = cross_sectional_rank_ic(df, "factor", "forward_20d")
        assert ic.mean() > 0.9

    def test_returns_nan_for_small_group(self):
        df = pd.DataFrame({
            "date": ["2020-01-01"] * 4,
            "ticker": ["A", "B", "C", "D"],
            "factor": [1, 2, 3, 4],
            "forward_20d": [4, 3, 2, 1],
        })
        ic = cross_sectional_rank_ic(df, "factor", "forward_20d")
        # Groups smaller than min_stocks=5 → NaN
        assert ic.isna().all()


class TestWinsorize:
    def test_winsorize_clamps_extremes(self):
        df = pd.DataFrame({
            "date": ["2020-01-01"] * 10,
            "factor": [-100.0] + [0.0] * 8 + [100.0],
        })
        out = winsorize_cross_section(df, "factor", limits=(0.1, 0.9))
        assert out["factor"].max() < 100.0
        assert out["factor"].min() > -100.0
