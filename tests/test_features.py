"""
Tests for feature computation modules.
"""

import pytest
import pandas as pd
import numpy as np
from src.features.momentum import compute_momentum_features
from src.features.quality import compute_quality_features
from src.features.earnings import compute_earnings_features


class TestMomentumFeatures:
    def _minimal_prices(self, n_days=300):
        """Build minimal price DataFrame with known tickers from the universe."""
        np.random.seed(0)
        dates = pd.date_range("2019-01-01", periods=n_days, freq="B")
        rows = []
        for ticker in ["TCS", "INFY", "HCLTECH"]:  # all IT sector
            price = 100.0
            for d in dates:
                price *= 1 + np.random.randn() * 0.01
                rows.append({"date": d, "ticker": ticker, "adj_close": price})
        return pd.DataFrame(rows)

    def test_sector_relative_momentum_demeaning(self):
        """For each (date, sector) group, sector-demeaned raw values must sum to ~0."""
        df = self._minimal_prices(300)
        mom_df = compute_momentum_features(df)
        # Drop NaN rows (early periods without enough history)
        valid = mom_df.dropna(subset=["sector_rel_60d_raw"])
        sec_sum = valid.groupby(["date", "sector"])["sector_rel_60d_raw"].sum()
        np.testing.assert_allclose(sec_sum.values, 0.0, atol=1e-9)

    def test_mom_12_1_not_same_as_252(self):
        """12-1 momentum should differ from raw 252D momentum."""
        df = self._minimal_prices(300)
        mom_df = compute_momentum_features(df)
        valid = mom_df.dropna(subset=["mom_252d", "mom_12_1"])
        assert not (valid["mom_252d"] == valid["mom_12_1"]).all()

    def test_output_contains_required_columns(self):
        df = self._minimal_prices(300)
        mom_df = compute_momentum_features(df)
        required = [
            "mom_5d", "mom_20d", "mom_60d", "mom_252d", "mom_12_1",
            "relative_momentum", "sector_relative_mom_60d", "sector_relative_mom_252d",
        ]
        for col in required:
            assert col in mom_df.columns, f"Missing column: {col}"


class TestQualityFeatures:
    def _sample_fundamental(self, ticker="TCS", is_financial=False):
        return pd.DataFrame([{
            "ticker": ticker,
            "quarter_end": "2023-03-31",
            "net_income_ttm": 200,
            "equity": 1000,
            "total_assets": 5000,
            "ebit_ttm": 300,
            "revenue_ttm": 1500,
            "fcf_ttm": 100,
            "ocf_ttm": 150,
            "short_term_debt": 50,
            "long_term_debt": 200,
            "interest_expense_ttm": 20,
        }])

    def test_financial_sector_exclusion(self):
        df = self._sample_fundamental(ticker="HDFCBANK")
        qual = compute_quality_features(df)
        # ROE/ROA must NOT be masked
        assert not np.isnan(qual["roe_ttm"].iloc[0]), "ROE should not be masked for financials"
        assert not np.isnan(qual["roa_ttm"].iloc[0]), "ROA should not be masked for financials"
        # Operational metrics MUST be masked
        for col in ["op_margin_ttm", "fcf_margin_ttm", "debt_to_equity", "ocf_margin_ttm"]:
            assert np.isnan(qual[col].iloc[0]), f"{col} should be NaN for financial sector"

    def test_non_financial_has_quality_metrics(self):
        df = self._sample_fundamental(ticker="TCS")
        qual = compute_quality_features(df)
        for col in ["roe_ttm", "roa_ttm", "op_margin_ttm", "fcf_margin_ttm"]:
            assert col in qual.columns
            # Should not be NaN for a non-financial with complete data
            assert not np.isnan(qual[col].iloc[0]), f"{col} should not be NaN for non-financial"

    def test_non_positive_equity_gives_nan_roe(self):
        df = self._sample_fundamental(ticker="TCS")
        df["equity"] = -500  # negative equity
        qual = compute_quality_features(df)
        assert np.isnan(qual["roe_ttm"].iloc[0]), "Non-positive equity must produce NaN ROE"


class TestEarningsFeatures:
    def _sample_quarterly(self, n_quarters=8):
        """8 quarters of data to allow 4-quarter YoY growth and 1-quarter accel."""
        dates = pd.date_range("2021-01-01", periods=n_quarters, freq="QE")
        rows = []
        for i, d in enumerate(dates):
            rows.append({
                "ticker": "INFY",
                "quarter_end": d,
                "eps": 10.0 + i * 0.5,
                "net_income": 1000 + i * 50,
                "revenue": 5000 + i * 100,
                "ebit": 800 + i * 30,
            })
        return pd.DataFrame(rows)

    def test_output_contains_acceleration_columns(self):
        df = self._sample_quarterly()
        earn = compute_earnings_features(df)
        for col in ["eps_growth", "eps_growth_acceleration", "profit_growth_acceleration"]:
            assert col in earn.columns, f"Missing column: {col}"

    def test_acceleration_is_first_diff_of_growth(self):
        df = self._sample_quarterly()
        earn = compute_earnings_features(df)
        # For a single ticker, diff(1) of eps_growth should equal eps_growth_acceleration
        valid = earn.dropna(subset=["eps_growth", "eps_growth_acceleration"])
        expected_accel = valid["eps_growth"].diff(1).dropna()
        actual_accel = valid["eps_growth_acceleration"].loc[expected_accel.index]
        np.testing.assert_allclose(expected_accel.values, actual_accel.values, atol=1e-10)
