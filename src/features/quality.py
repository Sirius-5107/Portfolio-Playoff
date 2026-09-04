"""
Quality Factors and Financial-Sector Masking Rules.

Financial-Sector Constraints (HDFCBANK, ICICIBANK, KOTAKBANK, SBIN, AXISBANK, BAJFINANCE):
  ROE, ROA            → Included (valid across all sectors)
  Operating Margin    → EXCLUDED (set to NaN)
  Net Margin          → EXCLUDED
  FCF Margin          → EXCLUDED
  OCF Margin          → EXCLUDED
  Debt/Equity         → EXCLUDED
  Interest Coverage   → EXCLUDED

Definitions (all TTM):
  ROE  = NetIncome_TTM / avg(Equity_t, Equity_{t-1Y})
  ROA  = NetIncome_TTM / avg(Assets_t, Assets_{t-1Y})
  OpMargin  = EBIT_TTM / Revenue_TTM
  NetMargin = NetIncome_TTM / Revenue_TTM
  FCFMargin = (OCF_TTM - CapEx_TTM) / Revenue_TTM
  OCFMargin = OCF_TTM / Revenue_TTM
  D/E  = (STDebt + LTDebt) / Equity
  IntCov = EBIT_TTM / InterestExpense_TTM

Non-positive equity → ROE becomes NaN (avoids sign-flip artifacts).
"""

import pandas as pd
import numpy as np
from config.sectors import FINANCIAL_TICKERS


def compute_quality_features(pit_fund_df: pd.DataFrame) -> pd.DataFrame:
    """Compute all quality metrics from a PIT-aligned fundamental DataFrame.

    Parameters
    ----------
    pit_fund_df : pd.DataFrame
        Must contain (at minimum): ticker, net_income_ttm, equity, total_assets.
        Optional: ebit_ttm, revenue_ttm, ocf_ttm, fcf_ttm,
                  short_term_debt, long_term_debt, interest_expense_ttm.

    Returns
    -------
    pd.DataFrame with quality columns appended.
    """
    df = pit_fund_df.copy()
    cols = df.columns  # capture column list once for membership checks

    # --- Lagged balance-sheet averages ---
    equity_lag4 = df.groupby("ticker")["equity"].shift(4)
    assets_lag4 = df.groupby("ticker")["total_assets"].shift(4)

    # Fallback to current period when lagged value unavailable (e.g. short history)
    df["equity_avg"] = (df["equity"] + equity_lag4.fillna(df["equity"])) / 2.0
    df["assets_avg"] = (df["total_assets"] + assets_lag4.fillna(df["total_assets"])) / 2.0

    # --- ROE and ROA — valid for ALL sectors including Financials ---
    # Replace non-positive equity with NaN to avoid sign-flip artifacts.
    equity_safe = df["equity"].where(df["equity"] > 0, np.nan)
    assets_avg_safe = df["assets_avg"].where(df["assets_avg"] != 0, np.nan)

    df["roe_ttm"] = df["net_income_ttm"] / equity_safe
    df["roa_ttm"] = df["net_income_ttm"] / assets_avg_safe

    # --- Non-financial operational / leverage metrics ---
    # Revenue denominator
    if "revenue_ttm" in cols:
        rev_safe = df["revenue_ttm"].where(df["revenue_ttm"] != 0, np.nan)
    else:
        rev_safe = np.nan

    if "ebit_ttm" in cols:
        df["op_margin_ttm"] = df["ebit_ttm"] / rev_safe
    else:
        df["op_margin_ttm"] = np.nan

    if "revenue_ttm" in cols:
        df["net_margin_ttm"] = df["net_income_ttm"] / rev_safe
    else:
        df["net_margin_ttm"] = np.nan

    if "fcf_ttm" in cols:
        df["fcf_margin_ttm"] = df["fcf_ttm"] / rev_safe
    else:
        df["fcf_margin_ttm"] = np.nan

    if "ocf_ttm" in cols:
        df["ocf_margin_ttm"] = df["ocf_ttm"] / rev_safe
    else:
        df["ocf_margin_ttm"] = np.nan

    if "short_term_debt" in cols and "long_term_debt" in cols:
        df["debt_to_equity"] = (df["short_term_debt"] + df["long_term_debt"]) / equity_safe
    else:
        df["debt_to_equity"] = np.nan

    if "ebit_ttm" in cols and "interest_expense_ttm" in cols:
        int_exp_safe = df["interest_expense_ttm"].where(df["interest_expense_ttm"] != 0, np.nan)
        df["interest_coverage"] = df["ebit_ttm"] / int_exp_safe
    else:
        df["interest_coverage"] = np.nan

    # --- Enforce Financial-Sector Exclusion Masking ---
    fin_mask = df["ticker"].isin(FINANCIAL_TICKERS)
    excluded_cols = [
        "op_margin_ttm",
        "net_margin_ttm",
        "fcf_margin_ttm",
        "ocf_margin_ttm",
        "debt_to_equity",
        "interest_coverage",
    ]
    for col in excluded_cols:
        if col in df.columns:
            df.loc[fin_mask, col] = np.nan

    return df
