"""
Earnings Momentum Features.

All growth calculations are YoY (quarter vs same quarter prior year).
Acceleration is strictly the 1-quarter first difference of YoY growth:

    EPS_Accel_t = EPS_Growth_t - EPS_Growth_{t-1}

No median imputation for missing values.
No lookahead — all features are computed on quarterly observations;
PIT alignment is handled downstream in pit_processor.py.

Features produced
-----------------
eps_growth                   YoY EPS growth
profit_growth                YoY net income growth
revenue_growth               YoY revenue growth
eps_growth_acceleration      1-quarter delta of eps_growth
profit_growth_acceleration   1-quarter delta of profit_growth
revenue_growth_acceleration  1-quarter delta of revenue_growth
op_margin_change             YoY change in operating margin (EBIT/Revenue)
days_since_earnings          Calendar days between quarter_end and usable_date
                             (populated downstream after PIT merge)
"""

import pandas as pd
import numpy as np


def compute_earnings_features(pit_fund_df: pd.DataFrame) -> pd.DataFrame:
    """Compute earnings momentum features from a quarterly fundamental DataFrame.

    Parameters
    ----------
    pit_fund_df : pd.DataFrame
        Must contain: ticker, quarter_end, eps, net_income, revenue, ebit.
        The DataFrame is sorted by (ticker, quarter_end) before computation.

    Returns
    -------
    pd.DataFrame with earnings features appended.
    """
    df = pit_fund_df.copy().sort_values(["ticker", "quarter_end"]).reset_index(drop=True)

    # --- YoY Level Growth (4-quarter shift = same quarter prior year) ---
    df["eps_growth"] = df.groupby("ticker")["eps"].pct_change(4)
    df["profit_growth"] = df.groupby("ticker")["net_income"].pct_change(4)
    df["revenue_growth"] = df.groupby("ticker")["revenue"].pct_change(4)

    # --- Acceleration (1-quarter first difference of YoY growth) ---
    df["eps_growth_acceleration"] = df.groupby("ticker")["eps_growth"].diff(1)
    df["profit_growth_acceleration"] = df.groupby("ticker")["profit_growth"].diff(1)
    df["revenue_growth_acceleration"] = df.groupby("ticker")["revenue_growth"].diff(1)

    # --- Operating Margin Change (YoY) ---
    df["op_margin"] = df["ebit"] / df["revenue"].replace(0, np.nan)
    df["op_margin_change"] = df.groupby("ticker")["op_margin"].diff(4)

    # days_since_earnings is populated AFTER the PIT merge, when usable_date is known.
    # Placeholder column added here so downstream code can always reference it.
    df["days_since_earnings"] = np.nan

    return df


def attach_days_since_earnings(merged_df: pd.DataFrame) -> pd.DataFrame:
    """Compute days_since_earnings after the PIT merge has added usable_date.

    Parameters
    ----------
    merged_df : pd.DataFrame
        Must contain columns: date (rebalance date), usable_date (PIT effective date).

    Returns
    -------
    pd.DataFrame with days_since_earnings populated.
    """
    df = merged_df.copy()
    if "usable_date" in df.columns and "date" in df.columns:
        df["days_since_earnings"] = (
            pd.to_datetime(df["date"]) - pd.to_datetime(df["usable_date"])
        ).dt.days
    return df
