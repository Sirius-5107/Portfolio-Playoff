"""
Momentum Feature Computation Engine.

Lookback periods computed:
  mom_5d    5-trading-day cumulative return
  mom_20d   20-trading-day cumulative return
  mom_60d   60-trading-day cumulative return
  mom_252d  252-trading-day (≈1-year) cumulative return
  mom_12_1  12-month minus 1-month momentum (price{t-21} / price{t-252} - 1)

Derived signals:
  relative_momentum          mom_60d minus cross-sectional mean of mom_60d
  sector_relative_mom_60d    Cross-sectional Z-score of (mom_60d - sector_mean_60d)
  sector_relative_mom_252d   Cross-sectional Z-score of (mom_252d - sector_mean_252d)

Sector demeaning is applied BEFORE Z-scoring.
"""

import pandas as pd
import numpy as np
from config.sectors import SECTOR_MAP
from .common import zscore_cross_section


def compute_momentum_features(prices_df: pd.DataFrame) -> pd.DataFrame:
    """Compute all momentum features from a daily price DataFrame.

    Parameters
    ----------
    prices_df : pd.DataFrame
        Must contain: date, ticker, adj_close.

    Returns
    -------
    pd.DataFrame with momentum feature columns appended.
    """
    df = prices_df.copy().sort_values(["ticker", "date"]).reset_index(drop=True)

    # --- Raw lookback returns ---
    for days in [5, 20, 60, 252]:
        df[f"mom_{days}d"] = df.groupby("ticker")["adj_close"].pct_change(days)

    # --- 12-1 Momentum (skip most-recent month to reduce short-term reversal noise) ---
    p_lag21 = df.groupby("ticker")["adj_close"].shift(21)
    p_lag252 = df.groupby("ticker")["adj_close"].shift(252)
    df["mom_12_1"] = (p_lag21 / p_lag252) - 1.0

    # --- Relative Momentum (vs universe cross-sectional mean of 60D return) ---
    unv_mean_60 = df.groupby("date")["mom_60d"].transform("mean")
    df["relative_momentum"] = df["mom_60d"] - unv_mean_60

    # --- Sector-Relative Momentum ---
    df["sector"] = df["ticker"].map(SECTOR_MAP)

    for lookback in [60, 252]:
        col = f"mom_{lookback}d"
        raw_col = f"sector_rel_{lookback}d_raw"
        out_col = f"sector_relative_mom_{lookback}d"

        sec_mean = df.groupby(["date", "sector"])[col].transform("mean")
        df[raw_col] = df[col] - sec_mean

        # Cross-sectional Z-score applied strictly AFTER sector demeaning
        df[out_col] = df.groupby("date")[raw_col].transform(zscore_cross_section)

    return df
