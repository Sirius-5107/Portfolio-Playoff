"""
Price ingestion and forward return computation.

Convention: uses total-return adjusted close prices to handle dividends
and corporate actions.

    ForwardReturn_{t, h} = P_{t+h} / P_{t} - 1
"""

import pandas as pd
import numpy as np
from config.universe import COMPETITION_UNIVERSE


def process_prices(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Filter to the competition universe and compute all forward return horizons.

    Parameters
    ----------
    raw_df : pd.DataFrame
        Must contain columns: date, ticker, adj_close

    Returns
    -------
    pd.DataFrame
        Sorted by (ticker, date) with forward_5d … forward_30d columns added.
    """
    df = raw_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = (
        df[df["ticker"].isin(COMPETITION_UNIVERSE)]
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )

    horizons = [5, 10, 20, 30]
    for h in horizons:
        df[f"forward_{h}d"] = (
            df.groupby("ticker")["adj_close"].shift(-h) / df["adj_close"] - 1.0
        )

    return df
