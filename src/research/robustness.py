"""
Robustness Testing: Leave-One-Stock-Out and Sub-period Sensitivity.
"""

import pandas as pd
import numpy as np
from config.universe import COMPETITION_UNIVERSE
from .statistics import cross_sectional_rank_ic


def leave_one_stock_out(
    df: pd.DataFrame,
    factor_col: str,
    target_col: str = "forward_20d",
) -> pd.DataFrame:
    """Compute IC series dropping one stock at a time.

    For each stock in COMPETITION_UNIVERSE, removes it from ``df``, computes
    the cross-sectional rank IC, and records summary statistics.

    Returns
    -------
    pd.DataFrame with columns: dropped_stock, mean_ic, std_ic, positive_pct
    """
    results = []
    for stock in COMPETITION_UNIVERSE:
        sub = df[df["ticker"] != stock]
        ic_series = cross_sectional_rank_ic(sub, factor_col, target_col)
        results.append(
            {
                "dropped_stock": stock,
                "mean_ic": ic_series.mean(),
                "std_ic": ic_series.std(),
                "positive_pct": (ic_series > 0).mean(),
            }
        )
    return pd.DataFrame(results)
