"""
Robustness Testing: Leave-One-Stock-Out and Sub-period Sensitivity.
"""

import pandas as pd
import numpy as np
from config.universe import UNIVERSE
from .statistics import cross_sectional_rank_ic

def leave_one_stock_out(df: pd.DataFrame, factor_col: str, target_col: str = 'forward_20d') -> pd.DataFrame:
    results = []
    for stock in UNIVERSE:
        sub = df[df['ticker'] != stock]
        ic_series = cross_sectional_rank_ic(sub, factor_col, target_col)
        results.append({
            'dropped_stock': stock,
            'mean_ic': ic_series.mean(),
            'std_ic': ic_series.std(),
            'positive_pct': (ic_series > 0).mean()
        })
    return pd.DataFrame(results)