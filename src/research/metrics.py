"""
Core Aggregation Metrics for Experiments.
"""

import pandas as pd
import numpy as np
from .statistics import calculate_hac_tstat

def summarize_ic_series(ic_series: pd.Series, hac_lags: list[int] = [0, 1, 3, 6, 12]) -> dict:
    clean = ic_series.dropna()
    res = {
        'mean_ic': clean.mean(),
        'median_ic': clean.median(),
        'std_ic': clean.std(),
        'positive_pct': (clean > 0).mean(),
        'count': len(clean)
    }
    
    for lag in hac_lags:
        t_stat, p_val = calculate_hac_tstat(clean, max_lag=lag)
        res[f'hac_t_lag_{lag}'] = t_stat
        res[f'hac_p_lag_{lag}'] = p_val
        
    return res