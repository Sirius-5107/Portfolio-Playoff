"""
EXPERIMENT 001 — HISTORICAL MOMENTUM
"""

import pandas as pd
import numpy as np
from config.research_config import Config
from src.utils.io import load_csv, save_csv
from src.data.prices import process_prices
from src.features.momentum import compute_momentum_features
from src.utils.dates import generate_rebalance_schedule
from src.research.statistics import cross_sectional_rank_ic
from src.research.metrics import summarize_ic_series
from src.research.robustness import leave_one_stock_out

def run():
    print("--- Running Experiment 001: Historical Momentum ---")
    raw_prices = load_csv("data/raw/prices_30.csv")
    prices_df = process_prices(raw_prices)
    mom_df = compute_momentum_features(prices_df)
    
    trading_days = pd.DatetimeIndex(mom_df['date'].unique()).sort_values()
    
    res_dates = generate_rebalance_schedule(trading_days, Config.RESEARCH_START, Config.RESEARCH_END)
    val_dates = generate_rebalance_schedule(trading_days, Config.VALIDATION_START, Config.VALIDATION_END)
    
    rebal_df = mom_df[mom_df['date'].isin(res_dates.union(val_dates))].copy()
    
    factors = [
        'mom_5d', 'mom_20d', 'mom_60d', 'mom_252d',
        'mom_12_1', 'sector_relative_mom_60d', 'sector_relative_mom_252d'
    ]
    
    res_mask = rebal_df['date'].isin(res_dates)
    val_mask = rebal_df['date'].isin(val_dates)
    
    summary = []
    for f in factors:
        ic_res = cross_sectional_rank_ic(rebal_df[res_mask], f, 'forward_20d')
        ic_val = cross_sectional_rank_ic(rebal_df[val_mask], f, 'forward_20d')
        
        m_res = summarize_ic_series(ic_res)
        m_val = summarize_ic_series(ic_val)
        
        summary.append({
            'factor': f,
            'research_ic': m_res['mean_ic'],
            'research_hac_t': m_res['hac_t_lag_3'],
            'validation_ic': m_val['mean_ic'],
            'validation_label': Config.VALIDATION_LABEL
        })
        
    out_df = pd.DataFrame(summary)
    save_csv(out_df, "outputs/tables/exp_001_ic_summary.csv")
    
    # Leave-one-stock-out for sector_relative_mom_60d
    loso = leave_one_stock_out(rebal_df[res_mask], 'sector_relative_mom_60d')
    save_csv(loso, "outputs/tables/exp_001_loso.csv")
    
    print("Experiment 001 Complete. Outputs saved.")
    return out_df

if __name__ == "__main__":
    run()