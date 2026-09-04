"""
EXPERIMENT 003 — MOMENTUM X EARNINGS ACCELERATION COMPLEMENTARITY
BaseScore = Z(sector_relative_mom_60d) + Z(eps_growth_acceleration)
Fixed 1:1 Equal Weights.
"""

import pandas as pd
import numpy as np
from config.research_config import Config
from src.utils.io import load_csv, save_csv
from src.data.prices import process_prices
from src.data.fundamentals import compute_ttm_fundamentals
from src.data.pit_processor import build_pit_fundamental_timeline, merge_pit_features_to_grid
from src.features.momentum import compute_momentum_features
from src.features.earnings import compute_earnings_features
from src.features.common import zscore_cross_section
from src.utils.dates import generate_rebalance_schedule
from src.research.statistics import (
    cross_sectional_rank_ic, residualize_cross_section, calculate_hac_tstat
)
from src.research.portfolio_sorts import double_sort_2x2
from src.research.metrics import summarize_ic_series

def run():
    print("--- Running Experiment 003: Complementarity ---")
    raw_prices = load_csv("data/raw/prices_30.csv")
    raw_fund = load_csv("data/raw/fundamentals_30.csv")
    raw_filings = load_csv("data/raw/filings_dates.csv")
    
    prices_df = process_prices(raw_prices)
    mom_df = compute_momentum_features(prices_df)
    trading_days = pd.DatetimeIndex(prices_df['date'].unique()).sort_values()
    
    ttm_fund = compute_ttm_fundamentals(raw_fund)
    earn_df = compute_earnings_features(ttm_fund)
    pit_fund = build_pit_fundamental_timeline(earn_df, raw_filings, trading_days)
    
    grid = mom_df[['date', 'ticker', 'forward_20d', 'sector_relative_mom_60d']].drop_duplicates()
    merged = merge_pit_features_to_grid(grid, pit_fund)
    
    res_dates = generate_rebalance_schedule(trading_days, Config.RESEARCH_START, Config.RESEARCH_END)
    val_dates = generate_rebalance_schedule(trading_days, Config.VALIDATION_START, Config.VALIDATION_END)
    
    rebal_df = merged[merged['date'].isin(res_dates.union(val_dates))].copy()
    
    # BaseScore Construction
    rebal_df['z_mom'] = rebal_df.groupby('date')['sector_relative_mom_60d'].transform(zscore_cross_section)
    rebal_df['z_eps'] = rebal_df.groupby('date')['eps_growth_acceleration'].transform(zscore_cross_section)
    rebal_df['BaseScore'] = rebal_df['z_mom'] + rebal_df['z_eps']
    
    res_mask = rebal_df['date'].isin(res_dates)
    val_mask = rebal_df['date'].isin(val_dates)
    res_df = rebal_df[res_mask].copy()
    
    # 1. Factor Correlation
    def calc_corr(group):
        valid = group[['z_mom', 'z_eps']].dropna()
        if len(valid) < 5: return np.nan
        return valid.corr(method='spearman').iloc[0, 1]
    
    corr_series = res_df.groupby('date').apply(calc_corr)
    
    # 2. BaseScore IC & Validation
    ic_base_res = cross_sectional_rank_ic(res_df, 'BaseScore')
    ic_base_val = cross_sectional_rank_ic(rebal_df[val_mask], 'BaseScore')
    
    base_summary = summarize_ic_series(ic_base_res)
    
    # 3. Residualization
    res_df['mom_res_eps'] = residualize_cross_section(res_df, 'z_mom', ['z_eps'])
    res_df['eps_res_mom'] = residualize_cross_section(res_df, 'z_eps', ['z_mom'])
    
    ic_mom_res = cross_sectional_rank_ic(res_df, 'mom_res_eps')
    ic_eps_res = cross_sectional_rank_ic(res_df, 'eps_res_mom')
    
    # 4. 2x2 Double Sort
    sorts_2x2 = double_sort_2x2(res_df, 'z_mom', 'z_eps', 'forward_20d')
    
    out_dict = {
        'metric': ['BaseScore_IC_Research', 'BaseScore_HAC_t', 'BaseScore_IC_Val', 'Factor_Correlation_Mean'],
        'value': [base_summary['mean_ic'], base_summary['hac_t_lag_3'], ic_base_val.mean(), corr_series.mean()]
    }
    
    save_csv(pd.DataFrame(out_dict), "outputs/tables/exp_003_summary.csv")
    save_csv(sorts_2x2.mean().to_frame('mean_20d_return').reset_index(), "outputs/tables/exp_003_2x2_sorts.csv")
    
    print("Experiment 003 Complete. Outputs saved.")
    return out_dict

if __name__ == "__main__":
    run()