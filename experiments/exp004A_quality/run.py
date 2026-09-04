"""
EXPERIMENT 004A — QUALITY INCREMENTAL SIGNAL AUDIT
Models:
Model A: Z(Mom)
Model B: Z(EPS)
Model C (BaseScore): Z(Mom) + Z(EPS)
Model D1: Z(Mom) + Z(EPS) + Z(ROE)
Model D2: Z(Mom) + Z(EPS) + Z(FCF)
PAIRED DELTA-IC TEST:
Delta_IC_t = IC_{Base+ROE, t} - IC_{Base, t}
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
from src.features.quality import compute_quality_features
from src.features.common import zscore_cross_section
from src.utils.dates import generate_rebalance_schedule
from src.research.statistics import (
    cross_sectional_rank_ic, calculate_hac_tstat, calculate_confidence_interval, residualize_cross_section
)
from src.research.metrics import summarize_ic_series
from src.research.robustness import leave_one_stock_out

def run():
    print("--- Running Experiment 004A: Quality Audit ---")
    raw_prices = load_csv("data/raw/prices_30.csv")
    raw_fund = load_csv("data/raw/fundamentals_30.csv")
    raw_filings = load_csv("data/raw/filings_dates.csv")
    
    prices_df = process_prices(raw_prices)
    mom_df = compute_momentum_features(prices_df)
    trading_days = pd.DatetimeIndex(prices_df['date'].unique()).sort_values()
    
    ttm_fund = compute_ttm_fundamentals(raw_fund)
    earn_df = compute_earnings_features(ttm_fund)
    qual_df = compute_quality_features(earn_df)
    
    pit_fund = build_pit_fundamental_timeline(qual_df, raw_filings, trading_days)
    
    grid = mom_df[['date', 'ticker', 'forward_20d', 'sector_relative_mom_60d']].drop_duplicates()
    merged = merge_pit_features_to_grid(grid, pit_fund)
    
    res_dates = generate_rebalance_schedule(trading_days, Config.RESEARCH_START, Config.RESEARCH_END)
    rebal_df = merged[merged['date'].isin(res_dates)].copy()
    
    # Composite Z-scores
    rebal_df['z_mom'] = rebal_df.groupby('date')['sector_relative_mom_60d'].transform(zscore_cross_section)
    rebal_df['z_eps'] = rebal_df.groupby('date')['eps_growth_acceleration'].transform(zscore_cross_section)
    rebal_df['z_roe'] = rebal_df.groupby('date')['roe_ttm'].transform(zscore_cross_section)
    rebal_df['z_fcf'] = rebal_df.groupby('date')['fcf_margin_ttm'].transform(zscore_cross_section)
    
    # Models
    rebal_df['Model_A'] = rebal_df['z_mom']
    rebal_df['Model_B'] = rebal_df['z_eps']
    rebal_df['Model_C'] = rebal_df['z_mom'] + rebal_df['z_eps']
    rebal_df['Model_D1'] = rebal_df['z_mom'] + rebal_df['z_eps'] + rebal_df['z_roe'].fillna(0)
    rebal_df['Model_D2'] = rebal_df['z_mom'] + rebal_df['z_eps'] + rebal_df['z_fcf'].fillna(0)
    
    ic_c = cross_sectional_rank_ic(rebal_df, 'Model_C')
    ic_d1 = cross_sectional_rank_ic(rebal_df, 'Model_D1')
    ic_d2 = cross_sectional_rank_ic(rebal_df, 'Model_D2')
    
    # CRITICAL PAIRED DELTA-IC TEST
    delta_ic_roe = ic_d1 - ic_c
    delta_ic_fcf = ic_d2 - ic_c
    
    roe_hac_t, roe_p = calculate_hac_tstat(delta_ic_roe, max_lag=3)
    roe_ci_low, roe_ci_high = calculate_confidence_interval(delta_ic_roe, max_lag=3)
    
    fcf_hac_t, fcf_p = calculate_hac_tstat(delta_ic_fcf, max_lag=3)
    
    delta_results = pd.DataFrame([
        {
            'comparison': 'Model_D1 (Base+ROE) vs Model_C (Base)',
            'mean_delta_ic': delta_ic_roe.mean(),
            'median_delta_ic': delta_ic_roe.median(),
            'std_delta_ic': delta_ic_roe.std(),
            'positive_pct': (delta_ic_roe > 0).mean(),
            'positive_count': f"{(delta_ic_roe > 0).sum()} / {len(delta_ic_roe)}",
            'hac_tstat': roe_hac_t,
            'p_value': roe_p,
            'ci_95_low': roe_ci_low,
            'ci_95_high': roe_ci_high
        },
        {
            'comparison': 'Model_D2 (Base+FCF) vs Model_C (Base)',
            'mean_delta_ic': delta_ic_fcf.mean(),
            'median_delta_ic': delta_ic_fcf.median(),
            'std_delta_ic': delta_ic_fcf.std(),
            'positive_pct': (delta_ic_fcf > 0).mean(),
            'positive_count': f"{(delta_ic_fcf > 0).sum()} / {len(delta_ic_fcf)}",
            'hac_tstat': fcf_hac_t,
            'p_value': fcf_p,
            'ci_95_low': np.nan,
            'ci_95_high': np.nan
        }
    ])
    
    save_csv(delta_results, "outputs/tables/exp_004A_paired_delta_ic.csv")
    
    # Residual ROE Test
    rebal_df['roe_res'] = residualize_cross_section(rebal_df, 'z_roe', ['z_mom', 'z_eps'])
    ic_roe_res = cross_sectional_rank_ic(rebal_df, 'roe_res')
    roe_res_t, roe_res_p = calculate_hac_tstat(ic_roe_res, max_lag=3)
    
    res_out = pd.DataFrame([{
        'metric': 'Residual_ROE_IC',
        'mean_ic': ic_roe_res.mean(),
        'hac_tstat': roe_res_t,
        'p_value': roe_res_p
    }])
    save_csv(res_out, "outputs/tables/exp_004A_residual_roe.csv")
    
    # Leave-One-Stock-Out Robustness for Delta IC
    loso_res = []
    for stock in Config.UNIVERSE:
        sub = rebal_df[rebal_df['ticker'] != stock]
        sub_c = cross_sectional_rank_ic(sub, 'Model_C')
        sub_d1 = cross_sectional_rank_ic(sub, 'Model_D1')
        sub_delta = sub_d1 - sub_c
        loso_res.append({
            'dropped_stock': stock,
            'mean_delta_ic': sub_delta.mean(),
            'is_positive': sub_delta.mean() > 0
        })
    save_csv(pd.DataFrame(loso_res), "outputs/tables/exp_004A_loso_delta.csv")
    
    print("Experiment 004A Complete. Outputs saved.")
    return delta_results

if __name__ == "__main__":
    run()