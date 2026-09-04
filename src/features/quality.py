"""
Quality Factors and Sector Masking Rules.
Financial Sector Constraints:
ROE, ROA -> Included
Operating Margin, Net Margin, FCF Margin, OCF Margin, D/E, Interest Coverage -> EXCLUDED (NaN)
"""

import pandas as pd
import numpy as np
from config.sectors import FINANCIAL_TICKERS


def compute_quality_features(pit_fund_df: pd.DataFrame) -> pd.DataFrame:
    df = pit_fund_df.copy()

    # Balance sheet averages with fallback for short history / unit tests
    equity_lag4 = df.groupby('ticker')['equity'].shift(4)
    assets_lag4 = df.groupby('ticker')['total_assets'].shift(4)

    df['equity_avg'] = (df['equity'] + equity_lag4.fillna(df['equity'])) / 2.0
    df['assets_avg'] = (df['total_assets'] + assets_lag4.fillna(df['total_assets'])) / 2.0

    # ROE and ROA (Valid for ALL sectors including financials)
    equity_safe = df['equity'].replace(0, np.nan)
    assets_avg_safe = df['assets_avg'].replace(0, np.nan)

    df['roe_ttm'] = df['net_income_ttm'] / equity_safe
    df['roa_ttm'] = df['net_income_ttm'] / assets_avg_safe

    # Non-financial operational/leverage metrics
    rev_safe = df['revenue_ttm'].replace(0, np.nan) if 'revenue_ttm' in df else np.nan
    int_exp_safe = df['interest_expense_ttm'].replace(0, np.nan) if 'interest_expense_ttm' in df else np.nan

    df['op_margin_ttm'] = df['ebit_ttm'] / rev_safe if 'ebit_ttm' in df else np.nan
    df['net_margin_ttm'] = df['net_income_ttm'] / rev_safe if 'revenue_ttm' in df else np.nan
    df['fcf_margin_ttm'] = df['fcf_ttm'] / rev_safe if 'fcf_ttm' in df else np.nan
    df['ocf_margin_ttm'] = df['ocf_ttm'] / rev_safe if 'ocf_ttm' in df else np.nan

    if 'short_term_debt' in df and 'long_term_debt' in df:
        df['debt_to_equity'] = (df['short_term_debt'] + df['long_term_debt']) / equity_safe
    else:
        df['debt_to_equity'] = np.nan

    if 'ebit_ttm' in df and 'interest_expense_ttm' in df:
        df['interest_coverage'] = df['ebit_ttm'] / int_exp_safe
    else:
        df['interest_coverage'] = np.nan

    # Enforce Financial Sector Exclusion Masking
    fin_mask = df['ticker'].isin(FINANCIAL_TICKERS)
    excluded_cols = [
        'op_margin_ttm', 'net_margin_ttm', 'fcf_margin_ttm', 
        'ocf_margin_ttm', 'debt_to_equity', 'interest_coverage'
    ]

    for col in excluded_cols:
        if col in df.columns:
            df.loc[fin_mask, col] = np.nan

    return df