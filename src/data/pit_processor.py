"""
Point-In-Time (PIT) Fundamental Data Alignment Engine.
Strict Accounting Rules:
quarter_end -> actual_filing_date -> first trading day after filing -> factor becomes eligible.
Fallback rule if filing_date missing: quarter_end + 45 days conservative fallback.
"""

import pandas as pd
import numpy as np

def build_pit_fundamental_timeline(fund_ttm: pd.DataFrame, filings: pd.DataFrame, trading_days: pd.DatetimeIndex) -> pd.DataFrame:
    df = pd.merge(fund_ttm, filings, on=['ticker', 'quarter_end'], how='left')
    
    # Conservative fallback flag & logic
    df['is_fallback_date'] = df['filing_date'].isna()
    df['actual_filing_date'] = pd.to_datetime(df['filing_date'])
    
    fallback_date = pd.to_datetime(df['quarter_end']) + pd.Timedelta(days=45)
    df['actual_filing_date'] = df['actual_filing_date'].fillna(fallback_date)
    
    # First trading day AFTER filing
    trading_series = pd.Series(trading_days, index=trading_days)
    
    def get_usable_date(f_date):
        future_days = trading_series[trading_series > f_date]
        if len(future_days) > 0:
            return future_days.iloc[0]
        return np.nan
        
    df['usable_date'] = df['actual_filing_date'].apply(get_usable_date)
    df = df.dropna(subset=['usable_date']).reset_index(drop=True)
    return df

def merge_pit_features_to_grid(grid_df: pd.DataFrame, pit_fund: pd.DataFrame) -> pd.DataFrame:
    """
    As-of point-in-time merge without forward-looking bias.
    """
    sorted_grid = grid_df.sort_values('date').reset_index(drop=True)
    sorted_pit = pit_fund.sort_values('usable_date').reset_index(drop=True)
    
    merged = pd.merge_asof(
        sorted_grid,
        sorted_pit,
        left_on='date',
        right_on='usable_date',
        by='ticker',
        direction='backward'
    )
    return merged