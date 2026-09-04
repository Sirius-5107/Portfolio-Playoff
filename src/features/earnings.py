"""
Earnings Momentum Features.
EPS Acceleration strictly follows locked 1-quarter delta:
EPS_Accel_t = Growth_t - Growth_{t-1}
"""

import pandas as pd
import numpy as np

def compute_earnings_features(pit_fund_df: pd.DataFrame) -> pd.DataFrame:
    df = pit_fund_df.copy().sort_values(['ticker', 'quarter_end']).reset_index(drop=True)
    
    # YoY Level Growth Metrics
    df['eps_growth'] = df.groupby('ticker')['eps'].pct_change(4)
    df['profit_growth'] = df.groupby('ticker')['net_income'].pct_change(4)
    df['revenue_growth'] = df.groupby('ticker')['revenue'].pct_change(4)
    
    # Acceleration Metrics (1-quarter delta of YoY growth)
    df['eps_growth_acceleration'] = df.groupby('ticker')['eps_growth'].diff(1)
    df['profit_growth_acceleration'] = df.groupby('ticker')['profit_growth'].diff(1)
    df['revenue_growth_acceleration'] = df.groupby('ticker')['revenue_growth'].diff(1)
    
    # Operating Margin Change
    df['op_margin'] = df['ebit'] / df['revenue']
    df['op_margin_change'] = df.groupby('ticker')['op_margin'].diff(4)
    
    return df