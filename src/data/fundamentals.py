"""
Fundamental metric raw processing and TTM aggregations.
"""

import pandas as pd

def compute_ttm_fundamentals(raw_fund: pd.DataFrame) -> pd.DataFrame:
    df = raw_fund.copy()
    df['quarter_end'] = pd.to_datetime(df['quarter_end'])
    df = df.sort_values(['ticker', 'quarter_end']).reset_index(drop=True)
    
    metrics = ['net_income', 'revenue', 'ebit', 'ocf', 'capex', 'interest_expense']
    for m in metrics:
        if m in df.columns:
            df[f'{m}_ttm'] = df.groupby('ticker')[m].transform(lambda x: x.rolling(4, min_periods=4).sum())
            
    df['fcf_ttm'] = df['ocf_ttm'] - df['capex_ttm']
    return df