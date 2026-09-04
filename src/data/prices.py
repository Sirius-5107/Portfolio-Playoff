"""
Price ingestion and forward return computation.
Conventions: Price convention uses total-return adjusted close prices to handle dividends and corporate actions.
ForwardReturn_{t, h} = P_{t+h} / P_{t} - 1
"""

import pandas as pd
import numpy as np
from config.universe import UNIVERSE

def process_prices(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['ticker'].isin(UNIVERSE)].sort_values(['ticker', 'date']).reset_index(drop=True)
    
    # Calculate forward returns per ticker
    horizons = [5, 10, 20, 30]
    for h in horizons:
        df[f'forward_{h}d'] = df.groupby('ticker')['adj_close'].shift(-h) / df['adj_close'] - 1.0
        
    return df