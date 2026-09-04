"""
Momentum Feature Computation Engine.
Sector-Relative 60D:
SectorRelMom_{i,t} = Mom_{i,t} - Mean(Mom_{sector,t})
Z-scoring is applied AFTER sector demeaning.
"""

import pandas as pd
import numpy as np
from config.sectors import SECTOR_MAP
from .common import zscore_cross_section

def compute_momentum_features(prices_df: pd.DataFrame) -> pd.DataFrame:
    df = prices_df.copy().sort_values(['ticker', 'date']).reset_index(drop=True)
    
    # Raw returns over lookbacks
    df['mom_5d'] = df.groupby('ticker')['adj_close'].pct_change(5)
    df['mom_20d'] = df.groupby('ticker')['adj_close'].pct_change(20)
    df['mom_60d'] = df.groupby('ticker')['adj_close'].pct_change(60)
    df['mom_252d'] = df.groupby('ticker')['adj_close'].pct_change(252)
    
    # 12-1 Momentum (252d lagged by 21d)
    p = df['adj_close']
    p_lag21 = df.groupby('ticker')['adj_close'].shift(21)
    p_lag252 = df.groupby('ticker')['adj_close'].shift(252)
    df['mom_12_1'] = (p_lag21 / p_lag252) - 1.0
    
    # Relative Momentum vs Universe Mean
    unv_mean_60 = df.groupby('date')['mom_60d'].transform('mean')
    df['relative_momentum'] = df['mom_60d'] - unv_mean_60
    
    # Sector relative momentum
    df['sector'] = df['ticker'].map(SECTOR_MAP)
    
    for lookback in [60, 252]:
        sec_mean = df.groupby(['date', 'sector'])[f'mom_{lookback}d'].transform('mean')
        df[f'sector_rel_{lookback}d_raw'] = df[f'mom_{lookback}d'] - sec_mean
        
        # Cross-sectional Z-score applied strictly AFTER demeaning
        df[f'sector_relative_mom_{lookback}d'] = df.groupby('date')[f'sector_rel_{lookback}d_raw'].transform(zscore_cross_section)
        
    return df