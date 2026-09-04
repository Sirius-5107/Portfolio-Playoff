"""
Portfolio Sort Utilities: Quintile Spreads and 2x2 Double Sorts.
"""

import pandas as pd
import numpy as np

def calculate_quintile_returns(df: pd.DataFrame, factor_col: str, target_col: str = 'forward_20d') -> pd.DataFrame:
    def _sort(group):
        valid = group.dropna(subset=[factor_col, target_col]).copy()
        if len(valid) < 5:
            return pd.Series({f'Q{i}': np.nan for i in range(1, 6)})
        
        valid['quintile'] = pd.qcut(valid[factor_col], 5, labels=[1, 2, 3, 4, 5], duplicates='drop')
        returns = valid.groupby('quintile')[target_col].mean()
        return pd.Series({f'Q{i}': returns.get(i, np.nan) for i in range(1, 6)})
        
    return df.groupby('date').apply(_sort)

def double_sort_2x2(df: pd.DataFrame, factor1: str, factor2: str, target_col: str = 'forward_20d') -> pd.DataFrame:
    def _sort2x2(group):
        valid = group.dropna(subset=[factor1, factor2, target_col]).copy()
        if len(valid) < 4:
            return pd.Series({'Low_Low': np.nan, 'Low_High': np.nan, 'High_Low': np.nan, 'High_High': np.nan})
        
        m1 = valid[factor1].median()
        m2 = valid[factor2].median()
        
        ll = valid[(valid[factor1] <= m1) & (valid[factor2] <= m2)][target_col].mean()
        lh = valid[(valid[factor1] <= m1) & (valid[factor2] > m2)][target_col].mean()
        hl = valid[(valid[factor1] > m1) & (valid[factor2] <= m2)][target_col].mean()
        hh = valid[(valid[factor1] > m1) & (valid[factor2] > m2)][target_col].mean()
        
        return pd.Series({'Low_Low': ll, 'Low_High': lh, 'High_Low': hl, 'High_High': hh})
        
    return df.groupby('date').apply(_sort2x2)