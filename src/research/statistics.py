"""
Centralized Statistical Infrastructure.
Includes HAC Newey-West standard errors, Winsorization, Rank IC, and p-value calculations.
"""

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

def cross_sectional_rank_ic(df: pd.DataFrame, factor_col: str, target_col: str = 'forward_20d') -> pd.Series:
    def calc_ic(group):
        valid = group[[factor_col, target_col]].dropna()
        if len(valid) < 5:
            return np.nan
        return stats.spearmanr(valid[factor_col], valid[target_col])[0]
        
    return df.groupby('date').apply(calc_ic)

def calculate_hac_tstat(series: pd.Series, max_lag: int = 3) -> tuple[float, float]:
    clean = series.dropna()
    N = len(clean)
    if N < 5:
        return np.nan, np.nan
    
    Y = clean.values
    X = np.ones(N)
    
    model = sm.OLS(Y, X)
    results = model.fit(cov_type='HAC', cov_kwds={'maxlags': max_lag})
    
    t_stat = results.tvalues[0]
    p_val = results.pvalues[0]
    return float(t_stat), float(p_val)

def calculate_confidence_interval(series: pd.Series, max_lag: int = 3, conf_level: float = 0.95) -> tuple[float, float]:
    clean = series.dropna()
    mean = clean.mean()
    t_stat, _ = calculate_hac_tstat(clean, max_lag=max_lag)
    if np.isna(t_stat) or t_stat == 0:
        return np.nan, np.nan
    se = mean / t_stat
    z = stats.norm.ppf((1 + conf_level) / 2)
    return mean - z * se, mean + z * se

def winsorize_cross_section(df: pd.DataFrame, factor_col: str, limits: tuple = (0.01, 0.99)) -> pd.DataFrame:
    if limits is None:
        return df.copy()
    
    out = df.copy()
    def _winsor(group):
        lower = group[factor_col].quantile(limits[0])
        upper = group[factor_col].quantile(limits[1])
        group[factor_col] = group[factor_col].clip(lower, upper)
        return group
        
    return out.groupby('date', group_keys=False).apply(_winsor)

import pandas as pd
import numpy as np

def residualize_cross_section(df: pd.DataFrame, target_col: str, regressor_cols: list[str]) -> pd.Series:
    """Residualizes target_col against regressor_cols cross-sectionally by date."""
    
    def _res_group(group):
        y = group[target_col].values
        # Stack 1s for intercept + regressor columns
        X = np.column_stack([np.ones(len(group)), group[regressor_cols].values])
        
        # OLS regression: y = X * beta + residual
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        residuals = y - X @ beta
        
        return pd.Series(residuals, index=group.index)

    # Use group_keys=False to preserve original row index alignment
    residuals_series = df.groupby('date', group_keys=False).apply(_res_group, include_groups=False)
    return residuals_series