"""
Centralized Statistical Infrastructure.

All reusable statistical functions live here.  Do not duplicate IC, HAC,
residualization or winsorization logic in experiment scripts.

Functions
---------
cross_sectional_rank_ic     Spearman rank IC per rebalance date
calculate_hac_tstat         Newey-West HAC t-statistic and p-value
calculate_confidence_interval  95% CI using HAC standard error
winsorize_cross_section     Per-date cross-sectional winsorization
residualize_cross_section   OLS residualization per date (returns flat Series)
"""

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


# ---------------------------------------------------------------------------
# IC calculations
# ---------------------------------------------------------------------------

def cross_sectional_rank_ic(
    df: pd.DataFrame,
    factor_col: str,
    target_col: str = "forward_20d",
    min_stocks: int = 5,
) -> pd.Series:
    """Compute Spearman rank IC per rebalance date.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain: date, ``factor_col``, ``target_col``.
    factor_col : str
        Name of the signal column.
    target_col : str
        Name of the forward-return column.
    min_stocks : int
        Minimum number of valid stock-date pairs to compute IC.
        Dates with fewer return NaN.

    Returns
    -------
    pd.Series indexed by date.
    """
    def _ic(group):
        valid = group[[factor_col, target_col]].dropna()
        if len(valid) < min_stocks:
            return np.nan
        return stats.spearmanr(valid[factor_col], valid[target_col])[0]

    return df.groupby("date").apply(_ic, include_groups=False)


# ---------------------------------------------------------------------------
# HAC / Newey-West
# ---------------------------------------------------------------------------

def calculate_hac_tstat(
    series: pd.Series,
    max_lag: int = 3,
    min_obs: int = 5,
) -> tuple:
    """Compute HAC (Newey-West) t-statistic for H0: mean = 0.

    Parameters
    ----------
    series : pd.Series
        Time series of IC observations.
    max_lag : int
        Maximum lag for Newey-West kernel.
    min_obs : int
        Minimum observations required.

    Returns
    -------
    (t_stat, p_value) — both NaN if insufficient data.
    """
    clean = series.dropna()
    if len(clean) < min_obs:
        return np.nan, np.nan

    Y = clean.values
    X = np.ones(len(Y))

    model = sm.OLS(Y, X)
    results = model.fit(cov_type="HAC", cov_kwds={"maxlags": max_lag})

    return float(results.tvalues[0]), float(results.pvalues[0])


def calculate_confidence_interval(
    series: pd.Series,
    max_lag: int = 3,
    conf_level: float = 0.95,
) -> tuple:
    """Compute confidence interval using HAC standard error.

    Returns
    -------
    (lower, upper) — both NaN if HAC fails.
    """
    clean = series.dropna()
    mean = clean.mean()
    t_stat, _ = calculate_hac_tstat(clean, max_lag=max_lag)
    if np.isnan(t_stat) or t_stat == 0:
        return np.nan, np.nan
    se = mean / t_stat
    z = stats.norm.ppf((1 + conf_level) / 2)
    return mean - z * se, mean + z * se


# ---------------------------------------------------------------------------
# Winsorization
# ---------------------------------------------------------------------------

def winsorize_cross_section(
    df: pd.DataFrame,
    factor_col: str,
    limits: tuple = (0.01, 0.99),
) -> pd.DataFrame:
    """Winsorize ``factor_col`` cross-sectionally at each date.

    Parameters
    ----------
    limits : tuple or None
        (lower_quantile, upper_quantile).  Pass None to skip winsorization.

    Returns
    -------
    Copy of ``df`` with ``factor_col`` clipped.
    """
    if limits is None:
        return df.copy()

    out = df.copy()

    def _winsor(group):
        lower = group[factor_col].quantile(limits[0])
        upper = group[factor_col].quantile(limits[1])
        group = group.copy()
        group[factor_col] = group[factor_col].clip(lower, upper)
        return group

    return out.groupby("date", group_keys=False).apply(_winsor)


# ---------------------------------------------------------------------------
# Residualization
# ---------------------------------------------------------------------------

def residualize_cross_section(
    df: pd.DataFrame,
    target_col: str,
    regressor_cols: list,
) -> pd.Series:
    """Residualize ``target_col`` against ``regressor_cols`` cross-sectionally.

    For each date, fits OLS:
        target_col = intercept + sum(beta_i * regressor_cols_i) + residual

    Returns the concatenated residuals as a flat pd.Series aligned to df's
    original index.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain: date, ``target_col``, and all columns in ``regressor_cols``.
    target_col : str
        Column to residualize.
    regressor_cols : list[str]
        Columns to regress out.

    Returns
    -------
    pd.Series with the same index as ``df``.
    """
    pieces = []

    for _, group in df.groupby("date"):
        y = group[target_col].values.astype(float)
        X = np.column_stack(
            [np.ones(len(group)), group[regressor_cols].values.astype(float)]
        )

        # Handle NaNs: only compute residuals for complete rows
        complete = ~(np.isnan(y) | np.isnan(X).any(axis=1))
        residuals = np.full(len(y), np.nan)

        if complete.sum() > len(regressor_cols) + 1:
            beta, _, _, _ = np.linalg.lstsq(X[complete], y[complete], rcond=None)
            residuals[complete] = y[complete] - X[complete] @ beta

        pieces.append(pd.Series(residuals, index=group.index))

    if not pieces:
        return pd.Series(dtype=float)

    return pd.concat(pieces).sort_index()
