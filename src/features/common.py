import numpy as np
import pandas as pd
from typing import Union

def zscore_cross_section(
    data: Union[pd.DataFrame, pd.Series],
    clip_lower: float = -3.0,
    clip_upper: float = 3.0
) -> Union[pd.DataFrame, pd.Series]:
    """Calculates cross-sectional Z-scores. Works on both DataFrames and Series."""
    if isinstance(data, pd.Series):
        mean = data.mean()
        std = data.std()
        if std == 0 or np.isnan(std):
            z_scores = pd.Series(0.0, index=data.index)
        else:
            z_scores = (data - mean) / std
    else:
        mean = data.mean(axis=1)
        std = data.std(axis=1).replace(0, np.nan)
        z_scores = data.sub(mean, axis=0).div(std, axis=0)

    if clip_lower is not None or clip_upper is not None:
        z_scores = z_scores.clip(lower=clip_lower, upper=clip_upper)

    return z_scores.fillna(0.0)


def winsorize_cross_section(
    df: pd.DataFrame, lower_quantile: float = 0.01, upper_quantile: float = 0.99
) -> pd.DataFrame:
    """Winsorizes cross-sectional feature distributions at specified upper and lower quantiles.
    """
    lower = df.quantile(lower_quantile, axis=1)
    upper = df.quantile(upper_quantile, axis=1)

    return df.clip(lower=lower, upper=upper, axis=0)