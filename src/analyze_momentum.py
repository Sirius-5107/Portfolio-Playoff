"""
Descriptive quantitative analysis module: ICs, Quintile Spreads, Non-overlapping Rebalances.
"""
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, ttest_1samp

def calculate_ic_metrics(df, factor_cols, target_cols, freq_step=1):
    """
    Computes cross-sectional Spearman Rank ICs per rebalance date.
    """
    dates = sorted(df['date'].unique())[::freq_step]
    df_sub = df[df['date'].isin(dates)]

    results = []
    for factor in factor_cols:
        for target in target_cols:
            ics = []
            for dt, group in df_sub.groupby('date'):
                valid = group[[factor, target]].dropna()
                if len(valid) >= 10:
                    corr, _ = spearmanr(valid[factor], valid[target])
                    if not np.isnan(corr):
                        ics.append(corr)

            ics = np.array(ics)
            if len(ics) > 0:
                mean_ic = np.mean(ics)
                std_ic = np.std(ics, ddof=1)
                t_stat, _ = ttest_1samp(ics, 0)
                results.append({
                    "factor": factor,
                    "target": target,
                    "mean_ic": mean_ic,
                    "median_ic": np.median(ics),
                    "std_ic": std_ic,
                    "pct_positive": np.mean(ics > 0) * 100,
                    "ic_tstat": t_stat,
                    "n_obs": len(ics)
                })
    return pd.DataFrame(results)

def calculate_quintile_performance(df, factor_cols, target_cols, freq_step=1):
    """
    Computes equal-weighted quintile returns and Q1-Q5 spreads across non-overlapping dates.
    """
    dates = sorted(df['date'].unique())[::freq_step]
    df_sub = df[df['date'].isin(dates)].copy()

    summary = []
    for factor in factor_cols:
        for target in target_cols:
            q_spreads = []
            q1_returns, q5_returns = [], []

            for dt, group in df_sub.groupby('date'):
                valid = group[[factor, target]].dropna().copy()
                if len(valid) >= 15:
                    valid['q'] = pd.qcut(valid[factor], 5, labels=[1, 2, 3, 4, 5], duplicates='drop')
                    q_means = valid.groupby('q', observed=False)[target].mean()
                    if 1 in q_means and 5 in q_means:
                        # Q1 = Top momentum, Q5 = Bottom momentum
                        spread = q_means[5] - q_means[1]
                        q_spreads.append(spread)
                        q1_returns.append(q_means[5])
                        q5_returns.append(q_means[1])

            q_spreads = np.array(q_spreads)
            if len(q_spreads) > 0:
                t_stat, _ = ttest_1samp(q_spreads, 0)
                summary.append({
                    "factor": factor,
                    "target": target,
                    "avg_q1": np.mean(q1_returns) * 100,
                    "avg_q5": np.mean(q5_returns) * 100,
                    "avg_spread_pct": np.mean(q_spreads) * 100,
                    "median_spread_pct": np.median(q_spreads) * 100,
                    "pct_pos_spread": np.mean(q_spreads > 0) * 100,
                    "vol_spread_pct": np.std(q_spreads, ddof=1) * 100,
                    "t_stat": t_stat,
                    "n_rebalances": len(q_spreads)
                })
    return pd.DataFrame(summary)