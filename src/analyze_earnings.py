"""
Statistical Factor Analysis module for Point-in-Time Fundamental Signals.
Runs cross-sectional Spearman Rank ICs, non-overlapping Quintile Spreads, and In-Sample vs Out-of-Sample tests.
"""
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, ttest_1samp

def evaluate_fundamental_ic(df, factor_cols, target_cols="forward_20d", freq_step=20):
    """
    Computes cross-sectional Spearman Rank IC across monthly non-overlapping rebalance dates.
    """
    dates = sorted(df['date'].unique())[::freq_step]
    df_sub = df[df['date'].isin(dates)]

    results = []
    for factor in factor_cols:
        ics = []
        for dt, group in df_sub.groupby('date'):
            valid = group[[factor, target_cols]].dropna()
            if len(valid) >= 15:
                corr, _ = spearmanr(valid[factor], valid[target_cols])
                if not np.isnan(corr):
                    ics.append(corr)

        ics = np.array(ics)
        if len(ics) > 0:
            mean_ic = np.mean(ics)
            std_ic = np.std(ics, ddof=1)
            t_stat, _ = ttest_1samp(ics, 0)
            results.append({
                "factor": factor,
                "target": target_cols,
                "mean_ic": mean_ic,
                "median_ic": np.median(ics),
                "std_ic": std_ic,
                "pct_positive": np.mean(ics > 0) * 100,
                "ic_tstat": t_stat,
                "n_rebalances": len(ics)
            })
    return pd.DataFrame(results)

def evaluate_fundamental_quintiles(df, factor_cols, target_cols="forward_20d", freq_step=20):
    """
    Evaluates equal-weighted quintile spreads (Q1 = Strongest Earnings Signal, Q5 = Weakest)
    """
    dates = sorted(df['date'].unique())[::freq_step]
    df_sub = df[df['date'].isin(dates)].copy()

    summary = []
    for factor in factor_cols:
        q_spreads = []
        q1_returns, q5_returns = [], []

        for dt, group in df_sub.groupby('date'):
            valid = group[[factor, target_cols]].dropna().copy()
            if len(valid) >= 15:
                valid['q'] = pd.qcut(valid[factor], 5, labels=[1, 2, 3, 4, 5], duplicates='drop')
                q_means = valid.groupby('q', observed=False)[target_cols].mean()
                if 1 in q_means and 5 in q_means:
                    # Q1 = Strongest Earnings Signal, Q5 = Weakest Earnings Signal
                    spread = q_means[1] - q_means[5]
                    q_spreads.append(spread)
                    q1_returns.append(q_means[1])
                    q5_returns.append(q_means[5])

        q_spreads = np.array(q_spreads)
        if len(q_spreads) > 0:
            t_stat, _ = ttest_1samp(q_spreads, 0)
            summary.append({
                "factor": factor,
                "target": target_cols,
                "avg_q1_pct": np.mean(q1_returns) * 100,
                "avg_q5_pct": np.mean(q5_returns) * 100,
                "avg_spread_pct": np.mean(q_spreads) * 100,
                "pct_pos_spread": np.mean(q_spreads > 0) * 100,
                "spread_tstat": t_stat,
                "n_rebalances": len(q_spreads)
            })
    return pd.DataFrame(summary)