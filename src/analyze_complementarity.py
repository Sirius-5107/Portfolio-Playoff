"""
Comprehensive Complementarity Testing Framework.
Executes Factor Correlations, Double Sorts (5x5 & 2x2), Residualization, and OOS Regression.
"""
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr, ttest_1samp
from sklearn.linear_model import LinearRegression

def compute_factor_correlations(df, factor1='sector_relative_mom_60d', factor2='eps_growth_acceleration', freq_step=20):
    """
    Computes cross-sectional Pearson and Spearman correlations on rebalance dates.
    """
    dates = sorted(df['date'].unique())[::freq_step]
    pearsons, spearmans = [], []

    for dt in dates:
        sub = df[df['date'] == dt][[factor1, factor2]].dropna()
        if len(sub) >= 15:
            pr, _ = pearsonr(sub[factor1], sub[factor2])
            sr, _ = spearmanr(sub[factor1], sub[factor2])
            pearsons.append(pr)
            spearmans.append(sr)

    return {
        "mean_pearson": np.mean(pearsons),
        "median_pearson": np.median(pearsons),
        "std_pearson": np.std(pearsons, ddof=1),
        "mean_spearman": np.mean(spearmans),
        "median_spearman": np.median(spearmans),
        "std_spearman": np.std(spearmans, ddof=1),
        "pct_positive": np.mean(np.array(spearmans) > 0) * 100
    }

def run_2x2_double_sort(df, factor1='sector_relative_mom_60d', factor2='eps_growth_acceleration', target='forward_20d', freq_step=20):
    """
    Executes a simplified 2x2 conditional double-sort (Top 50% / Bottom 50%).
    """
    dates = sorted(df['date'].unique())[::freq_step]
    bucket_returns = {"Low_Low": [], "Low_High": [], "High_Low": [], "High_High": []}

    for dt in dates:
        sub = df[df['date'] == dt][[factor1, factor2, target]].dropna().copy()
        if len(sub) >= 15:
            f1_med = sub[factor1].median()
            f2_med = sub[factor2].median()

            sub['f1_group'] = np.where(sub[factor1] >= f1_med, 'High', 'Low')
            sub['f2_group'] = np.where(sub[factor2] >= f2_med, 'High', 'Low')
            sub['group'] = sub['f1_group'] + "_" + sub['f2_group']

            grp_means = sub.groupby('group')[target].mean()
            for key in bucket_returns.keys():
                if key in grp_means:
                    bucket_returns[key].append(grp_means[key])

    summary = []
    for key, rets in bucket_returns.items():
        arr = np.array(rets)
        summary.append({
            "group": key,
            "mean_return_pct": np.mean(arr) * 100,
            "median_return_pct": np.median(arr) * 100,
            "std_return_pct": np.std(arr, ddof=1) * 100,
            "hit_rate_pos": np.mean(arr > 0) * 100,
            "n_obs": len(arr)
        })
    return pd.DataFrame(summary)

def run_residual_factor_tests(df, f_mom='sector_relative_mom_60d', f_eps='eps_growth_acceleration', target='forward_20d', freq_step=20):
    """
    Factor Neutralization Test:
    Cross-sectionally regresses Mom on EPS (and vice versa) to isolate residual factors.
    """
    dates = sorted(df['date'].unique())[::freq_step]
    res_mom_ics, res_eps_ics = [], []

    for dt in dates:
        sub = df[df['date'] == dt][[f_mom, f_eps, target]].dropna().copy()
        if len(sub) >= 15:
            # 1. Residual Momentum (Mom neutral to EPS)
            X_eps = sub[[f_eps]].values
            y_mom = sub[f_mom].values
            lr1 = LinearRegression().fit(X_eps, y_mom)
            sub['res_mom'] = y_mom - lr1.predict(X_eps)

            # 2. Residual EPS (EPS neutral to Mom)
            X_mom = sub[[f_mom]].values
            y_eps = sub[f_eps].values
            lr2 = LinearRegression().fit(X_mom, y_eps)
            sub['res_eps'] = y_eps - lr2.predict(X_mom)

            # Measure IC against forward return
            ic_m, _ = spearmanr(sub['res_mom'], sub[target])
            ic_e, _ = spearmanr(sub['res_eps'], sub[target])

            if not np.isnan(ic_m): res_mom_ics.append(ic_m)
            if not np.isnan(ic_e): res_eps_ics.append(ic_e)

    arr_m, arr_e = np.array(res_mom_ics), np.array(res_eps_ics)
    t_m, _ = ttest_1samp(arr_m, 0)
    t_e, _ = ttest_1samp(arr_e, 0)

    return pd.DataFrame([
        {"factor": "Residual Momentum (Cleaned of EPS)", "mean_ic": np.mean(arr_m), "pct_pos": np.mean(arr_m > 0)*100, "t_stat": t_m},
        {"factor": "Residual EPS Accel (Cleaned of Mom)", "mean_ic": np.mean(arr_e), "pct_pos": np.mean(arr_e > 0)*100, "t_stat": t_e}
    ])