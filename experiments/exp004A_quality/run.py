"""
EXPERIMENT 004A — QUALITY INCREMENTAL SIGNAL AUDIT

Models compared:
  Model A  (Mom only):          Z(sector_relative_mom_60d)
  Model B  (EPS only):          Z(eps_growth_acceleration)
  Model C  (BaseScore):         Z(Mom) + Z(EPS)
  Model D1 (Base + ROE):        Z(Mom) + Z(EPS) + Z(ROE)
  Model D2 (Base + FCF Margin): Z(Mom) + Z(EPS) + Z(FCF)

Primary test: PAIRED DELTA-IC
  Delta_IC_t = IC_{Model D1, t} - IC_{Model C, t}
  → HAC t-statistic on the paired difference series.

Quality factors evaluated standalone:
  ROE, ROA, Operating Margin, Net Margin,
  FCF Margin, OCF Margin, Debt/Equity, Interest Coverage

Financial-sector masking enforced in compute_quality_features().
"""

import pandas as pd
import numpy as np
from config.research_config import Config
from config.universe import COMPETITION_UNIVERSE
from src.utils.io import load_csv, save_csv
from src.data.prices import process_prices
from src.data.fundamentals import compute_ttm_fundamentals
from src.data.pit_processor import build_pit_fundamental_timeline, merge_pit_features_to_grid
from src.features.momentum import compute_momentum_features
from src.features.earnings import compute_earnings_features
from src.features.quality import compute_quality_features
from src.features.common import zscore_cross_section
from src.utils.dates import generate_rebalance_schedule
from src.research.statistics import (
    cross_sectional_rank_ic,
    calculate_hac_tstat,
    calculate_confidence_interval,
    residualize_cross_section,
)
from src.research.metrics import summarize_ic_series


def run() -> pd.DataFrame:
    print("--- Running Experiment 004A: Quality Audit ---")

    raw_prices = load_csv("data/raw/prices_30.csv")
    raw_fund = load_csv("data/raw/fundamentals_30.csv")
    raw_filings = load_csv("data/raw/filings_dates.csv")

    prices_df = process_prices(raw_prices)
    mom_df = compute_momentum_features(prices_df)
    trading_days = pd.DatetimeIndex(prices_df["date"].unique()).sort_values()

    ttm_fund = compute_ttm_fundamentals(raw_fund)
    earn_df = compute_earnings_features(ttm_fund)
    qual_df = compute_quality_features(earn_df)

    pit_fund = build_pit_fundamental_timeline(qual_df, raw_filings, trading_days)

    grid = mom_df[["date", "ticker", "forward_20d", "sector_relative_mom_60d"]].drop_duplicates()
    merged = merge_pit_features_to_grid(grid, pit_fund)

    res_dates = generate_rebalance_schedule(
        trading_days, Config.RESEARCH_START, Config.RESEARCH_END
    )
    rebal_df = merged[merged["date"].isin(res_dates)].copy()

    # --- Composite Z-scores ---
    rebal_df["z_mom"] = rebal_df.groupby("date")["sector_relative_mom_60d"].transform(
        zscore_cross_section
    )
    rebal_df["z_eps"] = rebal_df.groupby("date")["eps_growth_acceleration"].transform(
        zscore_cross_section
    )
    rebal_df["z_roe"] = rebal_df.groupby("date")["roe_ttm"].transform(zscore_cross_section)
    rebal_df["z_fcf"] = rebal_df.groupby("date")["fcf_margin_ttm"].transform(
        zscore_cross_section
    )

    # --- Standalone quality IC ---
    quality_factors = [
        ("roe_ttm", "z_roe"),
        ("roa_ttm", None),
        ("op_margin_ttm", None),
        ("net_margin_ttm", None),
        ("fcf_margin_ttm", "z_fcf"),
        ("ocf_margin_ttm", None),
        ("debt_to_equity", None),
        ("interest_coverage", None),
    ]
    qual_ic_rows = []
    for raw_col, z_col in quality_factors:
        col_to_use = z_col if z_col else raw_col
        if col_to_use not in rebal_df.columns:
            rebal_df[col_to_use] = rebal_df.groupby("date")[raw_col].transform(
                zscore_cross_section
            )
        ic_s = cross_sectional_rank_ic(rebal_df, col_to_use, "forward_20d")
        qual_ic_rows.append({"factor": raw_col, "research_ic": ic_s.mean()})
    save_csv(pd.DataFrame(qual_ic_rows), "outputs/tables/exp_004A_quality_ic.csv")

    # --- Model composites ---
    rebal_df["Model_A"] = rebal_df["z_mom"]
    rebal_df["Model_B"] = rebal_df["z_eps"]
    rebal_df["Model_C"] = rebal_df["z_mom"] + rebal_df["z_eps"]
    # fillna(0) for ROE/FCF: when a stock has no usable quality data, exclude
    # its quality contribution rather than dropping the row entirely.
    rebal_df["Model_D1"] = (
        rebal_df["z_mom"] + rebal_df["z_eps"] + rebal_df["z_roe"].fillna(0)
    )
    rebal_df["Model_D2"] = (
        rebal_df["z_mom"] + rebal_df["z_eps"] + rebal_df["z_fcf"].fillna(0)
    )

    ic_c = cross_sectional_rank_ic(rebal_df, "Model_C")
    ic_d1 = cross_sectional_rank_ic(rebal_df, "Model_D1")
    ic_d2 = cross_sectional_rank_ic(rebal_df, "Model_D2")

    model_summary = []
    for name, ic_s in [("Model_A", cross_sectional_rank_ic(rebal_df, "Model_A")),
                        ("Model_B", cross_sectional_rank_ic(rebal_df, "Model_B")),
                        ("Model_C", ic_c), ("Model_D1", ic_d1), ("Model_D2", ic_d2)]:
        m = summarize_ic_series(ic_s)
        model_summary.append({"model": name, "mean_ic": m["mean_ic"], "hac_t_lag_3": m["hac_t_lag_3"]})
    save_csv(pd.DataFrame(model_summary), "outputs/tables/exp_004A_model_summary.csv")

    # --- CRITICAL: Paired Delta-IC Test ---
    delta_ic_roe = ic_d1 - ic_c
    delta_ic_fcf = ic_d2 - ic_c

    roe_hac_t, roe_p = calculate_hac_tstat(delta_ic_roe, max_lag=3)
    roe_ci_low, roe_ci_high = calculate_confidence_interval(delta_ic_roe, max_lag=3)

    fcf_hac_t, fcf_p = calculate_hac_tstat(delta_ic_fcf, max_lag=3)

    delta_results = pd.DataFrame(
        [
            {
                "comparison": "Model_D1 (Base+ROE) vs Model_C (Base)",
                "mean_delta_ic": delta_ic_roe.mean(),
                "median_delta_ic": delta_ic_roe.median(),
                "std_delta_ic": delta_ic_roe.std(),
                "positive_pct": (delta_ic_roe > 0).mean(),
                "positive_count": f"{(delta_ic_roe > 0).sum()} / {len(delta_ic_roe)}",
                "hac_tstat": roe_hac_t,
                "p_value": roe_p,
                "ci_95_low": roe_ci_low,
                "ci_95_high": roe_ci_high,
            },
            {
                "comparison": "Model_D2 (Base+FCF) vs Model_C (Base)",
                "mean_delta_ic": delta_ic_fcf.mean(),
                "median_delta_ic": delta_ic_fcf.median(),
                "std_delta_ic": delta_ic_fcf.std(),
                "positive_pct": (delta_ic_fcf > 0).mean(),
                "positive_count": f"{(delta_ic_fcf > 0).sum()} / {len(delta_ic_fcf)}",
                "hac_tstat": fcf_hac_t,
                "p_value": fcf_p,
                "ci_95_low": np.nan,
                "ci_95_high": np.nan,
            },
        ]
    )
    save_csv(delta_results, "outputs/tables/exp_004A_paired_delta_ic.csv")

    # --- Residual ROE Test ---
    rebal_df["roe_res"] = residualize_cross_section(rebal_df, "z_roe", ["z_mom", "z_eps"])
    ic_roe_res = cross_sectional_rank_ic(rebal_df, "roe_res")
    roe_res_t, roe_res_p = calculate_hac_tstat(ic_roe_res, max_lag=3)

    res_out = pd.DataFrame(
        [
            {
                "metric": "Residual_ROE_IC",
                "mean_ic": ic_roe_res.mean(),
                "hac_tstat": roe_res_t,
                "p_value": roe_res_p,
            }
        ]
    )
    save_csv(res_out, "outputs/tables/exp_004A_residual_roe.csv")

    # --- Leave-One-Stock-Out Robustness for Delta-IC ---
    loso_res = []
    for stock in COMPETITION_UNIVERSE:
        sub = rebal_df[rebal_df["ticker"] != stock]
        sub_c = cross_sectional_rank_ic(sub, "Model_C")
        sub_d1 = cross_sectional_rank_ic(sub, "Model_D1")
        sub_delta = sub_d1 - sub_c
        loso_res.append(
            {
                "dropped_stock": stock,
                "mean_delta_ic": sub_delta.mean(),
                "is_positive": sub_delta.mean() > 0,
            }
        )
    save_csv(pd.DataFrame(loso_res), "outputs/tables/exp_004A_loso_delta.csv")

    print("Experiment 004A Complete. Outputs saved.")
    return delta_results


if __name__ == "__main__":
    run()
