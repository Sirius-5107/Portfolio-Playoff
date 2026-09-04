"""
EXPERIMENT 002 — HISTORICAL EARNINGS MOMENTUM

Strict point-in-time treatment:
  quarter_end → actual filing date → first trading day after filing → effective date

Factors evaluated:
  eps_growth               YoY EPS growth
  profit_growth            YoY net income growth
  revenue_growth           YoY revenue growth
  eps_growth_acceleration  1-quarter delta of YoY EPS growth  ← primary signal
  profit_growth_acceleration
  revenue_growth_acceleration
  op_margin_change         YoY operating margin change
  days_since_earnings      Days from filing date to rebalance date
"""

import pandas as pd
import numpy as np
from config.research_config import Config
from src.utils.io import load_csv, save_csv
from src.data.prices import process_prices
from src.data.fundamentals import compute_ttm_fundamentals
from src.data.pit_processor import build_pit_fundamental_timeline, merge_pit_features_to_grid
from src.features.earnings import compute_earnings_features, attach_days_since_earnings
from src.utils.dates import generate_rebalance_schedule
from src.research.statistics import cross_sectional_rank_ic
from src.research.metrics import summarize_ic_series


def run() -> pd.DataFrame:
    print("--- Running Experiment 002: Historical Earnings Momentum ---")

    raw_prices = load_csv("data/raw/prices_30.csv")
    raw_fund = load_csv("data/raw/fundamentals_30.csv")
    raw_filings = load_csv("data/raw/filings_dates.csv")

    prices_df = process_prices(raw_prices)
    trading_days = pd.DatetimeIndex(prices_df["date"].unique()).sort_values()

    ttm_fund = compute_ttm_fundamentals(raw_fund)
    earn_df = compute_earnings_features(ttm_fund)

    pit_fund = build_pit_fundamental_timeline(earn_df, raw_filings, trading_days)

    grid = prices_df[["date", "ticker", "forward_20d"]].drop_duplicates()
    merged = merge_pit_features_to_grid(grid, pit_fund)
    merged = attach_days_since_earnings(merged)

    res_dates = generate_rebalance_schedule(
        trading_days, Config.RESEARCH_START, Config.RESEARCH_END
    )
    val_dates = generate_rebalance_schedule(
        trading_days, Config.VALIDATION_START, Config.VALIDATION_END
    )

    rebal_df = merged[merged["date"].isin(res_dates.union(val_dates))].copy()

    factors = [
        "eps_growth",
        "profit_growth",
        "revenue_growth",
        "eps_growth_acceleration",
        "profit_growth_acceleration",
        "revenue_growth_acceleration",
        "op_margin_change",
        "days_since_earnings",
    ]

    res_mask = rebal_df["date"].isin(res_dates)
    val_mask = rebal_df["date"].isin(val_dates)

    summary = []
    for f in factors:
        if f not in rebal_df.columns:
            print(f"  WARNING: factor '{f}' not in merged DataFrame — skipping.")
            continue
        ic_res = cross_sectional_rank_ic(rebal_df[res_mask], f, "forward_20d")
        ic_val = cross_sectional_rank_ic(rebal_df[val_mask], f, "forward_20d")

        m_res = summarize_ic_series(ic_res)
        m_val = summarize_ic_series(ic_val)

        summary.append(
            {
                "factor": f,
                "research_ic": m_res["mean_ic"],
                "research_hac_t": m_res["hac_t_lag_3"],
                "validation_ic": m_val["mean_ic"],
                "validation_label": Config.VALIDATION_LABEL,
            }
        )

    out_df = pd.DataFrame(summary)
    save_csv(out_df, "outputs/tables/exp_002_ic_summary.csv")
    print("Experiment 002 Complete. Outputs saved.")
    return out_df


if __name__ == "__main__":
    run()
