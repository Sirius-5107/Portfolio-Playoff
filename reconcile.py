"""
Automated Reconciliation Engine.

Compares actual experiment output metrics against expected values stored in
config/locked_results.yaml.  Each check allows a ±tolerance window so that
minor floating-point drift does not produce spurious failures.

Exit code 0 = all checks passed.
Exit code 1 = at least one check failed.
"""

import sys
import yaml
import pandas as pd
import numpy as np
from src.utils.io import load_csv, get_path, save_csv


def main() -> None:
    locked_path = get_path("config/locked_results.yaml")
    with open(locked_path, "r") as f:
        locked = yaml.safe_load(f)

    e001_df = load_csv("outputs/tables/exp_001_ic_summary.csv")
    e002_df = load_csv("outputs/tables/exp_002_ic_summary.csv")
    e003_df = load_csv("outputs/tables/exp_003_summary.csv")
    e004_df = load_csv("outputs/tables/exp_004A_paired_delta_ic.csv")

    reconciliation = []

    def _check(experiment: str, metric: str, expected_cfg: dict, actual: float) -> None:
        exp = expected_cfg["expected"]
        tol = expected_cfg["tolerance"]
        diff = actual - exp
        passed = abs(diff) <= tol
        reconciliation.append(
            {
                "experiment": experiment,
                "metric": metric,
                "expected": exp,
                "actual": actual,
                "difference": diff,
                "tolerance": tol,
                "status": "PASS" if passed else "INVESTIGATE",
            }
        )

    # --- Experiment 001 ---
    for factor_key, df_factor in [
        ("sector_relative_60d_ic", "sector_relative_mom_60d"),
        ("mom_60d_ic", "mom_60d"),
        ("mom_12_1_ic", "mom_12_1"),
    ]:
        if factor_key in locked.get("experiment_001", {}):
            row = e001_df[e001_df["factor"] == df_factor]
            if len(row) > 0:
                _check("001", factor_key, locked["experiment_001"][factor_key], row.iloc[0]["research_ic"])

    # --- Experiment 002 ---
    for factor_key, df_factor in [
        ("eps_accel_ic", "eps_growth_acceleration"),
        ("eps_growth_ic", "eps_growth"),
        ("profit_accel_ic", "profit_growth_acceleration"),
    ]:
        if factor_key in locked.get("experiment_002", {}):
            row = e002_df[e002_df["factor"] == df_factor]
            if len(row) > 0:
                _check("002", factor_key, locked["experiment_002"][factor_key], row.iloc[0]["research_ic"])

    # --- Experiment 003 ---
    if "basescore_ic" in locked.get("experiment_003", {}):
        base_ic_rows = e003_df[e003_df["metric"] == "BaseScore_IC_Research"]
        if len(base_ic_rows) > 0:
            _check("003", "basescore_ic", locked["experiment_003"]["basescore_ic"], base_ic_rows.iloc[0]["value"])

    # --- Experiment 004A ---
    if "roe_delta_ic" in locked.get("experiment_004A", {}):
        roe_rows = e004_df[e004_df["comparison"].str.contains("ROE", na=False)]
        if len(roe_rows) > 0:
            _check("004A", "roe_delta_ic", locked["experiment_004A"]["roe_delta_ic"], roe_rows.iloc[0]["mean_delta_ic"])

    rec_df = pd.DataFrame(reconciliation)
    save_csv(rec_df, "outputs/reports/reconciliation_report.csv")

    print("\n" + "=" * 70)
    print("RECONCILIATION REPORT")
    print("=" * 70)
    print(rec_df.to_string(index=False))

    fails = (rec_df["status"] == "INVESTIGATE").any() if len(rec_df) > 0 else False
    if fails:
        print(
            "\nRECONCILIATION: Some metrics outside expected tolerance."
            "\nInvestigate whether data source, date range, or methodology"
            "\nhas changed relative to the locked results. DO NOT simply"
            "\nmanipulate code until numbers match."
        )
        sys.exit(1)
    else:
        print("\nRECONCILIATION: All checked metrics are within locked tolerances.")


if __name__ == "__main__":
    main()
