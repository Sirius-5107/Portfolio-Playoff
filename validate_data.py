#!/usr/bin/env python3
"""
Data Validation Script.

Run this BEFORE any experiment to confirm that required data files exist
and contain the expected columns, date ranges, and tickers.

Usage:
    python validate_data.py

Exit code 0 = all checks passed.
Exit code 1 = one or more problems found.
"""

import sys
import pandas as pd
from pathlib import Path
from config.universe import COMPETITION_UNIVERSE

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"

REQUIRED_FILES = {
    "prices_30.csv": {
        "required_columns": ["date", "ticker", "adj_close"],
        "expected_min_rows": 10_000,
    },
    "fundamentals_30.csv": {
        "required_columns": ["ticker", "quarter_end", "net_income", "revenue", "ebit", "eps",
                             "ocf", "capex", "equity", "total_assets", "interest_expense"],
        "expected_min_rows": 500,
    },
    "filings_dates.csv": {
        "required_columns": ["ticker", "quarter_end", "filing_date"],
        "expected_min_rows": 200,
    },
}

PRICE_DATE_RANGE = ("2010-01-04", "2026-09-04")
FUND_DATE_RANGE = ("2009-01-01", "2026-09-04")


def check_file(filename: str, spec: dict) -> list[str]:
    """Return list of error strings (empty = file passes all checks)."""
    path = DATA_RAW / filename
    errors = []

    if not path.exists():
        errors.append(
            f"MISSING: {path}\n"
            f"  → Obtain this file and place it at {path}\n"
            f"  → Required columns: {spec['required_columns']}"
        )
        return errors

    try:
        df = pd.read_csv(path, nrows=5)
        missing_cols = [c for c in spec["required_columns"] if c not in df.columns]
        if missing_cols:
            errors.append(f"MISSING COLUMNS in {filename}: {missing_cols}")
    except Exception as e:
        errors.append(f"UNREADABLE: {filename} → {e}")
        return errors

    # Row count check (approximate)
    try:
        df_full = pd.read_csv(path)
        if len(df_full) < spec["expected_min_rows"]:
            errors.append(
                f"TOO FEW ROWS in {filename}: got {len(df_full)}, "
                f"expected ≥ {spec['expected_min_rows']}"
            )
    except Exception as e:
        errors.append(f"READ ERROR for {filename}: {e}")

    return errors


def check_prices_coverage(filename="prices_30.csv") -> list[str]:
    """Check that prices cover the expected date range and all 30 tickers."""
    errors = []
    path = DATA_RAW / filename
    if not path.exists():
        return []  # already flagged in check_file

    df = pd.read_csv(path, parse_dates=["date"])
    tickers_in_file = set(df["ticker"].unique())
    missing_tickers = set(COMPETITION_UNIVERSE) - tickers_in_file
    if missing_tickers:
        errors.append(f"MISSING TICKERS in {filename}: {sorted(missing_tickers)}")

    start, end = PRICE_DATE_RANGE
    if df["date"].min() > pd.Timestamp(start):
        errors.append(
            f"PRICES start too late: earliest={df['date'].min().date()}, required≤{start}"
        )
    if df["date"].max() < pd.Timestamp("2024-01-01"):
        errors.append(
            f"PRICES end too early: latest={df['date'].max().date()}, required≥2024-01-01"
        )

    return errors


def check_no_duplicate_price_rows(filename="prices_30.csv") -> list[str]:
    errors = []
    path = DATA_RAW / filename
    if not path.exists():
        return []
    df = pd.read_csv(path, parse_dates=["date"])
    dupes = df.duplicated(subset=["date", "ticker"]).sum()
    if dupes:
        errors.append(f"DUPLICATE (date, ticker) rows in {filename}: {dupes} duplicates")
    return errors


def main() -> int:
    print("=" * 60)
    print("DATA VALIDATION")
    print("=" * 60)

    all_errors: list[str] = []

    for filename, spec in REQUIRED_FILES.items():
        errors = check_file(filename, spec)
        all_errors.extend(errors)

    all_errors.extend(check_prices_coverage())
    all_errors.extend(check_no_duplicate_price_rows())

    if all_errors:
        print(f"\nFOUND {len(all_errors)} PROBLEM(S):\n")
        for e in all_errors:
            print(f"  ✗ {e}\n")
        print("Fix the above before running experiments.")
        return 1
    else:
        print("\nAll data checks passed. Ready to run experiments.\n")
        return 0


if __name__ == "__main__":
    sys.exit(main())
