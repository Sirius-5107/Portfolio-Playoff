"""
Point-in-Time safe historical data loader with caching and reporting.
"""
import pandas as pd
import yfinance as yf
from pathlib import Path
from src.config import TICKERS, START_DATE, END_DATE, RAW_DATA_DIR

def fetch_and_cache_data():
    report_rows = []
    prices_list = []

    print(f"Fetching OHLCV data for {len(TICKERS)} tickers...")

    for ticker in TICKERS:
        file_path = RAW_DATA_DIR / f"{ticker}.csv"
        
        if file_path.exists():
            df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
        else:
            try:
                data = yf.Ticker(ticker)
                df = data.history(start=START_DATE, end=END_DATE, auto_adjust=True)
                if not df.empty:
                    df.to_csv(file_path)
            except Exception as e:
                print(f"Error downloading {ticker}: {e}")
                df = pd.DataFrame()

        if df.empty:
            report_rows.append({
                "ticker": ticker, "first_date": None, "last_date": None,
                "num_obs": 0, "missing_obs": 0, "pct_missing": 100.0, "status": "Failed"
            })
            continue

        df = df[~df.index.duplicated(keep='first')].sort_index()
        df['Ticker'] = ticker
        prices_list.append(df[['Close', 'Volume', 'Ticker']])

        full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq='B')
        missing_count = len(full_idx.difference(df.index))
        total_expected = len(df) + missing_count
        pct_missing = (missing_count / total_expected) * 100 if total_expected > 0 else 0.0

        report_rows.append({
            "ticker": ticker,
            "first_date": df.index.min().strftime('%Y-%m-%d'),
            "last_date": df.index.max().strftime('%Y-%m-%d'),
            "num_obs": len(df),
            "missing_obs": missing_count,
            "pct_missing": round(pct_missing, 2),
            "status": "Success"
        })

    report_df = pd.DataFrame(report_rows)
    all_prices = pd.concat(prices_list) if prices_list else pd.DataFrame()
    return all_prices, report_df

if __name__ == "__main__":
    prices, report = fetch_and_cache_data()
    print("\n--- DATA QUALITY REPORT ---")
    print(report.to_string())