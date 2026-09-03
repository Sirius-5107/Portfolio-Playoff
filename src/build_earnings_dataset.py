"""
Point-in-Time safe fundamental panel builder for Indian corporate filings.
Maps quarterly financial releases exclusively on or after official board result publication dates.
"""
import pandas as pd
import numpy as np
from src.config import SECTOR_MAP, FORWARD_HORIZONS, PROCESSED_DATA_DIR, RAW_DATA_DIR

def load_pit_fundamentals():
    """
    Loads raw quarterly statements alongside official announcement/filing dates.
    Strictly enforces zero look-ahead availability.
    """
    raw_path = RAW_DATA_DIR / "quarterly_earnings_pit.csv"
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Point-In-Time quarterly statements dataset missing at {raw_path}. "
            "Ensure actual NSE/BSE filing announcement timestamps are loaded."
        )
    
    df_raw = pd.read_csv(raw_path, parse_dates=['fiscal_quarter_end', 'announcement_date'])
    
    # Fundamental audit check: Announcement Date MUST be > Fiscal Quarter End Date
    invalid = df_raw[df_raw['announcement_date'] <= df_raw['fiscal_quarter_end']]
    if not invalid.empty:
        raise ValueError(f"Look-ahead violation detected in fundamental input! {len(invalid)} records fail PIT rules.")
        
    return df_raw

def construct_point_in_time_panel():
    fund_df = load_pit_fundamentals()
    
    # Calculate fundamental growth & acceleration metrics at earnings event level
    fund_df = fund_df.sort_values(['ticker', 'fiscal_quarter_end']).reset_index(drop=True)
    
    # YoY Quarterly Growth
    fund_df['revenue_growth_yoy'] = fund_df.groupby('ticker')['revenue'].pct_change(4)
    fund_df['profit_growth_yoy'] = fund_df.groupby('ticker')['net_profit'].pct_change(4)
    fund_df['eps_growth_yoy'] = fund_df.groupby('ticker')['eps'].pct_change(4)
    fund_df['op_margin'] = fund_df['operating_profit'] / fund_df['revenue']
    fund_df['operating_margin_change'] = fund_df.groupby('ticker')['op_margin'].diff(4)
    
    # Growth Acceleration (Delta in YoY Growth Rate vs Previous Quarter)
    fund_df['revenue_growth_acceleration'] = fund_df.groupby('ticker')['revenue_growth_yoy'].diff(1)
    fund_df['profit_growth_acceleration'] = fund_df.groupby('ticker')['profit_growth_yoy'].diff(1)
    fund_df['eps_growth_acceleration'] = fund_df.groupby('ticker')['eps_growth_yoy'].diff(1)

    # Load daily price grid for alignment
    prices_path = PROCESSED_DATA_DIR / "momentum_dataset.parquet"
    price_panel = pd.read_parquet(prices_path)[['date', 'ticker', 'price', 'forward_5d', 'forward_10d', 'forward_20d', 'forward_30d']].drop_duplicates()
    
    # Step-function expansion: Map fundamental disclosures to calendar days
    # Realized earnings metrics are expanded forward from announcement_date until replaced by next announcement
    merged_rows = []
    
    for ticker, p_group in price_panel.groupby('ticker'):
        p_group = p_group.sort_values('date').copy()
        f_group = fund_df[fund_df['ticker'] == ticker].sort_values('announcement_date').copy()
        
        # Merge asof: match every daily price date with the latest available announcement_date <= price_date
        merged = pd.merge_asof(
            p_group,
            f_group,
            left_on='date',
            right_on='announcement_date',
            by='ticker',
            direction='backward'
        )
        
        # Calculate days since last earnings announcement
        merged['days_since_earnings'] = (merged['date'] - merged['announcement_date']).dt.days
        merged_rows.append(merged)

    pit_dataset = pd.concat(merged_rows, ignore_index=True)
    pit_dataset['sector'] = pit_dataset['ticker'].map(SECTOR_MAP)
    
    # Filter out dates prior to a company's first public PIT filing record
    pit_dataset = pit_dataset.dropna(subset=['announcement_date']).reset_index(drop=True)
    
    pit_dataset.to_parquet(PROCESSED_DATA_DIR / "earnings_pit_dataset.parquet")
    print(f"Point-In-Time Earnings Dataset created successfully with shape: {pit_dataset.shape}")
    return pit_dataset

if __name__ == "__main__":
    construct_point_in_time_panel()