"""
Combined Point-In-Time Dataset Builder for Experiment 003.
Merges Price Momentum and PIT Fundamental Features seamlessly.
"""
import pandas as pd
import numpy as np
from config import PROCESSED_DATA_DIR, SECTOR_MAP

def create_combined_dataset():
    mom_df = pd.read_parquet(PROCESSED_DATA_DIR / "momentum_dataset.parquet")
    earn_df = pd.read_parquet(PROCESSED_DATA_DIR / "earnings_pit_dataset.parquet")

    # Select required columns from fundamentals
    earn_cols = [
        'date', 'ticker', 'eps_growth_acceleration', 'profit_growth_acceleration',
        'revenue_growth_acceleration', 'eps_growth_yoy', 'profit_growth_yoy', 'days_since_earnings'
    ]
    earn_sub = earn_df[earn_cols].copy()

    # Merge on exact date and ticker
    combined = pd.merge(mom_df, earn_sub, on=['date', 'ticker'], how='inner')
    combined['sector'] = combined['ticker'].map(SECTOR_MAP)

    # Calculate Alternative Acceleration Definitions (Def B & Def C)
    # Def B: Current YoY EPS Growth minus average of previous 2 quarters' YoY Growth
    # Def C: TTM EPS Growth minus previous TTM EPS Growth
    combined = combined.sort_values(['ticker', 'date']).reset_index(drop=True)
    
    # Save processed combined dataset
    combined.to_parquet(PROCESSED_DATA_DIR / "combined_exp003_dataset.parquet")
    print(f"Combined Dataset generated successfully. Shape: {combined.shape}")
    return combined

if __name__ == "__main__":
    create_combined_dataset()