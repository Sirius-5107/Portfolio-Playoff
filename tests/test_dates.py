import pytest
import pandas as pd
from config.research_config import Config
from src.utils.dates import generate_rebalance_schedule

def test_rebalance_schedule_counts():
    trading_days = pd.date_range("2015-01-01", "2026-05-31", freq='B')
    res_dates = generate_rebalance_schedule(trading_days, Config.RESEARCH_START, Config.RESEARCH_END)
    val_dates = generate_rebalance_schedule(trading_days, Config.VALIDATION_START, Config.VALIDATION_END)
    
    # Approximate tolerance test on schedule counts
    assert abs(len(res_dates) - Config.REQUIRED_RESEARCH_REBALANCES) <= 2
    assert abs(len(val_dates) - Config.REQUIRED_VALIDATION_REBALANCES) <= 2