import pandas as pd
from typing import List

def generate_rebalance_schedule(trading_days: pd.DatetimeIndex, start_date: pd.Timestamp, end_date: pd.Timestamp, freq_days: int = 20) -> pd.DatetimeIndex:
    sub_days = trading_days[(trading_days >= start_date) & (trading_days <= end_date)]
    if len(sub_days) == 0:
        return pd.DatetimeIndex([])
    rebal_dates = sub_days[::freq_days]
    return rebal_dates