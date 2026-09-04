import pytest
import pandas as pd
import numpy as np
from src.features.momentum import compute_momentum_features
from src.features.quality import compute_quality_features

def test_sector_relative_momentum_demeaning():
    # Verify sector relative momentum is demeaned before Z-scoring
    dates = pd.date_range("2021-01-01", periods=5, freq='D')
    data = []
    for d in dates:
        for t, s in [('T1', 'IT'), ('T2', 'IT'), ('T3', 'Energy')]:
            data.append({'date': d, 'ticker': t, 'adj_close': 100.0 + np.random.randn()})
    df = pd.DataFrame(data)
    mom_df = compute_momentum_features(df)
    
    # Check that for each sector date group, raw relative sum is zero
    sec_sum = mom_df.groupby(['date', 'sector'])['sector_rel_60d_raw'].sum()
    np.testing.assert_allclose(sec_sum.dropna().values, 0.0, atol=1e-7)

def test_financial_sector_quality_exclusion():
    raw_df = pd.DataFrame([{
        'ticker': 'HDFCBANK', 'quarter_end': '2023-03-31',
        'net_income_ttm': 100, 'equity': 500, 'total_assets': 1000,
        'ebit_ttm': 120, 'revenue_ttm': 300, 'fcf_ttm': 50, 'ocf_ttm': 60,
        'short_term_debt': 10, 'long_term_debt': 40, 'interest_expense_ttm': 5
    }])
    qual = compute_quality_features(raw_df)
    
    assert not np.isnan(qual['roe_ttm'].iloc[0])
    assert not np.isnan(qual['roa_ttm'].iloc[0])
    assert np.isnan(qual['op_margin_ttm'].iloc[0])
    assert np.isnan(qual['fcf_margin_ttm'].iloc[0])
    assert np.isnan(qual['debt_to_equity'].iloc[0])