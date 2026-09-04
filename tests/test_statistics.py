import pytest
import pandas as pd
import numpy as np
from src.research.statistics import residualize_cross_section, calculate_hac_tstat

def test_residualization_orthogonality():
    np.random.seed(42)
    df = pd.DataFrame({
        'date': ['2023-01-01'] * 30,
        'ticker': [f'T_{i}' for i in range(30)],
        'z_mom': np.random.randn(30),
        'z_eps': np.random.randn(30)
    })
    res = residualize_cross_section(df, 'z_mom', ['z_eps'])
    df['res'] = res
    corr = df[['res', 'z_eps']].corr().iloc[0, 1]
    np.testing.assert_allclose(corr, 0.0, atol=1e-7)