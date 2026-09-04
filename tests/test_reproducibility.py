import pytest
import pandas as pd
import numpy as np
from src.features.common import zscore_cross_section

def test_deterministic_output():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    z1 = zscore_cross_section(s)
    z2 = zscore_cross_section(s)
    pd.testing.assert_series_equal(z1, z2)