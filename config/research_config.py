"""
Global Research Configuration Parameters.
"""

import pandas as pd
from .universe import UNIVERSE

class Config:
    UNIVERSE = UNIVERSE
    
    RESEARCH_START = pd.Timestamp("2015-01-01")
    RESEARCH_END = pd.Timestamp("2023-12-31")
    
    VALIDATION_START = pd.Timestamp("2024-01-01")
    VALIDATION_END = pd.Timestamp("2026-05-31")
    
    VALIDATION_LABEL = "EXPOSED / CONTAMINATED / NON-BLIND"
    
    REBALANCE_FREQUENCY_DAYS = 20
    PRIMARY_HORIZON_DAYS = 20
    FORWARD_HORIZONS = [5, 10, 20, 30]
    
    PRIMARY_METRIC = "cross_sectional_spearman_ic"
    HAC_LAGS = [0, 1, 3, 6, 12]
    
    WINSOR_LEVELS = [
        None,
        (0.01, 0.99),
        (0.05, 0.95)
    ]
    
    REQUIRED_RESEARCH_REBALANCES = 118
    REQUIRED_VALIDATION_REBALANCES = 32