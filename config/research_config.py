"""
Global Research Configuration Parameters.

All date ranges, rebalance frequencies, and statistical settings live here.
Universe is imported from config.universe — do not duplicate it here.
"""

import pandas as pd
from .universe import COMPETITION_UNIVERSE, UNIVERSE


class Config:
    # Universe
    UNIVERSE = COMPETITION_UNIVERSE  # alias kept for backward-compat

    # Research window (in-sample, used for factor evaluation)
    RESEARCH_START = pd.Timestamp("2015-01-01")
    RESEARCH_END = pd.Timestamp("2023-12-31")

    # Validation window — explicitly labelled as EXPOSED / NOT BLIND
    # These periods saw the data during strategy development.
    VALIDATION_START = pd.Timestamp("2024-01-01")
    VALIDATION_END = pd.Timestamp("2026-09-04")  # through today's date

    VALIDATION_LABEL = "EXPOSED / CONTAMINATED / NON-BLIND"

    # Rebalance schedule
    REBALANCE_FREQUENCY_DAYS = 20          # every ~20 trading days (~monthly)
    PRIMARY_HORIZON_DAYS = 20
    FORWARD_HORIZONS = [5, 10, 20, 30]

    # Statistics
    PRIMARY_METRIC = "cross_sectional_spearman_ic"
    HAC_LAGS = [0, 1, 3, 6, 12]

    # Winsorization
    WINSOR_LEVELS = [
        None,
        (0.01, 0.99),
        (0.05, 0.95),
    ]

    # Expected schedule counts (approximate; tests allow ±2 tolerance)
    REQUIRED_RESEARCH_REBALANCES = 118
    REQUIRED_VALIDATION_REBALANCES = 32
