"""
Sector classifications for the 30-stock competition universe.
Includes explicit flags for financial sector accounting rules.

Re-exports canonical declarations from research.config.universe.
"""

from research.config.universe import (
    COMPETITION_UNIVERSE,
    FINANCIAL_TICKERS,
    SECTOR_MAP,
    validate_universe,
)

__all__ = [
    "COMPETITION_UNIVERSE",
    "SECTOR_MAP",
    "FINANCIAL_TICKERS",
    "validate_universe",
]