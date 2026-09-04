"""
Sector classifications for the 30-stock competition universe.
Includes explicit flags for financial-sector accounting rules.

Re-exports from config.universe — this is a thin convenience module.
All authoritative definitions live in config.universe.
"""

from config.universe import (
    COMPETITION_UNIVERSE,
    UNIVERSE,
    FINANCIAL_TICKERS,
    SECTOR_MAP,
    validate_universe,
)

__all__ = [
    "COMPETITION_UNIVERSE",
    "UNIVERSE",
    "SECTOR_MAP",
    "FINANCIAL_TICKERS",
    "validate_universe",
]
