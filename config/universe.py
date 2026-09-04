"""
Canonical Universe Configuration for Competition Experiments.

Single source of truth for COMPETITION_UNIVERSE and SECTOR_MAP.
`UNIVERSE` is an alias for COMPETITION_UNIVERSE — use either name.

Runs strict validation upon import to enforce non-negotiable universe constraints.
"""

COMPETITION_UNIVERSE = [
    # Information Technology (3)
    "TCS", "INFY", "HCLTECH",
    # Banking & Financial Services (6)
    "HDFCBANK", "ICICIBANK", "KOTAKBANK", "SBIN", "AXISBANK", "BAJFINANCE",
    # Energy & Utilities (4)
    "RELIANCE", "ONGC", "NTPC", "POWERGRID",
    # Automobile (3)
    "MARUTI", "TATAMOTORS", "M&M",
    # FMCG (3)
    "HINDUNILVR", "ITC", "NESTLE",
    # Pharma & Healthcare (3)
    "SUNPHARMA", "DRREDDY", "CIPLA",
    # Metals & Mining (3)
    "TATASTEEL", "JSWSTEEL", "HINDALCO",
    # Telecom (1)
    "BHARTIARTL",
    # Cement & Infrastructure (2)
    "ULTRACEMCO", "LT",
    # Consumer / Retail (2)
    "TITAN", "ASIANPAINT",
]

# Alias — all internal code should prefer COMPETITION_UNIVERSE but UNIVERSE is
# accepted for backward-compatibility with callers that used the shorter name.
UNIVERSE = COMPETITION_UNIVERSE

SECTOR_MAP = {
    # Information Technology
    "TCS": "IT",
    "INFY": "IT",
    "HCLTECH": "IT",
    # Banking & Financial Services
    "HDFCBANK": "Financials",
    "ICICIBANK": "Financials",
    "KOTAKBANK": "Financials",
    "SBIN": "Financials",
    "AXISBANK": "Financials",
    "BAJFINANCE": "Financials",
    # Energy & Utilities
    "RELIANCE": "Energy",
    "ONGC": "Energy",
    "NTPC": "Utilities",
    "POWERGRID": "Utilities",
    # Automobile
    "MARUTI": "Automobile",
    "TATAMOTORS": "Automobile",
    "M&M": "Automobile",
    # FMCG
    "HINDUNILVR": "Consumer Staples",
    "ITC": "Consumer Staples",
    "NESTLE": "Consumer Staples",
    # Pharma & Healthcare
    "SUNPHARMA": "Healthcare",
    "DRREDDY": "Healthcare",
    "CIPLA": "Healthcare",
    # Metals & Mining
    "TATASTEEL": "Materials",
    "JSWSTEEL": "Materials",
    "HINDALCO": "Materials",
    # Telecom
    "BHARTIARTL": "Telecom",
    # Cement & Infrastructure — ULTRACEMCO belongs here, not in Materials
    "ULTRACEMCO": "Cement & Infrastructure",
    "LT": "Cement & Infrastructure",
    # Consumer / Retail
    "TITAN": "Consumer Discretionary",
    "ASIANPAINT": "Consumer Discretionary",
}

FINANCIAL_TICKERS = [
    ticker for ticker, sector in SECTOR_MAP.items() if sector == "Financials"
]


def validate_universe(
    universe: list,
    sector_map: dict,
    expected_len: int = 30,
) -> None:
    """Validate COMPETITION_UNIVERSE and SECTOR_MAP integrity.

    Raises ValueError loudly if any rule is violated.
    """
    # 1. Length
    if len(universe) != expected_len:
        raise ValueError(
            f"Universe length validation failed: expected {expected_len}, got {len(universe)}"
        )

    # 2. Duplicates
    if len(universe) != len(set(universe)):
        duplicates = [t for t in set(universe) if universe.count(t) > 1]
        raise ValueError(
            f"Duplicate tickers in COMPETITION_UNIVERSE: {duplicates}"
        )

    # 3. Every universe member must be in SECTOR_MAP
    missing_sectors = [t for t in universe if t not in sector_map]
    if missing_sectors:
        raise ValueError(
            f"Tickers in UNIVERSE but missing from SECTOR_MAP: {missing_sectors}"
        )

    # 4. No extra tickers in SECTOR_MAP
    extra_sectors = [t for t in sector_map if t not in set(universe)]
    if extra_sectors:
        raise ValueError(
            f"Tickers in SECTOR_MAP but not in UNIVERSE: {extra_sectors}"
        )

    # 5. Explicitly banned tickers
    banned = {"LTIM", "GAIL", "ADANIPORTS"}
    found_banned = banned & set(universe)
    if found_banned:
        raise ValueError(
            f"Banned tickers found in universe: {found_banned}"
        )


# Validate on import — any mistake fails loudly here.
validate_universe(COMPETITION_UNIVERSE, SECTOR_MAP, expected_len=30)
