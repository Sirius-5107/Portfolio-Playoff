"""
Canonical Universe Configuration for Competition Experiments.

Centralizes COMPETITION_UNIVERSE and SECTOR_MAP.
Runs strict validation upon import to enforce non-negotiable universe constraints.
"""

COMPETITION_UNIVERSE = [
    # IT (3)
    "TCS", "INFY", "HCLTECH",
    # Financials (6)
    "HDFCBANK", "ICICIBANK", "KOTAKBANK", "SBIN", "AXISBANK", "BAJFINANCE",
    # Energy & Utilities (4)
    "RELIANCE", "ONGC", "NTPC", "POWERGRID",
    # Automobile (3)
    "MARUTI", "TATAMOTORS", "M&M",
    # Consumer Staples (3)
    "HINDUNILVR", "ITC", "NESTLE",
    # Healthcare (3)
    "SUNPHARMA", "DRREDDY", "CIPLA",
    # Materials / Metals (3)
    "TATASTEEL", "JSWSTEEL", "HINDALCO",
    # Telecom (1)
    "BHARTIARTL",
    # Industrials / Construction (2)
    "ULTRACEMCO", "LT",
    # Consumer Discretionary (2)
    "TITAN", "ASIANPAINT",
]

SECTOR_MAP = {
    "TCS": "IT",
    "INFY": "IT",
    "HCLTECH": "IT",
    "HDFCBANK": "Financials",
    "ICICIBANK": "Financials",
    "KOTAKBANK": "Financials",
    "SBIN": "Financials",
    "AXISBANK": "Financials",
    "BAJFINANCE": "Financials",
    "RELIANCE": "Energy",
    "ONGC": "Energy",
    "NTPC": "Utilities",
    "POWERGRID": "Utilities",
    "MARUTI": "Automobile",
    "TATAMOTORS": "Automobile",
    "M&M": "Automobile",
    "HINDUNILVR": "Consumer Staples",
    "ITC": "Consumer Staples",
    "NESTLE": "Consumer Staples",
    "SUNPHARMA": "Healthcare",
    "DRREDDY": "Healthcare",
    "CIPLA": "Healthcare",
    "TATASTEEL": "Materials",
    "JSWSTEEL": "Materials",
    "HINDALCO": "Materials",
    "BHARTIARTL": "Telecom",
    "ULTRACEMCO": "Materials",
    "LT": "Capital Goods",
    "TITAN": "Consumer Discretionary",
    "ASIANPAINT": "Consumer Discretionary",
}

FINANCIAL_TICKERS = [
    ticker for ticker, sector in SECTOR_MAP.items() if sector == "Financials"
]


def validate_universe(
    universe: list[str], sector_map: dict[str, str], expected_len: int = 30
) -> None:
    """Validates the structure and integrity of the universe configuration.

    Fails loudly with an explicit ValueError if any rule is broken.
    """
    # 1. Check length
    if len(universe) != expected_len:
        raise ValueError(
            f"Universe length validation failed: expected {expected_len}, got {len(universe)}"
        )

    # 2. Check duplicates
    if len(universe) != len(set(universe)):
        duplicates = [
            ticker for ticker in set(universe) if universe.count(ticker) > 1
        ]
        raise ValueError(
            f"Duplicate tickers found in COMPETITION_UNIVERSE: {duplicates}"
        )

    # 3. Missing from SECTOR_MAP
    missing_sectors = [
        ticker for ticker in universe if ticker not in sector_map
    ]
    if missing_sectors:
        raise ValueError(
            f"Tickers present in UNIVERSE but missing from SECTOR_MAP: {missing_sectors}"
        )

    # 4. Extra in SECTOR_MAP
    extra_sectors = [
        ticker for ticker in sector_map if ticker not in set(universe)
    ]
    if extra_sectors:
        raise ValueError(
            f"Tickers present in SECTOR_MAP but missing from UNIVERSE: {extra_sectors}"
        )


# Run strict validation upon import
validate_universe(COMPETITION_UNIVERSE, SECTOR_MAP, expected_len=30)