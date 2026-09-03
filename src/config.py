"""
Configuration settings for Portfolio Playoff - Experiment 001.
"""
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REPORTS_DIR = BASE_DIR / "reports"

# Ensure directories exist
for p in [RAW_DATA_DIR, PROCESSED_DATA_DIR, REPORTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Fixed Universe & Sector Definitions
SECTOR_MAP = {
    # IT
    "TCS.NS": "IT", "INFY.NS": "IT", "HCLTECH.NS": "IT",
    # Banking & Financial Services
    "HDFCBANK.NS": "Banking & Financial Services", "ICICIBANK.NS": "Banking & Financial Services",
    "KOTAKBANK.NS": "Banking & Financial Services", "SBIN.NS": "Banking & Financial Services",
    "AXISBANK.NS": "Banking & Financial Services", "BAJFINANCE.NS": "Banking & Financial Services",
    # Energy & Utilities
    "RELIANCE.NS": "Energy & Utilities", "ONGC.NS": "Energy & Utilities",
    "NTPC.NS": "Energy & Utilities", "POWERGRID.NS": "Energy & Utilities",
    # Automobile
    "MARUTI.NS": "Automobile", "TATAMOTORS.NS": "Automobile", "M&M.NS": "Automobile",
    # FMCG
    "HINDUNILVR.NS": "FMCG", "ITC.NS": "FMCG", "NESTLEIND.NS": "FMCG",
    # Pharma & Healthcare
    "SUNPHARMA.NS": "Pharma & Healthcare", "DRREDDY.NS": "Pharma & Healthcare", "CIPLA.NS": "Pharma & Healthcare",
    # Metals & Mining
    "TATASTEEL.NS": "Metals & Mining", "JSWSTEEL.NS": "Metals & Mining", "HINDALCO.NS": "Metals & Mining",
    # Telecom
    "BHARTIARTL.NS": "Telecom",
    # Cement & Infrastructure
    "ULTRACEMCO.NS": "Cement & Infrastructure", "LT.NS": "Cement & Infrastructure",
    # Consumer/Retail
    "TITAN.NS": "Consumer/Retail", "ASIANPAINT.NS": "Consumer/Retail"
}

TICKERS = list(SECTOR_MAP.keys())

# Research Timeframes
START_DATE = "2010-01-01"
END_DATE = "2026-09-07"
RESEARCH_PERIOD = ("2015-01-01", "2023-12-31")
VALIDATION_PERIOD = ("2024-01-01", "2026-09-07")

# Lookbacks & Targets
MOM_WINDOWS = [5, 10, 20, 60, 120, 252]
FORWARD_HORIZONS = [5, 10, 20, 30]