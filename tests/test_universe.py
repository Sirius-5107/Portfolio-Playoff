import pytest
from config.universe import UNIVERSE
from config.sectors import SECTOR_MAP, FINANCIAL_TICKERS

def test_universe_constraints():
    assert len(UNIVERSE) == 30, f"Expected 30 tickers, got {len(UNIVERSE)}"
    assert len(set(UNIVERSE)) == 30, "Duplicate tickers detected"
    assert "WIT" not in UNIVERSE
    assert "TECHM" not in UNIVERSE
    assert "ADANIENT" not in UNIVERSE

def test_sector_mappings():
    for ticker in UNIVERSE:
        assert ticker in SECTOR_MAP, f"Missing sector mapping for {ticker}"
    assert len(FINANCIAL_TICKERS) > 0, "No financials identified"