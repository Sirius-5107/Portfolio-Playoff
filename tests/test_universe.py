"""
Tests for universe integrity.

These tests are methodology guards: if the universe changes accidentally,
they will fail loudly before any experiment is run.
"""

import pytest
from config.universe import COMPETITION_UNIVERSE, UNIVERSE, SECTOR_MAP, FINANCIAL_TICKERS, validate_universe


def test_universe_length():
    assert len(COMPETITION_UNIVERSE) == 30, f"Expected 30 tickers, got {len(COMPETITION_UNIVERSE)}"


def test_universe_no_duplicates():
    assert len(COMPETITION_UNIVERSE) == len(set(COMPETITION_UNIVERSE)), (
        f"Duplicate tickers detected: "
        f"{[t for t in set(COMPETITION_UNIVERSE) if COMPETITION_UNIVERSE.count(t) > 1]}"
    )


def test_universe_alias_matches():
    """UNIVERSE alias must be identical to COMPETITION_UNIVERSE."""
    assert UNIVERSE is COMPETITION_UNIVERSE, "UNIVERSE alias does not point to COMPETITION_UNIVERSE"


def test_banned_tickers_absent():
    banned = {"LTIM", "GAIL", "ADANIPORTS"}
    found = banned & set(COMPETITION_UNIVERSE)
    assert not found, f"Banned tickers found in universe: {found}"


def test_expected_tickers_present():
    required = {
        "TCS", "INFY", "HCLTECH",
        "HDFCBANK", "ICICIBANK", "KOTAKBANK", "SBIN", "AXISBANK", "BAJFINANCE",
        "RELIANCE", "ONGC", "NTPC", "POWERGRID",
        "MARUTI", "TATAMOTORS", "M&M",
        "HINDUNILVR", "ITC", "NESTLE",
        "SUNPHARMA", "DRREDDY", "CIPLA",
        "TATASTEEL", "JSWSTEEL", "HINDALCO",
        "BHARTIARTL",
        "ULTRACEMCO", "LT",
        "TITAN", "ASIANPAINT",
    }
    assert required == set(COMPETITION_UNIVERSE), (
        f"Universe mismatch.\nExpected: {sorted(required)}\nGot: {sorted(COMPETITION_UNIVERSE)}"
    )


def test_sector_map_covers_universe():
    for ticker in COMPETITION_UNIVERSE:
        assert ticker in SECTOR_MAP, f"Missing sector mapping for {ticker}"


def test_sector_map_no_extra():
    for ticker in SECTOR_MAP:
        assert ticker in COMPETITION_UNIVERSE, (
            f"SECTOR_MAP contains '{ticker}' which is not in COMPETITION_UNIVERSE"
        )


def test_financial_tickers_non_empty():
    assert len(FINANCIAL_TICKERS) > 0, "No financials identified"
    expected_fins = {"HDFCBANK", "ICICIBANK", "KOTAKBANK", "SBIN", "AXISBANK", "BAJFINANCE"}
    assert expected_fins == set(FINANCIAL_TICKERS), (
        f"Financial tickers mismatch: expected {sorted(expected_fins)}, got {sorted(FINANCIAL_TICKERS)}"
    )


def test_validate_universe_passes():
    """validate_universe() must not raise on the canonical definitions."""
    validate_universe(COMPETITION_UNIVERSE, SECTOR_MAP, expected_len=30)


def test_validate_universe_catches_wrong_length():
    with pytest.raises(ValueError, match="length"):
        validate_universe(COMPETITION_UNIVERSE[:29], SECTOR_MAP)


def test_ultracemco_sector_is_cement():
    """ULTRACEMCO must not be in Materials — it belongs to Cement & Infrastructure."""
    assert SECTOR_MAP["ULTRACEMCO"] != "Materials", (
        "ULTRACEMCO is misclassified as Materials; should be Cement & Infrastructure"
    )
