"""
Tests for YahooClient.
All yfinance network calls are patched — no real HTTP requests.
"""
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
from src.yahoo_client import YahooClient
from src.exceptions import DataUnavailableError


# ─────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────

def _make_chain_df(strikes=(90.0, 100.0, 110.0)):
    rows = []
    for strike in strikes:
        rows.append({
            "strike": strike, "lastPrice": 5.0, "bid": 4.9, "ask": 5.1,
            "impliedVolatility": 0.20, "volume": 100, "openInterest": 500,
        })
    return pd.DataFrame(rows)


def _mock_ticker(spot=100.0, expiries=("2025-06-20", "2025-09-19"), calls_df=None, puts_df=None):
    """Return a mock yf.Ticker with sensible defaults."""
    if calls_df is None:
        calls_df = _make_chain_df()
    if puts_df is None:
        puts_df = _make_chain_df()

    chain = MagicMock()
    chain.calls = calls_df
    chain.puts = puts_df

    history_df = pd.DataFrame({"Close": [spot]})

    tkr = MagicMock()
    tkr.history.return_value = history_df
    tkr.options = expiries
    tkr.option_chain.return_value = chain
    return tkr


@pytest.fixture
def client():
    return YahooClient()


# ─────────────────────────────────────────
# get_spot
# ─────────────────────────────────────────

class TestGetSpot:
    def test_returns_float(self, client):
        with patch("yfinance.Ticker", return_value=_mock_ticker(spot=95.5)):
            assert client.get_spot("NESN.SW") == 95.5

    def test_raises_when_no_history(self, client):
        tkr = _mock_ticker()
        tkr.history.return_value = pd.DataFrame()
        with patch("yfinance.Ticker", return_value=tkr):
            with pytest.raises(DataUnavailableError, match="No price data"):
                client.get_spot("NESN.SW")


# ─────────────────────────────────────────
# get_expiries
# ─────────────────────────────────────────

class TestGetExpiries:
    def test_returns_tuple_of_dates(self, client):
        with patch("yfinance.Ticker", return_value=_mock_ticker(expiries=("2025-06-20", "2025-09-19"))):
            exps = client.get_expiries("AAPL")
        assert "2025-06-20" in exps
        assert "2025-09-19" in exps


# ─────────────────────────────────────────
# get_option_chain
# ─────────────────────────────────────────

class TestGetOptionChain:
    def test_returns_calls_and_puts(self, client):
        with patch("yfinance.Ticker", return_value=_mock_ticker()):
            df = client.get_option_chain("AAPL", "2025-06-20")
        assert set(df["type"].unique()) == {"call", "put"}

    def test_expected_columns_present(self, client):
        with patch("yfinance.Ticker", return_value=_mock_ticker()):
            df = client.get_option_chain("AAPL", "2025-06-20")
        expected = {"ticker", "expiry", "type", "strike", "last_price", "bid", "ask", "iv", "volume", "open_interest"}
        assert expected.issubset(df.columns)

    def test_ticker_and_expiry_stamped(self, client):
        with patch("yfinance.Ticker", return_value=_mock_ticker()):
            df = client.get_option_chain("AAPL", "2025-06-20")
        assert (df["ticker"] == "AAPL").all()
        assert (df["expiry"] == "2025-06-20").all()

    def test_row_count_is_calls_plus_puts(self, client):
        calls = _make_chain_df(strikes=(90.0, 100.0))
        puts  = _make_chain_df(strikes=(90.0, 100.0))
        with patch("yfinance.Ticker", return_value=_mock_ticker(calls_df=calls, puts_df=puts)):
            df = client.get_option_chain("AAPL", "2025-06-20")
        assert len(df) == len(calls) + len(puts)


# ─────────────────────────────────────────
# get_full_chain
# ─────────────────────────────────────────

class TestGetFullChain:
    def test_stacks_all_expiries(self, client):
        expiries = ("2025-06-20", "2025-09-19")
        with patch("yfinance.Ticker", return_value=_mock_ticker(expiries=expiries)):
            df = client.get_full_chain("AAPL")
        assert set(df["expiry"].unique()) == set(expiries)

    def test_raises_when_no_expiries(self, client):
        tkr = _mock_ticker(expiries=())
        with patch("yfinance.Ticker", return_value=tkr):
            with pytest.raises(DataUnavailableError, match="No option data"):
                client.get_full_chain("AAPL")

    def test_skips_failed_expiry_and_continues(self, client):
        expiries = ("2025-06-20", "2025-09-19")
        tkr = _mock_ticker(expiries=expiries)
        # First expiry raises, second succeeds
        tkr.option_chain.side_effect = [Exception("timeout"), tkr.option_chain.return_value]
        with patch("yfinance.Ticker", return_value=tkr):
            df = client.get_full_chain("AAPL")
        assert "2025-09-19" in df["expiry"].values


# ─────────────────────────────────────────
# get_atm_vol
# ─────────────────────────────────────────

class TestGetAtmVol:
    def test_returns_iv_of_closest_strike(self, client):
        calls = pd.DataFrame([
            {"strike": 90.0,  "lastPrice": 5.0, "bid": 4.9, "ask": 5.1, "impliedVolatility": 0.18, "volume": 100, "openInterest": 200},
            {"strike": 100.0, "lastPrice": 3.0, "bid": 2.9, "ask": 3.1, "impliedVolatility": 0.20, "volume": 80,  "openInterest": 150},
            {"strike": 110.0, "lastPrice": 1.0, "bid": 0.9, "ask": 1.1, "impliedVolatility": 0.22, "volume": 60,  "openInterest": 100},
        ])
        puts = calls.copy()
        # spot = 100 → ATM strike is 100.0 → iv = 0.20
        with patch("yfinance.Ticker", return_value=_mock_ticker(spot=100.0, calls_df=calls, puts_df=puts)):
            vol = client.get_atm_vol("AAPL", "2025-06-20")
        assert abs(vol - 0.20) < 1e-9

    def test_returns_float(self, client):
        with patch("yfinance.Ticker", return_value=_mock_ticker(spot=100.0)):
            vol = client.get_atm_vol("AAPL", "2025-06-20")
        assert isinstance(vol, float)
