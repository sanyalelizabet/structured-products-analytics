"""Thin wrapper around :mod:`yfinance`.

Each public method translates yfinance's opaque failures into the typed
exceptions defined in :mod:`src.exceptions`, so callers can distinguish
"no data for this ticker" from "network is down" and react accordingly.

yfinance already performs its own HTTP retries internally, so we do NOT
stack another retry decorator on top.
"""
from __future__ import annotations

import logging

import pandas as pd
import yfinance as yf

from src.exceptions import DataFetchError, DataUnavailableError, NetworkError

log = logging.getLogger(__name__)


class YahooClient:
    def __init__(self):
        pass

    # =========================
    # Spot price
    # =========================
    def get_spot(self, ticker: str) -> float:
        try:
            t = yf.Ticker(ticker)
            data = t.history(period="1d")
        except Exception as e:  # yfinance raises a variety of types
            raise NetworkError(f"Yahoo fetch failed for {ticker}") from e

        if data.empty:
            raise DataUnavailableError(f"No price data for {ticker}")

        return float(data["Close"].iloc[-1])

    # =========================
    # Available expiries
    # =========================
    def get_expiries(self, ticker: str) -> tuple:
        try:
            return yf.Ticker(ticker).options
        except Exception as e:
            raise NetworkError(
                f"Yahoo fetch failed for expiries of {ticker}"
            ) from e

    # =========================
    # Option chain for one expiry
    # =========================
    def get_option_chain(self, ticker: str, expiry: str) -> pd.DataFrame:
        try:
            chain = yf.Ticker(ticker).option_chain(expiry)
        except Exception as e:
            raise NetworkError(
                f"Yahoo option-chain fetch failed for {ticker} @ {expiry}"
            ) from e

        calls = chain.calls.copy()
        puts = chain.puts.copy()

        calls["type"] = "call"
        puts["type"] = "put"

        df = pd.concat([calls, puts], ignore_index=True)

        df = df.rename(
            columns={
                "lastPrice": "last_price",
                "impliedVolatility": "iv",
                "openInterest": "open_interest",
            }
        )

        df["expiry"] = expiry
        df["ticker"] = ticker

        return df[
            [
                "ticker",
                "expiry",
                "type",
                "strike",
                "last_price",
                "bid",
                "ask",
                "iv",
                "volume",
                "open_interest",
            ]
        ]

    # =========================
    # Full chain (all expiries)
    # =========================
    def get_full_chain(self, ticker: str) -> pd.DataFrame:
        expiries = self.get_expiries(ticker)

        all_data = []

        for exp in expiries:
            try:
                all_data.append(self.get_option_chain(ticker, exp))
            except DataFetchError as e:
                log.warning("Skipping expiry %s for %s: %s", exp, ticker, e)

        if not all_data:
            raise DataUnavailableError(f"No option data for {ticker}")

        return pd.concat(all_data, ignore_index=True)

    # =========================
    # ATM volatility
    # =========================
    def get_atm_vol(self, ticker: str, expiry: str) -> float:
        spot = self.get_spot(ticker)
        df = self.get_option_chain(ticker, expiry)

        df["distance"] = (df["strike"] - spot).abs()
        atm = df.loc[df["distance"].idxmin()]

        return float(atm["iv"])
