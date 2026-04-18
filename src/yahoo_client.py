
import yfinance as yf
import pandas as pd


class YahooClient:

    def __init__(self):
        pass

    # =========================
    # Spot price
    # =========================
    def get_spot(self, ticker: str):
        t = yf.Ticker(ticker)
        data = t.history(period="1d")

        if data.empty:
            raise ValueError(f"No price data for {ticker}")

        return float(data["Close"].iloc[-1])

    # =========================
    # Available expiries
    # =========================
    def get_expiries(self, ticker: str):
        t = yf.Ticker(ticker)
        return t.options

    # =========================
    # Option chain for one expiry
    # =========================
    def get_option_chain(self, ticker: str, expiry: str):
        t = yf.Ticker(ticker)

        chain = t.option_chain(expiry)

        calls = chain.calls.copy()
        puts = chain.puts.copy()

        calls["type"] = "call"
        puts["type"] = "put"

        df = pd.concat([calls, puts], ignore_index=True)

        # Normalize columns
        df = df.rename(columns={
            "strike": "strike",
            "lastPrice": "last_price",
            "bid": "bid",
            "ask": "ask",
            "impliedVolatility": "iv",
            "volume": "volume",
            "openInterest": "open_interest"
        })

        df["expiry"] = expiry
        df["ticker"] = ticker

        return df[[
            "ticker",
            "expiry",
            "type",
            "strike",
            "last_price",
            "bid",
            "ask",
            "iv",
            "volume",
            "open_interest"
        ]]

    # =========================
    # Full chain (all expiries)
    # =========================
    def get_full_chain(self, ticker: str):
        expiries = self.get_expiries(ticker)

        all_data = []

        for exp in expiries:
            try:
                df = self.get_option_chain(ticker, exp)
                all_data.append(df)
            except Exception as e:
                print(f"Skipping {exp}: {e}")

        if not all_data:
            raise ValueError(f"No option data for {ticker}")

        return pd.concat(all_data, ignore_index=True)

    # =========================
    # ATM volatility
    # =========================
    def get_atm_vol(self, ticker: str, expiry: str):
        spot = self.get_spot(ticker)
        df = self.get_option_chain(ticker, expiry)

        df["distance"] = (df["strike"] - spot).abs()

        atm = df.loc[df["distance"].idxmin()]

        return float(atm["iv"])
