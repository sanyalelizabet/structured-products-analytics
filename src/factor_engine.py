"""Factor data layer for the multi-factor stress engine.

Wraps ``MarketDataEngine``: factors are stored in ``prices.csv`` under
synthetic keys ``__FACTOR_<CODE>__``, reusing the existing refresh/caching
logic (same pattern as ``BetaEngine``'s ``__BENCHMARK__``).

Six liquid proxies spanning the systematic dimensions of a CHF/USD
equity-linked book:

    MKT     URTH.US        MSCI World ETF — global equity beta
    TECH    XLK.US         Technology Select Sector ETF
    HC      XLV.US         Healthcare Select Sector ETF
    FIN     XLF.US         Financials Select Sector ETF
    ENERGY  XLE.US         Energy Select Sector ETF — oil-price proxy
    FX      USDCHF.FOREX   USD/CHF spot

Consumers: ``FactorLoadingsEngine`` (OLS of asset returns on factors),
``FactorScenarioEngine`` (factor-path simulation under a scenario spec).
"""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd

from src.beta_engine import BENCHMARK_KEY, BENCHMARK_TICKER

log = logging.getLogger(__name__)


# Factor universe.  Each entry: (EOD ticker, storage key in prices.csv, label).
# The MKT factor reuses BetaEngine's benchmark row so URTH is fetched once,
# not twice.  Other factors use synthetic ISIN-like keys.
FACTORS: dict[str, tuple[str, str, str]] = {
    # code      ticker             storage key (real ISIN where available)   label
    "MKT":    (BENCHMARK_TICKER,   BENCHMARK_KEY,    "MSCI World"),       # US4642863926
    "TECH":   ("XLK.US",           "US81369Y8030",   "Tech sector"),
    "HC":     ("XLV.US",           "US81369Y2090",   "Healthcare sector"),
    "FIN":    ("XLF.US",           "US81369Y6059",   "Financials sector"),
    "ENERGY": ("XLE.US",           "US81369Y5069",   "Energy / oil proxy"),
    "FX":     ("USDCHF.FOREX",     "__FACTOR_FX__",  "USD/CHF"),          # FX has no ISIN
}


class FactorEngine:
    """Fetch (via ``MarketDataEngine``), align, and expose factor returns."""

    def __init__(self, market_data_engine):
        self.mde = market_data_engine

    # ---------------------------------------------------------------- fetch

    def fetch_factor_prices(
        self,
        years: int = 5,
        factors: Iterable[str] | None = None,
        force_refresh: bool = True,
    ) -> pd.DataFrame:
        """Delegate to ``MarketDataEngine.fetch_daily_prices``.

        Returns the slice of ``prices.csv`` belonging to factor rows.
        """
        codes = list(factors) if factors else list(FACTORS.keys())

        key_ticker_map = {
            FACTORS[code][1]: FACTORS[code][0]   # storage_key → ticker
            for code in codes
            if code in FACTORS
        }

        self.mde.fetch_daily_prices(
            key_ticker_map, years=years, force_refresh=force_refresh
        )

        db = self.mde.load_db()
        return db[db["isin"].isin(key_ticker_map.keys())].copy()

    # --------------------------------------------------------------- returns

    def build_returns(
        self,
        factors: Iterable[str] | None = None,
        years: int | None = None,
    ) -> pd.DataFrame:
        """Wide DataFrame of daily log-returns indexed by date.

        Columns are factor codes (e.g. "MKT"). NaN rows are dropped (listwise
        alignment so all factors are observed on every returned date).
        """
        if factors is None:
            factors = list(FACTORS.keys())
        keys = {FACTORS[c][1]: c for c in factors if c in FACTORS}

        db = self.mde.load_db()
        df = db[db["isin"].isin(keys.keys())].copy()
        if df.empty:
            raise RuntimeError(
                "Factor prices not in DB — call fetch_factor_prices first"
            )

        df["factor"] = df["isin"].map(keys)

        prices = (
            df.pivot_table(index="date", columns="factor", values="price", aggfunc="last")
            .sort_index()
        )

        if years is not None:
            cutoff = pd.Timestamp.today() - pd.DateOffset(years=years)
            prices = prices[prices.index >= cutoff]

        log_returns = np.log(prices / prices.shift(1)).dropna(how="any")
        return log_returns

    # --------------------------------------------------------- cov / corr

    def factor_cov(self, years: int | None = 5) -> pd.DataFrame:
        """Annualised factor covariance matrix from daily log-returns."""
        r = self.build_returns(years=years)
        return r.cov() * 252

    def factor_corr(self, years: int | None = 5) -> pd.DataFrame:
        """Factor correlation matrix from daily log-returns."""
        r = self.build_returns(years=years)
        return r.corr()

    def factor_vol(self, years: int | None = 5) -> pd.Series:
        """Annualised factor volatility from daily log-returns."""
        r = self.build_returns(years=years)
        return r.std() * np.sqrt(252)
