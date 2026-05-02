"""Single-factor β vs MSCI World — thin shim over ``FactorLoadingsEngine``.

This module exists for backwards compatibility with existing callers that
ask for ``{isin: β}``.  The underlying OLS lives exactly once, in
``FactorLoadingsEngine``, so single-factor β and the multi-factor loadings
share data, alignment, and regression code.

Constants kept for downstream imports:

* ``BENCHMARK_TICKER`` — EOD ticker of the market proxy (URTH).
* ``BENCHMARK_KEY``    — storage key in ``prices.csv``.  Reused by
  ``FactorEngine`` so the MKT factor and the legacy benchmark are stored
  in the same row.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

BENCHMARK_TICKER = "URTH.US"        # iShares MSCI World ETF on EOD
BENCHMARK_KEY    = "US4642863926"   # real ISIN of URTH


class BetaEngine:
    """Compute single-factor β_i = Cov(r_i, r_MKT) / Var(r_MKT) per ISIN.

    Implementation delegates to :class:`FactorLoadingsEngine` with
    ``factors=["MKT"]`` and extracts the MKT loading.  Result is identical
    to a univariate OLS — multivariate regression with a single regressor
    *is* univariate.
    """

    def __init__(self, market_data_engine):
        self.mde = market_data_engine

    def build_beta_map(
        self,
        isin_ticker_map: dict,
        years: int = 3,
        force_refresh: bool = False,
    ) -> dict[str, float]:
        """Return ``{isin: β_MKT}``. Defaults to 1.0 on insufficient data."""
        # Local import keeps the module-level dependency graph cycle-free
        # (FactorEngine → uses BENCHMARK_KEY/BENCHMARK_TICKER from here).
        from src.factor_loadings_engine import FactorLoadingsEngine

        loadings = FactorLoadingsEngine(self.mde).build_loadings(
            isin_ticker_map,
            factors=["MKT"],
            years=years,
            force_refresh=force_refresh,
        )
        return {isin: data["betas"]["MKT"] for isin, data in loadings.items()}
