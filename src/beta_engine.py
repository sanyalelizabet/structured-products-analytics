import logging
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

BENCHMARK_TICKER = "URTH.US"       # iShares MSCI World ETF on EOD — proxy for MSCI World
BENCHMARK_KEY    = "US4642863926"


class BetaEngine:
    """
    Compute realized OLS beta for each ISIN against MSCI World (URTH).

    β_i = Cov(R_i, R_m) / Var(R_m)

    Uses the same daily-price infrastructure as CorrelationEngine:
    prices are fetched once and cached in prices.csv.  The benchmark
    is stored under the synthetic key BENCHMARK_KEY so it coexists
    cleanly with ISIN-keyed rows.
    """

    def __init__(self, market_data_engine):
        self.mde = market_data_engine

    def build_beta_map(
        self,
        isin_ticker_map: dict,
        years: int = 3,
        force_refresh: bool = True,
    ) -> dict:
        """
        Compute beta for every ISIN in isin_ticker_map.

        Parameters
        ----------
        isin_ticker_map : dict  { isin: ticker }
        years           : int   years of daily history to use
        force_refresh   : bool  re-download prices

        Returns
        -------
        dict  { isin: beta }   floats, one per ISIN.
              ISINs with insufficient overlap default to 1.0.
        """
        # ── 1. Fetch stock prices ─────────────────────────────────────────
        db = self.mde.fetch_daily_prices(
            isin_ticker_map, years=years, force_refresh=force_refresh
        )
        db = db[db["isin"].isin(isin_ticker_map.keys())]

        # ── 2. Fetch benchmark (URTH / MSCI World) ────────────────────────
        benchmark_map = {BENCHMARK_KEY: BENCHMARK_TICKER}
        db_bm = self.mde.fetch_daily_prices(
            benchmark_map, years=years, force_refresh=force_refresh
        )
        db_bm = db_bm[db_bm["isin"] == BENCHMARK_KEY]

        if db_bm.empty:
            log.warning("No benchmark data for %s — all betas default to 1.0", BENCHMARK_TICKER)
            return {isin: 1.0 for isin in isin_ticker_map}

        # ── 3. Pivot & compute log returns ───────────────────────────────
        all_db = pd.concat([db, db_bm], ignore_index=True)

        prices_wide = (
            all_db.pivot_table(index="date", columns="isin", values="price", aggfunc="last")
            .sort_index()
        )

        log_returns = np.log(prices_wide / prices_wide.shift(1))

        if BENCHMARK_KEY not in log_returns.columns:
            log.warning("Benchmark column missing after pivot — all betas default to 1.0")
            return {isin: 1.0 for isin in isin_ticker_map}

        r_m = log_returns[BENCHMARK_KEY].dropna()

        # ── 4. OLS beta per ISIN ─────────────────────────────────────────
        beta_map = {}
        min_overlap = 252  # require at least one full trading year

        for isin in isin_ticker_map:
            if isin not in log_returns.columns:
                log.warning("No price data for %s — beta defaults to 1.0", isin)
                beta_map[isin] = 1.0
                continue

            r_i = log_returns[isin].dropna()

            # align on common dates
            common = r_i.index.intersection(r_m.index)

            if len(common) < min_overlap:
                log.warning(
                    "%s has only %d overlapping days with benchmark (need %d) "
                    "— beta defaults to 1.0",
                    isin, len(common), min_overlap,
                )
                beta_map[isin] = 1.0
                continue

            ri_aligned = r_i.loc[common].values
            rm_aligned = r_m.loc[common].values

            var_m = np.var(rm_aligned, ddof=1)
            if var_m < 1e-12:
                log.warning("Benchmark variance near zero for %s — beta defaults to 1.0", isin)
                beta_map[isin] = 1.0
                continue

            cov = np.cov(ri_aligned, rm_aligned, ddof=1)[0, 1]
            beta_map[isin] = round(cov / var_m, 4)

        return beta_map
