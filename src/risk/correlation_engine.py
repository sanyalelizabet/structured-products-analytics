import logging

import numpy as np
import pandas as pd

from src.numerics.linalg import is_positive_semidefinite, nearest_correlation_matrix

log = logging.getLogger(__name__)


class CorrelationEngine:

    def __init__(self, market_data_engine):
        self.mde = market_data_engine

    def build_corr_matrix(self, isin_ticker_map, years=5, force_refresh=False):
        db = self.mde.fetch_daily_prices(
            isin_ticker_map, years=years, force_refresh=force_refresh
        )

        db = db[db["isin"].isin(isin_ticker_map.keys())]

        prices_wide = (
            db.pivot_table(index="date", columns="isin", values="price", aggfunc="last")
            .sort_index()
        )

        # daily log returns (NaNs will exist where data is missing)
        log_returns = np.log(prices_wide / prices_wide.shift(1))

        # pairwise correlation — each pair uses its own overlapping window
        min_periods = 252  # one full trading year minimum per pair
        corr = log_returns.corr(min_periods=min_periods)

        # ISINs with insufficient data produce NaN rows/cols — fill with 0
        # (independence assumption) rather than crashing the whole matrix
        nan_isins = corr.columns[corr.isna().all()].tolist()
        if nan_isins:
            import logging
            logging.getLogger(__name__).warning(
                "Insufficient price history for %s — treating as uncorrelated",
                nan_isins,
            )
            arr = np.array(corr.fillna(0.0).values, dtype=float, copy=True)
            np.fill_diagonal(arr, 1.0)
            corr = pd.DataFrame(arr, index=corr.index, columns=corr.columns)

        # Pairwise estimation (per-pair overlap windows) and NaN→0 fills can
        # leave the matrix indefinite, which has no Cholesky factor. Project to
        # the nearest valid correlation matrix (Higham) so downstream samplers
        # always receive a PSD input.
        arr = corr.to_numpy(dtype=float)
        if not is_positive_semidefinite(arr):
            log.warning(
                "Correlation matrix not PSD; projecting to nearest correlation "
                "matrix (Higham) for %d underlyings.", arr.shape[0],
            )
            arr = nearest_correlation_matrix(arr)
            corr = pd.DataFrame(arr, index=corr.index, columns=corr.columns)

        return corr

    def build_translated_corr_matrix(
        self,
        isin_ticker_map: dict,
        isin_currency_map: dict,
        target_ccy: str = "USD",
        years: int = 5,
    ) -> pd.DataFrame:
        """Correlation matrix on prices translated into a single currency.

        Foreign-currency assets are converted to ``target_ccy`` per day using the
        daily FX rates persisted in ``fx_rates.csv`` (back-filled incrementally
        by :meth:`MarketDataEngine.fetch_fx_history`).  Correlation is then
        computed on log-returns of the translated series.

        This is the **investor view** in a single base currency and is intended
        for display only — the valuation engines keep using native-return
        correlation, which is correct for native-currency simulation.

        Assets already in ``target_ccy`` (or with no known native currency in
        the master) pass through unchanged.  An FX series for a foreign currency
        not carried by Frankfurter is skipped with a warning and the asset
        falls back to its native series.
        """
        db = self.mde.fetch_daily_prices(
            isin_ticker_map, years=years, force_refresh=False,
        )
        db = db[db["isin"].isin(isin_ticker_map.keys())]
        prices_wide = (
            db.pivot_table(index="date", columns="isin", values="price", aggfunc="last")
            .sort_index()
        )

        # Make sure we have an up-to-date FX history for the target base.
        self.mde.fetch_fx_history(base=target_ccy, years=years)
        fx_db = self.mde.load_fx_db()
        fx_base = fx_db[fx_db["base"] == target_ccy]
        # Wide: index=date, columns=quote_ccy, value=rate.  Frankfurter convention
        # is rate = units of quote per 1 unit of base (so 1 unit of quote =
        # 1 / rate units of base, which is how we translate a foreign price).
        fx_wide = (
            fx_base.pivot_table(index="date", columns="quote", values="rate", aggfunc="last")
            .sort_index()
        )

        translated = prices_wide.copy()
        for isin in translated.columns:
            ccy = isin_currency_map.get(isin)
            if ccy is None or ccy == target_ccy:
                continue
            if ccy not in fx_wide.columns:
                log.warning(
                    "No FX history for %s vs %s — leaving %s in native currency.",
                    ccy, target_ccy, isin,
                )
                continue
            # Align FX onto the price calendar; forward-fill across holidays in
            # the FX series so a missing weekend / FX-only holiday doesn't drop
            # a valid trading-day price.  Back-fill the leading edge.
            fx_aligned = fx_wide[ccy].reindex(translated.index).ffill().bfill()
            with np.errstate(divide="ignore", invalid="ignore"):
                translated[isin] = translated[isin] / fx_aligned

        log_returns = np.log(translated / translated.shift(1))
        min_periods = 252
        corr = log_returns.corr(min_periods=min_periods)

        nan_isins = corr.columns[corr.isna().all()].tolist()
        if nan_isins:
            log.warning(
                "Insufficient (translated) history for %s — treating as uncorrelated",
                nan_isins,
            )
            arr = np.array(corr.fillna(0.0).values, dtype=float, copy=True)
            np.fill_diagonal(arr, 1.0)
            corr = pd.DataFrame(arr, index=corr.index, columns=corr.columns)

        arr = corr.to_numpy(dtype=float)
        if not is_positive_semidefinite(arr):
            log.warning(
                "Translated correlation matrix not PSD; projecting (Higham) "
                "for %d underlyings.", arr.shape[0],
            )
            arr = nearest_correlation_matrix(arr)
            corr = pd.DataFrame(arr, index=corr.index, columns=corr.columns)

        return corr
