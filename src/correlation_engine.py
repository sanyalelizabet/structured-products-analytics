import logging

import numpy as np
import pandas as pd

from src.linalg import is_positive_semidefinite, nearest_correlation_matrix

log = logging.getLogger(__name__)


class CorrelationEngine:

    def __init__(self, market_data_engine):
        self.mde = market_data_engine

    def build_corr_matrix(self, isin_ticker_map, years=3, force_refresh=False):
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
