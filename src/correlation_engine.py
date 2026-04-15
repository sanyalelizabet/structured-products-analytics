import numpy as np


class CorrelationEngine:

    def __init__(self, market_data_engine):
        self.mde = market_data_engine

    def build_corr_matrix(self, isin_ticker_map, years=6, force_refresh=False):
        """
        Compute a realized pairwise correlation matrix from monthly log-returns.

        Steps
        -----
        1. fetch_monthly_prices  — download if not cached, else load from CSV
        2. Pivot wide            — rows = dates, columns = ISINs
        3. Resample monthly      — take last price per calendar month per ISIN
                                   (handles any mixed daily/monthly rows)
        4. dropna(how='any')     — align on common history window
        5. log returns           — log(P_t / P_{t-1})
        6. .corr()               — Pearson correlation matrix

        Parameters
        ----------
        isin_ticker_map : dict  { isin: ticker }
        years           : int   years of history (default 6)
        force_refresh   : bool  re-download monthly prices

        Returns
        -------
        pd.DataFrame  correlation matrix, ISIN as both index and columns
                      → pass directly to ScenarioEngine(corr_df=...)
        """
        db = self.mde.fetch_monthly_prices(
            isin_ticker_map, years=years, force_refresh=force_refresh
        )

        db = db[db["isin"].isin(isin_ticker_map.keys())]

        prices_wide = (
            db.pivot_table(index="date", columns="isin", values="price", aggfunc="last")
            .sort_index()
        )

        prices_wide = prices_wide.resample("ME").last()
        prices_wide = prices_wide.dropna(axis=0, how="any")

        n_obs    = len(prices_wide)
        n_assets = len(isin_ticker_map)

        if n_obs < max(12, n_assets * 3):
            raise ValueError(
                f"Only {n_obs} common monthly observations for {n_assets} assets — "
                f"need at least {max(12, n_assets * 3)}. "
                "Try increasing years= or check data availability."
            )

        log_returns = np.log(prices_wide / prices_wide.shift(1)).dropna()
        return log_returns.corr()
