import pandas as pd
import numpy as np
from pathlib import Path
from pandas.tseries.offsets import BDay

class MarketDataEngine:

    def __init__(self, client, db_path="data/prices.csv"):
        self.client = client
        self.db_path = Path(db_path)

    def load_db(self):
        if self.db_path.exists():
            return pd.read_csv(self.db_path, parse_dates=["date"])
        return pd.DataFrame(columns=["date", "isin", "ticker", "price"])

    def save_db(self, df):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.db_path, index=False)

    def fetch_latest_prices(self, portfolio):
        db = self.load_db()
        prev_trading_day = (pd.Timestamp.today() - BDay(1)).normalize()

        rows = []

        unique_pairs = set()
        for _, row in portfolio.iterrows():
            unique_pairs.update(zip(row["underlying_isins"], row["tickers"]))

        for isin, ticker in unique_pairs:
            try:
                # ----------------------------------
                # 1. Skip API call if previous trading day already exists
                # ----------------------------------
                exists_prev_day = (
                    (db["isin"] == isin) &
                    (db["date"] == prev_trading_day)
                ).any()

                if exists_prev_day:
                    continue

                # ----------------------------------
                # 2. Fetch quote
                # ----------------------------------
                quote = self.client.get_last_quote(ticker)
                quote_date = pd.to_datetime(quote["date"]).normalize()

                # ----------------------------------
                # 3. Check whether quoted date already exists
                # ----------------------------------
                exists_quote_date = (
                    (db["isin"] == isin) &
                    (db["date"] == quote_date)
                ).any()

                if exists_quote_date:
                    continue

                rows.append({
                    "date": quote_date,
                    "isin": isin,
                    "ticker": ticker,
                    "price": quote["price"]
                })

            except Exception as e:
                break

        new_df = pd.DataFrame(rows)

        if not new_df.empty:
            db = pd.concat([db, new_df], ignore_index=True)
            db = db.drop_duplicates(subset=["isin", "date"], keep="last")
            self.save_db(db)

        return db
    
    def update_spots(self, portfolio):
        portfolio = portfolio.copy()
        db = self.load_db()
    
        for i, row in portfolio.iterrows():
            new_spots = []
    
            for isin in row["underlying_isins"]:
                prices = db[db["isin"] == isin].sort_values("date")
    
                if prices.empty:
                    raise ValueError(f"No stored price found for ISIN {isin}")
    
                latest_price = prices.iloc[-1]["price"]
                new_spots.append(latest_price)
    
            portfolio.at[i, "current_spots"] = new_spots
    
        return portfolio

    def fetch_monthly_prices(self, isin_ticker_map, years=6, force_refresh=False):
        """
        Download monthly adjusted-close prices and append to prices.csv.
        Same schema as daily rows — month-end dates coexist naturally.

        Parameters
        ----------
        isin_ticker_map : dict  { isin: ticker }
        years           : int   calendar years of history (default 4)
        force_refresh   : bool  re-download even if data already exists

        Returns
        -------
        pd.DataFrame  full DB after update
        """
        db = self.load_db()

        rows = []

        for isin, ticker in isin_ticker_map.items():
            
            existing = db[db["isin"] == isin]
            if not force_refresh and len(existing) > 36:
                continue

            try:
                data = self.client.get_monthly_prices(ticker, years=years)

                for r in data:
                    rows.append({
                        "date"  : pd.to_datetime(r["date"]),
                        "isin"  : isin,
                        "ticker": ticker,
                        "price" : r["adjusted_close"],
                    })

            except Exception as e:
                print(f"Monthly fetch failed for {ticker} ({isin}): {e}")

        if rows:
            new_df = pd.DataFrame(rows)
            db = pd.concat([db, new_df], ignore_index=True)
            db = db.drop_duplicates(subset=["isin", "date"], keep="last")
            db = db.sort_values(["isin", "date"]).reset_index(drop=True)
            self.save_db(db)

        return db

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
                                   (Airbnb from Dec-2020 constrains the window)
        5. log returns           — log(P_t / P_{t-1})
        6. .corr()               — Pearson correlation matrix

        Parameters
        ----------
        isin_ticker_map : dict  { isin: ticker }
        years           : int   years of history (default 4)
        force_refresh   : bool  re-download monthly prices

        Returns
        -------
        pd.DataFrame  correlation matrix, ISIN as both index and columns
                      → pass directly to ScenarioEngine(corr_df=...)
        """
        db = self.fetch_monthly_prices(
            isin_ticker_map, years=years, force_refresh=force_refresh
        )

        # Keep only the requested ISINs
        db = db[db["isin"].isin(isin_ticker_map.keys())]

        # Pivot: rows = date, columns = isin
        prices_wide = (
            db.pivot_table(index="date", columns="isin", values="price", aggfunc="last")
            .sort_index()
        )

        # Resample to month-end so daily and monthly rows align on the same grid
        prices_wide = prices_wide.resample("ME").last()

        # Drop any month where at least one ISIN has no price
        prices_wide = prices_wide.dropna(axis=0, how="any")

        n_obs    = len(prices_wide)
        n_assets = len(isin_ticker_map)

        if n_obs < max(12, n_assets * 3):
            raise ValueError(
                f"Only {n_obs} common monthly observations for {n_assets} assets — "
                f"need at least {max(12, n_assets * 3)}. "
                "Try increasing years= or check data availability."
            )

        # Monthly log returns → Pearson correlation
        log_returns = np.log(prices_wide / prices_wide.shift(1)).dropna()
        corr_df     = log_returns.corr()
        
        

        return corr_df

    
    
    
    