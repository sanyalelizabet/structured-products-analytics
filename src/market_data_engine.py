import pandas as pd
from pathlib import Path
from pandas.tseries.offsets import BDay

class MarketDataEngine:

    MASTER_COLUMNS = ["isin", "ticker", "code", "exchange", "name", "type", "country", "currency"]

    def __init__(self, client, db_path="data/prices.csv"):
        self.client = client
        self.db_path = Path(db_path)
        self.master_path = self.db_path.parent / "securities_master_data.csv"



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

    def fetch_securities_master(self, isins, force_refresh=False):
        """
        Download master data for each ISIN and store in securities_master_data.csv.

        For each ISIN, calls search_by_isin and appends ALL exchange listings —
        one row per (isin, exchange) pair.  This means a security cross-listed
        on multiple exchanges will have multiple rows.

        Parameters
        ----------
        isins         : list[str]  list of ISINs
        force_refresh : bool       re-download even if ISIN already in CSV

        Returns
        -------
        pd.DataFrame  contents of securities_master_data.csv after update
        """
        if self.master_path.exists():
            master = pd.read_csv(self.master_path)
        else:
            master = pd.DataFrame(columns=self.MASTER_COLUMNS)

        rows = []

        for isin in isins:
            if not force_refresh and isin in master["isin"].values:
                continue

            try:
                listings = self.client.search_by_isin(isin)
                print(listings)

                if not listings:
                    print(f"No listings found for {isin}")
                    continue

                rows.extend(listings)

            except Exception as e:
                print(f"Master data fetch failed for {isin}: {e}")

        if rows:
            new_df = pd.DataFrame(rows)
            master = pd.concat([master, new_df], ignore_index=True)
            master = master.sort_values(["isin", "exchange"]).reset_index(drop=True)
            self.master_path.parent.mkdir(parents=True, exist_ok=True)
            master.to_csv(self.master_path, index=False)

        return master


