import pandas as pd
from pathlib import Path
from pandas.tseries.offsets import BDay

class MarketDataEngine:

    MASTER_COLUMNS = ["isin", "ticker", "code", "exchange", "name", "type", "country", "currency"]

    OPTIONS_COLUMNS = ["fetch_date", "isin", "ticker", "expiry", "type",
                       "strike", "last_price", "bid", "ask", "iv",
                       "volume", "open_interest"]

    def __init__(self, client, db_path="data/prices.csv"):
        self.client = client
        self.db_path = Path(db_path)
        self.master_path = self.db_path.parent / "securities_master_data.csv"
        self.options_path = self.db_path.parent / "options.csv"



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

    def _resolve_ticker(self, isin):

        if not self.master_path.exists():
            raise ValueError("Security master not initialized")

        master = pd.read_csv(self.master_path)
        matches = master[master["isin"] == isin]

        if matches.empty:
            raise ValueError(f"{isin} not found in master data")

        # Priority: US first
        us = matches[matches["country"].str.lower().isin(
            ["united states", "usa", "us"]
        )]
        if not us.empty:
            return us.iloc[0]["ticker"]

        # fallback
        return matches.iloc[0]["ticker"]

    def fetch_options_chain(self, isins, yahoo_client, force_refresh=False):
        """
        Download the full options chain for each ISIN using YahooClient.
        Tickers are resolved from securities_master_data.csv via _resolve_ticker.
        Results are stored in options.csv.

        Parameters
        ----------
        isins         : list[str]    ISINs to fetch options for
        yahoo_client  : YahooClient  Yahoo Finance client instance
        force_refresh : bool         re-download even if already fetched today

        Returns
        -------
        pd.DataFrame  contents of options.csv after update
        """

        if self.options_path.exists():
            options_db = pd.read_csv(self.options_path, parse_dates=["fetch_date"])
        else:
            options_db = pd.DataFrame(columns=self.OPTIONS_COLUMNS)

        today = pd.Timestamp.today().normalize()
        frames = []

        for isin in isins:
            ticker = self._resolve_ticker(isin)
            ticker = ticker.split(".")[0]

            already_fetched_today = (
                    not options_db.empty
                    and (
                            (options_db["isin"] == isin) &
                            (options_db["fetch_date"] == today)
                    ).any()
            )

            if not force_refresh and already_fetched_today:
                continue

            try:
                df = yahoo_client.get_full_chain(ticker)
                df["isin"] = isin
                df["fetch_date"] = today
                frames.append(df)
                print(f"Options fetched: {ticker} ({len(df)} rows)")

            except Exception as e:
                print(f"Options fetch failed for {ticker} ({isin}): {e}")

        if frames:
            new_df = pd.concat(frames, ignore_index=True)
            new_df = new_df[self.OPTIONS_COLUMNS]

            # Drop stale rows for the same ISINs fetched on a previous day
            if not options_db.empty:
                options_db = options_db[
                    ~options_db["isin"].isin(new_df["isin"].unique())
                ]

            options_db = pd.concat([options_db, new_df], ignore_index=True)
            options_db = options_db.sort_values(["isin", "expiry", "type", "strike"]).reset_index(drop=True)
            self.options_path.parent.mkdir(parents=True, exist_ok=True)
            options_db.to_csv(self.options_path, index=False)

        return options_db








