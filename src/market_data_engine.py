import logging

import numpy as np
import pandas as pd
from pathlib import Path
from pandas.tseries.offsets import BDay

log = logging.getLogger(__name__)

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

        unique_isins = {
            isin
            for _, row in portfolio.iterrows()
            for isin in row["underlying_isins"]
        }

        for isin in unique_isins:
            try:
                ticker = self._resolve_ticker(isin)
            except ValueError as e:
                log.warning("Skipping price fetch for %s: %s", isin, e)
                continue
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
                log.warning("Price fetch failed for %s (%s): %s", ticker, isin, e)
                continue

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
                log.warning("Monthly fetch failed for %s (%s): %s", ticker, isin, e)

        if rows:
            new_df = pd.DataFrame(rows)
            db = pd.concat([db, new_df], ignore_index=True)
            db = db.drop_duplicates(subset=["isin", "date"], keep="last")
            db = db.sort_values(["isin", "date"]).reset_index(drop=True)
            self.save_db(db)

        return db

    def fetch_daily_prices(self, isin_ticker_map, years=6, force_refresh=False):
        """
        Download daily adjusted-close prices and append to prices.csv.

        Parameters
        ----------
        isin_ticker_map : dict  { isin: ticker }
        years           : int   calendar years of history
        force_refresh   : bool  re-download even if data already exists

        Returns
        -------
        pd.DataFrame  full DB after update
        """
        db = self.load_db()
        rows = []

        for isin, ticker in isin_ticker_map.items():

            existing = db[db["isin"] == isin].copy()

            if not existing.empty:
                existing["date"] = pd.to_datetime(existing["date"])

            # Skip only if data covers the requested history window adequately.
            # Row-count alone is misleading: sparse daily quotes or old monthly
            # data can easily exceed 252 rows without covering the full window.
            if not force_refresh and not existing.empty:
                required_start = pd.Timestamp.today() - pd.DateOffset(years=years) - pd.Timedelta(days=30)
                has_history = existing["date"].min() <= required_start
                has_density = len(existing) >= years * 200  # ~200 trading days/year
                if has_history and has_density:
                    continue

            try:
                data = self.client.get_daily_prices(ticker, years=years)

                for r in data:
                    rows.append({
                        "date": pd.to_datetime(r["date"]),
                        "isin": isin,
                        "ticker": ticker,
                        "price": r["adjusted_close"],
                    })

            except Exception as e:
                log.warning("Daily fetch failed for %s (%s): %s", ticker, isin, e)

        if rows:
            new_df = pd.DataFrame(rows)

            db = pd.concat([db, new_df], ignore_index=True)
            db["date"] = pd.to_datetime(db["date"]).dt.normalize()

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
                    log.info("No listings found for %s", isin)
                    continue

                rows.extend(listings)

            except Exception as e:
                log.warning("Master data fetch failed for %s: %s", isin, e)

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
            try:
                ticker = self._resolve_ticker(isin).split(".")[0]
            except ValueError as e:
                log.info("Skipping options fetch for %s: %s", isin, e)
                continue

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
                log.info("Options fetched: %s (%d rows)", ticker, len(df))

            except Exception as e:
                log.warning("Options fetch failed for %s (%s): %s", ticker, isin, e)

        if frames:
            try:
                new_df = pd.concat(frames, ignore_index=True)
                log.debug("Columns in fetched data: %s", list(new_df.columns))
                new_df = new_df[self.OPTIONS_COLUMNS]

                # Drop stale rows for the same ISINs fetched on a previous day
                if not options_db.empty:
                    options_db = options_db[
                        ~options_db["isin"].isin(new_df["isin"].unique())
                    ]

                options_db = pd.concat([options_db, new_df], ignore_index=True)
                for col in ["volume", "open_interest"]:
                    options_db[col] = options_db[col].astype("Int64")
                options_db = options_db.sort_values(["isin", "expiry", "type", "strike"]).reset_index(drop=True)
                self.options_path.parent.mkdir(parents=True, exist_ok=True)
                options_db.to_csv(self.options_path, index=False)
                log.info("Options saved to %s (%d rows)", self.options_path, len(options_db))
            except Exception as e:
                log.error("Failed to save options: %s", e)

        return options_db


    def build_atm_vol_map(self, portfolio, fallback_vol=0.15):
        """
        Build a { isin: atm_implied_vol } map from stored options.csv and prices.csv.

        For each ISIN the expiry is matched to the product's maturity_date —
        the available option expiry closest to maturity is used, so the vol
        reflects the correct point on the term structure.

        If an ISIN appears in multiple products (different maturities), the
        longest maturity wins — conservative choice for a portfolio view.

        Parameters
        ----------
        portfolio    : pd.DataFrame  portfolio rows, must have underlying_isins
                                     and maturity_date columns
        fallback_vol : float         used if no options data exists for an ISIN

        Returns
        -------
        dict  { isin: float }  — drop-in replacement for the static vol_map
        """
        if not self.options_path.exists():
            raise ValueError("options.csv not found — run fetch_options_chain first")

        options_db = pd.read_csv(self.options_path)
        db = self.load_db()

        # Build isin -> maturity map (take longest maturity if ISIN appears in multiple products)
        isin_maturity = {}
        for _, row in portfolio.iterrows():
            maturity = pd.Timestamp(row["maturity_date"])
            for isin in row["underlying_isins"]:
                if isin not in isin_maturity or maturity > isin_maturity[isin]:
                    isin_maturity[isin] = maturity

        vol_map = {}

        for isin, maturity in isin_maturity.items():
            opts = options_db[options_db["isin"] == isin]
            if opts.empty:
                vol_map[isin] = fallback_vol
                continue

            prices = db[db["isin"] == isin].sort_values("date")
            if prices.empty:
                vol_map[isin] = fallback_vol
                continue

            spot = float(prices.iloc[-1]["price"])

            # Pick expiry closest to product maturity
            available = pd.to_datetime(opts["expiry"].unique())
            closest_expiry = min(available, key=lambda d: abs((d - maturity).days))

            chain = opts[opts["expiry"] == closest_expiry.strftime("%Y-%m-%d")].copy()
            chain["distance"] = (chain["strike"] - spot).abs()
            atm = chain.loc[chain["distance"].idxmin()]

            vol_map[isin] = float(atm["iv"])

        return vol_map

    def build_realised_vol_map(self, portfolio, window=252, fallback_vol=0.15):
        """
        Build a { isin: realised_vol } map from daily prices in prices.csv.

        σ = std(daily log returns, window) × √252

        Parameters
        ----------
        portfolio    : pd.DataFrame  must have underlying_isins column
        window       : int           rolling window in trading days (default 252 = 1 year)
        fallback_vol : float         used if insufficient price history

        Returns
        -------
        dict  { isin: float }  — drop-in replacement for the static vol_map
        """
        db = self.load_db()
        db["date"] = pd.to_datetime(db["date"])

        isins = list({isin for _, row in portfolio.iterrows() for isin in row["underlying_isins"]})

        vol_map = {}

        for isin in isins:
            prices = db[db["isin"] == isin].sort_values("date")

            if len(prices) < window // 2:
                log.warning("Insufficient price history for %s — using fallback vol", isin)
                vol_map[isin] = fallback_vol
                continue

            log_returns = np.log(prices["price"] / prices["price"].shift(1)).dropna()

            realized_var = (log_returns ** 2).mean() * 252
            realized_vol = np.sqrt(realized_var)

            if realized_vol <= 0 or np.isnan(realized_vol):
                vol_map[isin] = fallback_vol
            else:
                vol_map[isin] = round(realized_vol, 4)

        return vol_map

    def build_isin_ticker_map(self, isins):
        """
        Build { isin: ticker } from securities_master_data.csv for a list of ISINs.
        Uses the same priority logic as _resolve_ticker (US listings first).

        Parameters
        ----------
        isins : list[str]

        Returns
        -------
        dict  { isin: ticker }  — only includes ISINs found in master data
        """
        if not self.master_path.exists():
            raise ValueError("Security master not initialized")

        master = pd.read_csv(self.master_path)
        result = {}

        for isin in isins:
            try:
                result[isin] = self._resolve_ticker(isin)
            except ValueError:
                pass

        return result









