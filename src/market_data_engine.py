import pandas as pd
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