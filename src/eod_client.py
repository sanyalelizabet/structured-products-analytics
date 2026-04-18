import requests
from datetime import datetime
from datetime import timedelta
class EODClient:
    BASE_URL = "https://eodhistoricaldata.com/api"

    def __init__(self, api_key):
        self.api_key = api_key

    def get_last_quote(self, ticker):
        url = f"{self.BASE_URL}/real-time/{ticker}"
        params = {
            "api_token": self.api_key,
            "fmt": "json"
        }

        response = requests.get(url, params=params)

        if response.status_code != 200:
            raise ValueError(f"HTTP error {response.status_code}: {response.text}")

        data = response.json()

        if "close" not in data:
            raise ValueError(f"No price returned for {ticker}: {data}")

        ts = data.get("timestamp")
        market_timestamp = datetime.utcfromtimestamp(ts) if ts else None

        return {
            "ticker": ticker,
            "price": float(data["close"]),
            "date": market_timestamp,
            "source": "EOD"
        }
    
    
    def get_monthly_prices(self, ticker, years=4):
        """
        Download monthly adjusted-close prices for a ticker.

        Parameters
        ----------
        ticker : str   e.g. "ALC.SW", "ABNB"
        years  : int   how many calendar years of history to fetch (default 4)

        Returns
        -------
        list[dict]  each entry: { "date": "YYYY-MM-DD", "adjusted_close": float }
        """
        to_date   = datetime.today()
        from_date = to_date - timedelta(days=int(years * 365.25))

        url = f"{self.BASE_URL}/eod/{ticker}"
        params = {
            "api_token": self.api_key,
            "period"   : "m",
            "from"     : from_date.strftime("%Y-%m-%d"),
            "to"       : to_date.strftime("%Y-%m-%d"),
            "fmt"      : "json",
        }

        response = requests.get(url, params=params)

        if response.status_code != 200:
            raise ValueError(f"HTTP {response.status_code} for {ticker}: {response.text}")

        data = response.json()

        if not data:
            raise ValueError(f"No historical data returned for {ticker}")

        return [
            {
                "date"          : row["date"],
                "adjusted_close": float(row["adjusted_close"]),
            }
            for row in data
            if row.get("adjusted_close") is not None
        ]



    def search_by_isin(self, isin):
        """
        Uses Search API (ISIN supported).
        Returns ALL listings (not guaranteed complete).
        """
        url = f"{self.BASE_URL}/search/{isin}"
        params = {"api_token": self.api_key, "fmt": "json"}

        r = requests.get(url, params=params)

        if r.status_code != 200:
            raise ValueError(f"Search failed {isin}: {r.text}")

        results = r.json()

        listings = []

        for r in results:
            code = r.get("Code")
            exch = r.get("Exchange")

            if not code or not exch:
                continue

            ticker = f"{code}.{exch}"

            listings.append({
                "isin": isin,
                "ticker": ticker,
                "code": code,
                "exchange": exch,
                "name": r.get("Name"),
                "type": r.get("Type"),
                "country": r.get("Country"),
                "currency": r.get("Currency"),
            })

        return listings

