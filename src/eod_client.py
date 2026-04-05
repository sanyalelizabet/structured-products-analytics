import requests
from datetime import datetime

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