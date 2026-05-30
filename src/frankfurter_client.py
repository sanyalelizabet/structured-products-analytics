"""Thin wrapper around the Frankfurter FX API (ECB reference rates).

Mirrors :class:`EODClient` / :class:`YahooClient`: one job — make the request
and translate failures into the typed exceptions in :mod:`src.exceptions`, so
the data layer can distinguish "network down" from "bad/empty response". Holds
no persistence logic. Frankfurter is keyless and single-endpoint; swapping FX
providers means writing another client with the same interface.
"""
from __future__ import annotations

import logging
from datetime import date

import pandas as pd
import requests

from src.exceptions import DataUnavailableError, NetworkError

log = logging.getLogger(__name__)


class FrankfurterClient:
    # The legacy ``.app`` host 301-redirects to the canonical ``.dev/v1`` API.
    # ``get_latest_rates`` retains the old URL (still works via redirect);
    # ``get_history`` uses the canonical URL directly to avoid the extra hop.
    BASE_URL    = "https://api.frankfurter.app"
    HISTORY_URL = "https://api.frankfurter.dev/v1"

    def __init__(self, timeout: int = 10):
        self.timeout = timeout

    def get_latest_rates(self, base: str) -> dict:
        """Return the latest FX snapshot for ``base``.

        Payload shape: ``{"amount": 1.0, "base": <base>, "date": "YYYY-MM-DD",
        "rates": {quote: rate, ...}}`` where ``rate`` is units of ``quote`` per
        1 unit of ``base``.

        Raises :class:`NetworkError` on transport failure and
        :class:`DataUnavailableError` on a missing/malformed payload.
        """
        try:
            resp = requests.get(
                f"{self.BASE_URL}/latest", params={"from": base}, timeout=self.timeout
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            raise NetworkError(f"Frankfurter fetch failed for base {base}") from e

        try:
            payload = resp.json()
            rates = payload["rates"]
        except (ValueError, KeyError) as e:
            raise DataUnavailableError(
                f"Frankfurter returned no rates for base {base}"
            ) from e

        if not rates:
            raise DataUnavailableError(f"Frankfurter returned empty rates for {base}")
        return payload

    def get_history(
        self, base: str, from_date: date, to_date: date,
    ) -> pd.DataFrame:
        """Daily FX rates for ``base`` over ``[from_date, to_date]``.

        One HTTP call returns every quote currency Frankfurter publishes vs
        ``base`` (≈30 currencies), so this is the natural unit for back-filling
        the FX history.  Returned long-form for direct concatenation into the
        shared ``fx_rates.csv``: columns ``date, base, quote, rate``.

        ``rate`` follows Frankfurter's convention — units of ``quote`` per
        1 unit of ``base`` (so ``base=USD, quote=CHF, rate=0.78`` means
        1 USD = 0.78 CHF).

        Raises :class:`NetworkError` on transport failure and
        :class:`DataUnavailableError` on a missing or empty payload.
        """
        url = f"{self.HISTORY_URL}/{from_date.isoformat()}..{to_date.isoformat()}"
        try:
            resp = requests.get(url, params={"from": base}, timeout=self.timeout)
            resp.raise_for_status()
        except requests.RequestException as e:
            raise NetworkError(
                f"Frankfurter history fetch failed for base {base}"
            ) from e

        try:
            payload = resp.json()
            rates = payload["rates"]
        except (ValueError, KeyError) as e:
            raise DataUnavailableError(
                f"Frankfurter returned no history for base {base}"
            ) from e
        if not rates:
            raise DataUnavailableError(
                f"Frankfurter returned empty history for base {base}"
            )

        rows = []
        for date_str, quotes in rates.items():
            d = pd.Timestamp(date_str)
            for quote, rate in quotes.items():
                rows.append({
                    "date":  d,
                    "base":  base,
                    "quote": quote,
                    "rate":  float(rate),
                })
        return pd.DataFrame(rows, columns=["date", "base", "quote", "rate"])
