"""Thin wrapper around the Frankfurter FX API (ECB reference rates).

Mirrors :class:`EODClient` / :class:`YahooClient`: one job — make the request
and translate failures into the typed exceptions in :mod:`src.exceptions`, so
the data layer can distinguish "network down" from "bad/empty response". Holds
no persistence logic. Frankfurter is keyless and single-endpoint; swapping FX
providers means writing another client with the same ``get_latest_rates``
interface.
"""
from __future__ import annotations

import logging

import requests

from src.exceptions import DataUnavailableError, NetworkError

log = logging.getLogger(__name__)


class FrankfurterClient:
    BASE_URL = "https://api.frankfurter.app"

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
