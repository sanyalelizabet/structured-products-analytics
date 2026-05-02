"""Thin wrapper around the EOD Historical Data REST API.

All network calls go through :func:`_request_json`, which enforces a timeout,
translates transport errors into typed exceptions (see :mod:`src.exceptions`),
and is wrapped with a tenacity retry decorator that retries only transient
failures (``NetworkError`` / ``RateLimitError``).
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.exceptions import (
    DataUnavailableError,
    NetworkError,
    RateLimitError,
)

log = logging.getLogger(__name__)

# Wall-clock timeout per HTTP call (seconds). Applies to connect + read.
_REQUEST_TIMEOUT = 10

# Retry policy: only transient errors, exponential backoff, raise the
# underlying exception once attempts are exhausted.
_RETRY = retry(
    retry=retry_if_exception_type(NetworkError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)


def _numeric_or_none(value) -> float | None:
    """Return ``float(value)`` when meaningful, else ``None``.

    EOD frequently returns the literal string ``"NA"`` (or a NaN float)
    for fields that aren't quoting yet — both must coerce to ``None``
    rather than blowing up the caller.
    """
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    return f


def _parse_eod_date(data: dict) -> datetime | None:
    """Extract a timezone-naive ``datetime`` from an EOD real-time payload.

    EOD's real-time endpoint returns ``timestamp`` as a unix epoch — but
    for some virtual exchanges (notably GBOND) the field is a *string*,
    which crashes :func:`datetime.utcfromtimestamp`.  We accept either,
    and fall back to the ``date`` string if no usable timestamp is
    present, returning ``None`` only when the payload has neither.
    """
    ts = data.get("timestamp")
    if ts is not None:
        try:
            return datetime.utcfromtimestamp(int(float(ts)))
        except (TypeError, ValueError):
            pass

    date_str = data.get("date")
    if date_str:
        try:
            return datetime.strptime(str(date_str)[:10], "%Y-%m-%d")
        except ValueError:
            pass

    return None


def _request_json(url: str, params: dict) -> object:
    """GET ``url`` and return parsed JSON, translating failures to typed errors.

    Raises
    ------
    NetworkError         transport or 5xx failure (caller may retry).
    RateLimitError       HTTP 429 (retryable with backoff).
    DataUnavailableError non-retryable 4xx or invalid JSON body.
    """
    try:
        response = requests.get(url, params=params, timeout=_REQUEST_TIMEOUT)
    except requests.Timeout as e:
        raise NetworkError(f"Timeout calling {url}") from e
    except requests.ConnectionError as e:
        raise NetworkError(f"Connection error calling {url}") from e
    except requests.RequestException as e:
        raise NetworkError(f"Request failed for {url}: {e}") from e

    status = response.status_code

    if status == 429:
        raise RateLimitError(f"Rate limited by EOD at {url}")
    if 500 <= status < 600:
        raise NetworkError(f"Server error {status} at {url}")
    if status != 200:
        raise DataUnavailableError(
            f"HTTP error {status} for {url}: {response.text[:200]}"
        )

    try:
        return response.json()
    except ValueError as e:  # JSONDecodeError is a subclass of ValueError
        raise DataUnavailableError(f"Invalid JSON from {url}: {e}") from e


class EODClient:
    BASE_URL = "https://eodhistoricaldata.com/api"

    def __init__(self, api_key: str):
        self.api_key = api_key

    @_RETRY
    def get_last_quote(self, ticker: str) -> dict:
        url = f"{self.BASE_URL}/real-time/{ticker}"
        params = {"api_token": self.api_key, "fmt": "json"}

        data = _request_json(url, params)

        if "close" not in data:
            raise DataUnavailableError(
                f"No price returned for {ticker}: {data}"
            )

        return {
            "ticker": ticker,
            "price": float(data["close"]),
            "date": _parse_eod_date(data),
            "source": "EOD",
        }

    @_RETRY
    def get_monthly_prices(self, ticker: str, years: int = 4) -> list[dict]:
        """Download monthly adjusted-close prices for ``ticker``.

        Parameters
        ----------
        ticker : str   e.g. "ALC.SW", "ABNB"
        years  : int   how many calendar years of history to fetch (default 4)

        Returns
        -------
        list[dict]  each entry: ``{"date": "YYYY-MM-DD", "adjusted_close": float}``
        """
        to_date = datetime.today()
        from_date = to_date - timedelta(days=int(years * 365.25))

        url = f"{self.BASE_URL}/eod/{ticker}"
        params = {
            "api_token": self.api_key,
            "period": "m",
            "from": from_date.strftime("%Y-%m-%d"),
            "to": to_date.strftime("%Y-%m-%d"),
            "fmt": "json",
        }

        data = _request_json(url, params)

        if not data:
            raise DataUnavailableError(
                f"No historical data returned for {ticker}"
            )

        return [
            {
                "date": row["date"],
                "adjusted_close": float(row["adjusted_close"]),
            }
            for row in data
            if row.get("adjusted_close") is not None
        ]

    @_RETRY
    def get_daily_prices(self, ticker: str, years: int = 6) -> list[dict]:
        to_date = datetime.today()
        from_date = to_date - timedelta(days=int(years * 365.25))

        url = f"{self.BASE_URL}/eod/{ticker}"
        params = {
            "api_token": self.api_key,
            "period": "d",
            "from": from_date.strftime("%Y-%m-%d"),
            "to": to_date.strftime("%Y-%m-%d"),
            "fmt": "json",
        }

        data = _request_json(url, params)

        if not data:
            raise DataUnavailableError(
                f"No daily historical data returned for {ticker}"
            )

        return [
            {
                "date": row["date"],
                "adjusted_close": float(row["adjusted_close"]),
            }
            for row in data
            if row.get("adjusted_close") is not None
        ]

    @_RETRY
    def get_bond_yield(self, ticker: str) -> dict:
        """Latest government-bond yield for a ``.GBOND`` ticker.

        EOD exposes sovereign yields under the GBOND virtual exchange,
        e.g. ``US3M.GBOND``, ``CH3M.GBOND``, ``EU3M.GBOND``, ``GB3M.GBOND``.
        The ``close`` field is the yield in **percent** (e.g. ``4.35`` for
        4.35%).

        The real-time endpoint frequently returns ``"close": "NA"`` for
        GBOND tickers outside main quote windows.  When that happens we
        fall back to the historical (EOD) endpoint and return the most
        recent valid daily close — equivalent in meaning, just one day
        delayed.

        Returns
        -------
        dict
            ``{"ticker": str, "yield_pct": float, "date": datetime, "source": "EOD"}``
        """
        url = f"{self.BASE_URL}/real-time/{ticker}"
        params = {"api_token": self.api_key, "fmt": "json"}

        data = _request_json(url, params)

        # Step 1 — current "close" if numeric.
        close = data.get("close") if isinstance(data, dict) else None
        yield_pct = _numeric_or_none(close)
        if yield_pct is not None:
            return {
                "ticker":    ticker,
                "yield_pct": yield_pct,
                "date":      _parse_eod_date(data),
                "source":    "EOD",
            }

        # Step 2 — fall back to the same payload's "previousClose".  Saves
        # an API call when only `close` is "NA" (typical outside quote
        # windows for GBOND virtual exchange).
        prev_close = data.get("previousClose") if isinstance(data, dict) else None
        prev_yield = _numeric_or_none(prev_close)
        if prev_yield is not None:
            log.info("Real-time %s close was %r — using previousClose %.4f",
                     ticker, close, prev_yield)
            return {
                "ticker":    ticker,
                "yield_pct": prev_yield,
                "date":      _parse_eod_date(data),
                "source":    "EOD (previousClose)",
            }

        # Step 3 — historical endpoint.  Most reliable path when the
        # virtual exchange is quiet; raises DataUnavailableError if the
        # ticker isn't carried at all (HTTP 404 from /eod/...).
        log.info("Real-time %s had no usable yield — falling back to history",
                 ticker)
        history = self.get_bond_yield_history(ticker, years=1)
        if not history:
            raise DataUnavailableError(
                f"No real-time or historical yield available for {ticker}"
            )

        latest = history[-1]
        return {
            "ticker":    ticker,
            "yield_pct": float(latest["yield_pct"]),
            "date":      datetime.strptime(latest["date"][:10], "%Y-%m-%d"),
            "source":    "EOD (history)",
        }

    @_RETRY
    def get_bond_yield_history(self, ticker: str, years: int = 6) -> list[dict]:
        """Daily history of the GBOND yield for ``ticker``.

        Returns a list of ``{"date": "YYYY-MM-DD", "yield_pct": float}``,
        with yield in percent (EOD convention).
        """
        to_date = datetime.today()
        from_date = to_date - timedelta(days=int(years * 365.25))

        url = f"{self.BASE_URL}/eod/{ticker}"
        params = {
            "api_token": self.api_key,
            "period": "d",
            "from": from_date.strftime("%Y-%m-%d"),
            "to": to_date.strftime("%Y-%m-%d"),
            "fmt": "json",
        }

        data = _request_json(url, params)

        if not data:
            raise DataUnavailableError(
                f"No bond yield history for {ticker}"
            )

        return [
            {"date": row["date"], "yield_pct": float(row["adjusted_close"])}
            for row in data
            if row.get("adjusted_close") is not None
        ]

    @_RETRY
    def search_by_isin(self, isin: str) -> list[dict]:
        """Search the EOD catalogue by ISIN. Returns ALL listings found."""
        url = f"{self.BASE_URL}/search/{isin}"
        params = {"api_token": self.api_key, "fmt": "json"}

        results = _request_json(url, params)

        listings = []
        for r in results:
            code = r.get("Code")
            exch = r.get("Exchange")
            if not code or not exch:
                continue

            listings.append(
                {
                    "isin": isin,
                    "ticker": f"{code}.{exch}",
                    "code": code,
                    "exchange": exch,
                    "name": r.get("Name"),
                    "type": r.get("Type"),
                    "country": r.get("Country"),
                    "currency": r.get("Currency"),
                }
            )

        return listings
