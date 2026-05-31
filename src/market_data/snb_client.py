"""Thin wrapper around the Swiss National Bank open-data API (SARON rates).

Mirrors :class:`EODClient` / :class:`FrankfurterClient`: one job — fetch the
``zirepo`` cube and translate failures into the typed exceptions in
:mod:`src.exceptions`.  Holds no persistence logic.

The ``zirepo`` cube publishes the SARON complex.  We read the overnight rate and
the compounded term rates, which together form the CHF short-end risk-free
curve:

======  ============================  =====
series  meaning                       tenor
======  ============================  =====
H0      Overnight SARON               ON
H6      SARON 1M Compound Rate        1M
H7      SARON 3M Compound Rate        3M
H8      SARON 6M Compound Rate        6M
======  ============================  =====

SNB publishes the values in **percent** (e.g. ``-0.0473`` → −0.0473 %); this
client converts them to decimals (``yield = value / 100``) to match the
convention used by the EOD bond-yield path.
"""
from __future__ import annotations

import csv
import io
import logging
from datetime import date, timedelta

import pandas as pd
import requests

from src.exceptions import DataUnavailableError, NetworkError

log = logging.getLogger(__name__)


class SNBClient:
    BASE_URL = "https://data.snb.ch/api/cube/zirepo"

    # Tenor → SNB series id within dimension D0.
    SARON_SERIES: dict[str, str] = {
        "ON": "H0",   # Overnight SARON, close of trading
        "1M": "H6",   # SARON 1M Compound Rate
        "3M": "H7",   # SARON 3M Compound Rate
        "6M": "H8",   # SARON 6M Compound Rate
    }

    def __init__(self, timeout: int = 15):
        self.timeout = timeout
        # Reverse lookup: series id → tenor.
        self._series_to_tenor = {v: k for k, v in self.SARON_SERIES.items()}

    def get_saron_rates(
        self, from_date: date | None = None, to_date: date | None = None,
    ) -> dict[str, dict]:
        """Return the latest available SARON rate per tenor.

        A trailing window is requested (default: the last 30 days) and the most
        recent **non-empty** observation is kept per series, so weekends and
        holidays — which the cube returns as blank values — never mask the last
        real fixing.

        Returns
        -------
        dict
            ``{tenor: {"date": pd.Timestamp, "yield_pct": float, "yield": float}}``
            for each tenor that had at least one observation in the window.

        Raises
        ------
        NetworkError
            On transport failure.
        DataUnavailableError
            When the response carries no usable observation for any tenor.
        """
        to_date = to_date or date.today()
        from_date = from_date or (to_date - timedelta(days=30))

        codes = ",".join(self.SARON_SERIES.values())
        url = (
            f"{self.BASE_URL}/data/csv/en"
            f"?dimSel=D0({codes})"
            f"&fromDate={from_date.isoformat()}&toDate={to_date.isoformat()}"
        )

        try:
            resp = requests.get(url, timeout=self.timeout)
            resp.raise_for_status()
        except requests.RequestException as e:
            raise NetworkError("SNB SARON fetch failed") from e

        latest = self._parse_csv(resp.text)
        if not latest:
            raise DataUnavailableError(
                "SNB returned no usable SARON observations in the requested window"
            )
        return latest

    def _parse_csv(self, text: str) -> dict[str, dict]:
        """Parse the SNB ``;``-delimited CSV into the latest rate per tenor.

        The payload begins with metadata lines (``CubeId``, ``PublishingDate``,
        a blank line) followed by a ``Date;D0;Value`` header and the data rows.
        Blank values are skipped; values are percent and converted to decimals.
        """
        reader = csv.reader(io.StringIO(text), delimiter=";")
        latest: dict[str, dict] = {}
        in_data = False
        for parts in reader:
            if not parts:
                continue
            if not in_data:
                # The data section starts right after the "Date;D0;Value" header.
                if parts[0].strip().strip('"') == "Date":
                    in_data = True
                continue

            if len(parts) < 3:
                continue
            raw_date, series, raw_val = (p.strip().strip('"') for p in parts[:3])
            if not raw_val:                      # blank → non-trading day
                continue
            tenor = self._series_to_tenor.get(series)
            if tenor is None:
                continue

            try:
                yield_pct = float(raw_val)
                obs_date = pd.Timestamp(raw_date)
            except (ValueError, TypeError):
                continue

            prev = latest.get(tenor)
            if prev is None or obs_date > prev["date"]:
                latest[tenor] = {
                    "date":      obs_date,
                    "yield_pct": yield_pct,
                    "yield":     yield_pct / 100.0,
                }
        return latest
