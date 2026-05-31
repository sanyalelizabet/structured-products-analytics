"""Tests for :class:`SNBClient` — SARON parsing from the SNB zirepo cube."""
from __future__ import annotations

import pandas as pd
import pytest
import requests
from unittest.mock import MagicMock

from src.exceptions import DataUnavailableError, NetworkError
from src.market_data.snb_client import SNBClient


# A faithful slice of the SNB ``;``-delimited CSV: metadata header, a blank
# line, the column header, then data rows — including blank (non-trading-day)
# values that must be skipped.
_CSV = (
    '"CubeId";"zirepo"\n'
    '"PublishingDate";"2026-05-21 09:00"\n'
    "\n"
    '"Date";"D0";"Value"\n'
    '"2026-05-13";"H0";"-0.053133"\n'
    '"2026-05-13";"H7";"-0.0479"\n'
    '"2026-05-14";"H0";\n'              # holiday → blank, must be ignored
    '"2026-05-14";"H7";\n'
    '"2026-05-15";"H0";"-0.049175"\n'
    '"2026-05-15";"H6";"-0.0432"\n'
    '"2026-05-15";"H7";"-0.0482"\n'
    '"2026-05-15";"H8";"-0.0482"\n'
)


def _patch_response(monkeypatch, text=None, exc=None):
    resp = MagicMock()
    resp.text = text
    resp.raise_for_status.return_value = None

    def fake_get(url, timeout=None):
        if exc is not None:
            raise exc
        fake_get.url = url
        return resp

    monkeypatch.setattr(requests, "get", fake_get)
    return fake_get


class TestSNBClient:
    def test_parses_latest_non_empty_per_tenor(self, monkeypatch):
        _patch_response(monkeypatch, text=_CSV)
        out = SNBClient().get_saron_rates()
        assert set(out) == {"ON", "1M", "3M", "6M"}
        # Latest non-empty date wins (15 May over 13 May; 14 May blank skipped).
        assert out["ON"]["date"] == pd.Timestamp("2026-05-15")
        assert out["3M"]["date"] == pd.Timestamp("2026-05-15")

    def test_percent_converted_to_decimal(self, monkeypatch):
        _patch_response(monkeypatch, text=_CSV)
        out = SNBClient().get_saron_rates()
        assert out["3M"]["yield_pct"] == pytest.approx(-0.0482)
        assert out["3M"]["yield"] == pytest.approx(-0.0482 / 100.0)

    def test_requests_the_four_saron_series(self, monkeypatch):
        fake_get = _patch_response(monkeypatch, text=_CSV)
        SNBClient().get_saron_rates()
        assert "dimSel=D0(H0,H6,H7,H8)" in fake_get.url
        assert "zirepo/data/csv/en" in fake_get.url

    def test_network_failure_raises(self, monkeypatch):
        _patch_response(monkeypatch, exc=requests.ConnectionError("boom"))
        with pytest.raises(NetworkError):
            SNBClient().get_saron_rates()

    def test_all_blank_raises_data_unavailable(self, monkeypatch):
        blank = (
            '"CubeId";"zirepo"\n'
            '"Date";"D0";"Value"\n'
            '"2026-05-14";"H0";\n'
            '"2026-05-14";"H7";\n'
        )
        _patch_response(monkeypatch, text=blank)
        with pytest.raises(DataUnavailableError):
            SNBClient().get_saron_rates()
