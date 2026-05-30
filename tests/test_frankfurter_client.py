"""Tests for FrankfurterClient — HTTP mocked."""
from datetime import date
from unittest.mock import patch

import pytest
import requests

from src.frankfurter_client import FrankfurterClient
from src.exceptions import DataUnavailableError, NetworkError


def _resp(payload):
    class _R:
        def raise_for_status(self):
            pass

        def json(self):
            return payload
    return _R()


class TestGetLatestRates:
    def test_returns_payload(self):
        payload = {"base": "CHF", "date": "2026-05-22", "rates": {"USD": 1.27}}
        with patch("requests.get", return_value=_resp(payload)):
            out = FrankfurterClient().get_latest_rates("CHF")
        assert out["rates"]["USD"] == 1.27

    def test_transport_failure_raises_network_error(self):
        with patch("requests.get", side_effect=requests.RequestException("down")):
            with pytest.raises(NetworkError):
                FrankfurterClient().get_latest_rates("CHF")

    def test_missing_rates_key_raises_data_unavailable(self):
        with patch("requests.get", return_value=_resp({"base": "CHF"})):
            with pytest.raises(DataUnavailableError):
                FrankfurterClient().get_latest_rates("CHF")

    def test_empty_rates_raises_data_unavailable(self):
        with patch("requests.get", return_value=_resp({"rates": {}})):
            with pytest.raises(DataUnavailableError):
                FrankfurterClient().get_latest_rates("CHF")


class TestGetHistory:
    _HIST = {
        "amount": 1.0, "base": "USD",
        "start_date": "2025-01-02", "end_date": "2025-01-03",
        "rates": {
            "2025-01-02": {"CHF": 0.91, "EUR": 0.97},
            "2025-01-03": {"CHF": 0.92, "EUR": 0.96},
        },
    }

    def test_returns_long_form_dataframe(self):
        with patch("requests.get", return_value=_resp(self._HIST)) as g:
            df = FrankfurterClient().get_history(
                "USD", date(2025, 1, 2), date(2025, 1, 3),
            )
        # Columns + length: 2 dates × 2 quotes = 4 rows.
        assert list(df.columns) == ["date", "base", "quote", "rate"]
        assert len(df) == 4
        assert set(df["base"]) == {"USD"}
        assert set(df["quote"]) == {"CHF", "EUR"}
        # Canonical post-redirect URL used directly (avoids the 301 hop).
        called_url = g.call_args[0][0]
        assert "frankfurter.dev/v1/2025-01-02..2025-01-03" in called_url

    def test_transport_failure_raises_network_error(self):
        with patch("requests.get", side_effect=requests.RequestException("down")):
            with pytest.raises(NetworkError):
                FrankfurterClient().get_history(
                    "USD", date(2025, 1, 2), date(2025, 1, 3),
                )

    def test_empty_rates_raises_data_unavailable(self):
        with patch("requests.get", return_value=_resp({"rates": {}})):
            with pytest.raises(DataUnavailableError):
                FrankfurterClient().get_history(
                    "USD", date(2025, 1, 2), date(2025, 1, 3),
                )
