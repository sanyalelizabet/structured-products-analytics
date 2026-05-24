"""Tests for FrankfurterClient — HTTP mocked."""
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
