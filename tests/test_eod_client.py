"""
Tests for EODClient — all HTTP calls are mocked.
"""
import pytest
from unittest.mock import patch, MagicMock
from src.eod_client import EODClient


@pytest.fixture
def client():
    return EODClient(api_key="test_key")


def _mock_response(status_code=200, json_data=None):
    mock = MagicMock()
    mock.status_code = status_code
    mock.json.return_value = json_data or {}
    mock.text = "error"
    return mock


class TestGetLastQuote:
    def test_returns_correct_fields(self, client):
        payload = {"close": "95.50", "timestamp": 1704067200}
        with patch("requests.get", return_value=_mock_response(json_data=payload)):
            result = client.get_last_quote("NESN.SW")

        assert result["ticker"] == "NESN.SW"
        assert result["price"] == 95.50
        assert result["source"] == "EOD"
        assert result["date"] is not None

    def test_price_is_float(self, client):
        payload = {"close": "123.456", "timestamp": 1704067200}
        with patch("requests.get", return_value=_mock_response(json_data=payload)):
            result = client.get_last_quote("NESN.SW")
        assert isinstance(result["price"], float)

    def test_http_error_raises_value_error(self, client):
        with patch("requests.get", return_value=_mock_response(status_code=403)):
            with pytest.raises(ValueError, match="HTTP error 403"):
                client.get_last_quote("NESN.SW")

    def test_missing_close_raises_value_error(self, client):
        payload = {"open": 95.0}  # no "close" key
        with patch("requests.get", return_value=_mock_response(json_data=payload)):
            with pytest.raises(ValueError, match="No price returned"):
                client.get_last_quote("NESN.SW")

    def test_no_timestamp_gives_none_date(self, client):
        payload = {"close": "95.0"}  # no timestamp
        with patch("requests.get", return_value=_mock_response(json_data=payload)):
            result = client.get_last_quote("NESN.SW")
        assert result["date"] is None

    def test_correct_url_constructed(self, client):
        payload = {"close": "95.0", "timestamp": 1704067200}
        with patch("requests.get", return_value=_mock_response(json_data=payload)) as mock_get:
            client.get_last_quote("NESN.SW")
        call_url = mock_get.call_args[0][0]
        assert "NESN.SW" in call_url
        assert "real-time" in call_url

    def test_api_key_sent_in_params(self, client):
        payload = {"close": "95.0", "timestamp": 1704067200}
        with patch("requests.get", return_value=_mock_response(json_data=payload)) as mock_get:
            client.get_last_quote("NESN.SW")
        params = mock_get.call_args[1]["params"]
        assert params["api_token"] == "test_key"
