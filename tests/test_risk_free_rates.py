"""Tests for the GBOND-driven risk-free-rate pipeline.

Covers:

* :class:`EODClient` GBOND methods (``get_bond_yield`` / history).
* :class:`MarketDataEngine` rates DB IO and the
  ``fetch_latest_rates`` / ``build_risk_free_rate_map`` round trip.
* Fallback semantics — a failing API call leaves the previous CSV row
  intact, and ``build_risk_free_rate_map`` keeps returning it.
"""
from __future__ import annotations

import pandas as pd
import pytest
from unittest.mock import MagicMock

from src.eod_client import EODClient
from src.exceptions import DataUnavailableError, NetworkError
from src.market_data_engine import MarketDataEngine


# ─────────────────────────────────────────
# EODClient — GBOND endpoints
# ─────────────────────────────────────────

class TestEodClientBondYield:
    def test_returns_yield_pct(self, monkeypatch):
        client = EODClient(api_key="dummy")
        fake = {"close": 4.35, "timestamp": 1714521600}

        def fake_request(url, params):
            assert "real-time/US3M.GBOND" in url
            return fake

        monkeypatch.setattr("src.eod_client._request_json", fake_request)
        result = client.get_bond_yield("US3M.GBOND")
        assert result["ticker"]    == "US3M.GBOND"
        assert result["yield_pct"] == pytest.approx(4.35)
        assert result["source"]    == "EOD"

    def test_missing_close_falls_back_to_history(self, monkeypatch):
        """Real-time has no close → use the most recent daily row."""
        client = EODClient(api_key="dummy")
        calls = {"i": 0}

        def fake_request(url, params):
            calls["i"] += 1
            if "real-time" in url:
                return {"timestamp": 1}                 # no close
            return [
                {"date": "2025-01-14", "adjusted_close": 4.30},
                {"date": "2025-01-15", "adjusted_close": 4.35},
            ]
        monkeypatch.setattr("src.eod_client._request_json", fake_request)
        result = client.get_bond_yield("US3M.GBOND")
        assert result["yield_pct"] == pytest.approx(4.35)
        assert result["date"].strftime("%Y-%m-%d") == "2025-01-15"

    def test_close_NA_falls_back_to_history(self, monkeypatch):
        """EOD GBOND returns close="NA" outside quote windows → history."""
        client = EODClient(api_key="dummy")

        def fake_request(url, params):
            if "real-time" in url:
                return {"close": "NA", "timestamp": 1}
            return [{"date": "2025-01-15", "adjusted_close": 4.35}]
        monkeypatch.setattr("src.eod_client._request_json", fake_request)
        result = client.get_bond_yield("US3M.GBOND")
        assert result["yield_pct"] == pytest.approx(4.35)

    def test_real_time_and_history_both_unavailable_raises(self, monkeypatch):
        client = EODClient(api_key="dummy")

        def fake_request(url, params):
            if "real-time" in url:
                return {"close": "NA"}
            raise DataUnavailableError("no history")
        monkeypatch.setattr("src.eod_client._request_json", fake_request)
        with pytest.raises(DataUnavailableError):
            client.get_bond_yield("US3M.GBOND")

    def test_string_timestamp_does_not_crash(self, monkeypatch):
        """EOD returns ``timestamp`` as a string for some virtual
        exchanges (GBOND in particular).  The client must handle both."""
        client = EODClient(api_key="dummy")
        monkeypatch.setattr("src.eod_client._request_json",
                            lambda url, params: {"close": 4.35,
                                                  "timestamp": "1714521600"})
        result = client.get_bond_yield("US3M.GBOND")
        assert result["yield_pct"] == pytest.approx(4.35)
        assert result["date"] is not None  # parsed via int(float(ts))

    def test_falls_back_to_date_field_when_no_timestamp(self, monkeypatch):
        client = EODClient(api_key="dummy")
        monkeypatch.setattr("src.eod_client._request_json",
                            lambda url, params: {"close": 4.35,
                                                  "date": "2025-01-15"})
        result = client.get_bond_yield("US3M.GBOND")
        assert result["date"].strftime("%Y-%m-%d") == "2025-01-15"

    def test_no_timestamp_no_date_returns_none(self, monkeypatch):
        client = EODClient(api_key="dummy")
        monkeypatch.setattr("src.eod_client._request_json",
                            lambda url, params: {"close": 4.35})
        result = client.get_bond_yield("US3M.GBOND")
        assert result["date"] is None

    def test_history_returns_pct(self, monkeypatch):
        client = EODClient(api_key="dummy")
        monkeypatch.setattr("src.eod_client._request_json",
                            lambda url, params: [
                                {"date": "2025-01-01", "adjusted_close": 4.30},
                                {"date": "2025-01-02", "adjusted_close": 4.32},
                            ])
        rows = client.get_bond_yield_history("US3M.GBOND", years=1)
        assert len(rows) == 2
        assert rows[0]["yield_pct"] == pytest.approx(4.30)


# ─────────────────────────────────────────
# MarketDataEngine — rates DB IO
# ─────────────────────────────────────────

@pytest.fixture
def mock_client():
    c = MagicMock()
    # Default — successful 4.35% quote dated yesterday
    c.get_bond_yield.return_value = {
        "ticker": "US3M.GBOND",
        "yield_pct": 4.35,
        "date": pd.Timestamp("2025-01-15"),
        "source": "EOD",
    }
    return c


@pytest.fixture
def engine(mock_client, tmp_path):
    return MarketDataEngine(client=mock_client, db_path=str(tmp_path / "prices.csv"))


class TestRatesDbIo:
    def test_load_empty_when_no_file(self, engine):
        df = engine.load_rates_db()
        assert df.empty
        assert list(df.columns) == MarketDataEngine.RATES_COLUMNS

    def test_save_and_load_roundtrip(self, engine):
        df = pd.DataFrame([{
            "date": pd.Timestamp("2025-01-15"),
            "currency": "USD", "tenor": "3M", "ticker": "US3M.GBOND",
            "yield_pct": 4.35, "yield": 0.0435,
        }])
        engine.save_rates_db(df)
        loaded = engine.load_rates_db()
        assert len(loaded) == 1
        assert loaded.iloc[0]["yield"] == pytest.approx(0.0435)


class TestFetchLatestRates:
    def test_fetches_and_stores(self, engine, mock_client):
        db = engine.fetch_latest_rates(["USD"])
        assert not db.empty
        assert mock_client.get_bond_yield.called
        assert db.iloc[0]["yield"] == pytest.approx(0.0435)
        assert db.iloc[0]["ticker"] == "US3M.GBOND"

    def test_skips_when_quote_date_already_in_db(self, engine, mock_client):
        engine.fetch_latest_rates(["USD"])
        mock_client.get_bond_yield.reset_mock()
        engine.fetch_latest_rates(["USD"])
        # Second call should detect the already-present row and avoid re-store.
        db = engine.load_rates_db()
        assert len(db) == 1

    def test_unmapped_currency_is_skipped(self, engine, mock_client):
        # JPY is not in GBOND_TICKERS — should warn and skip silently.
        db = engine.fetch_latest_rates(["JPY"])
        assert db.empty
        assert not mock_client.get_bond_yield.called

    def test_api_failure_does_not_corrupt_db(self, engine, mock_client):
        """If the API blows up, the DB on disk must remain whatever it was."""
        # Seed with one good row first.
        engine.save_rates_db(pd.DataFrame([{
            "date": pd.Timestamp("2025-01-10"),
            "currency": "USD", "tenor": "3M", "ticker": "US3M.GBOND",
            "yield_pct": 4.10, "yield": 0.0410,
        }]))

        mock_client.get_bond_yield.side_effect = NetworkError("EOD down")
        db = engine.fetch_latest_rates(["USD"])
        # Old row still there, no new rows.
        assert len(db) == 1
        assert db.iloc[0]["yield"] == pytest.approx(0.0410)

    def test_multiple_currencies_independent(self, engine, mock_client):
        # CHF defaults to the 10Y ticker (EOD doesn't carry CH3M).
        def fake_quote(ticker):
            yields = {"US3M.GBOND": 4.35, "CH10Y.GBOND": 0.5,
                      "EU3M.GBOND": 2.5,  "GB3M.GBOND": 4.5}
            return {
                "ticker": ticker,
                "yield_pct": yields[ticker],
                "date": pd.Timestamp("2025-01-15"),
                "source": "EOD",
            }
        mock_client.get_bond_yield.side_effect = fake_quote

        db = engine.fetch_latest_rates(["USD", "CHF", "EUR", "GBP"])
        assert set(db["currency"]) == {"USD", "CHF", "EUR", "GBP"}
        # Check that CHF ended up under the 10Y tenor.
        chf_row = db[db["currency"] == "CHF"].iloc[0]
        assert chf_row["tenor"]  == "10Y"
        assert chf_row["ticker"] == "CH10Y.GBOND"


class TestBuildRiskFreeRateMap:
    def test_empty_db_returns_empty_map(self, engine):
        assert engine.build_risk_free_rate_map(["USD"]) == {}

    def test_returns_latest_per_currency(self, engine):
        engine.save_rates_db(pd.DataFrame([
            {"date": pd.Timestamp("2025-01-10"), "currency": "USD",
             "tenor": "3M", "ticker": "US3M.GBOND",
             "yield_pct": 4.10, "yield": 0.0410},
            {"date": pd.Timestamp("2025-01-15"), "currency": "USD",
             "tenor": "3M", "ticker": "US3M.GBOND",
             "yield_pct": 4.35, "yield": 0.0435},
        ]))
        m = engine.build_risk_free_rate_map(["USD"])
        assert m == {"USD": pytest.approx(0.0435)}

    def test_omits_currency_with_no_rows(self, engine):
        engine.save_rates_db(pd.DataFrame([{
            "date": pd.Timestamp("2025-01-15"), "currency": "USD",
            "tenor": "3M", "ticker": "US3M.GBOND",
            "yield_pct": 4.35, "yield": 0.0435,
        }]))
        m = engine.build_risk_free_rate_map(["USD", "JPY"])
        assert "USD" in m and "JPY" not in m

    def test_tenor_filter(self, engine):
        engine.save_rates_db(pd.DataFrame([
            {"date": pd.Timestamp("2025-01-15"), "currency": "USD",
             "tenor": "3M",  "ticker": "US3M.GBOND",
             "yield_pct": 4.35, "yield": 0.0435},
            {"date": pd.Timestamp("2025-01-15"), "currency": "USD",
             "tenor": "10Y", "ticker": "US10Y.GBOND",
             "yield_pct": 4.50, "yield": 0.0450},
        ]))
        m_3m  = engine.build_risk_free_rate_map(["USD"], tenor="3M")
        m_10y = engine.build_risk_free_rate_map(["USD"], tenor="10Y")
        assert m_3m["USD"]  == pytest.approx(0.0435)
        assert m_10y["USD"] == pytest.approx(0.0450)


class TestRoundTrip:
    """End-to-end: fetch → store → build map."""

    def test_fetch_then_build(self, engine, mock_client):
        engine.fetch_latest_rates(["USD"])
        m = engine.build_risk_free_rate_map(["USD"])
        assert m["USD"] == pytest.approx(0.0435)

    def test_api_failure_falls_back_to_existing_csv(self, engine, mock_client):
        # Pre-existing row in the CSV.
        engine.save_rates_db(pd.DataFrame([{
            "date": pd.Timestamp("2025-01-10"),
            "currency": "USD", "tenor": "3M", "ticker": "US3M.GBOND",
            "yield_pct": 4.10, "yield": 0.0410,
        }]))
        mock_client.get_bond_yield.side_effect = NetworkError("EOD down")
        engine.fetch_latest_rates(["USD"])
        m = engine.build_risk_free_rate_map(["USD"])
        # Falls back to the stored value.
        assert m["USD"] == pytest.approx(0.0410)
