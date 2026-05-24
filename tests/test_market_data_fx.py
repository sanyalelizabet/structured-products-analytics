"""Tests for the FX methods on MarketDataEngine (Frankfurter client mocked)."""
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.market_data_engine import MarketDataEngine
from src.exceptions import NetworkError


def _payload(rates, date="2026-05-22", base="CHF"):
    return {"amount": 1.0, "base": base, "date": date, "rates": rates}


@pytest.fixture
def fx_client():
    return MagicMock()


@pytest.fixture
def engine(fx_client, tmp_path):
    # EOD client is irrelevant for FX tests; pass a bare mock.
    return MarketDataEngine(client=MagicMock(),
                            db_path=str(tmp_path / "prices.csv"),
                            fx_client=fx_client)


class TestFetchLatestFx:
    def test_fetches_and_persists(self, engine, fx_client):
        fx_client.get_latest_rates.return_value = _payload({"EUR": 1.05, "USD": 1.27})
        db = engine.fetch_latest_fx("CHF")
        assert engine.fx_path.exists()
        assert set(db.columns) == set(engine.FX_COLUMNS)
        assert set(db["quote"]) == {"EUR", "USD", "CHF"}  # incl. self-pair

    def test_skips_when_today_present(self, engine, fx_client):
        today = pd.Timestamp.today().normalize().strftime("%Y-%m-%d")
        fx_client.get_latest_rates.return_value = _payload({"EUR": 1.05}, date=today)
        engine.fetch_latest_fx("CHF")
        engine.fetch_latest_fx("CHF")  # second should skip the client
        assert fx_client.get_latest_rates.call_count == 1

    def test_force_refresh_refetches(self, engine, fx_client):
        today = pd.Timestamp.today().normalize().strftime("%Y-%m-%d")
        fx_client.get_latest_rates.return_value = _payload({"EUR": 1.05}, date=today)
        engine.fetch_latest_fx("CHF")
        engine.fetch_latest_fx("CHF", force_refresh=True)
        assert fx_client.get_latest_rates.call_count == 2

    def test_deduplicates(self, engine, fx_client):
        fx_client.get_latest_rates.return_value = _payload({"EUR": 1.05})
        engine.fetch_latest_fx("CHF", force_refresh=True)
        engine.fetch_latest_fx("CHF", force_refresh=True)
        db = engine.load_fx_db()
        assert db.duplicated(subset=["base", "quote", "date"]).sum() == 0

    def test_client_failure_returns_existing_db(self, engine, fx_client):
        fx_client.get_latest_rates.return_value = _payload({"EUR": 1.05})
        engine.fetch_latest_fx("CHF", force_refresh=True)  # seed
        fx_client.get_latest_rates.side_effect = NetworkError("down")
        db = engine.fetch_latest_fx("CHF", force_refresh=True)
        assert "EUR" in set(db["quote"])  # stored snapshot survived


class TestBuildFxRateMap:
    def test_multiplier_is_inverse_of_rate(self, engine, fx_client):
        fx_client.get_latest_rates.return_value = _payload({"USD": 1.27})
        engine.fetch_latest_fx("CHF")
        fx, as_of = engine.build_fx_rate_map("CHF")
        assert abs(fx[("USD", "CHF")] - 1 / 1.27) < 1e-12
        assert fx[("CHF", "CHF")] == 1.0
        assert as_of == pd.Timestamp("2026-05-22")

    def test_latest_snapshot_wins(self, engine, fx_client):
        fx_client.get_latest_rates.return_value = _payload({"USD": 1.1}, date="2026-05-19")
        engine.fetch_latest_fx("CHF", force_refresh=True)
        fx_client.get_latest_rates.return_value = _payload({"USD": 1.2}, date="2026-05-20")
        engine.fetch_latest_fx("CHF", force_refresh=True)
        fx, as_of = engine.build_fx_rate_map("CHF")
        assert abs(fx[("USD", "CHF")] - 1 / 1.2) < 1e-12
        assert as_of == pd.Timestamp("2026-05-20")

    def test_empty_when_no_data(self, engine):
        fx, as_of = engine.build_fx_rate_map("CHF")
        assert fx == {}
        assert as_of is None
