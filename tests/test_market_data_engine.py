"""
Tests for MarketDataEngine.
"""
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
from src.market_data_engine import MarketDataEngine
from src.correlation_engine import CorrelationEngine
from tests.conftest import make_brc_row


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.get_last_quote.return_value = {
        "ticker": "NESN.SW",
        "price": 95.0,
        "date": pd.Timestamp("2024-12-31"),
        "source": "EOD",
    }
    return client


@pytest.fixture
def engine(mock_client, tmp_path):
    return MarketDataEngine(client=mock_client, db_path=str(tmp_path / "prices.csv"))


def _simple_portfolio():
    return pd.DataFrame([{
        "product_id": "BRC001",
        "underlying_isins": ["CH001"],
        "tickers": ["NESN.SW"],
    }])


def _two_ticker_portfolio():
    return pd.DataFrame([
        {"product_id": "BRC001", "underlying_isins": ["CH001"], "tickers": ["NESN.SW"]},
        {"product_id": "BRC002", "underlying_isins": ["CH002"], "tickers": ["NOVN.SW"]},
    ])


# ─────────────────────────────────────────
# load_db / save_db
# ─────────────────────────────────────────

class TestLoadDb:
    def test_empty_when_no_file(self, engine):
        df = engine.load_db()
        assert df.empty
        assert list(df.columns) == ["date", "isin", "ticker", "price"]

    def test_roundtrip(self, engine):
        db = pd.DataFrame([{
            "date": pd.Timestamp("2024-12-31"),
            "isin": "CH001", "ticker": "NESN.SW", "price": 95.0,
        }])
        engine.save_db(db)
        loaded = engine.load_db()
        assert len(loaded) == 1
        assert abs(loaded.iloc[0]["price"] - 95.0) < 1e-9


class TestSaveDb:
    def test_creates_parent_dirs(self, mock_client, tmp_path):
        engine = MarketDataEngine(
            client=mock_client,
            db_path=str(tmp_path / "nested" / "dir" / "prices.csv")
        )
        engine.save_db(pd.DataFrame(columns=["date", "isin", "ticker", "price"]))
        assert (tmp_path / "nested" / "dir" / "prices.csv").exists()


# ─────────────────────────────────────────
# fetch_latest_prices
# ─────────────────────────────────────────

class TestFetchLatestPrices:
    def test_fetches_and_stores_new_price(self, engine, mock_client):
        db = engine.fetch_latest_prices(_simple_portfolio())
        assert not db.empty
        assert mock_client.get_last_quote.called

    def test_skips_api_if_prev_day_already_in_db(self, engine, mock_client):
        from pandas.tseries.offsets import BDay
        prev_day = (pd.Timestamp.today() - BDay(1)).normalize()
        engine.save_db(pd.DataFrame([{
            "date": prev_day, "isin": "CH001", "ticker": "NESN.SW", "price": 95.0,
        }]))
        engine.fetch_latest_prices(_simple_portfolio())
        mock_client.get_last_quote.assert_not_called()

    def test_deduplicates_on_isin_and_date(self, engine, mock_client):
        engine.fetch_latest_prices(_simple_portfolio())
        engine.fetch_latest_prices(_simple_portfolio())
        db = engine.load_db()
        assert db.duplicated(subset=["isin", "date"]).sum() == 0

    @pytest.mark.xfail(reason="BUG: except block uses `break` instead of `continue` — one failed ticker aborts remaining tickers", strict=True)
    def test_api_error_on_one_ticker_continues_to_next(self, engine, mock_client):
        """
        If NESN fetch fails, NOVN should still be fetched.
        Current code uses `break` → stops the entire loop on first error.
        """
        def fail_first(ticker):
            if ticker == "NESN.SW":
                raise Exception("network error")
            return {"ticker": ticker, "price": 88.0, "date": pd.Timestamp("2024-12-31"), "source": "EOD"}

        mock_client.get_last_quote.side_effect = fail_first
        engine.fetch_latest_prices(_two_ticker_portfolio())

        db = engine.load_db()
        # NOVN should still have been fetched
        assert "CH002" in db["isin"].values


# ─────────────────────────────────────────
# update_spots
# ─────────────────────────────────────────

class TestUpdateSpots:
    def test_updates_from_db(self, engine):
        engine.save_db(pd.DataFrame([{
            "date": pd.Timestamp("2024-12-31"),
            "isin": "CH0012221716", "ticker": "NESN.SW", "price": 88.0,
        }]))
        pf = pd.DataFrame([make_brc_row(current_spot=95.0)])
        updated = engine.update_spots(pf)
        assert updated.iloc[0]["current_spots"] == [88.0]

    def test_raises_when_isin_missing(self, engine):
        pf = pd.DataFrame([make_brc_row()])
        with pytest.raises(ValueError, match="No stored price"):
            engine.update_spots(pf)

    def test_picks_latest_price(self, engine):
        engine.save_db(pd.DataFrame([
            {"date": pd.Timestamp("2024-12-29"), "isin": "CH0012221716", "ticker": "NESN.SW", "price": 90.0},
            {"date": pd.Timestamp("2024-12-31"), "isin": "CH0012221716", "ticker": "NESN.SW", "price": 95.0},
        ]))
        pf = pd.DataFrame([make_brc_row()])
        updated = engine.update_spots(pf)
        assert updated.iloc[0]["current_spots"] == [95.0]


# ─────────────────────────────────────────
# fetch_monthly_prices
# ─────────────────────────────────────────

class TestFetchMonthlyPrices:
    def _monthly_data(self):
        return [
            {"date": "2024-01-31", "adjusted_close": 90.0},
            {"date": "2024-02-29", "adjusted_close": 92.0},
            {"date": "2024-03-31", "adjusted_close": 95.0},
        ]

    def test_stores_monthly_prices(self, engine, mock_client):
        mock_client.get_monthly_prices.return_value = self._monthly_data()
        db = engine.fetch_monthly_prices({"CH001": "NESN.SW"})
        assert not db.empty
        assert "CH001" in db["isin"].values

    def test_skips_isin_with_enough_data(self, engine, mock_client):
        existing = pd.DataFrame([
            {"date": pd.Timestamp(f"2020-{m:02d}-28"), "isin": "CH001",
             "ticker": "NESN.SW", "price": 90.0}
            for m in range(1, 13)
        ] * 4)  # 48 rows — above the 36 threshold
        engine.save_db(existing)

        engine.fetch_monthly_prices({"CH001": "NESN.SW"})
        mock_client.get_monthly_prices.assert_not_called()

    def test_force_refresh_overrides_skip(self, engine, mock_client):
        existing = pd.DataFrame([
            {"date": pd.Timestamp(f"2020-{m:02d}-28"), "isin": "CH001",
             "ticker": "NESN.SW", "price": 90.0}
            for m in range(1, 13)
        ] * 4)
        engine.save_db(existing)

        mock_client.get_monthly_prices.return_value = self._monthly_data()
        engine.fetch_monthly_prices({"CH001": "NESN.SW"}, force_refresh=True)
        mock_client.get_monthly_prices.assert_called_once()

    def test_api_error_does_not_crash(self, engine, mock_client):
        mock_client.get_monthly_prices.side_effect = Exception("API down")
        db = engine.fetch_monthly_prices({"CH001": "NESN.SW"})
        assert db is not None


# ─────────────────────────────────────────
# build_corr_matrix
# ─────────────────────────────────────────

class TestBuildCorrMatrix:
    def _make_monthly_db(self, engine, isins):
        """Write 48 months of price data for each ISIN."""
        rows = []
        for isin in isins:
            for m in range(48):
                rows.append({
                    "date": pd.Timestamp("2020-01-31") + pd.DateOffset(months=m),
                    "isin": isin,
                    "ticker": isin + ".SW",
                    "price": 100.0 * (1 + 0.01 * m),
                })
        engine.save_db(pd.DataFrame(rows))

    def test_returns_dataframe_with_isin_index(self, engine, mock_client):
        self._make_monthly_db(engine, ["CH001", "CH002"])
        mock_client.get_monthly_prices.return_value = []
        corr = CorrelationEngine(engine).build_corr_matrix({"CH001": "A.SW", "CH002": "B.SW"})
        assert isinstance(corr, pd.DataFrame)
        assert "CH001" in corr.index
        assert "CH002" in corr.index

    def test_diagonal_is_one(self, engine, mock_client):
        self._make_monthly_db(engine, ["CH001", "CH002"])
        mock_client.get_monthly_prices.return_value = []
        corr = CorrelationEngine(engine).build_corr_matrix({"CH001": "A.SW", "CH002": "B.SW"})
        assert abs(corr.loc["CH001", "CH001"] - 1.0) < 1e-9
        assert abs(corr.loc["CH002", "CH002"] - 1.0) < 1e-9

    def test_off_diagonal_between_minus_one_and_one(self, engine, mock_client):
        self._make_monthly_db(engine, ["CH001", "CH002"])
        mock_client.get_monthly_prices.return_value = []
        corr = CorrelationEngine(engine).build_corr_matrix({"CH001": "A.SW", "CH002": "B.SW"})
        val = corr.loc["CH001", "CH002"]
        assert -1.0 <= val <= 1.0

    def test_raises_when_insufficient_observations(self, engine, mock_client):
        """Only 3 monthly rows per ISIN → well below the minimum threshold."""
        rows = []
        for isin in ["CH001", "CH002"]:
            for m in range(3):
                rows.append({
                    "date": pd.Timestamp("2024-01-31") + pd.DateOffset(months=m),
                    "isin": isin, "ticker": isin + ".SW", "price": 100.0,
                })
        engine.save_db(pd.DataFrame(rows))
        mock_client.get_monthly_prices.return_value = []

        with pytest.raises(ValueError, match="common monthly observations"):
            CorrelationEngine(engine).build_corr_matrix({"CH001": "A.SW", "CH002": "B.SW"})
