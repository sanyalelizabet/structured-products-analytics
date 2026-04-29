"""
Tests for MarketDataEngine.
"""
import pandas as pd
import pytest
import numpy as np
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


def _write_master(tmp_path):
    pd.DataFrame([
        {"isin": "CH001", "ticker": "NESN.SW", "code": "NESN", "exchange": "SW",
         "name": "Nestle", "type": "Common Stock", "country": "Switzerland", "currency": "CHF"},
        {"isin": "CH002", "ticker": "NOVN.SW", "code": "NOVN", "exchange": "SW",
         "name": "Novartis", "type": "Common Stock", "country": "Switzerland", "currency": "CHF"},
    ]).to_csv(tmp_path / "securities_master_data.csv", index=False)


@pytest.fixture
def engine(mock_client, tmp_path):
    _write_master(tmp_path)
    return MarketDataEngine(client=mock_client, db_path=str(tmp_path / "prices.csv"))


def _simple_portfolio():
    return pd.DataFrame([{
        "product_id": "BRC001",
        "underlying_isins": ["CH001"],
    }])


def _two_ticker_portfolio():
    return pd.DataFrame([
        {"product_id": "BRC001", "underlying_isins": ["CH001"]},
        {"product_id": "BRC002", "underlying_isins": ["CH002"]},
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

    def test_api_error_on_one_ticker_continues_to_next(self, engine, mock_client):
        """If NESN fetch fails, NOVN should still be fetched."""
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

class TestFetchDailyPrices:
    def _daily_data(self):
        return [
            {"date": "2024-01-01", "adjusted_close": 100.0},
            {"date": "2024-01-02", "adjusted_close": 101.0},
            {"date": "2024-01-03", "adjusted_close": 102.0},
        ]

    def test_stores_daily_prices(self, engine, mock_client):
        mock_client.get_daily_prices.return_value = self._daily_data()

        db = engine.fetch_daily_prices({"CH001": "NESN.SW"})

        assert not db.empty
        assert "CH001" in db["isin"].values
        assert mock_client.get_daily_prices.called

    def test_skips_isin_with_enough_daily_data(self, engine, mock_client):
        """Skip applies when (a) force_refresh=False, (b) date range covers
        the requested history window, and (c) row density is sufficient."""
        # Build a long history ending today: years=2 needs ≥400 rows and
        # min date ≤ today − 2y − 30d.
        end = pd.Timestamp.today().normalize()
        existing = pd.DataFrame([
            {
                "date": end - pd.offsets.BDay(d),
                "isin": "CH001",
                "ticker": "NESN.SW",
                "price": 100.0,
            }
            for d in range(600)
        ])

        engine.save_db(existing)

        engine.fetch_daily_prices(
            {"CH001": "NESN.SW"}, years=2, force_refresh=False
        )

        mock_client.get_daily_prices.assert_not_called()

    def test_force_refresh_overrides_skip(self, engine, mock_client):
        existing = pd.DataFrame([
            {
                "date": pd.Timestamp("2024-01-01") + pd.offsets.BDay(d),
                "isin": "CH001",
                "ticker": "NESN.SW",
                "price": 100.0,
            }
            for d in range(300)
        ])

        engine.save_db(existing)
        mock_client.get_daily_prices.return_value = self._daily_data()

        engine.fetch_daily_prices({"CH001": "NESN.SW"}, force_refresh=True)

        mock_client.get_daily_prices.assert_called_once_with("NESN.SW", years=6)

    def test_deduplicates_on_isin_and_date(self, engine, mock_client):
        mock_client.get_daily_prices.return_value = self._daily_data()

        engine.fetch_daily_prices({"CH001": "NESN.SW"})
        engine.fetch_daily_prices({"CH001": "NESN.SW"}, force_refresh=True)

        db = engine.load_db()

        assert db.duplicated(subset=["isin", "date"]).sum() == 0

    def test_api_error_does_not_crash(self, engine, mock_client):
        mock_client.get_daily_prices.side_effect = Exception("API down")

        db = engine.fetch_daily_prices({"CH001": "NESN.SW"})

        assert db is not None
# ─────────────────────────────────────────
# fetch_options_chain
# ─────────────────────────────────────────

def _make_options_df(ticker="NESN.SW", expiry="2025-06-20"):
    return pd.DataFrame([
        {"ticker": ticker, "expiry": expiry, "type": "call",
         "strike": 90.0, "last_price": 8.0, "bid": 7.9, "ask": 8.1,
         "iv": 0.20, "volume": 100, "open_interest": 500},
        {"ticker": ticker, "expiry": expiry, "type": "put",
         "strike": 90.0, "last_price": 3.0, "bid": 2.9, "ask": 3.1,
         "iv": 0.22, "volume": 80, "open_interest": 300},
    ])


def _master_csv(tmp_path):
    """Write a minimal securities_master_data.csv."""
    master = pd.DataFrame([{
        "isin": "CH0012221716", "ticker": "NESN.SW", "code": "NESN",
        "exchange": "SW", "name": "Nestle", "type": "Common Stock",
        "country": "Switzerland", "currency": "CHF",
    }])
    path = tmp_path / "securities_master_data.csv"
    master.to_csv(path, index=False)


@pytest.fixture
def yahoo_client():
    yc = MagicMock()
    yc.get_full_chain.return_value = _make_options_df()
    return yc


@pytest.fixture
def engine_with_master(mock_client, tmp_path):
    _master_csv(tmp_path)
    return MarketDataEngine(
        client=mock_client,
        db_path=str(tmp_path / "prices.csv"),
    )


class TestFetchOptionsChain:
    def test_fetches_and_saves_options(self, engine_with_master, yahoo_client):
        df = engine_with_master.fetch_options_chain(["CH0012221716"], yahoo_client)
        assert not df.empty
        assert set(["isin", "ticker", "expiry", "type", "strike", "iv"]).issubset(df.columns)
        assert engine_with_master.options_path.exists()

    def test_isin_stored_in_output(self, engine_with_master, yahoo_client):
        df = engine_with_master.fetch_options_chain(["CH0012221716"], yahoo_client)
        assert "CH0012221716" in df["isin"].values

    def test_skips_if_already_fetched_today(self, engine_with_master, yahoo_client):
        engine_with_master.fetch_options_chain(["CH0012221716"], yahoo_client)
        engine_with_master.fetch_options_chain(["CH0012221716"], yahoo_client)
        assert yahoo_client.get_full_chain.call_count == 1

    def test_force_refresh_re_fetches(self, engine_with_master, yahoo_client):
        engine_with_master.fetch_options_chain(["CH0012221716"], yahoo_client)
        engine_with_master.fetch_options_chain(["CH0012221716"], yahoo_client, force_refresh=True)
        assert yahoo_client.get_full_chain.call_count == 2

    def test_api_error_does_not_crash(self, engine_with_master, yahoo_client):
        yahoo_client.get_full_chain.side_effect = Exception("Yahoo down")
        df = engine_with_master.fetch_options_chain(["CH0012221716"], yahoo_client)
        assert df is not None


class TestBuildCorrMatrix:

    def _make_daily_db(self, engine, isins, n_days=500, seed=0):
        """Write realistic daily price paths with mild randomness."""
        rng = np.random.default_rng(seed)
        rows = []

        for isin in isins:
            price = 100.0
            for d in range(n_days):
                shock = rng.normal(0, 0.01)
                price *= (1 + shock)

                rows.append({
                    "date": pd.Timestamp("2020-01-01") + pd.offsets.BDay(d),
                    "isin": isin,
                    "ticker": isin + ".SW",
                    "price": price,
                })

        engine.save_db(pd.DataFrame(rows))


    def test_returns_dataframe_with_isin_index(self, engine):
        self._make_daily_db(engine, ["CH001", "CH002"])

        corr = CorrelationEngine(engine).build_corr_matrix({
            "CH001": "CH001.SW", "CH002": "CH002.SW"
        })

        assert isinstance(corr, pd.DataFrame)
        assert set(corr.index) == {"CH001", "CH002"}


    def test_diagonal_is_one(self, engine):
        self._make_daily_db(engine, ["CH001", "CH002"])

        corr = CorrelationEngine(engine).build_corr_matrix({
            "CH001": "CH001.SW", "CH002": "CH002.SW"
        })

        assert np.allclose(np.diag(corr.values), 1.0)


    def test_off_diagonal_between_minus_one_and_one(self, engine):
        self._make_daily_db(engine, ["CH001", "CH002"])

        corr = CorrelationEngine(engine).build_corr_matrix({
            "CH001": "CH001.SW", "CH002": "CH002.SW"
        })

        val = corr.loc["CH001", "CH002"]
        assert -1.0 <= val <= 1.0


    def test_identical_series_gives_correlation_one(self, engine):
        """Same price path → correlation ≈ 1"""
        rows = []

        price = 100.0
        for d in range(500):
            price *= 1.001
            date = pd.Timestamp("2020-01-01") + pd.offsets.BDay(d)

            rows.append({"date": date, "isin": "CH001", "ticker": "A", "price": price})
            rows.append({"date": date, "isin": "CH002", "ticker": "B", "price": price})

        engine.save_db(pd.DataFrame(rows))

        corr = CorrelationEngine(engine).build_corr_matrix({
            "CH001": "A", "CH002": "B"
        })

        assert abs(corr.loc["CH001", "CH002"] - 1.0) < 1e-3


    def test_opposite_series_gives_correlation_minus_one(self, engine):
        """Perfect opposite returns → correlation ≈ -1"""
        rows = []

        price_a = 100.0
        price_b = 100.0

        for d in range(500):
            shock = 0.01 if d % 2 == 0 else -0.01
            price_a *= (1 + shock)
            price_b *= (1 - shock)

            date = pd.Timestamp("2020-01-01") + pd.offsets.BDay(d)

            rows.append({"date": date, "isin": "CH001", "ticker": "A", "price": price_a})
            rows.append({"date": date, "isin": "CH002", "ticker": "B", "price": price_b})

        engine.save_db(pd.DataFrame(rows))

        corr = CorrelationEngine(engine).build_corr_matrix({
            "CH001": "A", "CH002": "B"
        })

        assert abs(corr.loc["CH001", "CH002"] + 1.0) < 1e-3


    def test_falls_back_to_identity_when_insufficient_observations(self, engine):
        """ISINs with fewer than the min-period overlap (252 days) should be
        treated as uncorrelated — diagonal 1, off-diagonal 0 — instead of
        crashing the whole matrix.  This protects the rest of the portfolio
        when one ticker has thin history."""
        rows = []

        for isin in ["CH001", "CH002"]:
            for d in range(50):  # well below the 252-day pairwise threshold
                rows.append({
                    "date": pd.Timestamp("2024-01-01") + pd.offsets.BDay(d),
                    "isin": isin,
                    "ticker": isin + ".SW",
                    "price": 100.0 * (1 + 0.001 * d),
                })

        engine.save_db(pd.DataFrame(rows))

        corr = CorrelationEngine(engine).build_corr_matrix({
            "CH001": "CH001.SW",
            "CH002": "CH002.SW",
        })

        # Diagonal exactly 1, off-diagonal exactly 0 (independence fallback)
        assert corr.loc["CH001", "CH001"] == pytest.approx(1.0)
        assert corr.loc["CH002", "CH002"] == pytest.approx(1.0)
        assert corr.loc["CH001", "CH002"] == pytest.approx(0.0)
        assert corr.loc["CH002", "CH001"] == pytest.approx(0.0)


    def test_pairwise_correlation_invariant_to_extra_assets(self, engine):
        """
        Correlation between two assets should not change when adding a third asset.
        """

        rows = []

        price_a = 100.0
        price_b = 100.0
        price_c = 100.0

        for d in range(500):
            date = pd.Timestamp("2020-01-01") + pd.offsets.BDay(d)

            price_a *= 1.001
            price_b *= 1.0012
            price_c *= 1.0008

            rows.append({"date": date, "isin": "A", "ticker": "A", "price": price_a})
            rows.append({"date": date, "isin": "B", "ticker": "B", "price": price_b})
            rows.append({"date": date, "isin": "C", "ticker": "C", "price": price_c})

        engine.save_db(pd.DataFrame(rows))

        corr_engine = CorrelationEngine(engine)

        corr_pair = corr_engine.build_corr_matrix({"A": "A", "B": "B"})
        corr_full = corr_engine.build_corr_matrix({"A": "A", "B": "B", "C": "C"})

        corr_ab_pair = corr_pair.loc["A", "B"]
        corr_ab_full = corr_full.loc["A", "B"]

        assert abs(corr_ab_pair - corr_ab_full) < 1e-3

# ─────────────────────────────────────────
# _resolve_ticker — home-market priority
# ─────────────────────────────────────────

class TestResolveTicker:
    """The ticker chosen for daily-price fetches must be the issuer's
    home-market listing.  Picking a US OTC pink-sheet for a Swiss-listed
    stock yields stale prices and corrupts every downstream regression.
    """

    def _master_with(self, tmp_path, rows):
        pd.DataFrame(rows).to_csv(tmp_path / "securities_master_data.csv", index=False)

    def test_swiss_isin_picks_swiss_listing(self, mock_client, tmp_path):
        """CH-prefixed ISIN with Swiss + US-OTC listings → Swiss wins."""
        self._master_with(tmp_path, [
            {"isin": "CH0012005267", "ticker": "NVSEF.US", "code": "NVSEF",
             "exchange": "US", "name": "Novartis OTC",
             "type": "Common Stock", "country": "USA", "currency": "USD"},
            {"isin": "CH0012005267", "ticker": "NOVN.SW",  "code": "NOVN",
             "exchange": "SW", "name": "Novartis",
             "type": "Common Stock", "country": "Switzerland", "currency": "CHF"},
        ])
        engine = MarketDataEngine(client=mock_client, db_path=str(tmp_path / "prices.csv"))
        assert engine._resolve_ticker("CH0012005267") == "NOVN.SW"

    def test_us_isin_picks_us_listing(self, mock_client, tmp_path):
        self._master_with(tmp_path, [
            {"isin": "US0079031078", "ticker": "AMD.US", "code": "AMD",
             "exchange": "US", "name": "AMD",
             "type": "Common Stock", "country": "USA", "currency": "USD"},
            {"isin": "US0079031078", "ticker": "0HEL.LSE", "code": "0HEL",
             "exchange": "LSE", "name": "AMD LSE",
             "type": "Common Stock", "country": "UK", "currency": "GBP"},
        ])
        engine = MarketDataEngine(client=mock_client, db_path=str(tmp_path / "prices.csv"))
        assert engine._resolve_ticker("US0079031078") == "AMD.US"

    def test_german_isin_picks_german_listing(self, mock_client, tmp_path):
        self._master_with(tmp_path, [
            {"isin": "DE000BASF111", "ticker": "BAS.XETRA", "code": "BAS",
             "exchange": "XETRA", "name": "BASF",
             "type": "Common Stock", "country": "Germany", "currency": "EUR"},
            {"isin": "DE000BASF111", "ticker": "BASFY.US", "code": "BASFY",
             "exchange": "US", "name": "BASF ADR",
             "type": "Common Stock", "country": "USA", "currency": "USD"},
        ])
        engine = MarketDataEngine(client=mock_client, db_path=str(tmp_path / "prices.csv"))
        assert engine._resolve_ticker("DE000BASF111") == "BAS.XETRA"

    def test_falls_back_to_us_when_home_country_missing(self, mock_client, tmp_path):
        """Country prefix not in the home-market map → fall back to US listing."""
        self._master_with(tmp_path, [
            {"isin": "BR0011112222", "ticker": "VALE.US", "code": "VALE",
             "exchange": "US", "name": "Vale ADR",
             "type": "Common Stock", "country": "USA", "currency": "USD"},
            {"isin": "BR0011112222", "ticker": "0HAH.LSE", "code": "0HAH",
             "exchange": "LSE", "name": "Vale LSE",
             "type": "Common Stock", "country": "UK", "currency": "GBP"},
        ])
        engine = MarketDataEngine(client=mock_client, db_path=str(tmp_path / "prices.csv"))
        assert engine._resolve_ticker("BR0011112222") == "VALE.US"

    def test_falls_back_to_first_when_neither_home_nor_us(self, mock_client, tmp_path):
        self._master_with(tmp_path, [
            {"isin": "BR0011112222", "ticker": "0HAH.LSE", "code": "0HAH",
             "exchange": "LSE", "name": "Vale LSE",
             "type": "Common Stock", "country": "UK", "currency": "GBP"},
        ])
        engine = MarketDataEngine(client=mock_client, db_path=str(tmp_path / "prices.csv"))
        assert engine._resolve_ticker("BR0011112222") == "0HAH.LSE"

    def test_missing_isin_raises(self, mock_client, tmp_path):
        self._master_with(tmp_path, [
            {"isin": "CH001", "ticker": "NESN.SW", "code": "NESN",
             "exchange": "SW", "name": "Nestle",
             "type": "Common Stock", "country": "Switzerland", "currency": "CHF"},
        ])
        engine = MarketDataEngine(client=mock_client, db_path=str(tmp_path / "prices.csv"))
        with pytest.raises(ValueError, match="not found in master"):
            engine._resolve_ticker("XX999999")


# ─────────────────────────────────────────
# fetch_securities_master — dedup
# ─────────────────────────────────────────

class TestFetchSecuritiesMasterDedup:
    """Repeat fetches must not append duplicate (isin, ticker) rows."""

    def test_repeated_force_refresh_does_not_duplicate(self, mock_client, tmp_path):
        listings = [{
            "isin": "CH0012005267", "ticker": "NOVN.SW", "code": "NOVN",
            "exchange": "SW", "name": "Novartis",
            "type": "Common Stock", "country": "Switzerland", "currency": "CHF",
        }]
        mock_client.search_by_isin.return_value = listings
        engine = MarketDataEngine(client=mock_client, db_path=str(tmp_path / "prices.csv"))

        # Refresh three times — master should still have exactly one row.
        for _ in range(3):
            engine.fetch_securities_master(["CH0012005267"], force_refresh=True)

        master = pd.read_csv(tmp_path / "securities_master_data.csv")
        assert len(master[master["isin"] == "CH0012005267"]) == 1


# ─────────────────────────────────────────
# fetch_daily_prices — auto-purge stale-ticker rows
# ─────────────────────────────────────────

class TestFetchDailyPricesAutoPurge:
    """If existing prices were stored under a *different* ticker than the one
    now being requested, drop them before re-fetching.  This prevents the
    silent CHF/USD mixing bug we saw with NVSEF.US polluting NOVN.SW data.
    """

    def test_stale_ticker_rows_purged_on_refetch(self, engine, mock_client, tmp_path):
        # Seed with existing rows for CH001 under the WRONG ticker (US OTC).
        existing = pd.DataFrame([
            {
                "date": pd.Timestamp("2024-01-01") + pd.offsets.BDay(d),
                "isin": "CH001", "ticker": "NESN_US_OTC.US",  # wrong listing
                "price": 100.0,
            }
            for d in range(50)
        ])
        engine.save_db(existing)

        # New fetch under the correct ticker.
        mock_client.get_daily_prices.return_value = [
            {"date": "2024-06-01", "adjusted_close": 105.0},
            {"date": "2024-06-02", "adjusted_close": 106.0},
        ]
        engine.fetch_daily_prices({"CH001": "NESN.SW"}, force_refresh=True)

        db = engine.load_db()
        sub = db[db["isin"] == "CH001"]
        # All remaining rows must be under the new ticker; the 50 stale rows are gone.
        assert (sub["ticker"] == "NESN.SW").all()
        assert (sub["ticker"] == "NESN_US_OTC.US").sum() == 0

    def test_matching_ticker_rows_preserved(self, engine, mock_client, tmp_path):
        existing = pd.DataFrame([
            {
                "date": pd.Timestamp("2024-01-01") + pd.offsets.BDay(d),
                "isin": "CH001", "ticker": "NESN.SW",
                "price": 100.0,
            }
            for d in range(50)
        ])
        engine.save_db(existing)

        mock_client.get_daily_prices.return_value = [
            {"date": "2024-06-01", "adjusted_close": 105.0},
        ]
        engine.fetch_daily_prices({"CH001": "NESN.SW"}, force_refresh=True)

        db = engine.load_db()
        # Original 50 rows still there; one new row appended.
        assert len(db[db["isin"] == "CH001"]) >= 50
