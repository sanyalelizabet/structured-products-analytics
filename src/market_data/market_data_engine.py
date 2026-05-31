import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from pathlib import Path
from pandas.tseries.offsets import BDay

from src.exceptions import DataFetchError
from src.market_data.frankfurter_client import FrankfurterClient
from src.market_data.snb_client import SNBClient

log = logging.getLogger(__name__)


# Default network concurrency.  EOD's rate limit is generous (~1000 req/min
# on most plans) so 8 simultaneous requests is comfortably under it while
# delivering a clear speedup on cold start.
_DEFAULT_MAX_WORKERS = 8


def _parallel_map(fn, items, max_workers: int = _DEFAULT_MAX_WORKERS):
    """Run ``fn(item)`` over ``items`` in a thread pool, preserving order.

    Threads are correct here because every consumer is HTTP I/O-bound and
    releases the GIL during the request.  Order preservation matters for
    deterministic concatenation into the persisted DBs.
    """
    if not items:
        return []
    with ThreadPoolExecutor(max_workers=min(max_workers, len(items))) as pool:
        return list(pool.map(fn, items))


def _append_rows(db: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """Append non-empty rows while avoiding pandas empty-frame concat warnings."""
    if rows.empty:
        return db.copy()
    if db.empty:
        return rows.copy()
    return pd.concat([db, rows], ignore_index=True)


def _append_records(db: pd.DataFrame, records: list[dict]) -> pd.DataFrame:
    """Append record dicts to a DB frame without concat warnings on empty DBs."""
    if not records:
        return db.copy()
    return _append_rows(db, pd.DataFrame(records))


class MarketDataEngine:

    MASTER_COLUMNS = ["isin", "ticker", "code", "exchange", "name", "type", "country", "currency"]

    OPTIONS_COLUMNS = ["fetch_date", "isin", "ticker", "expiry", "type",
                       "strike", "last_price", "bid", "ask", "iv",
                       "volume", "open_interest"]

    RATES_COLUMNS = ["date", "currency", "tenor", "ticker",
                     "yield_pct", "yield"]

    # FX store: rate = units of ``quote`` per 1 unit of ``base`` (Frankfurter
    # convention). One base per fetch; reverse direction handled by inversion
    # in build_fx_rate_map.
    FX_COLUMNS = ["date", "base", "quote", "rate"]

    # Default GBOND ticker per (currency, tenor).  Extend by inserting
    # additional tenors as needed.
    #
    # CHF is intentionally absent: EOD's GBOND virtual exchange does not carry
    # a usable CH3M point, so the Swiss risk-free rate is sourced from the SNB
    # SARON complex instead (see ``_fetch_saron_rows`` / :class:`SNBClient`).
    GBOND_TICKERS = {
        ("USD", "3M"):  "US3M.GBOND",
        ("EUR", "3M"):  "DE3M.GBOND",
        ("GBP", "3M"):  "UK3M.GBOND",
    }

    # Per-currency preferred tenor — used when a caller doesn't pin one
    # explicitly.  CHF uses the SARON 3M compound rate (sourced from SNB);
    # the majors use their 3M GBOND yield.
    DEFAULT_TENORS = {
        "USD": "3M",
        "CHF": "3M",
        "EUR": "3M",
        "GBP": "3M",
    }

    # SNB SARON tickers persisted into the rates DB (provenance), per tenor.
    SARON_TICKERS = {
        "ON": "SARON.SNB",
        "1M": "SARON1M.SNB",
        "3M": "SARON3M.SNB",
        "6M": "SARON6M.SNB",
    }

    def __init__(self, client, db_path="data/prices.csv", fx_client=None,
                 snb_client=None):
        self.client = client
        # FX provider — injected for testing/substitution; defaults to
        # Frankfurter. Same role the EOD ``client`` plays for prices/yields.
        self.fx_client = fx_client if fx_client is not None else FrankfurterClient()
        # CHF risk-free provider — SNB SARON. Injected for testing; same role
        # the EOD ``client`` plays for the majors' GBOND yields.
        self.snb_client = snb_client if snb_client is not None else SNBClient()
        self.db_path = Path(db_path)
        self.master_path  = self.db_path.parent / "securities_master_data.csv"
        self.options_path = self.db_path.parent / "options.csv"
        self.rates_path   = self.db_path.parent / "risk_free_rates.csv"
        self.fx_path      = self.db_path.parent / "fx_rates.csv"

        # Per-engine "fetched-today" memo for fetch_daily_prices.  In a
        # given session BetaEngine, FactorLoadingsEngine, CorrelationEngine,
        # and FactorEngine all call ``fetch_daily_prices`` with overlapping
        # ISINs.  After the first verification we don't need to re-load the
        # CSV or re-run the skip-logic — record (isin, ticker, years) as
        # "verified today" and short-circuit on the next call.  Resets
        # automatically when the calendar date rolls over.
        self._daily_check_date: pd.Timestamp | None = None
        self._daily_checked: set[tuple[str, str, int]] = set()

        # In-memory cache of the parsed prices DB.  Invalidated when the
        # CSV's mtime changes — so any save_db() call elsewhere refreshes
        # the cache transparently.  Saves ~15 ms per call by skipping the
        # CSV parse when nothing has changed on disk.
        self._db_cache:       pd.DataFrame | None = None
        self._db_cache_mtime: float | None        = None



    def load_db(self):
        """Load and parse ``prices.csv``, with an mtime-based memo.

        The parsed DataFrame is cached in memory; subsequent calls return
        a copy of the cached frame as long as the file's mtime is
        unchanged.  Any ``save_db`` call (here or in another process)
        bumps mtime and invalidates the cache transparently.
        """
        if not self.db_path.exists():
            return pd.DataFrame(columns=["date", "isin", "ticker", "price"])
        mtime = self.db_path.stat().st_mtime
        if (self._db_cache is not None
                and self._db_cache_mtime == mtime):
            return self._db_cache.copy()
        df = pd.read_csv(self.db_path, parse_dates=["date"])
        self._db_cache       = df
        self._db_cache_mtime = mtime
        return df.copy()

    def save_db(self, df):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.db_path, index=False)
        # Refresh the in-memory cache so the next load_db() returns the
        # frame we just wrote without re-parsing the CSV.
        self._db_cache       = df.copy()
        self._db_cache_mtime = self.db_path.stat().st_mtime

    def fetch_latest_prices(self, portfolio,
                            max_workers: int = _DEFAULT_MAX_WORKERS):
        """Refresh the latest spot price for every ISIN in ``portfolio``.

        Network calls run in parallel; ticker resolution and the
        already-have-this-date short-circuit happen sequentially first
        so the pool only fires for ISINs that actually need a fetch.
        """
        db = self.load_db()
        prev_trading_day = (pd.Timestamp.today() - BDay(1)).normalize()

        unique_isins = sorted({
            isin
            for _, row in portfolio.iterrows()
            for isin in row["underlying_isins"]
        })

        # Pre-resolve tickers and skip any ISIN whose prev-day price is
        # already in the DB.
        to_fetch: list[tuple[str, str]] = []
        for isin in unique_isins:
            try:
                ticker = self._resolve_ticker(isin)
            except ValueError as e:
                log.warning("Skipping price fetch for %s: %s", isin, e)
                continue

            exists_prev_day = (
                (db["isin"] == isin) & (db["date"] == prev_trading_day)
            ).any()
            if exists_prev_day:
                continue

            to_fetch.append((isin, ticker))

        def _fetch_one(item):
            isin, ticker = item
            try:
                quote = self.client.get_last_quote(ticker)
            except Exception as e:
                log.warning("Price fetch failed for %s (%s): %s", ticker, isin, e)
                return None
            quote_date = pd.to_datetime(quote["date"]).normalize()
            # Already have this quote date — nothing to append.
            if ((db["isin"] == isin) & (db["date"] == quote_date)).any():
                return None
            return {
                "date":   quote_date,
                "isin":   isin,
                "ticker": ticker,
                "price":  quote["price"],
            }

        results = _parallel_map(_fetch_one, to_fetch, max_workers=max_workers)
        rows = [r for r in results if r is not None]

        if rows:
            db = _append_records(db, rows)
            db = db.drop_duplicates(subset=["isin", "date"], keep="last")
            self.save_db(db)

        return db

    def update_spots(self, portfolio):
        """Refresh ``current_spots`` from the local price DB.

        Also writes ``current_spot_dates`` — one date per underlying, taken
        from the same row that produced the spot.  Existing portfolios that
        do not carry ``current_spot_dates`` get the column added on the fly.
        """
        portfolio = portfolio.copy()
        db = self.load_db()

        if "current_spot_dates" not in portfolio.columns:
            portfolio["current_spot_dates"] = [
                [None] * len(row["underlying_isins"])
                for _, row in portfolio.iterrows()
            ]

        for i, row in portfolio.iterrows():
            new_spots: list[float] = []
            new_dates: list = []

            for isin in row["underlying_isins"]:
                prices = db[db["isin"] == isin].sort_values("date")

                if prices.empty:
                    raise ValueError(f"No stored price found for ISIN {isin}")

                latest = prices.iloc[-1]
                new_spots.append(float(latest["price"]))
                new_dates.append(pd.Timestamp(latest["date"]).normalize())

            portfolio.at[i, "current_spots"]      = new_spots
            portfolio.at[i, "current_spot_dates"] = new_dates

        return portfolio

    # ─────────────────────────────────────────
    # Risk-free rates (GBOND)
    # ─────────────────────────────────────────

    def load_rates_db(self) -> pd.DataFrame:
        """Load the persisted risk-free-rates DB. Empty frame if absent."""
        if self.rates_path.exists():
            return pd.read_csv(self.rates_path, parse_dates=["date"])
        return pd.DataFrame(columns=self.RATES_COLUMNS)

    def save_rates_db(self, df: pd.DataFrame) -> None:
        self.rates_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.rates_path, index=False)

    def fetch_latest_rates(self, currencies, tenor: str | None = None,
                           max_workers: int = _DEFAULT_MAX_WORKERS) -> pd.DataFrame:
        """Refresh latest risk-free yields for ``currencies``.

        CHF is sourced from the SNB SARON complex (overnight + 1M/3M/6M
        compound) and the majors (USD/EUR/GBP) from their EOD GBOND yields.

        ``tenor`` is the GBOND point to fetch for the majors.  When ``None``
        (default) each uses its preferred tenor from :attr:`DEFAULT_TENORS`
        (3M).  It does not apply to CHF, whose SARON tenors are fixed.

        Behaviour mirrors :meth:`fetch_latest_prices`:
        * Skip the API call when the previous business day is already in
          the DB for that (ccy, tenor).
        * Skip when the quote date is already in the DB.
        * On any API failure, log and continue — the DB itself is the
          fallback (last available row remains the active rate).

        Returns the full rates DB (after any append), in long form:
        ``date, currency, tenor, ticker, yield_pct, yield``.
        """
        db = self.load_rates_db()
        prev_trading_day = (pd.Timestamp.today() - BDay(1)).normalize()

        # One-time, self-healing migration: CHF risk-free moved from EOD GBOND
        # (the CH10Y point, mislabeled as 3M) to the SNB SARON complex.  Drop any
        # stale GBOND-sourced CHF rows so they cannot win the latest-date
        # selection over SARON — and so that, if SNB is unreachable, CHF falls
        # back to the static anchor rather than a wrong stored value.
        purged = False
        if not db.empty:
            stale = (
                (db["currency"] == "CHF")
                & db["ticker"].astype(str).str.contains("GBOND", na=False)
            )
            if stale.any():
                db = db[~stale].reset_index(drop=True)
                purged = True

        # CHF risk-free comes from SNB SARON, not EOD GBOND.
        saron_rows = (
            self._fetch_saron_rows(db) if "CHF" in currencies else []
        )

        # Pre-filter: build the list of (ccy, tenor, ticker) triples that
        # still need a network call.  Any ticker not in GBOND_TICKERS or
        # already up-to-date is dropped here so the thread pool only fires
        # for real work.  CHF is handled above via SNB and skipped here.
        to_fetch: list[tuple[str, str, str]] = []
        for ccy in currencies:
            if ccy == "CHF":
                continue
            ccy_tenor = tenor or self.DEFAULT_TENORS.get(ccy, "3M")
            ticker = self.GBOND_TICKERS.get((ccy, ccy_tenor))
            if ticker is None:
                log.warning("No GBOND ticker mapped for %s %s — skipping",
                            ccy, ccy_tenor)
                continue

            already_recent = (
                (db["currency"] == ccy)
                & (db["tenor"] == ccy_tenor)
                & (db["date"] == prev_trading_day)
            ).any()
            if already_recent:
                continue

            to_fetch.append((ccy, ccy_tenor, ticker))

        def _fetch_one(item):
            ccy, ccy_tenor, ticker = item
            try:
                quote = self.client.get_bond_yield(ticker)
            except Exception as e:
                log.warning("Bond yield fetch failed for %s (%s): %s",
                            ticker, ccy, e)
                return None

            # EOD's real-time payload sometimes carries no usable timestamp
            # (``_parse_eod_date`` returns None).  ``pd.to_datetime(None)``
            # returns NaT — but some pandas builds let it leak through as
            # ``None``, which then crashes on ``.normalize()``.  Fall back
            # to today's date when no timestamp is present.
            raw_date = quote.get("date") if isinstance(quote, dict) else None
            if raw_date is None:
                log.info("Bond yield for %s (%s) had no timestamp — "
                          "stamping today's date", ticker, ccy)
                quote_date = pd.Timestamp.today().normalize()
            else:
                ts = pd.to_datetime(raw_date)
                if ts is None or pd.isna(ts):
                    quote_date = pd.Timestamp.today().normalize()
                else:
                    quote_date = ts.normalize()
            already_have = (
                (db["currency"] == ccy)
                & (db["tenor"] == ccy_tenor)
                & (db["date"] == quote_date)
            ).any()
            if already_have:
                return None

            yield_pct = float(quote["yield_pct"])
            return {
                "date":      quote_date,
                "currency":  ccy,
                "tenor":     ccy_tenor,
                "ticker":    ticker,
                "yield_pct": yield_pct,
                "yield":     yield_pct / 100.0,
            }

        results = _parallel_map(_fetch_one, to_fetch, max_workers=max_workers)
        rows = saron_rows + [r for r in results if r is not None]

        if rows:
            db = _append_records(db, rows)
            db = db.drop_duplicates(subset=["currency", "tenor", "date"], keep="last")

        # Persist when new rows arrived or the stale-CHF purge changed the DB.
        if rows or purged:
            db = db.sort_values(["currency", "tenor", "date"]).reset_index(drop=True)
            self.save_rates_db(db)

        return db

    def _fetch_saron_rows(self, db: pd.DataFrame) -> list[dict]:
        """Fetch the latest SARON rates from SNB as rates-DB records.

        Returns one record per tenor (ON / 1M / 3M / 6M) for any observation
        not already stored for that (CHF, tenor, date).  On any SNB failure the
        DB is the fallback — we log and return no new rows.
        """
        try:
            rates = self.snb_client.get_saron_rates()
        except DataFetchError as e:
            log.warning("SARON fetch from SNB failed: %s", e)
            return []

        have_chf = (not db.empty) and (db["currency"] == "CHF").any()
        rows: list[dict] = []
        for tenor, obs in rates.items():
            quote_date = pd.Timestamp(obs["date"]).normalize()
            if have_chf and (
                (db["currency"] == "CHF")
                & (db["tenor"] == tenor)
                & (db["date"] == quote_date)
            ).any():
                continue
            rows.append({
                "date":      quote_date,
                "currency":  "CHF",
                "tenor":     tenor,
                "ticker":    self.SARON_TICKERS.get(tenor, f"SARON{tenor}.SNB"),
                "yield_pct": obs["yield_pct"],
                "yield":     obs["yield"],
            })
        return rows

    def build_risk_free_rate_map(
        self, currencies, tenor: str | None = None,
    ) -> dict[str, float]:
        """Build the ``{currency: rate}`` map consumed by pricers/engines.

        Reads the rates DB and picks the latest row per requested
        currency.  When ``tenor`` is ``None``, the per-currency default
        from :attr:`DEFAULT_TENORS` is used (3M for all currencies — for CHF
        this resolves to the SARON 3M compound rate sourced from SNB).

        Yields are returned as **decimals** (e.g. ``0.0435``) — same
        convention as the legacy static dict.  Currencies with no rows
        in the DB are simply omitted.
        """
        db = self.load_rates_db()
        if db.empty:
            return {}

        result: dict[str, float] = {}
        for ccy in currencies:
            ccy_tenor = tenor or self.DEFAULT_TENORS.get(ccy, "3M")
            sub = db[(db["currency"] == ccy) & (db["tenor"] == ccy_tenor)]
            if sub.empty:
                continue
            latest = sub.sort_values("date").iloc[-1]
            result[ccy] = float(latest["yield"])
        return result

    # ─────────────────────────────────────────
    # FX rates (Frankfurter)
    # ─────────────────────────────────────────

    def load_fx_db(self) -> pd.DataFrame:
        """Load the persisted FX DB. Empty frame if absent."""
        if self.fx_path.exists():
            return pd.read_csv(self.fx_path, parse_dates=["date"])
        return pd.DataFrame(columns=self.FX_COLUMNS)

    def save_fx_db(self, df: pd.DataFrame) -> None:
        self.fx_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.fx_path, index=False)

    def fetch_latest_fx(self, base: str, force_refresh: bool = False) -> pd.DataFrame:
        """Refresh and persist FX rates for ``base`` (long form:
        ``date, base, quote, rate``).

        Mirrors :meth:`fetch_latest_rates`: skip the call when today's snapshot
        for ``base`` is already stored; on any client failure log and return the
        existing DB (the stored snapshot is the fallback). Returns the full DB.
        """
        db = self.load_fx_db()
        today = pd.Timestamp.today().normalize()

        if not force_refresh and not db.empty:
            if ((db["base"] == base) & (db["date"] == today)).any():
                return db

        try:
            payload = self.fx_client.get_latest_rates(base)
        except DataFetchError as e:
            log.warning("FX fetch failed for base %s (%s); using stored rates.",
                        base, e)
            return db

        rates = payload["rates"]
        quote_date = pd.to_datetime(payload.get("date"))
        if quote_date is None or pd.isna(quote_date):
            quote_date = today
        quote_date = quote_date.normalize()

        rows = [{"date": quote_date, "base": base, "quote": q, "rate": float(r)}
                for q, r in rates.items()]
        rows.append({"date": quote_date, "base": base, "quote": base, "rate": 1.0})

        new_df = pd.DataFrame(rows)
        db = new_df if db.empty else pd.concat([db, new_df], ignore_index=True)
        db = db.drop_duplicates(subset=["base", "quote", "date"], keep="last")
        db = db.sort_values(["base", "quote", "date"]).reset_index(drop=True)
        self.save_fx_db(db)
        return db

    def fetch_fx_history(self, base: str, years: int = 5) -> pd.DataFrame:
        """Incrementally back-fill daily FX history for ``base`` into the
        shared ``fx_rates.csv``.

        On the first call for ``base`` the full ``[today - years, today]``
        window is fetched in a single Frankfurter request (one response carries
        every quote currency at once).  Every subsequent call resolves the most
        recent ``date`` already stored for ``base`` and asks Frankfurter only
        for the tail beyond that — no historical rows are re-downloaded.

        Schema is the same long-form table that :meth:`fetch_latest_fx` writes
        (``date, base, quote, rate``), so the *latest-rate* consumers
        (``build_fx_rate_map``) continue to work unchanged on the extended DB.

        On any client failure we log and return the existing DB — the stored
        history is the fallback.  Returns the full DB.
        """
        db = self.load_fx_db()
        today = pd.Timestamp.today().normalize()

        base_db = db[db["base"] == base] if not db.empty else db
        if base_db.empty:
            from_date = (today - pd.DateOffset(years=years)).normalize()
        else:
            last = pd.Timestamp(base_db["date"].max()).normalize()
            from_date = last + pd.Timedelta(days=1)

        if from_date > today:
            return db                                       # cache already current

        try:
            new_df = self.fx_client.get_history(
                base, from_date.date(), today.date(),
            )
        except DataFetchError as e:
            log.warning("FX history fetch failed for base %s (%s); using stored.",
                        base, e)
            return db
        if new_df.empty:
            return db

        # Frankfurter omits the base→base self-row; add it at every fetched date
        # so ``build_fx_rate_map`` lookups for (base, base) keep returning 1.0.
        self_rows = pd.DataFrame([
            {"date": d, "base": base, "quote": base, "rate": 1.0}
            for d in new_df["date"].unique()
        ])
        new_df = pd.concat([new_df, self_rows], ignore_index=True)

        db = _append_rows(db, new_df)
        db = db.drop_duplicates(subset=["base", "quote", "date"], keep="last")
        db = db.sort_values(["base", "quote", "date"]).reset_index(drop=True)
        self.save_fx_db(db)
        return db

    def build_fx_rate_map(
        self, base: str,
    ) -> tuple[dict[tuple[str, str], float], "pd.Timestamp | None"]:
        """Return ``({(quote, base): multiplier_to_base}, as_of)``.

        ``multiplier_to_base`` converts 1 unit of ``quote`` into ``base`` and
        equals ``1 / rate``. Uses the latest stored snapshot per quote. Returns
        ``({}, None)`` when no data exists for ``base``.
        """
        db = self.load_fx_db()
        if db.empty:
            return {}, None
        sub = db[db["base"] == base]
        if sub.empty:
            return {}, None

        as_of = sub["date"].max()
        latest = sub.sort_values("date").groupby("quote", as_index=False).last()

        fx: dict[tuple[str, str], float] = {}
        for _, r in latest.iterrows():
            rate = float(r["rate"])
            if rate != 0.0:
                fx[(str(r["quote"]), base)] = 1.0 / rate
        fx[(base, base)] = 1.0
        return fx, as_of

    def fetch_monthly_prices(self, isin_ticker_map, years=6, force_refresh=False,
                             max_workers: int = _DEFAULT_MAX_WORKERS):
        """
        Download monthly adjusted-close prices and append to prices.csv.
        Same schema as daily rows — month-end dates coexist naturally.

        Parameters
        ----------
        isin_ticker_map : dict  { isin: ticker }
        years           : int   calendar years of history (default 4)
        force_refresh   : bool  re-download even if data already exists

        Returns
        -------
        pd.DataFrame  full DB after update
        """
        db = self.load_db()

        # Sequential prefilter — anything already dense gets skipped.
        to_fetch: list[tuple[str, str]] = []
        for isin, ticker in isin_ticker_map.items():
            existing = db[db["isin"] == isin]
            if not force_refresh and len(existing) > 36:
                continue
            to_fetch.append((isin, ticker))

        def _fetch_one(item):
            isin, ticker = item
            try:
                data = self.client.get_monthly_prices(ticker, years=years)
                return [
                    {"date":   pd.to_datetime(r["date"]),
                     "isin":   isin,
                     "ticker": ticker,
                     "price":  r["adjusted_close"]}
                    for r in data
                ]
            except Exception as e:
                log.warning("Monthly fetch failed for %s (%s): %s", ticker, isin, e)
                return []

        rows: list[dict] = []
        for batch in _parallel_map(_fetch_one, to_fetch, max_workers=max_workers):
            rows.extend(batch)

        if rows:
            db = _append_records(db, rows)
            db = db.drop_duplicates(subset=["isin", "date"], keep="last")
            db = db.sort_values(["isin", "date"]).reset_index(drop=True)
            self.save_db(db)

        return db

    def fetch_daily_prices(self, isin_ticker_map, years=6, force_refresh=False,
                           max_workers: int = _DEFAULT_MAX_WORKERS):
        """
        Download daily adjusted-close prices and append to prices.csv.

        HTTP fetches run in parallel via a thread pool — each ISIN's
        ``get_daily_prices`` call is independent I/O.  The decide-what-to-
        fetch and merge-into-DB phases run sequentially because they
        mutate shared state (``db``, the on-disk CSV).

        Parameters
        ----------
        isin_ticker_map : dict  { isin: ticker }
        years           : int   calendar years of history
        force_refresh   : bool  re-download even if data already exists
        max_workers     : int   max concurrent HTTP requests (default 8)

        Returns
        -------
        pd.DataFrame  full DB after update
        """
        # ── Phase 0: daily memo short-circuit ───────────────────────────
        # If we've already verified every requested (isin, ticker, years)
        # today, return the persisted DB directly — no scan, no parse, no
        # disk I/O beyond the load itself.
        today = pd.Timestamp.today().normalize()
        if self._daily_check_date != today:
            self._daily_check_date = today
            self._daily_checked.clear()

        if not force_refresh:
            requested = {(isin, ticker, years)
                         for isin, ticker in isin_ticker_map.items()}
            if requested.issubset(self._daily_checked):
                return self.load_db()

        db = self.load_db()

        # ── Phase 1a: vectorised stale-ticker purge ─────────────────────
        # Self-heal: drop rows whose ISIN we're tracking but whose stored
        # ticker no longer matches the resolved one.  Mixing prices across
        # listings (e.g. NOVN.SW in CHF + NVSEF.US in USD) silently
        # corrupts every downstream regression and stress run.
        purged_isins: list[str] = []
        if not db.empty:
            wanted_ticker = db["isin"].map(isin_ticker_map)
            stale_mask    = wanted_ticker.notna() & (db["ticker"] != wanted_ticker)
            if stale_mask.any():
                purged_isins = sorted(db.loc[stale_mask, "isin"].unique().tolist())
                db = db.loc[~stale_mask].reset_index(drop=True)

        # ── Phase 1b: decide who needs fetching using a single groupby ──
        # `db["date"]` is already datetime64 (load_db parses it on read),
        # so no per-ISIN to_datetime() reparse is needed.
        required_start   = pd.Timestamp.today() - pd.DateOffset(years=years) - pd.Timedelta(days=30)
        required_density = years * 200  # ~200 trading days per year

        if db.empty:
            min_dates: dict[str, pd.Timestamp] = {}
            counts:    dict[str, int]          = {}
        else:
            grp = db.groupby("isin", sort=False)
            min_dates = grp["date"].min().to_dict()
            counts    = grp.size().to_dict()

        to_fetch: list[tuple[str, str]] = []
        for isin, ticker in isin_ticker_map.items():
            if not force_refresh:
                count    = counts.get(isin, 0)
                min_date = min_dates.get(isin)
                if (
                    count >= required_density
                    and min_date is not None
                    and min_date <= required_start
                ):
                    # Already dense enough — mark as verified for the rest
                    # of the day so subsequent calls skip the scan entirely.
                    self._daily_checked.add((isin, ticker, years))
                    continue
            to_fetch.append((isin, ticker))

        # ── Phase 2 (parallel): fetch in a thread pool ─────────────────────
        def _fetch_one(item):
            isin, ticker = item
            try:
                data = self.client.get_daily_prices(ticker, years=years)
                return isin, ticker, [
                    {"date": pd.to_datetime(r["date"]),
                     "isin": isin,
                     "ticker": ticker,
                     "price": r["adjusted_close"]}
                    for r in data
                ]
            except Exception as e:
                log.warning("Daily fetch failed for %s (%s): %s", ticker, isin, e)
                return isin, ticker, None      # None signals failure

        rows: list[dict] = []
        for isin, ticker, batch in _parallel_map(_fetch_one, to_fetch,
                                                  max_workers=max_workers):
            if batch is None:
                # Failed fetches stay UN-marked so the next call retries.
                continue
            rows.extend(batch)
            self._daily_checked.add((isin, ticker, years))

        # ── Phase 3 (sequential): merge, dedupe, save ──────────────────────
        if purged_isins:
            log.info("Purged stale-ticker rows for %d ISIN(s) before re-fetch: %s",
                     len(purged_isins), purged_isins)

        if rows or purged_isins:
            db = _append_records(db, rows)
            db["date"] = pd.to_datetime(db["date"]).dt.normalize()

            db = db.drop_duplicates(subset=["isin", "date"], keep="last")
            db = db.sort_values(["isin", "date"]).reset_index(drop=True)

            self.save_db(db)

        return db

    def fetch_securities_master(self, isins, force_refresh=False,
                                max_workers: int = _DEFAULT_MAX_WORKERS):
        """
        Download master data for each ISIN and store in securities_master_data.csv.

        For each ISIN, calls search_by_isin and appends ALL exchange listings —
        one row per (isin, exchange) pair.  This means a security cross-listed
        on multiple exchanges will have multiple rows.

        Parameters
        ----------
        isins         : list[str]  list of ISINs
        force_refresh : bool       re-download even if ISIN already in CSV

        Returns
        -------
        pd.DataFrame  contents of securities_master_data.csv after update
        """
        if self.master_path.exists():
            master = pd.read_csv(self.master_path)
        else:
            master = pd.DataFrame(columns=self.MASTER_COLUMNS)

        # Filter out ISINs that already have a row (unless force_refresh).
        existing = set(master["isin"].values) if not master.empty else set()
        to_fetch = [
            isin for isin in isins
            if force_refresh or isin not in existing
        ]

        def _fetch_one(isin):
            try:
                listings = self.client.search_by_isin(isin)
                if not listings:
                    log.info("No listings found for %s", isin)
                    return []
                return listings
            except Exception as e:
                log.warning("Master data fetch failed for %s: %s", isin, e)
                return []

        rows: list[dict] = []
        for batch in _parallel_map(_fetch_one, to_fetch, max_workers=max_workers):
            rows.extend(batch)

        if rows:
            new_df = pd.DataFrame(rows)
            master = pd.concat([master, new_df], ignore_index=True)
            master = (
                master.drop_duplicates(subset=["isin", "ticker"], keep="last")
                       .sort_values(["isin", "exchange"])
                       .reset_index(drop=True)
            )
            self.master_path.parent.mkdir(parents=True, exist_ok=True)
            master.to_csv(self.master_path, index=False)

        return master

    # ── ISIN country-prefix → master ``country`` field synonyms ─────────
    # Used by ``_resolve_ticker`` to pick the issuer's home-market listing.
    # Picking a US ADR / OTC pink-sheet for a Swiss-listed stock yields
    # stale, illiquid prices — see e.g. NVSEF.US for Novartis.
    _ISIN_COUNTRY_TO_NAMES: dict[str, list[str]] = {
        "US": ["united states", "usa", "us"],
        "CH": ["switzerland", "ch"],
        "DE": ["germany", "de"],
        "FR": ["france", "fr"],
        "GB": ["united kingdom", "uk", "gb", "britain"],
        "NL": ["netherlands", "nl"],
        "IT": ["italy", "it"],
        "ES": ["spain", "es"],
        "BE": ["belgium", "be"],
        "AT": ["austria", "at"],
        "DK": ["denmark", "dk"],
        "SE": ["sweden", "se"],
        "NO": ["norway", "no"],
        "FI": ["finland", "fi"],
        "JP": ["japan", "jp"],
        "CA": ["canada", "ca"],
        "AU": ["australia", "au"],
    }

    def _resolve_ticker(self, isin):
        """Pick the listing best suited for daily-price fetches.

        Priority order:
        1. **Issuer's home market** — derived from the first two letters of
           the ISIN.  CH-ISINs get the Swiss listing, US-ISINs the US
           listing, etc.  This avoids picking thinly-traded ADRs / OTC
           pink-sheets when a liquid home-market listing is available.
        2. US listing — fallback for ISINs whose country prefix isn't in
           the map.
        3. Anything in the master row set.
        """
        if not self.master_path.exists():
            raise ValueError("Security master not initialized")

        master = pd.read_csv(self.master_path)
        matches = master[master["isin"] == isin]

        if matches.empty:
            raise ValueError(f"{isin} not found in master data")

        # 1. Home market — match by ISIN country prefix
        country_prefix = (isin[:2].upper() if isinstance(isin, str) and len(isin) >= 2 else "")
        home_names = self._ISIN_COUNTRY_TO_NAMES.get(country_prefix, [])
        if home_names:
            home = matches[matches["country"].str.lower().isin(home_names)]
            if not home.empty:
                return home.iloc[0]["ticker"]

        # 2. US fallback for ISINs whose home market isn't in the map
        us = matches[matches["country"].str.lower().isin(
            ["united states", "usa", "us"]
        )]
        if not us.empty:
            return us.iloc[0]["ticker"]

        # 3. anything
        return matches.iloc[0]["ticker"]

    def fetch_options_chain(self, isins, yahoo_client, force_refresh=False,
                            max_workers: int = _DEFAULT_MAX_WORKERS):
        """
        Download the full options chain for each ISIN using YahooClient.
        Tickers are resolved from securities_master_data.csv via _resolve_ticker.
        Results are stored in options.csv.

        Parameters
        ----------
        isins         : list[str]    ISINs to fetch options for
        yahoo_client  : YahooClient  Yahoo Finance client instance
        force_refresh : bool         re-download even if already fetched today

        Returns
        -------
        pd.DataFrame  contents of options.csv after update
        """

        if self.options_path.exists():
            options_db = pd.read_csv(self.options_path, parse_dates=["fetch_date"])
        else:
            options_db = pd.DataFrame(columns=self.OPTIONS_COLUMNS)

        today = pd.Timestamp.today().normalize()

        # Sequential prefilter — resolve tickers and apply the once-a-day skip.
        to_fetch: list[tuple[str, str]] = []
        for isin in isins:
            try:
                ticker = self._resolve_ticker(isin).split(".")[0]
            except ValueError as e:
                log.info("Skipping options fetch for %s: %s", isin, e)
                continue

            already_fetched_today = (
                not options_db.empty
                and (
                    (options_db["isin"] == isin)
                    & (options_db["fetch_date"] == today)
                ).any()
            )
            if not force_refresh and already_fetched_today:
                continue

            to_fetch.append((isin, ticker))

        def _fetch_one(item):
            isin, ticker = item
            try:
                df = yahoo_client.get_full_chain(ticker)
                df["isin"]       = isin
                df["fetch_date"] = today
                log.info("Options fetched: %s (%d rows)", ticker, len(df))
                return df
            except Exception as e:
                log.warning("Options fetch failed for %s (%s): %s", ticker, isin, e)
                return None

        results = _parallel_map(_fetch_one, to_fetch, max_workers=max_workers)
        frames = [df for df in results if df is not None]

        if frames:
            try:
                new_df = pd.concat(frames, ignore_index=True)
                log.debug("Columns in fetched data: %s", list(new_df.columns))
                new_df = new_df[self.OPTIONS_COLUMNS]

                # Drop stale rows for the same ISINs fetched on a previous day
                if not options_db.empty:
                    options_db = options_db[
                        ~options_db["isin"].isin(new_df["isin"].unique())
                    ]

                options_db = _append_rows(options_db, new_df)
                for col in ["volume", "open_interest"]:
                    options_db[col] = options_db[col].astype("Int64")
                options_db = options_db.sort_values(["isin", "expiry", "type", "strike"]).reset_index(drop=True)
                self.options_path.parent.mkdir(parents=True, exist_ok=True)
                options_db.to_csv(self.options_path, index=False)
                log.info("Options saved to %s (%d rows)", self.options_path, len(options_db))
            except Exception as e:
                log.error("Failed to save options: %s", e)

        return options_db


    def build_atm_vol_map(self, portfolio, fallback_vol=0.15):
        """
        Build a { isin: atm_implied_vol } map from stored options.csv and prices.csv.

        For each ISIN the expiry is matched to the product's maturity_date —
        the available option expiry closest to maturity is used, so the vol
        reflects the correct point on the term structure.

        If an ISIN appears in multiple products (different maturities), the
        longest maturity wins — conservative choice for a portfolio view.

        Parameters
        ----------
        portfolio    : pd.DataFrame  portfolio rows, must have underlying_isins
                                     and maturity_date columns
        fallback_vol : float         used if no options data exists for an ISIN

        Returns
        -------
        dict  { isin: float }  — drop-in replacement for the static vol_map
        """
        if not self.options_path.exists():
            raise ValueError("options.csv not found — run fetch_options_chain first")

        options_db = pd.read_csv(self.options_path)
        db = self.load_db()

        # Build isin -> maturity map (take longest maturity if ISIN appears in multiple products)
        isin_maturity = {}
        for _, row in portfolio.iterrows():
            maturity = pd.Timestamp(row["maturity_date"])
            for isin in row["underlying_isins"]:
                if isin not in isin_maturity or maturity > isin_maturity[isin]:
                    isin_maturity[isin] = maturity

        vol_map = {}

        for isin, maturity in isin_maturity.items():
            opts = options_db[options_db["isin"] == isin]
            if opts.empty:
                vol_map[isin] = fallback_vol
                continue

            prices = db[db["isin"] == isin].sort_values("date")
            if prices.empty:
                vol_map[isin] = fallback_vol
                continue

            spot = float(prices.iloc[-1]["price"])

            # Pick expiry closest to product maturity
            available = pd.to_datetime(opts["expiry"].unique())
            closest_expiry = min(available, key=lambda d: abs((d - maturity).days))

            chain = opts[opts["expiry"] == closest_expiry.strftime("%Y-%m-%d")].copy()
            chain["distance"] = (chain["strike"] - spot).abs()
            atm = chain.loc[chain["distance"].idxmin()]

            vol_map[isin] = float(atm["iv"])

        return vol_map

    def build_realised_vol_map(self, portfolio, window=252, fallback_vol=0.15):
        """
        Build a { isin: realised_vol } map from daily prices in prices.csv.

        σ = std(daily log returns, window) × √252

        Parameters
        ----------
        portfolio    : pd.DataFrame  must have underlying_isins column
        window       : int           rolling window in trading days (default 252 = 1 year)
        fallback_vol : float         used if insufficient price history

        Returns
        -------
        dict  { isin: float }  — drop-in replacement for the static vol_map
        """
        db = self.load_db()
        db["date"] = pd.to_datetime(db["date"])

        isins = list({isin for _, row in portfolio.iterrows() for isin in row["underlying_isins"]})

        vol_map = {}

        for isin in isins:
            prices = db[db["isin"] == isin].sort_values("date")

            if len(prices) < window // 2:
                log.warning("Insufficient price history for %s — using fallback vol", isin)
                vol_map[isin] = fallback_vol
                continue

            log_returns = np.log(prices["price"] / prices["price"].shift(1)).dropna()

            realised = float(log_returns.tail(window).std() * np.sqrt(252))

            if realised <= 0 or np.isnan(realised):
                vol_map[isin] = fallback_vol
            else:
                vol_map[isin] = round(realised, 4)

        return vol_map

    def build_vol_surface_map(self, portfolio, valuation_date=None):
        """Build per-(ISIN, listed expiry) implied-volatility slice surfaces.

        For every underlying in the portfolio the method assembles, from
        the stored options chain in ``options.csv``, one
        :class:`VolSliceSurface` per available expiry. Each slice is the
        product of the full calibration pipeline: SVI fit, butterfly
        and wing-bound arbitrage gates, data-quality gate, and the
        chain-proxy or constant-volatility fallback when any of those
        checks fail. The decision and reason are recorded on the
        returned slice so that the user interface can badge the result
        without re-running the calibration.

        Conventions
        -----------
        The forward price at each expiry is approximated by the spot
        price of the underlying (i.e. ``F = S``). This is the same
        convention adopted by the existing ATM vol map and is
        acceptable for SVI calibration because the smile-translation
        parameter ``m`` absorbs any small offset between the true
        forward and the spot. Removing the simplification, in concert
        with a dividend-yield estimator, is recorded as future work.
        Only out-of-the-money options are retained per expiry —
        out-of-the-money calls above the spot, out-of-the-money puts
        below the spot — to avoid the early-exercise bid-ask
        distortion that affects in-the-money American equity options.

        Parameters
        ----------
        portfolio : pd.DataFrame
            Portfolio rows; must expose an ``underlying_isins`` column.
        valuation_date : pd.Timestamp or str, optional
            As-of date used to compute the tenor of each expiry. When
            ``None`` the most recent ``fetch_date`` available in the
            options database is used.

        Returns
        -------
        dict
            ``{ isin: { expiry_iso: VolSliceSurface } }``. ISINs for
            which no option chain is available are absent from the
            outer dictionary rather than mapped to an empty inner one,
            so that callers can detect missing data unambiguously.
        """
        # Local import to avoid a hard dependency from market_data_engine
        # on the pricing layer when it is imported in lightweight
        # contexts (notebooks, data-collection scripts).
        from src.pricing.vol_surface import VolSliceSurface

        if not self.options_path.exists():
            log.warning("options.csv not found; vol surface map is empty")
            return {}

        options_db = pd.read_csv(self.options_path)
        if options_db.empty:
            return {}

        price_db = self.load_db()
        if price_db.empty:
            log.warning("price DB empty; cannot determine spots for vol surfaces")
            return {}

        # As-of: explicit argument wins, then the most recent fetch_date
        # recorded in the options DB.
        if valuation_date is None:
            valuation_date = pd.to_datetime(options_db["fetch_date"]).max()
        else:
            valuation_date = pd.Timestamp(valuation_date)

        # Active-portfolio ISINs: every distinct underlying referenced
        # in any row of the portfolio.
        portfolio_isins = sorted({
            isin
            for _, row in portfolio.iterrows()
            for isin in row["underlying_isins"]
        })

        surfaces: dict[str, dict[str, "VolSliceSurface"]] = {}

        for isin in portfolio_isins:
            isin_options = options_db[options_db["isin"] == isin]
            if isin_options.empty:
                continue

            # Spot from the most recent quote in the price DB.
            isin_prices = price_db[price_db["isin"] == isin].sort_values("date")
            if isin_prices.empty:
                continue
            spot = float(isin_prices.iloc[-1]["price"])
            if spot <= 0.0 or not np.isfinite(spot):
                continue

            isin_slices: dict[str, "VolSliceSurface"] = {}
            for expiry_iso, expiry_options in isin_options.groupby("expiry"):
                expiry_ts = pd.Timestamp(expiry_iso)
                tenor_years = (expiry_ts - valuation_date).days / 365.25
                if tenor_years <= 0.0:
                    continue   # expired

                # OTM filter: keep calls with K > spot and puts with K < spot.
                # Discard rows with missing/invalid IV or bid-ask quotes.
                clean = expiry_options[
                    (expiry_options["iv"].between(0.01, 5.0))
                    & (expiry_options["strike"] > 0.0)
                    & (expiry_options["bid"] > 0.0)
                    & (expiry_options["ask"] >= expiry_options["bid"])
                ].copy()
                otm = clean[
                    ((clean["type"] == "call") & (clean["strike"] >= spot))
                    | ((clean["type"] == "put") & (clean["strike"] <= spot))
                ]
                if otm.empty:
                    continue

                # Deduplicate by strike: keep the row with the tighter
                # bid-ask spread when both a call and a put are present
                # at the same strike (typically straddle-of-the-money).
                otm = otm.assign(spread=otm["ask"] - otm["bid"])
                otm = (otm.sort_values("spread")
                          .drop_duplicates(subset="strike", keep="first")
                          .sort_values("strike"))

                strikes = otm["strike"].to_numpy(dtype=float)
                ivs = otm["iv"].to_numpy(dtype=float)
                bid_asks = (otm["ask"] - otm["bid"]).to_numpy(dtype=float)
                mids = ((otm["ask"] + otm["bid"]) / 2.0).to_numpy(dtype=float)

                slice_surface = VolSliceSurface.from_chain(
                    isin=isin,
                    T=tenor_years,
                    forward=spot,   # F = S simplification, per module convention
                    strikes=strikes,
                    implied_vols=ivs,
                    bid_asks=bid_asks,
                    mids=mids,
                )
                isin_slices[str(expiry_iso)] = slice_surface

            if isin_slices:
                surfaces[isin] = isin_slices

        return surfaces

    def build_vol_surfaces(self, portfolio, valuation_date=None):
        """Build per-ISIN term-structure-consistent implied volatility surfaces.

        The method composes the per-(ISIN, expiry) slices produced by
        :meth:`build_vol_surface_map` into a single :class:`VolSurface`
        per underlying. The resulting object exposes the surface as a
        function of strike *and* tenor, interpolating between listed
        expiries by the linear-in-total-variance recipe of Gatheral
        (2006) and extrapolating beyond the listed range by vol-flat
        scaling of the appropriate anchor slice. The slice-level map
        returned by :meth:`build_vol_surface_map` remains available
        through the same engine instance for diagnostic access to
        individual slices.

        Parameters
        ----------
        portfolio : pd.DataFrame
            Portfolio rows; must expose an ``underlying_isins`` column.
        valuation_date : pd.Timestamp or str, optional
            As-of date used to compute the tenor of each expiry. When
            ``None`` the most recent ``fetch_date`` available in the
            options database is used.

        Returns
        -------
        dict
            ``{ isin: VolSurface }``. ISINs without any chain coverage
            are absent from the mapping. ISINs whose slices all fall
            back to the chain proxy or constant fallback at the slice
            level are present but in the surface's
            :data:`SURFACE_STATUS_FALLBACK` regime, which the caller
            may interrogate via ``surface.surface_status_at(T)``.
        """
        from src.pricing.vol_surface import VolSurface

        slice_map = self.build_vol_surface_map(portfolio, valuation_date=valuation_date)
        return {
            isin: VolSurface.from_slice_map(isin, isin_slices)
            for isin, isin_slices in slice_map.items()
        }

    def build_isin_ticker_map(self, isins):
        """
        Build { isin: ticker } from securities_master_data.csv for a list of ISINs.
        Uses the same priority logic as _resolve_ticker (US listings first).

        Parameters
        ----------
        isins : list[str]

        Returns
        -------
        dict  { isin: ticker }  — only includes ISINs found in master data
        """
        if not self.master_path.exists():
            raise ValueError("Security master not initialized")

        master = pd.read_csv(self.master_path)
        result = {}

        for isin in isins:
            try:
                result[isin] = self._resolve_ticker(isin)
            except ValueError:
                pass

        return result

    def build_isin_currency_map(self, isins) -> dict[str, str]:
        """Build ``{isin: native_currency}`` from the securities master.

        Mirrors :meth:`build_isin_ticker_map` — only includes ISINs that have a
        row with a non-empty ``currency`` in the master CSV; the caller decides
        how to handle missing entries (typically by treating them as already in
        the target currency, i.e. no translation).
        """
        if not self.master_path.exists():
            return {}
        master = pd.read_csv(self.master_path)
        if "currency" not in master.columns:
            return {}
        sub = master[["isin", "currency"]].dropna()
        sub = sub[sub["isin"].isin(set(isins))]
        return {row["isin"]: str(row["currency"]).upper() for _, row in sub.iterrows()}









