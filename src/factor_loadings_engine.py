"""Multivariate OLS factor loadings — one computation point for all
asset-vs-factor regressions.

Per ISIN, fits over aligned daily log-returns:

    r_{i,t} = α_i + Σ_k β_{i,k} F_{k,t} + ε_{i,t}

where ``F`` is any subset of ``FactorEngine.FACTORS``. Covers both single-factor
β vs MSCI World (``factors=["MKT"]``) and the full multi-factor set.

Output per ISIN
---------------
``{
    "betas":     {factor_code: β_k},
    "alpha":     α,
    "idio_vol":  annualised residual σ,
    "r_squared": R²,
    "n_obs":     observations used,
}``

Fallbacks: pairwise date alignment per ISIN; overlap < ``min_obs`` or a
near-singular design matrix returns a default loading set (β_MKT=1, others=0,
idio_vol=total_vol, R²=0) and logs a warning.
"""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd

from src.factor_engine import FACTORS, FactorEngine

log = logging.getLogger(__name__)


# Default fallback when a stock has insufficient history. β_MKT=1 keeps it
# CAPM-like; all other factor exposures default to zero (no sector/FX tilt).
def _default_loading(factor_codes: list[str], total_vol: float) -> dict:
    return {
        "betas":     {f: (1.0 if f == "MKT" else 0.0) for f in factor_codes},
        "alpha":     0.0,
        "idio_vol":  float(total_vol) if np.isfinite(total_vol) else 0.15,
        "r_squared": 0.0,
        "n_obs":     0,
    }


class FactorLoadingsEngine:
    """Compute multivariate factor loadings (β-vector + idio σ + R²) per ISIN."""

    def __init__(self, market_data_engine, factor_engine: FactorEngine | None = None):
        self.mde = market_data_engine
        self.fe = factor_engine or FactorEngine(market_data_engine)

    # ------------------------------------------------------------------ API

    def build_loadings(
        self,
        isin_ticker_map: dict,
        factors: Iterable[str] | None = None,
        years: int = 5,
        force_refresh: bool = False,
        min_obs: int = 252,
    ) -> dict[str, dict]:
        """Fit OLS loadings for every ISIN in ``isin_ticker_map``.

        Parameters
        ----------
        isin_ticker_map : dict   { isin: ticker }  — what to regress
        factors         : list   subset of FACTORS keys; defaults to all
        years           : int    daily-return window
        force_refresh   : bool   re-download stock & factor prices
        min_obs         : int    minimum aligned daily observations required

        Returns
        -------
        dict[isin → loading-dict]
        """
        factor_codes = list(factors) if factors else list(FACTORS.keys())
        for f in factor_codes:
            if f not in FACTORS:
                raise ValueError(f"Unknown factor code: {f}")

        # ── 1. Fetch stock and factor prices ─────────────────────────────
        self.mde.fetch_daily_prices(
            isin_ticker_map, years=years, force_refresh=force_refresh
        )
        self.fe.fetch_factor_prices(
            years=years, factors=factor_codes, force_refresh=force_refresh
        )

        # ── 2. Build factor returns (wide, listwise-aligned) ─────────────
        try:
            factor_returns = self.fe.build_returns(factors=factor_codes, years=years)
        except RuntimeError as e:
            log.warning("Factor returns unavailable (%s) — all loadings default", e)
            return {
                isin: _default_loading(factor_codes, np.nan)
                for isin in isin_ticker_map
            }

        if len(factor_returns) < min_obs:
            log.warning(
                "Only %d aligned factor observations (need %d) — all loadings default",
                len(factor_returns), min_obs,
            )
            return {
                isin: _default_loading(factor_codes, np.nan)
                for isin in isin_ticker_map
            }

        # ── 3. Build stock log-returns ───────────────────────────────────
        db = self.mde.load_db()
        db = db[db["isin"].isin(isin_ticker_map.keys())]
        stock_prices = (
            db.pivot_table(index="date", columns="isin", values="price", aggfunc="last")
            .sort_index()
        )
        cutoff = pd.Timestamp.today() - pd.DateOffset(years=years)
        stock_prices = stock_prices[stock_prices.index >= cutoff]
        stock_returns = np.log(stock_prices / stock_prices.shift(1))

        # ── 4. OLS per ISIN on the (date)-aligned overlap ────────────────
        loadings: dict[str, dict] = {}
        for isin in isin_ticker_map:
            if isin not in stock_returns.columns:
                log.warning("No price data for %s — using default loadings", isin)
                loadings[isin] = _default_loading(factor_codes, np.nan)
                continue

            r_i = stock_returns[isin].dropna()
            common = r_i.index.intersection(factor_returns.index)

            if len(common) < min_obs:
                total_vol = float(r_i.std() * np.sqrt(252)) if len(r_i) > 1 else np.nan
                log.warning(
                    "%s has only %d aligned days (need %d) — using default loadings",
                    isin, len(common), min_obs,
                )
                loadings[isin] = _default_loading(factor_codes, total_vol)
                continue

            y = r_i.loc[common].to_numpy()
            X = factor_returns.loc[common, factor_codes].to_numpy()
            X_aug = np.column_stack([np.ones(len(X)), X])  # intercept

            try:
                coef, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
            except np.linalg.LinAlgError as e:
                total_vol = float(r_i.std() * np.sqrt(252))
                log.warning("OLS failed for %s (%s) — using default loadings", isin, e)
                loadings[isin] = _default_loading(factor_codes, total_vol)
                continue

            alpha = float(coef[0])
            betas = {f: float(coef[1 + k]) for k, f in enumerate(factor_codes)}

            y_hat = X_aug @ coef
            resid = y - y_hat
            idio_vol = float(resid.std(ddof=1) * np.sqrt(252))

            ss_res = float(np.sum(resid ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

            loadings[isin] = {
                "betas":     {f: round(b, 4) for f, b in betas.items()},
                "alpha":     round(alpha, 6),
                "idio_vol":  round(idio_vol, 4),
                "r_squared": round(r_squared, 4),
                "n_obs":     int(len(common)),
            }

        return loadings

    # ----------------------------------------------------- summary helpers

    def loadings_to_dataframe(self, loadings: dict[str, dict]) -> pd.DataFrame:
        """Flatten ``build_loadings`` output to a tidy DataFrame for inspection."""
        rows = []
        for isin, d in loadings.items():
            row = {"isin": isin, "alpha": d["alpha"], "idio_vol": d["idio_vol"],
                   "r_squared": d["r_squared"], "n_obs": d["n_obs"]}
            for f, b in d["betas"].items():
                row[f"β_{f}"] = b
            rows.append(row)
        return pd.DataFrame(rows)
