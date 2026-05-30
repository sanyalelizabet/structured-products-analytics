"""American (continuous) barrier monitoring in the stress engines.

Phase 1b: an American BRC must observe its barrier continuously in stress, not
just at maturity.  Continuous monitoring can only find *more* knock-ins than the
final-fixing check, so under the same scenario and Common Random Numbers an
American BRC must show a higher barrier-breach frequency and a no-better tail
than the otherwise-identical European note.  The comparison is fair because the
two runs share the sampler seed → identical price paths; only the observation
style differs.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.scenario_engine import ScenarioEngine
from src.factor_scenario_engine import FactorScenarioEngine
from tests.conftest import make_brc_row, BETA_MAP, VOL_MAP


def _brc_portfolio(style: str) -> pd.DataFrame:
    # Barrier at 90 vs spot 95, ~1y, 20% vol → frequent intraday dips that often
    # recover by maturity: exactly where American and European diverge.
    row = make_brc_row(barrier_pct=0.90, current_spot=95.0)
    row["type_style"] = style
    return pd.DataFrame([row])


_SCENARIO = {
    "market_shock": 0, "n_shocks": 1, "shock_in_days": 0, "shock_spacing_days": 0,
    "pre_shock_drift_pa": 0.0, "post_shock_drift_pa": 0.0,
}


def _run_single_factor(style: str) -> pd.Series:
    eng = ScenarioEngine(
        portfolio=_brc_portfolio(style), beta_map=BETA_MAP, vol_map=VOL_MAP,
        n_paths=500, mean_reversion_kappa=0.0,
    )
    return eng.run_path_scenario(_SCENARIO)["product_df"].iloc[0]


class TestSingleFactorAmericanBarrier:
    def test_american_breaches_more_and_tail_no_better(self):
        eur = _run_single_factor("european")
        am  = _run_single_factor("american")
        assert am["barrier_breach_freq"] > eur["barrier_breach_freq"]
        assert am["pnl_p5"] <= eur["pnl_p5"] + 1e-6      # worse (or equal) tail

    def test_european_unchanged_is_terminal_only(self):
        # Sanity: with a barrier the path cannot reach, both styles agree.
        def run(style):
            row = make_brc_row(barrier_pct=0.01, current_spot=95.0)
            row["type_style"] = style
            eng = ScenarioEngine(portfolio=pd.DataFrame([row]), beta_map=BETA_MAP,
                                 vol_map=VOL_MAP, n_paths=200, mean_reversion_kappa=0.0)
            return eng.run_path_scenario(_SCENARIO)["product_df"].iloc[0]
        assert run("european")["barrier_breach_freq"] == 0.0
        assert run("american")["barrier_breach_freq"] == 0.0

    def test_reproducible_under_crn(self):
        a = _run_single_factor("american")["barrier_breach_freq"]
        b = _run_single_factor("american")["barrier_breach_freq"]
        assert a == b


class TestAutocallableAmericanBarrier:
    """An autocallable whose uncalled paths observe the barrier continuously
    must breach more often than its European-observed twin.  With no autocall
    observation dates every path runs to maturity, isolating the barrier effect."""

    def _run(self, style):
        row = make_brc_row(barrier_pct=0.90, current_spot=95.0)
        row["product_type"] = "AC_BRC"
        row["type_style"]   = style
        row["autocall_obs_dates"]   = []          # never autocalls → pure barrier
        row["autocall_trigger_pct"] = 1.0
        eng = ScenarioEngine(portfolio=pd.DataFrame([row]), beta_map=BETA_MAP,
                             vol_map=VOL_MAP, n_paths=500, mean_reversion_kappa=0.0)
        return eng.run_path_scenario(_SCENARIO)["product_df"].iloc[0]

    def test_american_uncalled_breaches_more_than_european(self):
        eur = self._run("european")
        am  = self._run("american")
        assert am["barrier_breach_freq"] > eur["barrier_breach_freq"]


def _factor_loadings(isin):
    from src.factor_engine import FACTORS
    return {
        isin: {
            "betas": {"MKT": 1.0, "TECH": 0.0, "HC": 0.0,
                      "FIN": 0.0, "ENERGY": 0.0, "FX": 0.0},
            "alpha": 0.0, "idio_vol": 0.20, "r_squared": 0.6, "n_obs": 750,
        }
    }


def _run_factor(tmp_path, style: str):
    from unittest.mock import MagicMock
    from src.factor_engine import FactorEngine
    from src.market_data_engine import MarketDataEngine
    from tests.test_factor_scenario_engine import _seed_factor_db

    isin = "TEST_AMD"
    row = make_brc_row(barrier_pct=0.90, current_spot=95.0)
    row["type_style"] = style
    row["underlyings"] = ["AMD_TEST"]
    row["underlying_isins"] = [isin]
    portfolio = pd.DataFrame([row])

    _seed_factor_db(tmp_path)
    mde = MarketDataEngine(client=MagicMock(), db_path=str(tmp_path / "prices.csv"))
    mde.fetch_daily_prices = MagicMock(return_value=None)
    fe = FactorEngine(mde)

    eng = FactorScenarioEngine(
        portfolio=portfolio, loadings=_factor_loadings(isin), factor_engine=fe,
        risk_free_rates={"CHF": 0.0}, idio_intensity=0.3,
        mean_reversion_kappa=0.0, n_paths=500,
    )
    ui_scenario = {"events": [], "initial_market_state": "Flat"}
    return eng.run_path_scenario(ui_scenario)["product_df"].iloc[0]


class TestFactorAmericanBarrier:
    def test_american_breaches_more_than_european(self, tmp_path):
        eur = _run_factor(tmp_path, "european")
        am  = _run_factor(tmp_path, "american")
        assert am["barrier_breach_freq"] > eur["barrier_breach_freq"]
