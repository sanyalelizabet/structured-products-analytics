"""Named-scenario behaviour tests.

For each preset in :data:`FACTOR_SCENARIO_PRESETS` we run the engine and
assert that the **simulated factor paths** exhibit the qualitative
behaviour the scenario name implies.  This is the user-facing contract:
the name on the dropdown must mean what it says.

Strategy
--------
* Use a small synthetic factor DB with realistic vols.
* Run with ``idio_intensity = 0.0`` and a deterministic ``NoiseSampler``
  so the sole source of structure is the scenario itself — making
  per-path noise unable to flip qualitative claims.
* Use ``n_paths = 25`` so the median path is stable.
* Test on **factor index paths** (``res["factor_paths"][code]["median"]``)
  base 100, since presets define factor-level shocks.

Scenarios covered
-----------------
* COVID March 2020   — V-shape, sector dispersion, USD safe-haven, recovery
* Inflation 2022     — energy up, tech down hardest, defensive HC holds
* Tech Wreck         — TECH << MKT, slow grinding recovery
* Tariffs            — gradual broad decline, USD up, deal rally
* Oil Spike          — ENERGY ↑↑, TECH < MKT, mean-reversion of oil
* V-Shape Sell-off   — sharp drop + sharp bounce in factor paths
* Slow Bleed         — monotone downward staircase, no recovery
* Custom             — flat baseline (no events)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

from data.factor_scenarios import FACTOR_SCENARIO_PRESETS
from src.factor_engine import FACTORS, FactorEngine
from src.factor_scenario_engine import FactorScenarioEngine
from src.market_data_engine import MarketDataEngine
from tests.conftest import make_mbrc_row


FACTOR_CODES = list(FACTORS.keys())


# ──────────────────────────────────────────────────────────────────────────
# Synthetic world builder
# ──────────────────────────────────────────────────────────────────────────

def _seed_factor_db(tmp_path, n_days=900, seed=29):
    rng   = np.random.default_rng(seed)
    end   = pd.Timestamp.today().normalize()
    dates = pd.bdate_range(end=end, periods=n_days)

    sigmas = {"MKT": 0.15, "TECH": 0.22, "HC": 0.12,
              "FIN": 0.20, "ENERGY": 0.28, "FX": 0.08}

    rows = []
    for code in FACTOR_CODES:
        ticker, key, _ = FACTORS[code]
        rets = rng.normal(0, sigmas[code] / np.sqrt(252), n_days)
        prices = 100.0 * np.exp(np.cumsum(rets))
        for d, p in zip(dates, prices):
            rows.append({"date": d, "isin": key, "ticker": ticker, "price": p})

    pd.DataFrame(rows).to_csv(tmp_path / "prices.csv", index=False)


def _portfolio(maturity="2028-06-01"):
    """Two-asset portfolio with a long enough horizon to span the scenarios."""
    row = make_mbrc_row(
        initial_levels=[100.0, 100.0],
        strikes=[100.0, 100.0],
        current_spots=[100.0, 100.0],
        maturity_date=maturity,
    )
    row["underlyings"]      = ["AMD_TEST", "NESN_TEST"]
    row["underlying_isins"] = ["TEST_AMD", "TEST_NESN"]
    return pd.DataFrame([row])


def _loadings():
    return {
        "TEST_AMD": {
            "betas":     {"MKT": 1.3, "TECH": 1.2, "HC": 0.0, "FIN": 0.0,
                          "ENERGY": 0.0, "FX": 0.1},
            "alpha": 0.0, "idio_vol": 0.30, "r_squared": 0.65, "n_obs": 750,
        },
        "TEST_NESN": {
            "betas":     {"MKT": 0.6, "TECH": 0.0, "HC": 0.4, "FIN": 0.0,
                          "ENERGY": 0.0, "FX": -0.05},
            "alpha": 0.0, "idio_vol": 0.12, "r_squared": 0.55, "n_obs": 750,
        },
    }


@pytest.fixture
def engine(tmp_path):
    _seed_factor_db(tmp_path)
    mock_client = MagicMock()
    mde = MarketDataEngine(client=mock_client, db_path=str(tmp_path / "prices.csv"))
    mde.fetch_daily_prices = MagicMock(return_value=None)
    fe = FactorEngine(mde)
    return FactorScenarioEngine(
        portfolio=_portfolio(),
        loadings=_loadings(),
        factor_engine=fe,
        idio_intensity=0.0,           # deterministic — no idio noise
        mean_reversion_kappa=0.5,
        n_paths=25,
    )


def _run(engine, name):
    """Run a named preset and return the result dict."""
    return engine.run_path_scenario(FACTOR_SCENARIO_PRESETS[name])


def _factor_median(res, code):
    """Median factor index path (base 100), as ndarray."""
    return res["factor_paths"][code]["median"].to_numpy()


def _factor_dates(res, code):
    return res["factor_paths"][code]["date"].to_list()


def _terminal_pct(res, code):
    """Terminal % return of the median factor path vs base 100."""
    arr = _factor_median(res, code)
    return float(arr[-1] / arr[0] - 1.0) * 100


def _trough_pct(res, code):
    """Deepest drawdown of the median factor path, in %."""
    arr = _factor_median(res, code)
    return float(arr.min() / arr[0] - 1.0) * 100


# ══════════════════════════════════════════════════════════════════════════
# COVID March 2020 — crash + V-shaped recovery
# ══════════════════════════════════════════════════════════════════════════

class TestCovid:

    def test_mkt_drops_at_least_15pct_at_some_point(self, engine):
        res = _run(engine, "COVID March 2020")
        assert _trough_pct(res, "MKT") < -15.0, \
            f"COVID should produce a meaningful drawdown — got {_trough_pct(res,'MKT'):.1f}%"

    def test_v_shape_terminal_above_trough(self, engine):
        """V-shape: terminal level should be materially above the trough."""
        res = _run(engine, "COVID March 2020")
        arr = _factor_median(res, "MKT")
        gap = (arr[-1] - arr.min()) / arr[0] * 100
        assert gap > 10.0, f"COVID V-shape: expected >10pp recovery, got {gap:.1f}pp"

    def test_energy_worst_hit_at_trough(self, engine):
        """ENERGY should bottom deeper than MKT and TECH at trough."""
        res = _run(engine, "COVID March 2020")
        assert _trough_pct(res, "ENERGY") < _trough_pct(res, "MKT")
        assert _trough_pct(res, "ENERGY") < _trough_pct(res, "TECH")

    def test_hc_less_hit_than_mkt(self, engine):
        """Healthcare is defensive — drawdown shallower than broad market."""
        res = _run(engine, "COVID March 2020")
        assert _trough_pct(res, "HC") > _trough_pct(res, "MKT")

    def test_fx_safe_haven_rises_during_crash(self, engine):
        """USD/CHF (FX) ticks UP at the crash trough as risk-off flow."""
        res = _run(engine, "COVID March 2020")
        # FX index at the crash window (day ~30-90) > base 100
        fx = _factor_median(res, "FX")
        # Take a window around day 60 (50 business days in)
        window_max = fx[20:90].max()
        assert window_max > 102.0, f"FX safe-haven spike expected, got peak {window_max:.1f}"


# ══════════════════════════════════════════════════════════════════════════
# Inflation 2022 — equity bear + energy bull, tech worst
# ══════════════════════════════════════════════════════════════════════════

class TestInflation:

    def test_energy_ends_significantly_above_baseline(self, engine):
        res = _run(engine, "Inflation 2022")
        assert _terminal_pct(res, "ENERGY") > 20.0, \
            f"ENERGY should rally substantially — got {_terminal_pct(res,'ENERGY'):.1f}%"

    def test_tech_ends_well_below_mkt(self, engine):
        """Rate-sensitive tech should underperform the broad market."""
        res = _run(engine, "Inflation 2022")
        assert _terminal_pct(res, "TECH") < _terminal_pct(res, "MKT") - 5.0

    def test_hc_outperforms_tech(self, engine):
        """Defensive HC > rate-sensitive TECH."""
        res = _run(engine, "Inflation 2022")
        assert _terminal_pct(res, "HC") > _terminal_pct(res, "TECH")

    def test_cross_sectional_ordering(self, engine):
        """ENERGY > HC > TECH at terminal."""
        res = _run(engine, "Inflation 2022")
        e = _terminal_pct(res, "ENERGY")
        h = _terminal_pct(res, "HC")
        t = _terminal_pct(res, "TECH")
        assert e > h > t, f"Inflation ordering: ENERGY={e:.1f}% HC={h:.1f}% TECH={t:.1f}%"


# ══════════════════════════════════════════════════════════════════════════
# Tech Wreck — concentrated tech selloff
# ══════════════════════════════════════════════════════════════════════════

class TestTechWreck:

    def test_tech_terminal_far_below_mkt(self, engine):
        res = _run(engine, "Tech Wreck")
        gap = _terminal_pct(res, "MKT") - _terminal_pct(res, "TECH")
        assert gap > 15.0, f"Tech Wreck: TECH should trail MKT by >15pp, got {gap:.1f}"

    def test_hc_holds_while_tech_collapses(self, engine):
        """HC should be near-flat or up while TECH plunges."""
        res = _run(engine, "Tech Wreck")
        hc_term   = _terminal_pct(res, "HC")
        tech_term = _terminal_pct(res, "TECH")
        assert hc_term > tech_term + 25.0

    def test_energy_unaffected(self, engine):
        """ENERGY shouldn't move much in a tech-specific drawdown."""
        res = _run(engine, "Tech Wreck")
        assert abs(_terminal_pct(res, "ENERGY")) < 15.0

    def test_no_v_shape_unlike_covid(self, engine):
        """Tech wreck has slow recovery — no V-shape signature."""
        covid = _run(engine, "COVID March 2020")
        wreck = _run(engine, "Tech Wreck")
        # COVID gap (terminal − trough) should be much bigger than Tech Wreck
        covid_gap = (_factor_median(covid, "MKT")[-1] -
                     _factor_median(covid, "MKT").min())
        wreck_gap = (_factor_median(wreck, "MKT")[-1] -
                     _factor_median(wreck, "MKT").min())
        assert covid_gap > wreck_gap


# ══════════════════════════════════════════════════════════════════════════
# Tariffs / Trade War — gradual decline + deal rally
# ══════════════════════════════════════════════════════════════════════════

class TestTariffs:

    def test_tech_underperforms_mkt_due_to_china_exposure(self, engine):
        res = _run(engine, "Tariffs / Trade War")
        assert _terminal_pct(res, "TECH") < _terminal_pct(res, "MKT")

    def test_fx_strengthens(self, engine):
        """USD strengthens vs CHF on risk-off flows."""
        res = _run(engine, "Tariffs / Trade War")
        # FX should peak above baseline at some point during the drawdown
        fx = _factor_median(res, "FX")
        assert fx.max() > 100.5

    def test_intermediate_drawdown_before_rally(self, engine):
        """Path goes negative before the eventual deal rally — i.e. there's
        an intermediate trough below the initial level."""
        res = _run(engine, "Tariffs / Trade War")
        assert _trough_pct(res, "MKT") < -3.0


# ══════════════════════════════════════════════════════════════════════════
# Oil Spike — energy explosion that reverts
# ══════════════════════════════════════════════════════════════════════════

class TestOilSpike:

    def test_energy_peaks_far_above_mkt(self, engine):
        res = _run(engine, "Oil Spike")
        # Within the spike window energy is dramatically up
        e = _factor_median(res, "ENERGY")
        m = _factor_median(res, "MKT")
        spike_idx = int(np.argmax(e))
        gap = (e[spike_idx] - m[spike_idx])
        assert gap > 20.0, f"Oil spike vs MKT gap at peak too small: {gap:.1f}"

    def test_tech_underperforms_mkt(self, engine):
        """Tech weakens on rate fears triggered by the inflation impulse."""
        res = _run(engine, "Oil Spike")
        # Within the spike phase (before reversion), TECH < MKT
        t = _factor_median(res, "TECH")
        m = _factor_median(res, "MKT")
        # mid-window
        mid = len(t) // 3
        assert t[mid] < m[mid]

    def test_energy_reverts_by_end(self, engine):
        """Oil eventually normalises — terminal energy < peak."""
        res = _run(engine, "Oil Spike")
        e = _factor_median(res, "ENERGY")
        assert e[-1] < e.max() - 10.0

    def test_fx_strengthens_during_spike(self, engine):
        res = _run(engine, "Oil Spike")
        fx = _factor_median(res, "FX")
        assert fx.max() > 102.0


# ══════════════════════════════════════════════════════════════════════════
# V-Shape Sell-off — textbook V signature
# ══════════════════════════════════════════════════════════════════════════

class TestVShape:

    def test_path_drops_then_strongly_recovers(self, engine):
        """V-shape essence: a deep trough, with terminal materially above
        the trough.  (We don't pin the terminal at baseline because Fast
        recovery's drift keeps running for the full post-event segment;
        what matters is the *shape*: down hard, then up.)"""
        res = _run(engine, "V-Shape Sell-off")
        arr = _factor_median(res, "MKT")
        trough = arr.min()
        terminal = arr[-1]
        assert trough < 90.0,                  f"V-trough not deep enough: {trough:.1f}"
        assert terminal - trough > 15.0,       f"V-recovery too weak: {terminal - trough:.1f}pp"
        assert terminal >= arr[0] - 5.0,       f"Terminal still well below baseline: {terminal:.1f}"

    def test_trough_in_first_half(self, engine):
        res = _run(engine, "V-Shape Sell-off")
        arr = _factor_median(res, "MKT")
        trough_idx = int(np.argmin(arr))
        assert trough_idx < len(arr) // 2

    def test_recovery_completes_within_horizon(self, engine):
        """After the trough, path must be visibly rising toward baseline."""
        res = _run(engine, "V-Shape Sell-off")
        arr = _factor_median(res, "MKT")
        trough_idx = int(np.argmin(arr))
        post_trough = arr[trough_idx:]
        # Last quarter of post-trough is well above the trough level
        late = post_trough[-len(post_trough) // 4:]
        assert late.mean() > arr[trough_idx] + 5.0


# ══════════════════════════════════════════════════════════════════════════
# Slow Bleed — monotone bear, no recovery
# ══════════════════════════════════════════════════════════════════════════

class TestSlowBleed:

    def test_terminal_well_below_baseline(self, engine):
        res = _run(engine, "Slow Bleed")
        assert _terminal_pct(res, "MKT") < -15.0

    def test_no_meaningful_recovery_at_terminal(self, engine):
        """Terminal level stays close to the trough (no V)."""
        res = _run(engine, "Slow Bleed")
        arr = _factor_median(res, "MKT")
        gap = (arr[-1] - arr.min()) / arr[0] * 100
        assert gap < 5.0, f"Slow Bleed should NOT recover — terminal-trough gap {gap:.1f}pp"

    def test_three_distinct_legs_down(self, engine):
        """Path should make new lows after each shock event."""
        res = _run(engine, "Slow Bleed")
        arr = _factor_median(res, "MKT")
        # Split into three chunks and check each chunk's min is lower than the previous
        third = len(arr) // 3
        m1 = arr[:third].min()
        m2 = arr[third:2 * third].min()
        m3 = arr[2 * third:].min()
        assert m2 < m1 and m3 < m2


# ══════════════════════════════════════════════════════════════════════════
# Custom — empty events, baseline behaviour
# ══════════════════════════════════════════════════════════════════════════

class TestCustom:

    def test_no_events_means_no_shocks(self, engine):
        res = _run(engine, "Custom")
        # With idio=0 and stable initial drift, factor paths stay near base 100.
        for code in FACTOR_CODES:
            arr = _factor_median(res, code)
            # Mean reversion + zero drift + no shocks → stays close to baseline
            assert 80.0 < arr[-1] < 130.0, \
                f"Custom (no shocks) for {code} drifted to {arr[-1]:.1f}"


# ══════════════════════════════════════════════════════════════════════════
# Cross-scenario — distinguishing properties
# ══════════════════════════════════════════════════════════════════════════

class TestCrossScenarioContrasts:
    """Ensure scenarios are *meaningfully different* from each other —
    not just the same shape labelled differently."""

    def test_inflation_vs_covid_energy_diverges(self, engine):
        """Inflation has ENERGY rallying; COVID has ENERGY crashing."""
        infl = _run(engine, "Inflation 2022")
        covid = _run(engine, "COVID March 2020")
        # Inflation: terminal energy positive; COVID: trough deeply negative
        assert _terminal_pct(infl, "ENERGY") > 20.0
        assert _trough_pct(covid, "ENERGY")  < -25.0

    def test_v_shape_recovers_better_than_slow_bleed(self, engine):
        v = _run(engine, "V-Shape Sell-off")
        b = _run(engine, "Slow Bleed")
        assert _terminal_pct(v, "MKT") > _terminal_pct(b, "MKT") + 15.0

    def test_oil_spike_unique_in_energy_outperformance(self, engine):
        """Across COVID, Tech Wreck, Slow Bleed — none should produce ENERGY
        outperforming the broad market the way Oil Spike does."""
        oil  = _run(engine, "Oil Spike")
        tech = _run(engine, "Tech Wreck")
        bleed = _run(engine, "Slow Bleed")

        oil_outperf = max(_factor_median(oil, "ENERGY") - _factor_median(oil, "MKT"))
        tech_outperf = max(_factor_median(tech, "ENERGY") - _factor_median(tech, "MKT"))
        bleed_outperf = max(_factor_median(bleed, "ENERGY") - _factor_median(bleed, "MKT"))

        assert oil_outperf > tech_outperf
        assert oil_outperf > bleed_outperf


# ══════════════════════════════════════════════════════════════════════════
# All presets run without error end-to-end
# ══════════════════════════════════════════════════════════════════════════

class TestAllPresetsRun:
    def test_every_preset_runs_cleanly(self, engine):
        for name in FACTOR_SCENARIO_PRESETS:
            res = _run(engine, name)
            assert "factor_paths" in res
            assert "asset_paths" in res
            for code in FACTOR_CODES:
                arr = _factor_median(res, code)
                assert np.isfinite(arr).all(), f"[{name}] non-finite in factor {code}"
