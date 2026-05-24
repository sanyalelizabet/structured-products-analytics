"""Factor scenario presets — rich event-timeline stress vectors.

Each preset is an **event timeline**: an initial market state, plus a list
of dated events.  Each event applies per-factor shocks AND chooses a
*recovery archetype* that, coupled with the shock magnitude, defines the
drift in the segment that follows.

Schema (UI-friendly, archetype labels)::

    {
        "label":                 str,
        "description":           str,
        "initial_market_state":  str,    # see scenario_archetypes.INITIAL_MARKET_STATES
        "events": [
            {"day": int,
             "factor_shock": {factor: % move},
             "recovery":     str},        # see scenario_archetypes.EVENT_RECOVERY_ARCHETYPES
            …
        ],
        "idio_intensity":        float,
        "mean_reversion_kappa":  float,
    }

Use :func:`src.scenario_archetypes.translate_ui_scenario` to convert to
the engine's numerical form.

Calibration draws on documented historical episodes; these are *coherent
shock vectors*, not literal historical replays.  Each preset is tuned so
that — when run through the engine — the qualitative behaviour matches
the scenario's name.  The accompanying tests in
``tests/test_named_scenarios_behavior.py`` enforce this.
"""
from __future__ import annotations


FACTOR_SCENARIO_PRESETS: dict[str, dict] = {

    # ──────────────────────────────────────────────────────────────────────
    # Default preset — V-Shape sell-off with a very fast (~1 month) snap-back.
    # Listed first so the Factor Stress selectbox lands on it by default.
    "V-Shape Sell-off": {
        "label":       "V-Shape Sell-off",
        "description": "Sharp sell-off, snaps back within a month.",
        "initial_market_state": "Flat",
        "events": [
            {"day":  30,
             "factor_shock": {"MKT": -20.0, "TECH": -25.0, "HC": -10.0,
                              "FIN": -22.0, "ENERGY": -25.0, "FX":  +4.0},
             "recovery": "Very fast recovery (~1mo)"},
        ],
        "idio_intensity":       0.3,
        "mean_reversion_kappa": 0.5,
    },

    # ──────────────────────────────────────────────────────────────────────
    "COVID March 2020": {
        "label":       "COVID March 2020",
        "description": (
            "Pandemic crash + V-shaped recovery.  Initial panic across all "
            "sectors with energy worst hit, defensive HC less affected, USD "
            "spikes as safe-haven, then a fast snap-back rally as central banks "
            "and fiscal policy flood the system.  The recovery is attached to "
            "the crash itself and runs uninterrupted, so the V completes."
        ),
        "initial_market_state": "Flat",
        "events": [
            # Day 30: the crash — the fast (~6mo) snap-back is attached HERE, to
            # the down-shock it must reverse, and no later down-leg interrupts it.
            {"day":  30,
             "factor_shock": {"MKT": -25.0, "TECH": -18.0, "HC": -12.0,
                              "FIN": -32.0, "ENERGY": -45.0, "FX":  +5.0},
             "recovery": "Fast recovery (~6mo)"},
            # Day 270: mild post-recovery normalisation above the pre-crash level.
            {"day": 270,
             "factor_shock": {"MKT":  +3.0, "TECH":  +4.0, "HC":  +2.0,
                              "FIN":  +3.0, "ENERGY":  +3.0, "FX":  -1.0},
             "recovery": "Stable (no drift)"},
        ],
        "idio_intensity":       0.3,
        "mean_reversion_kappa": 0.5,
    },

    # ──────────────────────────────────────────────────────────────────────
    "Inflation 2022": {
        "label":       "Inflation 2022",
        "description": (
            "Sustained rate-driven repricing.  Equities decline in stages "
            "while energy rallies on supply tightness.  Tech crushed by "
            "duration risk; HC defensive.  Final central-bank pivot brings "
            "a slow recovery and energy reversion."
        ),
        "initial_market_state": "Bull",
        "events": [
            # Day 60: first rate-shock
            {"day":  60,
             "factor_shock": {"MKT":  -7.0, "TECH": -12.0, "HC":  -1.0,
                              "FIN":  -3.0, "ENERGY": +20.0, "FX":  +3.0},
             "recovery": "Continued bear"},
            # Day 180: persistent inflation pressure
            {"day": 180,
             "factor_shock": {"MKT":  -8.0, "TECH": -13.0, "HC":  -1.0,
                              "FIN":  -4.0, "ENERGY": +18.0, "FX":  +2.0},
             "recovery": "Continued bear"},
            # Day 300: peak inflation / CB pivot — the final equity down-leg,
            # with energy reverting down. A *slow* (~2y) recovery begins here
            # and runs uninterrupted to the horizon (no later event truncates
            # it), so equities grind back partway — not a V-shape.
            {"day": 300,
             "factor_shock": {"MKT":  -5.0, "TECH":  -9.0, "HC":   0.0,
                              "FIN":  -4.0, "ENERGY": -18.0, "FX":  +1.0},
             "recovery": "Slow recovery (~2y)"},
        ],
        "idio_intensity":       0.3,
        "mean_reversion_kappa": 0.5,
    },

    # ──────────────────────────────────────────────────────────────────────
    "Tech Wreck": {
        "label":       "Tech Wreck",
        "description": (
            "Concentrated tech-sector drawdown (Nasdaq 2000-style).  Tech "
            "down sharply with broader market following modestly; defensives "
            "hold.  Slow grinding recovery, no V-shape."
        ),
        "initial_market_state": "Bull",
        "events": [
            # Day 30: first leg down
            {"day":  30,
             "factor_shock": {"MKT": -10.0, "TECH": -28.0, "HC":  -2.0,
                              "FIN":  -5.0, "ENERGY":  +2.0, "FX":  +2.0},
             "recovery": "Continued bear"},
            # Day 120: aftershock — tech keeps bleeding to the trough, from
            # which a *slow* (~2y) grind back begins and runs uninterrupted to
            # the horizon (no V-shape, only a partial recovery).
            {"day": 120,
             "factor_shock": {"MKT":  -5.0, "TECH": -15.0, "HC":  -1.0,
                              "FIN":  -3.0, "ENERGY":  +1.0, "FX":  +1.0},
             "recovery": "Slow recovery (~2y)"},
        ],
        "idio_intensity":       0.3,
        "mean_reversion_kappa": 0.5,
    },

    # ──────────────────────────────────────────────────────────────────────
    "Tariffs / Trade War": {
        "label":       "Tariffs / Trade War",
        "description": (
            "Gradual deterioration as tariffs escalate; cyclicals and "
            "China-exposed tech hit hardest, USD rallies on risk-off flows.  "
            "Eventually a deal is struck, triggering a rally and recovery."
        ),
        "initial_market_state": "Flat",
        "events": [
            # Day 30: first round of tariffs
            {"day":  30,
             "factor_shock": {"MKT":  -5.0, "TECH": -10.0, "HC":  -2.0,
                              "FIN":  -4.0, "ENERGY": -3.0, "FX":  +3.0},
             "recovery": "Continued bear"},
            # Day 180: escalation to the trough — recovery attached to this
            # down-shock so equities climb back over ~6mo after it.
            {"day": 180,
             "factor_shock": {"MKT":  -6.0, "TECH":  -8.0, "HC":  -2.0,
                              "FIN":  -4.0, "ENERGY": -2.0, "FX":  +3.0},
             "recovery": "Fast recovery (~6mo)"},
            # Day 360: deal struck, relief rally on top of the recovery.
            {"day": 360,
             "factor_shock": {"MKT":  +8.0, "TECH": +14.0, "HC":  +3.0,
                              "FIN":  +6.0, "ENERGY": +4.0, "FX":  -3.0},
             "recovery": "Stable (no drift)"},
        ],
        "idio_intensity":       0.3,
        "mean_reversion_kappa": 0.5,
    },

    # ──────────────────────────────────────────────────────────────────────
    "Oil Spike": {
        "label":       "Oil Spike",
        "description": (
            "Geopolitical / OPEC supply shock — energy spikes, broader "
            "equities decline modestly, tech weakens on rate fears, USD up.  "
            "Oil eventually normalises and equities recover."
        ),
        "initial_market_state": "Flat",
        "events": [
            # Day 30: oil crisis — equities dip and recover over ~6mo (recovery
            # attached to the equity down-shock); energy spikes.
            {"day":  30,
             "factor_shock": {"MKT":  -8.0, "TECH": -15.0, "HC":  -3.0,
                              "FIN":  -5.0, "ENERGY": +35.0, "FX":  +5.0},
             "recovery": "Fast recovery (~6mo)"},
            # Day 240: oil normalises (energy reverts down); equities steady.
            {"day": 240,
             "factor_shock": {"MKT":  +2.0, "TECH":  +3.0, "HC":  +1.0,
                              "FIN":  +2.0, "ENERGY": -30.0, "FX":  -3.0},
             "recovery": "Stable (no drift)"},
        ],
        "idio_intensity":       0.3,
        "mean_reversion_kappa": 0.5,
    },

    # ──────────────────────────────────────────────────────────────────────
    "Slow Bleed": {
        "label":       "Slow Bleed",
        "description": (
            "Drawn-out bear market — three consecutive moderate down-legs "
            "with no relief in between.  No recovery rally; final state "
            "stays in continuation."
        ),
        "initial_market_state": "Flat",
        "events": [
            {"day":  60,
             "factor_shock": {"MKT":  -7.0, "TECH": -10.0, "HC":  -3.0,
                              "FIN":  -8.0, "ENERGY":  -5.0, "FX":  +2.0},
             "recovery": "Continued bear"},
            {"day": 180,
             "factor_shock": {"MKT":  -7.0, "TECH":  -9.0, "HC":  -3.0,
                              "FIN":  -7.0, "ENERGY":  -5.0, "FX":  +2.0},
             "recovery": "Continued bear"},
            {"day": 300,
             "factor_shock": {"MKT":  -7.0, "TECH":  -8.0, "HC":  -3.0,
                              "FIN":  -7.0, "ENERGY":  -5.0, "FX":  +2.0},
             "recovery": "Stable (no drift)"},
        ],
        "idio_intensity":       0.3,
        "mean_reversion_kappa": 0.5,
    },

    # ──────────────────────────────────────────────────────────────────────
    "Custom": {
        "label":       "Custom",
        "description": "Empty starting point.  Add your own events.",
        "initial_market_state": "Flat",
        "events": [],
        "idio_intensity":       0.3,
        "mean_reversion_kappa": 0.5,
    },
}
