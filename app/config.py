# =========================
# UI Configuration
# =========================
#
# Single-factor "Stress Testing" view presets.  The post-shock drift is
# expressed as a *Recovery archetype* (coupled to the market shock) and
# the pre-shock state is an Initial market-state archetype, both
# consistent with the Factor Stress view.  The view translates these
# labels to numerical drifts via :mod:`src.scenario_archetypes` before
# passing them to ``ScenarioEngine``.

SCENARIO_PRESETS = {
    "Custom": None,
    "Current": {
        "market_shock": 0,
        "n_shocks": 1,
        "shock_in_days": 2,
        "shock_spacing_days": 0,
        "initial_market_state": "Stable",
        "recovery":             "Stable (no drift)",
    },
    "Down 5%": {
        "market_shock": -5,
        "n_shocks": 1,
        "shock_in_days": 2,
        "shock_spacing_days": 0,
        "initial_market_state": "Moderate bull",
        "recovery":             "Slow recovery (~2y)",
    },
    "Down 10%": {
        "market_shock": -10,
        "n_shocks": 1,
        "shock_in_days": 2,
        "shock_spacing_days": 0,
        "initial_market_state": "Moderate bull",
        "recovery":             "Slow recovery (~2y)",
    },
    "Crash (-20%)": {
        "market_shock": -20,
        "n_shocks": 1,
        "shock_in_days": 1,
        "shock_spacing_days": 0,
        "initial_market_state": "Stable",
        "recovery":             "Continued bear",
    },
    "Crash + Fast V-Recovery": {
        "market_shock": -20,
        "n_shocks": 1,
        "shock_in_days": 1,
        "shock_spacing_days": 0,
        "initial_market_state": "Stable",
        "recovery":             "Fast recovery (~6mo)",
    },
}

# Fallback values used when Custom is selected
SCENARIO_CUSTOM_DEFAULT = {
    "market_shock": -10,
    "n_shocks": 1,
    "shock_in_days": 15,
    "shock_spacing_days": 0,
    "initial_market_state": "Stable",
    "recovery":             "Slow recovery (~2y)",
}
