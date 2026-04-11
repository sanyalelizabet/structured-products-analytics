# =========================
# UI Configuration
# =========================

# Scenario presets for the Stress Testing view.
# Each preset maps to a dict of scenario parameters, or None for Custom.
SCENARIO_PRESETS = {
    "Custom": None,
    "Current": {
        "market_shock": 0,
        "n_shocks": 1,
        "shock_in_days": 2,
        "shock_spacing_days": 0,
        "pre_shock_drift_pa": 0.0,
        "post_shock_drift_pa": 0.0,
    },
    "Down 5%": {
        "market_shock": -5,
        "n_shocks": 1,
        "shock_in_days": 2,
        "shock_spacing_days": 0,
        "pre_shock_drift_pa": 0.05,
        "post_shock_drift_pa": 0.05,
    },
    "Down 10%": {
        "market_shock": -10,
        "n_shocks": 1,
        "shock_in_days": 2,
        "shock_spacing_days": 0,
        "pre_shock_drift_pa": 0.05,
        "post_shock_drift_pa": 0.05,
    },
    "Crash (-20%)": {
        "market_shock": -20,
        "n_shocks": 1,
        "shock_in_days": 0,
        "shock_spacing_days": 0,
        "pre_shock_drift_pa": 0.05,
        "post_shock_drift_pa": 0.05,
    },
    "Recovery (+10% + ERP)": {
        "market_shock": -20,
        "n_shocks": 1,
        "shock_in_days": 1,
        "shock_spacing_days": 0,
        "pre_shock_drift_pa": 0.05,
        "post_shock_drift_pa": 0.05,
    },
}

# Fallback values used when Custom is selected
SCENARIO_CUSTOM_DEFAULT = {
    "market_shock": -10,
    "n_shocks": 1,
    "shock_in_days": 15,
    "shock_spacing_days": 0,
    "pre_shock_drift_pa": 0.05,
    "post_shock_drift_pa": 0.0,
}
