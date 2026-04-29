"""Factor scenario presets — coherent multi-factor stress vectors.

Each preset specifies per-factor shock magnitudes (in %), event timing,
pre/post drift assumptions and a recommended idiosyncratic intensity.

Calibration draws on documented historical episodes:

* **COVID March 2020**:   Feb 19 – Mar 23 2020 peak-to-trough drawdown
* **Inflation 2022**:     2022 calendar year repricing
* **Tech Wreck**:         Nasdaq 2000-style concentrated tech selloff
* **Tariffs**:            2018 trade-war template, scaled
* **Oil Spike**:          2022 H1 / hypothetical Middle-East supply shock

Scenarios are *coherent* shock vectors calibrated to each episode's factor
cross-section — not literal historical replays. They serve as the seed for
the stress run; the user can override any field via the UI before running.

Each preset has the following schema::

    {
        "label":                  str,    # display name
        "description":            str,    # 1-line UI description
        "factor_shock":           dict,   # % move per shock event, by factor code
        "n_shocks":               int,
        "shock_in_days":          int,
        "shock_spacing_days":     int,
        "factor_drift_pre_pa":    dict,   # annualised drift before first shock
        "factor_drift_post_pa":   dict,   # annualised drift after last shock
        "idio_intensity":         float,  # recommended λ
        "mean_reversion_kappa":   float,  # recommended κ
    }
"""
from __future__ import annotations


# Default zero-drift dict used as a base, overridden per preset where useful.
_ZERO_DRIFT = {f: 0.0 for f in ["MKT", "TECH", "HC", "FIN", "ENERGY", "FX"]}


FACTOR_SCENARIO_PRESETS: dict[str, dict] = {

    # ──────────────────────────────────────────────────────────────────────
    "COVID March 2020": {
        "label":       "COVID March 2020",
        "description": (
            "Pandemic shock — equities crash in 5 weeks, oil collapses, "
            "USD spikes as safe haven. Tech and Healthcare relatively resilient."
        ),
        "factor_shock": {
            "MKT":    -25.0,   # S&P -34% peak-to-trough; calibrated to ~25% over horizon
            "TECH":   -18.0,   # XLK held up better than broad market
            "HC":     -15.0,   # XLV defensive
            "FIN":    -32.0,   # XLF -42% peak-to-trough
            "ENERGY": -45.0,   # XLE -52%; oil briefly negative
            "FX":      +5.0,   # USD up vs CHF (safe-haven flight)
        },
        "n_shocks":           1,
        "shock_in_days":      30,
        "shock_spacing_days": 0,
        "factor_drift_pre_pa": {"MKT":  0.05, "TECH":  0.10, "HC":  0.05,
                                "FIN":  0.05, "ENERGY": 0.0, "FX":  0.0},
        "factor_drift_post_pa":{"MKT":  0.15, "TECH":  0.25, "HC":  0.10,
                                "FIN":  0.10, "ENERGY": 0.10, "FX": -0.02},
        "idio_intensity":      0.3,
        "mean_reversion_kappa": 0.5,
    },

    # ──────────────────────────────────────────────────────────────────────
    "Inflation 2022": {
        "label":       "Inflation 2022",
        "description": (
            "Sustained rate-driven repricing. Tech crushed by duration risk, "
            "energy rallies on supply tightness, USD strengthens on rate spread."
        ),
        "factor_shock": {
            "MKT":    -19.0,
            "TECH":   -33.0,   # Nasdaq -33% in 2022
            "HC":      -2.0,   # XLV barely down
            "FIN":    -11.0,
            "ENERGY": +59.0,   # XLE +64% in 2022
            "FX":      +1.3,   # mild USD strength vs CHF
        },
        "n_shocks":           3,    # gradual repricing through the year
        "shock_in_days":      60,
        "shock_spacing_days": 90,
        "factor_drift_pre_pa": {"MKT":  0.0,  "TECH": -0.10, "HC":  0.05,
                                "FIN":  0.0,  "ENERGY": 0.20, "FX": 0.05},
        "factor_drift_post_pa":{"MKT":  0.05, "TECH":  0.10, "HC":  0.07,
                                "FIN":  0.05, "ENERGY": 0.0,  "FX": -0.02},
        "idio_intensity":      0.3,
        "mean_reversion_kappa": 0.5,
    },

    # ──────────────────────────────────────────────────────────────────────
    "Tech Wreck": {
        "label":       "Tech Wreck",
        "description": (
            "Concentrated tech-sector drawdown (Nasdaq 2000-style). "
            "Tech down hard, broader market follows but defensives hold."
        ),
        "factor_shock": {
            "MKT":    -15.0,
            "TECH":   -45.0,
            "HC":      -5.0,
            "FIN":    -10.0,
            "ENERGY":  +3.0,
            "FX":      +3.0,
        },
        "n_shocks":           2,
        "shock_in_days":      45,
        "shock_spacing_days": 90,
        "factor_drift_pre_pa": {"MKT":  0.0,  "TECH": -0.10, "HC":  0.05,
                                "FIN":  0.0,  "ENERGY": 0.05, "FX": 0.02},
        "factor_drift_post_pa":{"MKT":  0.05, "TECH": -0.05, "HC":  0.05,
                                "FIN":  0.05, "ENERGY": 0.0,  "FX": 0.0},
        "idio_intensity":      0.3,
        "mean_reversion_kappa": 0.5,
    },

    # ──────────────────────────────────────────────────────────────────────
    "Tariffs": {
        "label":       "Tariffs",
        "description": (
            "Trade-war shock — cyclicals and China-exposed tech hit hardest. "
            "Modest broad-market drawdown, USD up vs CHF on risk-off."
        ),
        "factor_shock": {
            "MKT":    -10.0,
            "TECH":   -20.0,   # high China supply-chain exposure
            "HC":      -5.0,
            "FIN":     -8.0,
            "ENERGY":  -5.0,
            "FX":      +5.0,
        },
        "n_shocks":           2,
        "shock_in_days":      30,
        "shock_spacing_days": 60,
        "factor_drift_pre_pa": {"MKT":  0.0,  "TECH": -0.05, "HC":  0.03,
                                "FIN":  0.0,  "ENERGY": 0.0,  "FX": 0.03},
        "factor_drift_post_pa":{"MKT":  0.03, "TECH":  0.05, "HC":  0.05,
                                "FIN":  0.03, "ENERGY": 0.02, "FX": 0.0},
        "idio_intensity":      0.3,
        "mean_reversion_kappa": 0.5,
    },

    # ──────────────────────────────────────────────────────────────────────
    "Oil Spike": {
        "label":       "Oil Spike",
        "description": (
            "Supply-side oil shock (geopolitical or OPEC). Energy rallies, "
            "broader equities decline modestly, USD up, tech weakens on rates."
        ),
        "factor_shock": {
            "MKT":     -8.0,
            "TECH":   -15.0,
            "HC":      -3.0,
            "FIN":     -5.0,
            "ENERGY": +40.0,
            "FX":      +5.0,
        },
        "n_shocks":           1,
        "shock_in_days":      30,
        "shock_spacing_days": 0,
        "factor_drift_pre_pa": {"MKT":  0.0,  "TECH": -0.05, "HC":  0.03,
                                "FIN":  0.0,  "ENERGY": 0.20, "FX": 0.03},
        "factor_drift_post_pa":{"MKT":  0.03, "TECH":  0.05, "HC":  0.05,
                                "FIN":  0.03, "ENERGY": 0.0,  "FX": 0.0},
        "idio_intensity":      0.3,
        "mean_reversion_kappa": 0.5,
    },

    # ──────────────────────────────────────────────────────────────────────
    "Custom": {
        "label":       "Custom",
        "description": "Define your own factor shock vector and timing.",
        "factor_shock": {
            "MKT": -10.0, "TECH": -10.0, "HC": -5.0,
            "FIN": -10.0, "ENERGY":  0.0, "FX":  0.0,
        },
        "n_shocks":             1,
        "shock_in_days":        30,
        "shock_spacing_days":   60,
        "factor_drift_pre_pa":  dict(_ZERO_DRIFT),
        "factor_drift_post_pa": dict(_ZERO_DRIFT),
        "idio_intensity":       0.3,
        "mean_reversion_kappa": 0.5,
    },
}
