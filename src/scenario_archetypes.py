"""Scenario archetypes — map named presets to per-factor drift dicts.

The UI picks an archetype (e.g. "Fast recovery"); the engine consumes a
numerical per-factor drift dict. This module does the translation, keeping
the engine purely numerical.

For events, the post-event drift is coupled to the shock magnitude, so a
−25 % shock recovers on a steeper slope than a −5 % shock under the same
archetype. Per factor::

    drift_i = sign × log(1 + shock_i/100) / horizon

with ``(sign, horizon)`` from the archetype (:data:`EVENT_RECOVERY_ARCHETYPES`).
``sign = -1`` recovery (drift reverses the shock); ``+1`` continuation; ``0`` flat.

Concrete examples (per-factor), with shock −25 %:

* Continued bear      → drift = +log(0.75)/1   ≈ −0.288  /y  (continues down)
* Stable              → drift = 0
* Slow recovery (2y)  → drift = −log(0.75)/2   ≈ +0.144  /y  (recovers in 2y)
* Fast recovery (6mo) → drift = −log(0.75)/0.5 ≈ +0.575  /y  (V-shape)

For **initial market state** (before any event), there is no shock to
couple to, so a simpler vocabulary is used — see
:data:`INITIAL_MARKET_STATES`.
"""
from __future__ import annotations

import math
from typing import Iterable


# ──────────────────────────────────────────────────────────────────────────
# Archetype tables
# ──────────────────────────────────────────────────────────────────────────

# Event recovery: (sign relative to shock direction, horizon in years).
# The drift formula is ``sign × log(1 + shock/100) / horizon``.
EVENT_RECOVERY_ARCHETYPES: dict[str, tuple[int, float]] = {
    "Continued bear":             (+1, 1.0),
    "Stable (no drift)":          (0,  1.0),
    "Slow recovery (~2y)":        (-1, 2.0),
    "Fast recovery (~6mo)":       (-1, 0.5),
    "Very fast recovery (~1mo)":  (-1, 1.0 / 12.0),
}

# Initial market state — UI label → internal regime key.
#
# Drifts are no longer single scalars broadcast across factors.  Each
# regime carries a **per-factor** drift vector pulled from
# :mod:`src.factor_premiums`, derived from historical returns conditional
# on the trailing-12mo MKT return.  The dropdown stays short and
# regime-flavoured; the engine sees vectors.
INITIAL_MARKET_STATES: dict[str, str] = {
    "Bull": "bull",
    "Flat": "flat",
    "Bear": "bear",
}

DEFAULT_INITIAL_MARKET_STATE = "Flat"
DEFAULT_RECOVERY_ARCHETYPE   = "Stable (no drift)"


# ──────────────────────────────────────────────────────────────────────────
# Translators
# ──────────────────────────────────────────────────────────────────────────

def event_drift_for_factor(shock_pct: float, archetype: str) -> float:
    """Per-factor annualised drift given (shock %, recovery archetype).

    Semantic by archetype sign:

      * ``sign == 0``  → no drift (e.g. "Stable").
      * ``sign == +1`` → continue in the shock direction (e.g. "Continued
        bear" continues bear after a negative shock, or continues bull
        after a positive shock).  Drift = +log(1+shock)/horizon.
      * ``sign == -1`` → recovery toward baseline, **only when the shock
        was negative**.  Positive-shock factors do nothing — the level
        stays where the shock left it.  This matches user intuition:
        "recovery" is meaningful only for downside; an upside shock with a
        recovery archetype shouldn't reverse the rally.
    """
    if archetype not in EVENT_RECOVERY_ARCHETYPES:
        raise ValueError(f"Unknown recovery archetype: {archetype}")
    sign, horizon = EVENT_RECOVERY_ARCHETYPES[archetype]
    if sign == 0:
        return 0.0
    # Recovery-style archetypes (sign = -1) are no-ops for non-negative
    # shocks: there's nothing to recover from.
    if sign == -1 and shock_pct >= 0:
        return 0.0
    # Guard against pathological shocks that would give log of non-positive.
    multiplier = max(1.0 + float(shock_pct) / 100.0, 1e-8)
    log_shock = math.log(multiplier)
    return float(sign) * log_shock / float(horizon)


def event_next_drift_dict(
    shock_dict: dict,
    archetype: str,
    factor_codes: Iterable[str],
) -> dict[str, float]:
    """``next_drift_pa`` dict for an event, per factor, from shock + archetype."""
    return {
        c: event_drift_for_factor(float(shock_dict.get(c, 0.0)), archetype)
        for c in factor_codes
    }


def initial_drift_dict(
    market_state: str,
    factor_codes: Iterable[str],
    premiums=None,
) -> dict[str, float]:
    """Translate the initial-market-state archetype → per-factor drift dict.

    Drifts come from :func:`src.factor_premiums.get_factor_drift`. ``premiums``,
    when supplied, is the regime×factor table to read (the chosen estimator);
    otherwise the default cache is used.
    """
    if market_state not in INITIAL_MARKET_STATES:
        raise ValueError(f"Unknown initial market state: {market_state}")
    regime = INITIAL_MARKET_STATES[market_state]
    from src.factor_premiums import get_factor_drift
    return get_factor_drift(regime, factor_codes, premiums=premiums)


def translate_ui_scenario(ui_scenario: dict, factor_codes: Iterable[str],
                          premiums=None) -> dict:
    """Translate a UI/preset scenario (archetype labels) → engine scenario
    (numerical ``initial_drift_pa`` + per-event ``next_drift_pa``).

    Pass-through keys (``idio_intensity``, ``mean_reversion_kappa``,
    ``label``, ``description`` …) are preserved.
    """
    factor_codes = list(factor_codes)

    initial_state    = ui_scenario.get("initial_market_state", DEFAULT_INITIAL_MARKET_STATE)
    initial_drift_pa = initial_drift_dict(initial_state, factor_codes, premiums=premiums)

    out_events = []
    for ev in ui_scenario.get("events", []) or []:
        shock_dict = dict(ev.get("factor_shock", {}) or {})
        archetype  = ev.get("recovery", DEFAULT_RECOVERY_ARCHETYPE)
        # Each archetype has a (sign, horizon) pair — the horizon bounds
        # how long the recovery drift is meant to run.  Past it the
        # engine reverts to ``initial_drift_pa`` (unless a later event
        # supersedes earlier).
        sign, horizon = EVENT_RECOVERY_ARCHETYPES[archetype]

        # Per-factor drift for the post-event segment.
        #
        # For factors where the recovery archetype is a no-op (sign = −1
        # and shock ≥ 0), we substitute the initial-market-state drift
        # instead of leaving the factor perfectly flat.  Reasoning: the
        # "recovery" semantic only makes sense for downside shocks; on the
        # upside, the realistic post-shock dynamic is to continue at the
        # ambient market regime (bull / stable / bear) — not to freeze.
        next_drift_pa = {}
        for c in factor_codes:
            shock_c = float(shock_dict.get(c, 0.0))
            d = event_drift_for_factor(shock_c, archetype)
            if sign == -1 and shock_c >= 0:
                d = float(initial_drift_pa[c])
            next_drift_pa[c] = d

        out_events.append({
            "day":                      int(ev["day"]),
            "factor_shock":             shock_dict,
            "next_drift_pa":            next_drift_pa,
            "next_drift_horizon_years": float(horizon),
            "recovery":                 archetype,   # kept for UI round-trip
        })

    passthrough = {
        k: v for k, v in ui_scenario.items()
        if k not in {"initial_market_state", "events"}
    }
    return {
        **passthrough,
        "initial_drift_pa": initial_drift_pa,
        "events":            out_events,
    }
