"""Tests for the portfolio-source dispatcher and the user-entry schema.

The dispatcher (:mod:`app.portfolio_source`) is the single source of
truth for which portfolio the app is analysing.  These tests pin:

* mode dispatch (``demo`` / ``user`` / unset → demo);
* schema parity — a user-entered row of any product type carries every
  key the corresponding demo row does, so downstream analytics see one
  unified DataFrame shape.
"""
from __future__ import annotations

import pandas as pd
import pytest


# ──────────────────────────────────────────────────────────────────────────
# Streamlit session-state shim
# ──────────────────────────────────────────────────────────────────────────
# Tests run outside Streamlit, but ``portfolio_source`` reads
# ``st.session_state``.  The fixture below substitutes a plain dict so
# the dispatcher works in a plain pytest context.


@pytest.fixture
def st_state(monkeypatch):
    import streamlit as st
    fake_state: dict = {}
    monkeypatch.setattr(st, "session_state", fake_state, raising=False)
    return fake_state


# ──────────────────────────────────────────────────────────────────────────
# Dispatcher
# ──────────────────────────────────────────────────────────────────────────

class TestDispatcher:

    def test_default_returns_demo(self, st_state):
        from app.portfolio_source import get_active_portfolio
        from data.portfolio import portfolio as demo
        out = get_active_portfolio()
        assert out is demo  # singleton identity, not just equality

    def test_explicit_demo_mode_returns_demo(self, st_state):
        from app.portfolio_source import get_active_portfolio, set_mode
        from data.portfolio import portfolio as demo
        set_mode("demo")
        assert get_active_portfolio() is demo

    def test_user_mode_with_empty_rows_returns_empty_df(self, st_state):
        from app.portfolio_source import get_active_portfolio, set_mode
        set_mode("user")
        out = get_active_portfolio()
        assert isinstance(out, pd.DataFrame)
        assert len(out) == 0

    def test_user_mode_returns_rows_as_dataframe(self, st_state):
        from app.portfolio_source import (
            get_active_portfolio, set_mode, append_user_row,
        )
        set_mode("user")
        append_user_row({"product_id": "X", "product_type": "CPN"})
        out = get_active_portfolio()
        assert isinstance(out, pd.DataFrame)
        assert len(out) == 1
        assert out.iloc[0]["product_id"] == "X"

    def test_clear_mode_falls_back_to_demo(self, st_state):
        from app.portfolio_source import (
            get_active_portfolio, set_mode, clear_mode,
        )
        from data.portfolio import portfolio as demo
        set_mode("user")
        clear_mode()
        assert get_active_portfolio() is demo

    def test_invalid_mode_raises(self, st_state):
        from app.portfolio_source import set_mode
        with pytest.raises(ValueError, match="Invalid portfolio mode"):
            set_mode("euphoric")

    def test_clear_mode_preserves_user_rows(self, st_state):
        """Switching back to user mode after clearing should restore the
        in-progress portfolio — clear_mode only resets the dispatcher."""
        from app.portfolio_source import (
            set_mode, clear_mode, append_user_row,
            get_active_portfolio,
        )
        set_mode("user")
        append_user_row({"product_id": "X", "product_type": "CPN"})
        clear_mode()
        set_mode("user")
        assert len(get_active_portfolio()) == 1


# ──────────────────────────────────────────────────────────────────────────
# Schema parity — user-entered rows must be a superset of demo rows
# ──────────────────────────────────────────────────────────────────────────

class TestSchemaParity:
    """The user-entry form's finalise_row output must include every key
    that the corresponding demo row in ``data/portfolio.py`` carries.

    Having extra keys (e.g. issuer chain, disclosures) is fine — pandas
    builds the union and the demo rows simply have NaN in those columns.
    Having *fewer* keys would break downstream code that does direct
    column access on the merged DataFrame.
    """

    @pytest.fixture(scope="class")
    def demo_rows_by_type(self):
        from data.portfolio import p1, p2, p6, p7
        return {"BRC": p1, "MBRC": p2, "AC_BRC": p6, "CPN": p7}

    def _mock_user_row(self, ptype: str, ref: dict) -> dict:
        """Build a finalised user row using values copied from the demo
        row (for shared keys) and reasonable placeholders elsewhere."""
        from src.portfolio_entry import fields_for_product, finalise_row
        row = {"product_type": ptype}
        for f in fields_for_product(ptype):
            if f.name == "product_type":
                continue
            v = ref.get(f.name)
            if v is None:
                if f.kind.startswith("list"):
                    v = []
                elif f.kind == "float":
                    v = 1.0
                elif f.kind == "int":
                    v = 1
                elif f.kind == "iso_date":
                    v = "2026-01-01"
                else:
                    v = "X"
            row[f.name] = v
        return finalise_row(row)

    @pytest.mark.parametrize("ptype", ["BRC", "MBRC", "AC_BRC", "CPN"])
    def test_user_row_is_superset_of_demo_row(self, demo_rows_by_type, ptype):
        ref = demo_rows_by_type[ptype]
        user_row = self._mock_user_row(ptype, ref)
        missing = set(ref.keys()) - set(user_row.keys())
        assert not missing, (
            f"User-entered {ptype} row is missing demo keys: {sorted(missing)}"
        )

    def test_user_rows_aggregate_into_a_valid_dataframe(
        self, demo_rows_by_type,
    ):
        """A mixed-type user portfolio should build into a single
        DataFrame with no column-key surprises."""
        rows = [
            self._mock_user_row(t, ref)
            for t, ref in demo_rows_by_type.items()
        ]
        df = pd.DataFrame(rows)
        assert len(df) == 4
        assert set(df["product_type"]) == {"BRC", "MBRC", "AC_BRC", "CPN"}


# ──────────────────────────────────────────────────────────────────────────
# CPN-specific: round-trip user row through the product class
# ──────────────────────────────────────────────────────────────────────────

class TestRoundTripCPNThroughProductClass:

    def test_user_built_cpn_row_works_in_product_class(self):
        """A row produced by finalise_row for a CPN must be consumable
        by ``CapitalProtectionNote`` without errors."""
        from src.capital_protection_note import CapitalProtectionNote
        from src.portfolio_entry import finalise_row

        row = finalise_row({
            "product_type":       "CPN",
            "product_id":         "CH_TEST",
            "type_style":         "European",
            "currency":           "USD",
            "issuer":             "Test issuer",
            "underlyings":        ["Amazon"],
            "underlying_isins":   ["US0231351067"],
            "initial_levels":     [209.07],
            "strike":             [209.07],
            "denomination":       1000.0,
            "position_units":     10,
            "cost_price":         1.00,
            "purchase_date":      "2026-03-04",
            "initial_fixing_date":"2026-02-25",
            "maturity_date":      "2026-09-01",
            "day_count":          "ACT/360",
            "coupon":             0.0,
            "protection_pct":     0.95,
            "participation_pct":  0.52,
        })
        prod = CapitalProtectionNote(pd.Series(row), final_level=0.0)
        s = prod.summary()
        assert s["product_type"] == "CPN"
        assert s["total_notional"] == 10_000.0
        assert s["protection_pct"] == 0.95
