"""
Tests for product-level information added to ReverseConvertible:
  - denomination (per-piece nominal) vs total_notional (position)
  - issuer / issuer_rating / issuer_country
"""
import pytest
import pandas as pd
from src.reverse_convertible import ReverseConvertible
from tests.conftest import make_brc_row


_BRC_ROW_KWARGS = {
    "notional", "cost_price", "coupon", "barrier_pct",
    "initial_level", "strike", "current_spot",
    "initial_fixing_date", "maturity_date",
}


def row_with(**overrides):
    """
    Build a BRC row, routing make_brc_row arguments through the helper
    (so list-shaped fields get wrapped properly) and applying any extra
    keys directly via Series mutation.
    """
    constructor_kwargs = {k: v for k, v in overrides.items() if k in _BRC_ROW_KWARGS}
    extra = {k: v for k, v in overrides.items() if k not in _BRC_ROW_KWARGS}
    row = make_brc_row(**constructor_kwargs)
    for k, v in extra.items():
        row[k] = v
    return row


# ─────────────────────────────────────────
# Denomination / total notional
# ─────────────────────────────────────────

class TestDenomination:
    def test_explicit_denomination_drives_total(self):
        rc = ReverseConvertible(row_with(denomination=1000, position_units=10, notional=99999))
        # denomination wins — total = 1000 × 10
        assert rc.denomination == 1000
        assert rc.total_notional == 10_000

    def test_legacy_notional_only_derives_denomination(self):
        # No denomination given → fall back to legacy notional
        rc = ReverseConvertible(row_with(notional=10_000, position_units=10))
        assert rc.total_notional == 10_000
        assert rc.denomination == 1_000

    def test_legacy_notional_one_piece_denomination_equals_notional(self):
        rc = ReverseConvertible(row_with(notional=100_000, position_units=1))
        assert rc.denomination == 100_000
        assert rc.total_notional == 100_000

    def test_notional_alias_equals_total(self):
        rc = ReverseConvertible(row_with(denomination=500, position_units=20))
        # Backwards-compat: self.notional remains and equals total_notional
        assert rc.notional == rc.total_notional == 10_000

    def test_missing_both_raises(self):
        row = make_brc_row()
        # Drop notional and don't add denomination
        row = row.drop("notional")
        with pytest.raises(ValueError, match="denomination|notional"):
            ReverseConvertible(row)

    def test_coupon_payment_uses_total_notional(self):
        rc = ReverseConvertible(
            row_with(denomination=1_000, position_units=10, coupon=0.08,
                     initial_fixing_date="2024-01-01", maturity_date="2025-01-01")
        )
        # Total notional = 10_000 → coupon over 366 days at ACT/360
        expected = 10_000 * 0.08 * 366 / 360
        assert abs(rc.coupon_payment() - expected) < 1e-6

    def test_redemption_scales_with_total_notional(self):
        rc = ReverseConvertible(
            row_with(denomination=1_000, position_units=5, current_spot=110.0, strike=100.0),
            final_levels=[10.0],
        )
        # No breach → redemption = total_notional
        assert rc.redemption() == 5_000


# ─────────────────────────────────────────
# Issuer info
# ─────────────────────────────────────────

class TestIssuer:
    def test_issuer_defaults_to_none_when_absent(self):
        rc = ReverseConvertible(row_with())
        assert rc.issuer is None
        assert rc.issuer_rating is None
        assert rc.issuer_country is None

    def test_empty_string_issuer_treated_as_none(self):
        rc = ReverseConvertible(row_with(issuer="", issuer_rating="   "))
        assert rc.issuer is None
        assert rc.issuer_rating is None

    def test_issuer_fields_read_when_set(self):
        rc = ReverseConvertible(
            row_with(issuer="UBS AG", issuer_rating="A+", issuer_country="CH")
        )
        assert rc.issuer == "UBS AG"
        assert rc.issuer_rating == "A+"
        assert rc.issuer_country == "CH"

    def test_summary_exposes_issuer_fields(self):
        rc = ReverseConvertible(
            row_with(issuer="Leonteq Securities AG", issuer_rating="BBB",
                     issuer_country="CH"),
            final_levels=[10.0],
        )
        s = rc.summary()
        assert s["issuer"] == "Leonteq Securities AG"
        assert s["issuer_rating"] == "BBB"
        assert s["issuer_country"] == "CH"

    def test_summary_exposes_denomination_and_total_notional(self):
        rc = ReverseConvertible(
            row_with(denomination=1000, position_units=10),
            final_levels=[10.0],
        )
        s = rc.summary()
        assert s["denomination"] == 1000
        assert s["position_units"] == 10
        assert s["total_notional"] == 10_000
