"""Manual portfolio-row entry.

Schema-driven so the same field metadata can drive a CLI prompter today
and a Streamlit form tomorrow.  Front-ends call into the shared parse
helpers (``parse_float``, ``parse_iso_date`` …) for consistent validation.

CLI usage:

    py -3.10 -m src.portfolio_entry

Produces a complete row dict matching ``data/portfolio.py`` schema,
printed as JSON.  Press ``Ctrl+C`` to abort mid-entry.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable


# ──────────────────────────────────────────────────────────────────────────
# Field metadata (single source of truth for CLI + future Streamlit form)
# ──────────────────────────────────────────────────────────────────────────

# Which product types this field applies to.  None = always.
ALL_TYPES = {"BRC", "MBRC", "AC_BRC", "CPN"}
BARRIER_TYPES = {"BRC", "MBRC", "AC_BRC"}

# Human-readable product names.  Single source of truth for any UI that
# surfaces product types to end users (entry form, product detail view,
# portfolio table headers).  The abbreviation stays visible in parens so
# power users can still cross-reference.
PRODUCT_TYPE_FULL_NAME: dict[str, str] = {
    "BRC":    "Barrier Reverse Convertible (BRC)",
    "MBRC":   "Multi-asset Barrier RC, worst-of (MBRC)",
    "AC_BRC": "Autocallable Barrier RC (AC_BRC)",
    "CPN":    "Capital Protection Note (CPN)",
}


@dataclass
class Field:
    name: str
    label: str
    kind: str                          # str / float / int / iso_date / choice / list_str / list_float / list_iso_date
    required: bool = True
    default: Any = None
    choices: list[str] | None = None    # for kind="choice"
    applies_to: set[str] | None = None  # None = all product types
    help: str = ""
    placeholder: str = ""               # inline example shown when widget is empty


# Curated to the **minimum useful set** for manual entry — every dropped
# field is either rarely populated in real term sheets, easily derived,
# or covered by a sensible default in :func:`finalise_row`.  Fields that
# arrive from PDF extraction (which can be richer than what we ask the
# user to type) are still preserved in the row dict; they just aren't
# user-editable from the manual form.
#
# Sections, in entry order: identifier → issuer → underlying →
# sizing & purchase → dates → product-specific.
FIELDS: list[Field] = [
    # ----- Identification --------------------------------------------------
    Field("product_type", "Product type", "choice", choices=list(sorted(ALL_TYPES)),
          help="BRC = Barrier RC, MBRC = Multi-asset worst-of, "
               "AC_BRC = Autocallable, CPN = Capital Protection."),
    Field("product_id",   "Product ID (ISIN)", "str",
          placeholder="e.g. CH1537766565",
          help="12-character ISIN printed on the term sheet."),
    Field("type_style",   "Settlement style", "choice",
          choices=["European", "American"], default="European",
          help="European = exercised at maturity only. Almost always European."),
    Field("currency",     "Currency", "choice",
          choices=["CHF", "USD", "EUR", "GBP"], default="CHF",
          help="Trading and settlement currency of the certificate."),

    # ----- Issuer ----------------------------------------------------------
    Field("issuer",        "Issuer (legal entity name)", "str",
          placeholder="e.g. Vontobel Financial Products Ltd., DIFC Dubai",
          help="The legal entity that issued the certificate (not the bank brand)."),
    Field("issuer_rating", "Issuer credit rating", "str", required=False,
          placeholder="e.g. A3, Aa2",
          help="Moody's or S&P long-term issuer rating. Leave blank if not disclosed."),

    # ----- Underlying ------------------------------------------------------
    # Single name for BRC / AC_BRC / CPN; multiple comma-separated for MBRC.
    Field("underlyings",      "Underlying name(s)", "list_str",
          placeholder="e.g. Amazon.com Inc.  — or for MBRC: Apple, Microsoft, Tesla",
          help="One name for single-underlying products; comma-separated names for MBRC (worst-of)."),
    Field("underlying_isins", "Underlying ISIN(s)", "list_str",
          placeholder="e.g. US0231351067",
          help="ISIN per underlying, in the same order. Comma-separated for multi-underlying."),
    Field("initial_levels",   "Initial fixing level(s)", "list_float",
          placeholder="e.g. 209.07",
          help="Spot price at the initial fixing date, per underlying."),
    Field("strike",           "Strike level(s)", "list_float",
          placeholder="e.g. 209.07",
          help="Strike per underlying. For ATM products this equals the initial level."),

    # ----- Sizing & your purchase -----------------------------------------
    Field("denomination",   "Denomination (face value per certificate)", "float",
          placeholder="e.g. 1000",
          help="Face value of one certificate, in the product currency."),
    Field("position_units", "Position units (certificates you hold)", "int",
          default=1, placeholder="e.g. 10",
          help="How many certificates you own."),
    Field("cost_price",    "Cost price (fraction of denomination)", "float",
          default=1.00, placeholder="e.g. 1.00 for at-par, 0.98 for 98 % of par",
          help="What you actually paid, as a fraction of denomination. 1.00 = at par."),
    Field("purchase_date", "Purchase date", "iso_date",
          help="The date you actually bought the certificate."),

    # ----- Dates -----------------------------------------------------------
    Field("initial_fixing_date", "Initial fixing date", "iso_date",
          help="When the strike / barrier reference prices were locked in."),
    Field("maturity_date",       "Maturity / repayment date", "iso_date",
          help="When the certificate matures and the payoff is settled."),

    # ----- BRC / MBRC / AC_BRC --------------------------------------------
    Field("barrier_pct", "Barrier (fraction of strike)", "float",
          applies_to=BARRIER_TYPES,
          placeholder="e.g. 0.60 for a 60 % barrier",
          help="Knock-in barrier as a fraction of strike. 0.60 means the barrier "
               "triggers if the underlying ever falls below 60 % of strike."),
    Field("coupon",      "Coupon (annualised, as a fraction)", "float",
          applies_to=BARRIER_TYPES, default=0.0,
          placeholder="e.g. 0.08 for 8 %/yr",
          help="Annualised coupon rate as a decimal. 0.08 means 8 %/yr."),

    # ----- AC_BRC only ----------------------------------------------------
    Field("autocall_obs_dates",   "Autocall observation dates", "list_iso_date",
          applies_to={"AC_BRC"}, required=False,
          placeholder="e.g. 2026-06-01, 2026-12-01",
          help="Comma-separated YYYY-MM-DD dates when the autocall trigger is checked."),
    Field("autocall_trigger_pct", "Autocall trigger (× strike)", "float",
          applies_to={"AC_BRC"}, required=False,
          placeholder="e.g. 1.00 for 100 % of strike",
          help="If the underlying is at-or-above this level on an obs date, the note autocalls."),
    Field("autocall_coupon_memory", "Coupon memory (snowball)", "choice",
          applies_to={"AC_BRC"}, required=False, default="false",
          choices=["false", "true"],
          help="If true, any missed coupons accumulate and are paid at the next trigger date."),

    # ----- CPN-specific ---------------------------------------------------
    Field("protection_pct",    "Capital protection (as a fraction)", "float",
          applies_to={"CPN"},
          placeholder="e.g. 0.95 for 95 % protected",
          help="What share of denomination is guaranteed at maturity. 0.95 = 95 % protected."),
    Field("participation_pct", "Participation (as a fraction)", "float",
          applies_to={"CPN"},
          placeholder="e.g. 0.52 for 52 % participation",
          help="What share of upside above strike the note participates in. 0.52 = 52 %."),
    Field("coupon",            "Coupon (annualised, 0 for zero-coupon)", "float",
          applies_to={"CPN"}, default=0.0,
          placeholder="e.g. 0 for zero-coupon CPN",
          help="Most CPNs are zero-coupon. Set to 0 unless the term sheet specifies."),
]


# ──────────────────────────────────────────────────────────────────────────
# Parse helpers (front-end agnostic — Streamlit will reuse these)
# ──────────────────────────────────────────────────────────────────────────

class ParseError(ValueError):
    pass


def parse_float(s: str) -> float:
    try:
        return float(s.replace(",", "").strip())
    except ValueError:
        raise ParseError(f"Expected a number, got {s!r}")


def parse_int(s: str) -> int:
    try:
        return int(s.replace(",", "").strip())
    except ValueError:
        raise ParseError(f"Expected an integer, got {s!r}")


def parse_iso_date(s: str) -> str:
    """Validate YYYY-MM-DD format and return the trimmed string."""
    s = s.strip()
    try:
        dt.datetime.strptime(s, "%Y-%m-%d")
    except ValueError:
        raise ParseError(f"Expected YYYY-MM-DD, got {s!r}")
    return s


def parse_list_str(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_list_float(s: str) -> list[float]:
    return [parse_float(x) for x in parse_list_str(s)]


def parse_list_iso_date(s: str) -> list[str]:
    return [parse_iso_date(x) for x in parse_list_str(s)]


_PARSERS: dict[str, Callable[[str], Any]] = {
    "str":            lambda s: s.strip(),
    "float":          parse_float,
    "int":            parse_int,
    "iso_date":       parse_iso_date,
    "list_str":       parse_list_str,
    "list_float":     parse_list_float,
    "list_iso_date":  parse_list_iso_date,
}


def parse_field(field: Field, raw: str) -> Any:
    """Parse a raw user-input string into the field's typed value."""
    if field.kind == "choice":
        s = raw.strip()
        if s not in (field.choices or []):
            raise ParseError(
                f"Must be one of {field.choices}; got {s!r}"
            )
        return s
    parser = _PARSERS.get(field.kind)
    if parser is None:
        raise NotImplementedError(f"No parser for kind {field.kind!r}")
    return parser(raw)


# ──────────────────────────────────────────────────────────────────────────
# Field filtering by selected product type
# ──────────────────────────────────────────────────────────────────────────

def fields_for_product(product_type: str) -> list[Field]:
    """Return only the fields that apply to a given product type."""
    return [
        f for f in FIELDS
        if f.applies_to is None or product_type in f.applies_to
    ]


# ──────────────────────────────────────────────────────────────────────────
# Row post-processing (derive notional, default fields, validate consistency)
# ──────────────────────────────────────────────────────────────────────────

def finalise_row(row: dict[str, Any]) -> dict[str, Any]:
    """Fill in derived / defaulted fields. Run AFTER all prompts."""
    # notional = denomination × position_units
    denom = row.get("denomination")
    units = row.get("position_units") or 1
    if denom is not None:
        row["notional"] = float(denom) * int(units)

    # issue_price defaults to denomination if not provided
    if row.get("issue_price") is None and denom is not None:
        row["issue_price"] = float(denom)

    # purchase_date defaults to initial_fixing_date
    if row.get("purchase_date") is None:
        row["purchase_date"] = row.get("initial_fixing_date")

    # coupon_dates defaults to bullet at maturity (single entry)
    if not row.get("coupon_dates") and row.get("maturity_date"):
        row["coupon_dates"] = [row["maturity_date"]]

    # CPN convenience: spot_reference_price defaults to initial_levels[0]
    if row.get("product_type") == "CPN":
        levels = row.get("initial_levels") or []
        if row.get("spot_reference_price") is None and levels:
            row["spot_reference_price"] = float(levels[0])

    # Live-market placeholders.  Defaulting ``current_spots`` to a copy
    # of ``initial_levels`` (rather than ``[None] * n``) keeps the row
    # self-sufficient: it can be passed straight to a product class
    # without first going through the market-data refresh step.  The
    # streamlit pipeline overwrites these with live spots when available.
    n_u = len(row.get("underlyings") or [])
    initial = row.get("initial_levels") or []
    row.setdefault("current_spots",
                   list(initial) if len(initial) == n_u else [None] * n_u)
    row.setdefault("current_spot_dates", [None] * n_u)
    row.setdefault("barrier_breached", False)

    # Schema parity with demo rows in ``data/portfolio.py``: CPN rows carry
    # an explicit ``barrier_pct = None`` so the resulting DataFrame has the
    # same column set whether the row came from the demo or the entry form.
    if row.get("product_type") == "CPN":
        row.setdefault("barrier_pct", None)

    # Coerce the AC_BRC string toggle to a real bool.
    if row.get("product_type") == "AC_BRC" and isinstance(row.get("autocall_coupon_memory"), str):
        row["autocall_coupon_memory"] = (row["autocall_coupon_memory"].lower() == "true")

    # strike_pct_of_spot — informational, useful for the UI.
    levels = row.get("initial_levels") or []
    strikes = row.get("strike") or []
    if levels and strikes and len(levels) == len(strikes) and levels[0]:
        row.setdefault("strike_pct_of_spot", strikes[0] / levels[0])

    # day_count default keeps cash-flow math consistent across all rows.
    row.setdefault("day_count", "ACT/360")

    # ── Schema-parity fills for optional/disclosure fields ─────────────
    # These columns exist in ``data/portfolio.py`` rows; the trimmed
    # entry form deliberately doesn't ask the user for them.  We emit
    # ``None`` so the merged DataFrame produced by pandas has a stable
    # column set regardless of where the row originated (demo / manual
    # entry / PDF extraction / saved JSON).
    _OPTIONAL_DEFAULT_NONE = (
        "issuer_country", "guarantor", "guarantor_rating",
        "keep_well_agreement", "issue_size", "payment_date",
        "last_trading_day", "final_fixing_date",
        "bond_npv_at_issue", "implied_irr_at_issue",
    )
    for k in _OPTIONAL_DEFAULT_NONE:
        row.setdefault(k, None)

    # CPN disclosure fields that the demo row carries but our form does
    # not ask for.  Set to None for parity; the analytic pricer doesn't
    # depend on them.
    if row.get("product_type") == "CPN":
        row.setdefault("capital_protection_amount", None)
        row.setdefault("number_of_underlyings", None)

    return row


def validate_row_errors(row: dict[str, Any]) -> list[str]:
    """Hard errors that would crash downstream product-class construction.

    These are *contract* violations — the resulting row cannot be
    instantiated as a product, so the form must refuse to add it.
    """
    errs: list[str] = []
    ptype = row.get("product_type")

    lens = {
        k: len(row.get(k) or [])
        for k in ("underlyings", "underlying_isins", "initial_levels", "strike")
    }
    if len(set(lens.values())) != 1:
        errs.append(
            "Underlyings, ISINs, initial levels and strikes must all have "
            f"the same number of entries (got {lens})."
        )

    # Single-underlying product types — list cardinality is enforced by
    # the product classes (see CapitalProtectionNote.__init__).
    if ptype in {"BRC", "AC_BRC", "CPN"} and lens["underlyings"] != 1:
        errs.append(
            f"{ptype} requires exactly one underlying; "
            f"got {lens['underlyings']}."
        )

    # MBRC must have at least 2 underlyings to be worst-of.
    if ptype == "MBRC" and lens["underlyings"] < 2:
        errs.append(
            f"MBRC (worst-of) needs at least 2 underlyings; "
            f"got {lens['underlyings']}."
        )

    return errs


def validate_row(row: dict[str, Any]) -> list[str]:
    """Soft warnings: row is consumable but values look inconsistent.

    Returned alongside any hard errors from :func:`validate_row_errors`
    so the UI can surface both: errors block the add, warnings do not.
    """
    warns: list[str] = []
    ptype = row.get("product_type")

    if ptype == "CPN":
        cp_amt = row.get("capital_protection_amount")
        denom = row.get("denomination")
        prot = row.get("protection_pct")
        if cp_amt is not None and denom is not None and prot is not None:
            derived = denom * prot
            if abs(cp_amt - derived) > 0.01:
                warns.append(
                    f"capital_protection_amount={cp_amt} disagrees with "
                    f"denomination × protection_pct = {derived:.2f}."
                )
        nU = row.get("number_of_underlyings")
        strikes = row.get("strike") or []
        if nU is not None and denom is not None and strikes:
            derived_b = denom / strikes[0]
            if abs(nU - derived_b) > 1e-3:
                warns.append(
                    f"number_of_underlyings={nU} disagrees with "
                    f"denomination/strike = {derived_b:.5f}."
                )

    return warns


# ──────────────────────────────────────────────────────────────────────────
# CLI prompter
# ──────────────────────────────────────────────────────────────────────────

def _format_label(f: Field) -> str:
    parts = [f.label]
    if f.choices:
        parts.append(f"[{'/'.join(f.choices)}]")
    if f.default is not None:
        parts.append(f"(default: {f.default})")
    elif not f.required:
        parts.append("(optional, blank to skip)")
    return " ".join(parts) + ": "


def _prompt_one_cli(f: Field) -> Any:
    """Loop until the user enters a parseable value (or skips/defaults)."""
    while True:
        if f.help:
            print(f"  - {f.help}")
        raw = input(_format_label(f))
        if not raw.strip():
            if f.default is not None:
                return f.default
            if not f.required:
                return None
            print("  ! Required field — please enter a value.")
            continue
        try:
            return parse_field(f, raw)
        except ParseError as e:
            print(f"  ! {e}")


_SECTION_HEADERS = {
    "product_type":         "---Identifier---",
    "issuer":               "\n---Issuer chain---",
    "underlyings":          "\n---Underlying(s)---",
    "denomination":         "\n---Sizing---",
    "cost_price":           "\n---Trading-side (your purchase)---",
    "initial_fixing_date":  "\n---Dates---",
    "barrier_pct":          "\n---Barrier terms---",
    "autocall_obs_dates":   "\n---Autocall terms---",
    "protection_pct":       "\n---CPN payoff terms---",
    "capital_protection_amount": "\n---Optional CPN disclosures---",
}


def enter_product_cli() -> dict[str, Any]:
    """Walk the user through entering one product. Returns a finalised row."""
    print("\n=== Add a new structured product ===\n")

    # First field is always product_type — we need it to filter the rest.
    ptype_field = next(f for f in FIELDS if f.name == "product_type")
    print(_SECTION_HEADERS[ptype_field.name])
    row: dict[str, Any] = {}
    row["product_type"] = _prompt_one_cli(ptype_field)

    ptype = row["product_type"]
    relevant = [f for f in fields_for_product(ptype) if f.name != "product_type"]

    for f in relevant:
        if f.name in _SECTION_HEADERS:
            print(_SECTION_HEADERS[f.name])
        row[f.name] = _prompt_one_cli(f)

    row = finalise_row(row)

    warns = validate_row(row)
    if warns:
        print("\n---Validation warnings---")
        for w in warns:
            print(f"  ! {w}")
    else:
        print("\nValidation: clean.")
    return row


# ──────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────

def main() -> int:
    try:
        row = enter_product_cli()
    except KeyboardInterrupt:
        print("\nAborted.", file=sys.stderr)
        return 130

    print("\n=== Final row ===")
    print(json.dumps(row, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
