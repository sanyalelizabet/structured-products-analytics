"""Term-sheet extractor — POC using Google Gemini.

Takes a structured-product term sheet PDF and returns a dict matching the
portfolio-row schema used by ``data/portfolio.py``.  Fields the term
sheet can't possibly know (``position_units``, ``cost_price``,
``purchase_date``, live ``current_spots``…) are returned as ``None`` for
the user to fill in.

Usage (CLI):

    py -3.10 -m src.term_sheet_extractor data/term_sheets_samples/your_file.pdf

Usage (importable):

    from src.ingestion.term_sheet_extractor import extract_term_sheet
    row = extract_term_sheet("data/term_sheets_samples/your_file.pdf")
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

try:
    import tomllib  # py >= 3.11
except ImportError:
    import tomli as tomllib  # type: ignore


# Schema description handed to Gemini.  Mirrors the row dicts in
# ``data/portfolio.py``.  We embed it in the prompt as a structured spec
# so the model knows exactly what to fill and what to leave null.
SCHEMA_SPEC = """
{
  "product_id":                 "string  // ISIN of the certificate",
  "product_type":               "string  // one of: BRC, MBRC, AC_BRC, CPN",
  "type_style":                 "string  // 'European' or 'American'",

  "issuer":                     "string  // legal entity issuing the note",
  "issuer_rating":              "string  // e.g. 'A3' if disclosed, else null",
  "issuer_country":             "string  // ISO-2 country code",
  "guarantor":                  "string  // null if no guarantor",
  "guarantor_rating":           "string  // null if no guarantor",
  "keep_well_agreement":        "string  // null if none",

  "underlyings":                "list[string]   // names of underlying assets",
  "underlying_isins":           "list[string]   // ISINs of underlyings",
  "currency":                   "string         // ISO-3 currency code",

  "denomination":               "number  // face value per certificate",
  "issue_price":                "number  // issue price per certificate (usually = denomination at par)",
  "issue_size":                 "number  // total number of certificates issued",

  "spot_reference_price":       "number  // initial spot reference (single-underlying)",
  "initial_levels":             "list[number]  // initial fixing levels per underlying",
  "strike":                     "list[number]  // strike per underlying",
  "strike_pct_of_spot":         "number        // strike / spot_reference, typically 0.6-1.0",

  "protection_pct":             "number  // CPN only — capital protection as fraction (e.g. 0.95)",
  "capital_protection_amount":  "number  // CPN only — protection amount in currency per certificate",
  "participation_pct":          "number  // CPN only — upside participation rate (e.g. 0.52)",
  "number_of_underlyings":      "number  // CPN only — disclosed B = denomination / strike",

  "barrier_pct":                "number  // BRC/MBRC/AC_BRC only — barrier as fraction of the initial fixing level (e.g. 0.60)",

  "coupon":                     "number  // annualised coupon rate as fraction (0.08 for 8%)",
  "coupon_dates":               "list[string]  // ISO dates YYYY-MM-DD; for zero-coupon use ['maturity_date']",
  "day_count":                  "string  // typically 'ACT/360' or '30/360'",

  "initial_fixing_date":        "string  // YYYY-MM-DD",
  "payment_date":               "string  // YYYY-MM-DD — issue settlement date",
  "last_trading_day":           "string  // YYYY-MM-DD",
  "final_fixing_date":          "string  // YYYY-MM-DD",
  "maturity_date":              "string  // YYYY-MM-DD — repayment date",

  "bond_npv_at_issue":          "number  // issuer-disclosed bond-leg NPV at issue (CPN)",
  "implied_irr_at_issue":       "number  // issuer-disclosed implied IRR at issue (CPN)"
}
""".strip()


EXTRACTION_PROMPT = f"""You are extracting structured-product terms from a single term sheet PDF.

Return a single JSON object matching exactly the schema below.  Rules:

  1. Fields that don't apply to this product type → null.
     - CPN-only fields (protection_pct, participation_pct,
       capital_protection_amount, number_of_underlyings,
       bond_npv_at_issue, implied_irr_at_issue) → null for BRC/MBRC/AC_BRC.
     - barrier_pct → null for CPN.
  2. Fields the term sheet doesn't disclose → null.  Do not guess.
  3. Dates → ISO format YYYY-MM-DD.
  4. Percentages → fractions (8% → 0.08), not basis points or %.
  5. ``product_type`` must be one of: BRC, MBRC, AC_BRC, CPN.  Infer from
     payoff: "capital protection" + "participation" = CPN, "barrier" +
     "worst-of" = MBRC, "barrier" + single underlying = BRC, "autocall"
     or "early redemption" observations = AC_BRC.
  6. For single-underlying products, ``underlyings``, ``underlying_isins``,
     ``initial_levels``, and ``strike`` are length-1 lists.
  7. Output ONLY the JSON object, no markdown fences, no prose.

Schema:
{SCHEMA_SPEC}
"""


# ──────────────────────────────────────────────────────────────────────────
# API key loading
# ──────────────────────────────────────────────────────────────────────────

def _load_api_key() -> str:
    """Read GEMINI_API_KEY from .streamlit/secrets.toml."""
    secrets_path = Path(".streamlit") / "secrets.toml"
    if not secrets_path.exists():
        raise RuntimeError(
            f"Cannot find {secrets_path}.  Run from project root and ensure "
            "the secrets file exists."
        )
    with open(secrets_path, "rb") as f:
        secrets = tomllib.load(f)
    key = secrets.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY not set in .streamlit/secrets.toml.  Add a line:\n"
            '    GEMINI_API_KEY = "your-key-here"'
        )
    return key


# ──────────────────────────────────────────────────────────────────────────
# Extraction
# ──────────────────────────────────────────────────────────────────────────

def extract_term_sheet(
    pdf_source: str | Path | bytes | bytearray,
    model: str = "gemini-2.5-flash",
) -> dict[str, Any]:
    """Send a term sheet PDF to Gemini and return the parsed row dict.

    ``pdf_source`` may be a filesystem path *or* the raw PDF bytes (the
    latter is what Streamlit's ``file_uploader`` hands you).  Raises on
    API errors or malformed JSON.  Caller is responsible for cross-
    checking the result against the live portfolio schema.
    """
    from google import genai
    from google.genai import types

    if isinstance(pdf_source, (bytes, bytearray)):
        pdf_bytes = bytes(pdf_source)
    else:
        path = Path(pdf_source)
        if not path.exists():
            raise FileNotFoundError(path)
        pdf_bytes = path.read_bytes()

    client = genai.Client(api_key=_load_api_key())

    response = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(
                data=pdf_bytes,
                mime_type="application/pdf",
            ),
            EXTRACTION_PROMPT,
        ],
    )

    raw = (response.text or "").strip()
    # Defensive: strip accidental markdown fences if the model adds them.
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.rsplit("```", 1)[0].strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Gemini returned non-JSON output:\n{raw[:500]}..."
        ) from e


# ──────────────────────────────────────────────────────────────────────────
# Validation helper — flags fields that look wrong
# ──────────────────────────────────────────────────────────────────────────

REQUIRED_ALL = {
    "product_id", "product_type", "currency", "issuer",
    "underlyings", "underlying_isins",
    "denomination", "initial_fixing_date", "maturity_date",
}

REQUIRED_BY_TYPE = {
    "CPN":    {"protection_pct", "participation_pct", "strike"},
    "BRC":    {"barrier_pct", "strike", "coupon"},
    "MBRC":   {"barrier_pct", "strike", "coupon"},
    "AC_BRC": {"barrier_pct", "strike", "coupon"},
}


def validate(row: dict[str, Any]) -> list[str]:
    """Return a list of validation warnings (empty list = clean)."""
    warnings: list[str] = []

    missing_all = [f for f in REQUIRED_ALL if row.get(f) in (None, "", [])]
    for f in missing_all:
        warnings.append(f"Missing required field: {f!r}")

    ptype = row.get("product_type")
    if ptype not in REQUIRED_BY_TYPE:
        warnings.append(f"Unknown product_type: {ptype!r}")
    else:
        for f in REQUIRED_BY_TYPE[ptype]:
            if row.get(f) in (None, "", []):
                warnings.append(f"Missing required field for {ptype}: {f!r}")

    # Sanity checks on list lengths
    n_u = len(row.get("underlyings") or [])
    n_i = len(row.get("underlying_isins") or [])
    n_k = len(row.get("strike") or [])
    n_l = len(row.get("initial_levels") or [])
    if not (n_u == n_i == n_k == n_l) and n_u > 0:
        warnings.append(
            f"Mismatched list lengths: underlyings={n_u} isins={n_i} "
            f"strike={n_k} initial_levels={n_l}"
        )

    return warnings


# ──────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────

def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: py -3.10 -m src.term_sheet_extractor <pdf-path>",
              file=sys.stderr)
        return 2
    pdf_path = Path(sys.argv[1])
    print(f"Extracting from: {pdf_path}", file=sys.stderr)
    row = extract_term_sheet(pdf_path)
    print(json.dumps(row, indent=2, ensure_ascii=False))

    warnings = validate(row)
    if warnings:
        print("\n--- Validation warnings ---", file=sys.stderr)
        for w in warnings:
            print(f"  ! {w}", file=sys.stderr)
    else:
        print("\nValidation: clean.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
