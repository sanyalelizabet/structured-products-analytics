"""Streamlit view: manually add a product to the active user portfolio.

Reuses the field schema in :mod:`src.portfolio_entry` so this form, the
CLI prompter, and any future JSON-template generator all share one
source of truth.

State is held in ``st.session_state`` via :mod:`app.portfolio_source` —
the same dispatcher every other view reads from.  This view is only
mounted in **user mode**; demo mode is read-only and routes elsewhere.
A "Download as JSON" button lets the user export the running portfolio;
persistent server-side storage is a planned follow-up.
"""
from __future__ import annotations

import datetime as dt
import json
from typing import Any

import pandas as pd
import streamlit as st

import hashlib

from app import portfolio_source as ps
from app import portfolio_storage as storage
from app.formatting import chf
from src.portfolio_entry import (
    ALL_TYPES,
    FIELDS,
    Field,
    ParseError,
    PRODUCT_TYPE_FULL_NAME,
    fields_for_product,
    finalise_row,
    parse_list_float,
    parse_list_iso_date,
    parse_list_str,
    validate_row,
    validate_row_errors,
)


# Section headers — match the FIELDS grouping in src.portfolio_entry.
# Entries whose anchor field isn't in the active product type are simply
# skipped at render time (harmless no-op).
_SECTION_HEADERS = {
    "product_id":            "**1 · Identification**",
    "issuer":                "**2 · Issuer**",
    "underlyings":           "**3 · Underlying**",
    "denomination":          "**4 · Sizing & your purchase**",
    "initial_fixing_date":   "**5 · Dates**",
    "barrier_pct":           "**6 · Barrier & coupon**",
    "autocall_obs_dates":    "**7 · Autocall features**",
    "protection_pct":        "**6 · CPN payoff**",
}


# ──────────────────────────────────────────────────────────────────────────
# Widget rendering — one Streamlit widget per Field kind
# ──────────────────────────────────────────────────────────────────────────

def _label(f: Field) -> str:
    """Streamlit widget labels render markdown (incl. coloured text).

    Required fields get a red ":red[*]" marker — explicit and obvious
    without the visual noise of starring every field.  Optional fields
    get a muted "(optional)" suffix.
    """
    if f.required:
        return f"{f.label}  :red[*]"
    return f"{f.label}  *(optional)*"


def _render_field(f: Field, key: str) -> Any:
    """Render the right Streamlit widget for ``f`` and return its raw value.

    Returns the widget value directly — typed conversion (e.g. list parsing)
    happens in :func:`_collect_field_values` so we can show field-level
    errors next to the offending widget.
    """
    help_text = f.help or None
    placeholder = f.placeholder or None

    if f.kind == "choice":
        idx = 0
        if f.default and f.choices and f.default in f.choices:
            idx = f.choices.index(f.default)
        return st.selectbox(_label(f), f.choices or [], index=idx,
                            key=key, help=help_text)

    if f.kind == "str":
        return st.text_input(_label(f), value=str(f.default or ""),
                             key=key, help=help_text,
                             placeholder=placeholder)

    if f.kind == "float":
        default = float(f.default) if f.default is not None else 0.0
        # format="%g" keeps the box readable for both 0.95 and 209.07.
        # ``placeholder`` on st.number_input doesn't render, so we fall
        # back to surfacing the example in the help tooltip.
        merged_help = (
            f"{help_text}\n\nExample: {placeholder}"
            if (help_text and placeholder)
            else (placeholder or help_text)
        )
        return st.number_input(_label(f), value=default, key=key,
                               help=merged_help, format="%g", step=0.01)

    if f.kind == "int":
        default = int(f.default) if f.default is not None else 0
        merged_help = (
            f"{help_text}\n\nExample: {placeholder}"
            if (help_text and placeholder)
            else (placeholder or help_text)
        )
        return st.number_input(_label(f), value=default, key=key,
                               help=merged_help, step=1, format="%d")

    if f.kind == "iso_date":
        # Use a sentinel "blank" value for optional dates.  Streamlit
        # date_input always returns *something*, so we can't represent
        # "not entered" — we add a checkbox toggle for optional dates.
        if not f.required:
            include = st.checkbox(f"Include {f.label}", value=False,
                                  key=f"{key}_inc")
            if not include:
                return None
        return st.date_input(_label(f), value=dt.date.today(),
                             key=key, help=help_text)

    if f.kind.startswith("list_"):
        # Lists are entered as comma-separated text — the placeholder
        # carries the most concrete guidance (one example value), so
        # surface it directly on the input.
        return st.text_input(
            _label(f) + "  (comma-separated)",
            value="", key=key,
            help=help_text,
            placeholder=placeholder or "e.g. AAPL.US, MSFT.US",
        )

    raise NotImplementedError(f"No widget for kind {f.kind!r}")


def _parse_widget_value(f: Field, raw: Any) -> tuple[Any, str | None]:
    """Convert a widget's raw value to the schema-typed value.

    Returns ``(value, error_message)``. On success, ``error_message`` is
    None. On failure (e.g. unparseable list), returns ``(None, "...")``.
    """
    if raw is None or raw == "":
        if f.required:
            return None, "Required field"
        return None, None

    try:
        if f.kind == "iso_date":
            # raw is a datetime.date
            return raw.strftime("%Y-%m-%d"), None
        if f.kind == "list_str":
            v = parse_list_str(raw)
        elif f.kind == "list_float":
            v = parse_list_float(raw)
        elif f.kind == "list_iso_date":
            v = parse_list_iso_date(raw)
        else:
            v = raw
    except ParseError as e:
        return None, str(e)

    if f.required and (v is None or v == [] or v == ""):
        return None, "Required field"
    return v, None


# ──────────────────────────────────────────────────────────────────────────
# Form: walk FIELDS, render section headers, render widgets, submit
# ──────────────────────────────────────────────────────────────────────────

def _render_form(
    ptype: str,
    *,
    form_key: str | None = None,
    widget_key_prefix: str | None = None,
    submit_label: str = "Add to portfolio",
    extra_button_label: str | None = None,
) -> tuple[dict[str, Any] | None, bool]:
    """Render the form for ``ptype``.

    Reused by both the main "Add product" flow and the Edit dialog —
    the caller supplies a unique ``form_key`` and ``widget_key_prefix``
    so widget state doesn't collide between the two contexts.

    Returns
    -------
    (row, extra_clicked)
        ``row``           : parsed + finalised row dict on successful submit,
                            else ``None``.
        ``extra_clicked`` : ``True`` iff the optional secondary button was
                            clicked (used by the Edit dialog for "Cancel").
    """
    form_key = form_key or f"add_product_{ptype}"
    widget_key_prefix = widget_key_prefix or f"fld_{ptype}"

    relevant = [f for f in fields_for_product(ptype) if f.name != "product_type"]

    # Group fields into sections so each numbered section can render
    # inside its own bordered box.  A field starts a new section if its
    # ``name`` is a key in ``_SECTION_HEADERS``; otherwise it joins the
    # current section.  Fields appearing before any section anchor land
    # in a fall-through "Other" bucket (shouldn't happen with the curated
    # FIELDS list, but the form stays robust if it does).
    sections: list[tuple[str, list[Field]]] = []
    current_header: str | None = None
    current_fields: list[Field] = []
    for f in relevant:
        if f.name in _SECTION_HEADERS:
            if current_fields:
                sections.append((current_header or "**Other**", current_fields))
            current_header = _SECTION_HEADERS[f.name]
            current_fields = [f]
        else:
            current_fields.append(f)
    if current_fields:
        sections.append((current_header or "**Other**", current_fields))

    raw_inputs: dict[str, Any] = {}
    with st.form(key=form_key, clear_on_submit=False):
        for header, fields in sections:
            with st.container(border=True):
                st.markdown(header)
                # Two-column grid inside each section so the form scans
                # vertically without becoming a single long stripe.
                cols = st.columns(2)
                col_idx = 0
                for f in fields:
                    with cols[col_idx]:
                        raw_inputs[f.name] = _render_field(
                            f, key=f"{widget_key_prefix}_{f.name}",
                        )
                    col_idx = 1 - col_idx

        # Action buttons live outside the section boxes — clear visual
        # break between "what to fill in" and "what to do with it".
        st.markdown("")
        if extra_button_label is not None:
            btn_cols = st.columns(2)
            with btn_cols[0]:
                extra_clicked = st.form_submit_button(extra_button_label,
                                                     width="stretch")
            with btn_cols[1]:
                submitted = st.form_submit_button(submit_label, type="primary",
                                                  width="stretch")
        else:
            extra_clicked = False
            submitted = st.form_submit_button(submit_label, type="primary",
                                              width="stretch")

    if extra_clicked:
        return None, True
    if not submitted:
        return None, False

    row: dict[str, Any] = {"product_type": ptype}
    field_errors: list[str] = []
    for f in relevant:
        v, err = _parse_widget_value(f, raw_inputs[f.name])
        if err:
            field_errors.append(f"{f.label}: {err}")
        row[f.name] = v

    if field_errors:
        st.error("Could not add product — please fix:\n\n"
                 + "\n".join(f"- {e}" for e in field_errors))
        return None, False

    return finalise_row(row), False


# ──────────────────────────────────────────────────────────────────────────
# Running user portfolio (read via app.portfolio_source)
# ──────────────────────────────────────────────────────────────────────────

def _is_empty(v: Any) -> bool:
    """Treat None / empty string / empty list / NaN as 'missing'."""
    if v is None:
        return True
    if isinstance(v, str) and not v.strip():
        return True
    if isinstance(v, list) and not v:
        return True
    if isinstance(v, float):
        import math
        return math.isnan(v)
    return False


def _row_completeness(row: dict) -> tuple[int, int]:
    """Count empty optional fields for the running-portfolio summary."""
    ptype = str(row.get("product_type", "")).upper()
    n_empty_opt = 0
    n_total_opt = 0
    for f in fields_for_product(ptype):
        if f.name == "product_type" or f.required:
            continue
        n_total_opt += 1
        if _is_empty(row.get(f.name)):
            n_empty_opt += 1
    return n_empty_opt, n_total_opt


def _missing_required(row: dict) -> list[str]:
    """Required fields that are empty — flagged in red on the card."""
    ptype = str(row.get("product_type", "")).upper()
    missing = []
    for f in fields_for_product(ptype):
        if f.name == "product_type" or not f.required:
            continue
        if _is_empty(row.get(f.name)):
            missing.append(f.label)
    return missing


def _render_product_card(idx: int, row: dict) -> None:
    """One product card with key fields + Edit / Delete actions.

    Missing required fields are flagged in red; missing optional fields
    are summarised as "N optional empty" so the user can spot rows that
    need filling in before save/analytics."""
    with st.container(border=True):
        cols = st.columns([4, 1, 1])
        ptype = str(row.get("product_type", "?")).upper()
        pid = row.get("product_id", "?") or "(no ID)"
        issuer = row.get("issuer") or "—"
        maturity = row.get("maturity_date") or "—"
        notional = row.get("notional")
        ccy = row.get("currency") or ""

        with cols[0]:
            full_name = PRODUCT_TYPE_FULL_NAME.get(ptype, ptype)
            st.markdown(f"**{full_name}** · `{pid}`")
            st.caption(f"Issuer: {issuer}")

            meta_cols = st.columns(3)
            meta_cols[0].caption(f"Maturity: **{maturity}**")
            if notional is not None and not _is_empty(notional):
                meta_cols[1].caption(f"Notional: **{chf(notional, 2)} {ccy}**")
            else:
                meta_cols[1].caption(":red[Notional: —]")

            n_empty, _ = _row_completeness(row)
            missing_req = _missing_required(row)
            if missing_req:
                meta_cols[2].caption(
                    f":red[⚠ {len(missing_req)} required missing]"
                )
            elif n_empty:
                meta_cols[2].caption(f"_{n_empty} optional empty_")
            else:
                meta_cols[2].caption(":green[✓ Complete]")

            if missing_req:
                st.error(
                    "Missing required fields: " + ", ".join(missing_req[:5])
                    + ("…" if len(missing_req) > 5 else "")
                )

        with cols[1]:
            if st.button("Edit", key=f"edit_{idx}",
                         width="stretch"):
                st.session_state["_editing_idx"] = idx
                st.rerun()
        with cols[2]:
            if st.button("Delete", key=f"del_{idx}",
                         width="stretch"):
                rows = st.session_state.get(ps.SESSION_ROWS_KEY, [])
                if 0 <= idx < len(rows):
                    removed = rows.pop(idx)
                    st.toast(
                        f"Removed {removed.get('product_type','?')} "
                        f"{removed.get('product_id','?')}",
                        icon="❌",
                    )
                st.rerun()


def _render_running_portfolio() -> None:
    """Show the running user portfolio with per-product cards + actions."""
    rows = st.session_state.get(ps.SESSION_ROWS_KEY, [])
    st.markdown("---")
    st.subheader(
        f"Your portfolio ({len(rows)} product{'s' if len(rows) != 1 else ''})"
    )

    if not rows:
        st.info(
            "No products added yet. Fill in the form above and click "
            "**Add to portfolio**."
        )
        return

    for i, row in enumerate(rows):
        _render_product_card(i, row)

    st.markdown("")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Save portfolio", type="primary",
                     width="stretch", key="save_open"):
            # Reset dialog state every time the dialog opens so a stale
            # half-completed flow from earlier doesn't bleed through.
            # render() reopens the dialog on the rerun while a flow is active.
            st.session_state.pop("save_name_input", None)
            st.session_state["_save_dialog_state"] = "initial"
            st.session_state["_save_dialog_msg"] = None
            st.rerun()
    with c2:
        st.download_button(
            "Download as JSON",
            data=json.dumps(rows, indent=2, ensure_ascii=False, default=str),
            file_name="user_portfolio.json",
            mime="application/json",
            help="Local download — re-upload from the splash next time.",
            width="stretch",
        )
    with c3:
        if st.button("Clear portfolio", width="stretch",
                     key="clear_user"):
            ps.clear_user_rows()
            st.rerun()


# ──────────────────────────────────────────────────────────────────────────
# Term-sheet quick-fill — Gemini extraction wired to the form's widget state
# ──────────────────────────────────────────────────────────────────────────

def _render_pdf_quickfill() -> None:
    """Show a PDF uploader; on a new upload extract + prefill the form.

    Sits **outside** ``st.form`` so it can trigger reruns on file
    change.  Deduplication is by content SHA-256 — re-renders don't
    re-hit the Gemini API for an already-extracted file.

    Rendered as the *primary* path on the page (always visible, not in
    an expander) so users encounter the fastest entry method first.
    """
    with st.container(border=True):
        st.markdown("**📄 Upload term sheet (PDF) — fastest way**")
        st.caption(
            "We'll extract the product terms with AI and prefill the "
            "form below. You'll be able to review and edit everything "
            "before adding it to your portfolio."
        )
        uploaded = st.file_uploader(
            "PDF term sheet", type=["pdf"],
            key="pdf_quickfill_upload",
            label_visibility="collapsed",
        )
        if uploaded is None:
            return

        pdf_bytes = uploaded.getvalue()
        digest = hashlib.sha256(pdf_bytes).hexdigest()
        already = st.session_state.get("_extracted_for_sha")

        if already == digest:
            # Same file as last extraction — silent no-op so reruns from
            # other widgets don't burn API calls.
            st.success(
                f"Extracted from **{uploaded.name}** — review the fields "
                "below."
            )
            return

        # New file: run extraction once.
        try:
            from src.term_sheet_extractor import extract_term_sheet, validate
            with st.spinner(f"Extracting from {uploaded.name}…"):
                row = extract_term_sheet(pdf_bytes)
        except RuntimeError as e:
            if "GEMINI_API_KEY" in str(e):
                st.error(
                    "Term-sheet extraction needs a Gemini API key.  Add a "
                    "`GEMINI_API_KEY` line to `.streamlit/secrets.toml` "
                    "and rerun."
                )
            else:
                st.error(f"Extraction failed: {e}")
            return
        except Exception as e:                    # noqa: BLE001
            st.error(f"Extraction failed: {type(e).__name__}: {e}")
            return

        # Flip the product-type dropdown to whatever was inferred.
        ptype = str(row.get("product_type") or "").upper()
        if ptype in ALL_TYPES:
            st.session_state["ptype_select"] = ptype
        else:
            st.warning(
                f"Could not classify product type from PDF (got "
                f"{ptype!r}). Pick one manually below."
            )
            ptype = st.session_state.get("ptype_select", "CPN")

        _prefill_form_widgets(ptype, row)

        # Surface any validation warnings from the extractor's own check
        # so the user knows which fields they'll need to fill in by hand.
        warns = validate(row)
        st.session_state["_extracted_warnings"] = warns
        st.session_state["_extracted_for_sha"] = digest
        st.rerun()


def _prefill_form_widgets(ptype: str, row: dict) -> None:
    """Push extracted values into the session_state keys the form uses.

    The form widgets read their initial value from
    ``st.session_state[<key>]`` if present; setting these before the
    form re-renders makes the form appear "auto-filled".
    """
    for f in fields_for_product(ptype):
        if f.name == "product_type":
            continue
        v = row.get(f.name)
        if v is None or v == "":
            continue
        widget_key = f"fld_{ptype}_{f.name}"

        try:
            if f.kind == "iso_date":
                d = dt.datetime.strptime(str(v), "%Y-%m-%d").date()
                st.session_state[widget_key] = d
                if not f.required:
                    # Tick the "include this date" companion checkbox.
                    st.session_state[f"{widget_key}_inc"] = True

            elif f.kind in ("list_str", "list_float", "list_iso_date"):
                if isinstance(v, list):
                    st.session_state[widget_key] = ", ".join(str(x) for x in v)
                else:
                    st.session_state[widget_key] = str(v)

            elif f.kind == "choice":
                if v in (f.choices or []):
                    st.session_state[widget_key] = v

            elif f.kind == "float":
                st.session_state[widget_key] = float(v)

            elif f.kind == "int":
                st.session_state[widget_key] = int(v)

            else:                                  # str
                st.session_state[widget_key] = str(v)
        except (TypeError, ValueError):
            # Don't crash the prefill on a single bad value — the user
            # will see an empty/default widget and can correct manually.
            continue


# ──────────────────────────────────────────────────────────────────────────
# Edit dialog — in-place edit of an already-added product
# ──────────────────────────────────────────────────────────────────────────
# Opens when ``st.session_state['_editing_idx']`` is set by the Edit
# button on a product card.  Reuses the main ``_render_form`` with a
# scoped widget-key prefix so the dialog's widget state can't collide
# with the page's "Add product" form below.

@st.dialog("Edit product", width="large")
def _edit_product_dialog() -> None:
    idx = st.session_state.get("_editing_idx")
    rows = st.session_state.get(ps.SESSION_ROWS_KEY, [])
    if idx is None or idx >= len(rows):
        st.session_state.pop("_editing_idx", None)
        return

    row = rows[idx]
    ptype = str(row.get("product_type", "CPN")).upper()
    if ptype not in ALL_TYPES:
        st.error(f"Cannot edit row {idx}: unknown product_type {ptype!r}.")
        return

    # On first render of this dialog instance, push the row's values
    # into the dialog's widget keys so the form shows current data.
    init_flag_key = f"_edit_dialog_initialised_{idx}"
    if not st.session_state.get(init_flag_key):
        _seed_edit_widgets(ptype, row, idx)
        st.session_state[init_flag_key] = True

    st.caption(
        f"Editing product **{idx + 1}** of "
        f"{len(rows)} — {row.get('product_type','?')} "
        f"{row.get('product_id','?')}"
    )

    updated, cancelled = _render_form(
        ptype,
        form_key=f"edit_product_form_{idx}",
        widget_key_prefix=f"editfld_{idx}",
        submit_label="Save changes",
        extra_button_label="Cancel",
    )

    if cancelled:
        _clear_edit_widgets(ptype, idx)
        st.rerun()

    if updated is not None:
        errors = validate_row_errors(updated)
        if errors:
            st.error(
                "Cannot save changes — please fix:\n\n"
                + "\n".join(f"- {e}" for e in errors)
            )
            # Don't clear widgets / rerun: keep the dialog open with the
            # user's edits intact so they can correct and resubmit.
        else:
            rows[idx] = updated
            _clear_edit_widgets(ptype, idx)
            st.toast("Product updated.", icon="✅")
            st.rerun()


def _seed_edit_widgets(ptype: str, row: dict, idx: int) -> None:
    """Push ``row``'s values into the edit-dialog's widget keys."""
    for f in fields_for_product(ptype):
        if f.name == "product_type":
            continue
        widget_key = f"editfld_{idx}_{f.name}"
        v = row.get(f.name)
        if _is_empty(v):
            continue
        try:
            if f.kind == "iso_date":
                d = dt.datetime.strptime(str(v), "%Y-%m-%d").date()
                st.session_state[widget_key] = d
                if not f.required:
                    st.session_state[f"{widget_key}_inc"] = True
            elif f.kind in ("list_str", "list_float", "list_iso_date"):
                if isinstance(v, list):
                    st.session_state[widget_key] = ", ".join(str(x) for x in v)
                else:
                    st.session_state[widget_key] = str(v)
            elif f.kind == "choice":
                if v in (f.choices or []):
                    st.session_state[widget_key] = v
            elif f.kind == "float":
                st.session_state[widget_key] = float(v)
            elif f.kind == "int":
                st.session_state[widget_key] = int(v)
            else:
                st.session_state[widget_key] = str(v)
        except (TypeError, ValueError):
            continue


def _clear_edit_widgets(ptype: str, idx: int) -> None:
    """Wipe the edit dialog's widget keys so the next Edit starts fresh."""
    for f in fields_for_product(ptype):
        st.session_state.pop(f"editfld_{idx}_{f.name}", None)
        st.session_state.pop(f"editfld_{idx}_{f.name}_inc", None)
    st.session_state.pop(f"_edit_dialog_initialised_{idx}", None)
    st.session_state.pop("_editing_idx", None)


# ──────────────────────────────────────────────────────────────────────────
# Save dialog — owner-key pastebin flow
# ──────────────────────────────────────────────────────────────────────────
# State machine (kept in ``st.session_state['_save_dialog_state']``):
#
#     initial      → name + public-disclosure → Save
#                    ├── new name      → save_new → "reveal_key"
#                    └── existing name → "needs_key"
#     needs_key    → owner-key prompt → Overwrite
#                    ├── correct key → "success"
#                    └── wrong key   → stay (show error in dialog)
#     reveal_key   → show generated key, confirm checkbox, Done
#     success      → confirmation, Done
#
# The dialog stays open across reruns; ``st.rerun()`` from the parent
# Streamlit page closes it.

@st.dialog("Save portfolio")
def _save_dialog() -> None:
    state = st.session_state.get("_save_dialog_state", "initial")

    if state == "initial":
        _render_initial_save()
    elif state == "needs_key":
        _render_overwrite_check()
    elif state == "reveal_key":
        _render_key_reveal()
    elif state == "success":
        _render_save_success()


def _render_initial_save() -> None:
    st.caption(
        "Your portfolio will be saved publicly — anyone using this app "
        "can read it. Updates and deletion require the owner key the "
        "app generates for you below."
    )
    # Pre-fill with the active portfolio's name so saving an edited,
    # previously-loaded portfolio routes through the overwrite/owner-key
    # prompt instead of silently creating a new one.
    if "save_name_input" not in st.session_state:
        active_name = ps.get_name()
        st.session_state["save_name_input"] = (
            active_name if active_name != ps.UNSAVED_PORTFOLIO_NAME else ""
        )
    name = st.text_input("Portfolio name", key="save_name_input",
                         placeholder="e.g. my-structured-portfolio")
    public_ok = st.checkbox(
        "I understand this portfolio will be publicly readable.",
        key="save_public_ack",
    )

    if st.session_state.get("_save_dialog_msg"):
        st.error(st.session_state["_save_dialog_msg"])

    if st.button("Save", type="primary", key="save_submit_initial"):
        if not name.strip():
            st.session_state["_save_dialog_msg"] = "Name is required."
            st.rerun()
        if not public_ok:
            st.session_state["_save_dialog_msg"] = (
                "Please confirm the public-disclosure checkbox."
            )
            st.rerun()
        try:
            storage.slugify(name)
        except ValueError:
            st.session_state["_save_dialog_msg"] = (
                "Name must contain at least one letter or digit."
            )
            st.rerun()

        rows = st.session_state.get(ps.SESSION_ROWS_KEY, [])
        ccy = ps.get_reference_currency()
        if storage.exists(name):
            st.session_state["_save_dialog_msg"] = None
            st.session_state["_save_pending_name"] = name
            st.session_state["_save_dialog_state"] = "needs_key"
            st.rerun()
        else:
            slug, key = storage.save_new(name, rows, reference_currency=ccy)
            st.session_state["_save_generated_key"] = key
            st.session_state["_save_saved_name"] = name
            st.session_state["_save_dialog_msg"] = None
            st.session_state["_save_dialog_state"] = "reveal_key"
            # Bind the in-session portfolio to its newly-saved identity.
            ps.set_name(name)
            st.rerun()


def _render_overwrite_check() -> None:
    name = st.session_state.get("_save_pending_name", "")
    st.warning(
        f"A portfolio called **{name}** already exists. Enter its owner "
        "key to overwrite, or go back and pick a different name."
    )
    owner_key = st.text_input(
        "Owner key", type="password", key="save_key_input",
        placeholder="the key shown when the portfolio was first saved",
    )

    if st.session_state.get("_save_dialog_msg"):
        st.error(st.session_state["_save_dialog_msg"])

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("← Back", key="save_back"):
            st.session_state["_save_dialog_state"] = "initial"
            st.session_state["_save_dialog_msg"] = None
            st.rerun()
    with col_b:
        if st.button("Overwrite", type="primary", key="save_do_overwrite"):
            rows = st.session_state.get(ps.SESSION_ROWS_KEY, [])
            ccy = ps.get_reference_currency()
            try:
                storage.overwrite(name, owner_key, rows,
                                  reference_currency=ccy)
            except storage.AuthError:
                st.session_state["_save_dialog_msg"] = (
                    "Owner key doesn't match. Check the key, or use a "
                    "different name."
                )
                st.rerun()
            else:
                st.session_state["_save_saved_name"] = name
                st.session_state["_save_dialog_state"] = "success"
                st.session_state["_save_dialog_msg"] = None
                ps.set_name(name)
                st.rerun()


def _render_key_reveal() -> None:
    name = st.session_state.get("_save_saved_name", "")
    key = st.session_state.get("_save_generated_key", "")
    st.success(f"Saved as **{name}**.")
    st.error(
        "**Save the owner key shown below.** This is the only time it "
        "will be displayed — if you lose it you won't be able to update "
        "or delete this portfolio (you can always Save As under a new "
        "name)."
    )
    st.code(key, language="text")
    confirmed = st.checkbox(
        "I've saved my owner key somewhere safe.",
        key="save_key_confirmed",
    )
    if st.button("Done", type="primary", disabled=not confirmed,
                 key="save_reveal_done"):
        # Wipe the plaintext key from session state once the user has
        # acknowledged saving it.
        st.session_state.pop("_save_generated_key", None)
        st.session_state["_save_last_message"] = (
            f"Saved as '{name}'."
        )
        st.session_state.pop("_save_dialog_state", None)
        st.rerun()


def _render_save_success() -> None:
    name = st.session_state.get("_save_saved_name", "")
    st.success(f"Updated **{name}**.")
    if st.button("Done", type="primary", key="save_success_done"):
        st.session_state["_save_last_message"] = f"Updated '{name}'."
        st.session_state.pop("_save_dialog_state", None)
        st.rerun()


# ──────────────────────────────────────────────────────────────────────────
# Public entry
# ──────────────────────────────────────────────────────────────────────────

def _render_page_header() -> None:
    """Title (with portfolio name), reference-currency picker, prominent Save."""
    name = ps.get_name()

    # Top row: title + Save button.  Putting Save up here (rather than
    # buried under the product list) makes it the obvious next step once
    # the user has finished editing.
    head_title, head_save = st.columns([4, 1])
    with head_title:
        st.title(f"Portfolio Manager — {name}")
    with head_save:
        st.write("")  # vertical alignment under the title baseline
        n_rows = len(st.session_state.get(ps.SESSION_ROWS_KEY, []))
        save_disabled = (n_rows == 0)
        if st.button(
            "Save portfolio",
            type="primary",
            width="stretch",
            disabled=save_disabled,
            help=(
                "Save this portfolio publicly with a name and owner key. "
                "Add at least one product first."
                if save_disabled else
                "Save this portfolio publicly with a name and owner key. "
                "Anyone can read it; only the owner-key holder can "
                "update or delete it."
            ),
            key="header_save_portfolio",
        ):
            st.session_state.pop("save_name_input", None)
            st.session_state["_save_dialog_state"] = "initial"
            st.session_state["_save_dialog_msg"] = None
            st.rerun()

    st.caption(
        "Manage your portfolio: add products manually, extract them from "
        "term-sheet PDFs, save with a name + owner key, or download as JSON."
    )

    # Currency picker — drives the reference currency used by every
    # analytics view via PortfolioAnalytics(reference_currency=...).
    current_ccy = ps.get_reference_currency()
    try:
        idx = ps.SUPPORTED_CURRENCIES.index(current_ccy)
    except ValueError:
        idx = 0
    col_a, _ = st.columns([1, 4])
    with col_a:
        picked = st.selectbox(
            "Reference currency",
            options=ps.SUPPORTED_CURRENCIES,
            index=idx,
            help=(
                "All analytics views roll the portfolio up into this "
                "currency. Stored with the portfolio when you save it."
            ),
            key="portfolio_currency_picker",
        )
    if picked != current_ccy:
        ps.set_reference_currency(picked)
        st.rerun()


def render() -> None:
    """Streamlit view entry point. Wired from ``app/streamlit_app.py``."""
    # Re-open the save dialog on every rerun while a save flow is active.
    # A @st.dialog only renders when its function is called during a run;
    # the internal st.rerun() between steps (initial → reveal_key /
    # needs_key → success) would otherwise close it before those later
    # steps — including the owner-key reveal and overwrite prompt — show.
    if st.session_state.get("_save_dialog_state"):
        _save_dialog()

    # After a successful save, show a persistent banner with a "View
    # analytics" CTA so the user can immediately use their portfolio.
    last_msg = st.session_state.pop("_save_last_message", None)
    if last_msg:
        with st.container(border=True):
            cta_a, cta_b = st.columns([3, 1])
            cta_a.success(
                f"{last_msg}  This portfolio is now active for the "
                "**Product / Portfolio / Stress Testing / Factor Stress** views."
            )
            with cta_b:
                st.write("")  # vertical alignment
                if st.button("View analytics", type="primary",
                             width="stretch",
                             key="post_save_view_analytics"):
                    # The sidebar reads ``active_view`` directly when
                    # rendering its button group, so a single state set
                    # is enough to highlight the Portfolio button on the
                    # next render. Land on the portfolio-level overview —
                    # the natural view after building a whole portfolio.
                    st.session_state["active_view"] = "Portfolio"
                    st.rerun()

    _render_page_header()
    st.markdown("---")

    st.subheader("Add a product")

    # ── Primary action: upload a term-sheet PDF for AI extraction ────────
    _render_pdf_quickfill()
    extracted_warns = st.session_state.get("_extracted_warnings")
    if extracted_warns:
        st.warning(
            "Term sheet extracted, but some fields couldn't be filled:\n\n"
            + "\n".join(f"- {w}" for w in extracted_warns)
            + "\n\nPlease complete them manually below."
        )

    # OR-divider before the manual fallback.
    st.markdown(
        "<div style='text-align:center; color:#666; "
        "margin: 1.0rem 0 0.75rem 0; font-size: 0.85rem; "
        "letter-spacing: 0.1em;'>OR ENTER MANUALLY</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "All fields are required unless labelled _(optional)_. "
        "Leave optional fields blank if you don't have the data."
    )

    # Codes stay as the option values (downstream code keys on these),
    # but the user only ever sees the full names via format_func.
    ptype_codes = sorted(ALL_TYPES)
    # Establish a stable session-state key so the PDF extractor can flip
    # the dropdown to whatever product_type it inferred.
    if "ptype_select" not in st.session_state:
        st.session_state["ptype_select"] = "CPN"
    ptype = st.selectbox(
        "Product type",
        options=ptype_codes,
        format_func=lambda c: PRODUCT_TYPE_FULL_NAME.get(c, c),
        help="Pick the product family the term sheet you're entering belongs to.",
        key="ptype_select",
    )

    row, _ = _render_form(ptype)
    if row is not None:
        # Hard contract violations (cardinality, etc.) must block — they
        # would crash the analytics views downstream.
        errors = validate_row_errors(row)
        if errors:
            st.error(
                "Cannot add product — please fix and resubmit:\n\n"
                + "\n".join(f"- {e}" for e in errors)
            )
        else:
            warns = validate_row(row)
            if warns:
                st.warning(
                    "Product added with validation warnings:\n\n"
                    + "\n".join(f"- {w}" for w in warns)
                )
            else:
                st.success(
                    f"Added {row.get('product_type')} {row.get('product_id')} "
                    "to your portfolio."
                )
            ps.append_user_row(row)

    # Open the edit dialog if a row's Edit button was clicked on the
    # previous render — Streamlit dialogs are invoked from the script
    # body, not from inside the button handler.
    if st.session_state.get("_editing_idx") is not None:
        _edit_product_dialog()

    _render_running_portfolio()
