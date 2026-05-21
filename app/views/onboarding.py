"""Onboarding splash — pick a portfolio mode before the main app starts.

Two-step state machine driven by ``st.session_state['onboarding_step']``:

* ``"start"`` (default) — choose between Load existing / Create new.
* ``"load_choose"``     — Load was picked; choose between Demo and
                          Upload JSON.

When a terminal choice is made (Demo, Upload success, or Create), this
module sets the portfolio mode via :mod:`app.portfolio_source` and clears
the onboarding step.  The caller in ``streamlit_app.py`` then re-renders
into the main app.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st

from app import portfolio_source as ps
from app import portfolio_storage as storage
from data.portfolio import portfolio as demo_portfolio


SESSION_STEP_KEY = "onboarding_step"

# Splash logo — transparent PNG sits cleanly on the dark background.
# Falls back to the boxed logo if the transparent version is missing,
# and silently skips if neither exists (no app crash on a partial repo).
_ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"
_LOGO_PATH = next(
    (p for p in (_ASSETS_DIR / "logo_no_bg.png", _ASSETS_DIR / "logo.png")
     if p.exists()),
    None,
)


def _render_logo(column_ratio: tuple[int, int, int] = (3, 2, 3)) -> None:
    """App logo centred above the splash heading.  No-op if no logo file.

    Uses a three-column split with the logo column filled responsively
    (``width="stretch"``).  The column ratio controls how wide
    the logo is relative to the page — ``(3, 2, 3)`` puts it at ~25 %
    of page width, comfortably visible without dominating.
    """
    if _LOGO_PATH is None:
        return
    _, mid, _ = st.columns(list(column_ratio))
    with mid:
        st.image(str(_LOGO_PATH), width="stretch")


def is_active() -> bool:
    """True iff the splash should be shown (no mode picked yet)."""
    return ps.get_mode() is None


def _go_to(step: str) -> None:
    st.session_state[SESSION_STEP_KEY] = step
    st.rerun()


def _finish(mode: ps.PortfolioMode) -> None:
    ps.set_mode(mode)
    st.session_state.pop(SESSION_STEP_KEY, None)
    st.rerun()


# ──────────────────────────────────────────────────────────────────────────
# Screen 1 — start
# ──────────────────────────────────────────────────────────────────────────

def _render_start() -> None:
    _render_logo()
    st.markdown(
        "<h1 style='text-align: center; margin-top: 0.5rem;'>"
        "Structured Products Analytics</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align: center; font-size: 1.1rem; color: #aaa; "
        "margin-bottom: 2rem;'>How would you like to start?</p>",
        unsafe_allow_html=True,
    )

    # Fixed-height cards so the two columns line up regardless of how
    # much copy each one carries.  Inside Card A we insert an invisible
    # spacer matching the height that Card B's text_input occupies —
    # that makes the buttons (last element of each card) sit on the
    # same horizontal baseline.
    #
    # The descriptions are deliberately tuned to similar lengths so they
    # wrap to the same number of lines at typical column widths.  Tune
    # both copy and spacer together — they're coupled.
    _CARD_HEIGHT  = 330
    _INPUT_SPACER = 86   # matches a default-sized st.text_input

    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        with st.container(border=True, height=_CARD_HEIGHT):
            st.subheader("Open an existing portfolio")
            st.write(
                "Continue with a portfolio you have previously saved, "
                "or open the demonstration portfolio for an overview "
                "of the analytics."
            )
            # Invisible spacer to align this card's button with Card B's
            # — Card B has a text_input below its description that we
            # don't render here.
            st.markdown(
                f"<div style='height: {_INPUT_SPACER}px;'></div>",
                unsafe_allow_html=True,
            )
            if st.button("Open", type="primary",
                         width="stretch", key="onboard_load"):
                _go_to("load_choose")

    with col_b:
        with st.container(border=True, height=_CARD_HEIGHT):
            st.subheader("Create a new portfolio")
            st.write(
                "Begin a new portfolio. Products can be added manually, "
                "or extracted automatically from uploaded term sheets."
            )
            new_name = st.text_input(
                "Portfolio name",
                placeholder="e.g. my-structured-portfolio",
                key="onboard_new_name",
                help=(
                    "Displayed at the top of the Portfolio Manager. "
                    "You can change it before saving. On save, an owner "
                    "key is issued to authorise later updates."
                ),
            )
            if st.button("Create", type="primary",
                         width="stretch", key="onboard_create"):
                ps.clear_user_rows()
                ps.clear_identity()
                ps.set_reference_currency(ps.DEFAULT_REFERENCE_CURRENCY)
                # Trimmed non-empty name is set immediately so the
                # Portfolio Manager title reflects the user's choice;
                # blank falls through to the "New portfolio" placeholder.
                if new_name and new_name.strip():
                    ps.set_name(new_name.strip())
                _finish("user")


# ──────────────────────────────────────────────────────────────────────────
# Screen 2 — load choose
# ──────────────────────────────────────────────────────────────────────────

def _validate_uploaded_rows(rows: Any) -> tuple[list[dict] | None, str | None]:
    """Lightweight schema check on an uploaded JSON portfolio."""
    if not isinstance(rows, list):
        return None, "Expected a JSON list of product rows at the top level."
    if not rows:
        return None, "Uploaded portfolio is empty."
    required_per_row = {"product_type", "product_id", "currency",
                        "underlyings", "underlying_isins"}
    for i, r in enumerate(rows):
        if not isinstance(r, dict):
            return None, f"Row {i}: expected a JSON object, got {type(r).__name__}."
        missing = required_per_row - set(r.keys())
        if missing:
            return None, (
                f"Row {i} (product_id={r.get('product_id', '?')!r}) is missing "
                f"required field(s): {sorted(missing)}."
            )
    return rows, None


def _render_load_choose() -> None:
    # Slightly smaller (narrower centre column) on the secondary screen.
    _render_logo(column_ratio=(4, 2, 4))
    st.markdown(
        "<h1 style='text-align: center; margin-top: 0.5rem;'>"
        "Load a portfolio</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align: center; color: #aaa; margin-bottom: 1.5rem;'>"
        "Choose a source.</p>",
        unsafe_allow_html=True,
    )

    # Same alignment strategy as the start screen: fixed-height cards
    # with measured spacers that compensate for the differently-sized
    # action widgets in each card (button only / selectbox + button /
    # file_uploader), so the primary action buttons sit on the same
    # horizontal baseline.
    _CARD_HEIGHT       = 360
    # A label-collapsed st.selectbox is shorter than a full selectbox —
    # ~50 px of total vertical space (box + below-margin).  This spacer
    # goes in the Demo card and the Saved-empty branch to compensate
    # for that selectbox in the Saved-populated branch, so all three
    # primary buttons sit on the same baseline.
    _SELECTBOX_SPACER  = 50
    _FILEUPLOAD_TOPPAD = 16   # small top-pad so file uploader doesn't crowd the text

    col_a, col_b, col_c = st.columns(3, gap="large")

    # ---- Demonstration portfolio (button only) ----------------------------
    with col_a:
        with st.container(border=True, height=_CARD_HEIGHT):
            st.subheader("Demonstration portfolio")
            st.write(
                f"A pre-populated portfolio of **{len(demo_portfolio)} "
                "representative products**. Read-only."
            )
            # Spacer so this card's button aligns with the Saved card's
            # button (Saved has a selectbox above its button that this
            # card lacks).
            st.markdown(
                f"<div style='height: {_SELECTBOX_SPACER}px;'></div>",
                unsafe_allow_html=True,
            )
            if st.button("Open demonstration", type="primary",
                         width="stretch", key="onboard_demo"):
                _finish("demo")

    # ---- Saved portfolios — selectbox + button ----------------------------
    with col_b:
        with st.container(border=True, height=_CARD_HEIGHT):
            st.subheader("Saved portfolios")
            summaries = storage.list_saved()
            if not summaries:
                st.write(
                    "No portfolios saved yet. To begin, choose "
                    "**Create a new portfolio** from the previous screen."
                )
                # No portfolios → no selectbox.  Insert the same spacer
                # as the Demo card so the (disabled) action sits at the
                # right height for visual symmetry.
                st.markdown(
                    f"<div style='height: {_SELECTBOX_SPACER}px;'></div>",
                    unsafe_allow_html=True,
                )
                st.button("Open", type="primary",
                          width="stretch",
                          disabled=True, key="onboard_saved_open_disabled")
            else:
                st.write(
                    f"**{len(summaries)} portfolio(s)** available. "
                    "Select one to open."
                )
                picked = st.selectbox(
                    "Pick one",
                    options=[s.name for s in summaries],
                    format_func=lambda n: next(
                        f"{s.name}  —  {s.n_products} products"
                        for s in summaries if s.name == n
                    ),
                    key="onboard_saved_pick",
                    label_visibility="collapsed",
                )
                if st.button("Open", type="primary",
                             width="stretch", key="onboard_saved_open"):
                    payload = storage.load(picked)
                    ps.replace_user_rows(payload.get("products", []))
                    ps.set_name(payload.get("name", picked))
                    ps.set_reference_currency(
                        payload.get("reference_currency",
                                    ps.DEFAULT_REFERENCE_CURRENCY)
                    )
                    _finish("user")

    # ---- Upload from file — file_uploader IS the action -------------------
    # The file_uploader is taller than a selectbox+button stack and acts as
    # both input and action.  Add a small top-pad and no extra spacer; the
    # uploader naturally fills the card's lower half, lining up its top
    # edge approximately with the buttons in the other two cards.
    with col_c:
        with st.container(border=True, height=_CARD_HEIGHT):
            st.subheader("Upload from file")
            st.write(
                "Restore a portfolio from a JSON file previously "
                "downloaded from this application."
            )
            st.markdown(
                f"<div style='height: {_FILEUPLOAD_TOPPAD}px;'></div>",
                unsafe_allow_html=True,
            )
            uploaded = st.file_uploader(
                "Portfolio file (.json)", type=["json"],
                key="onboard_upload",
                label_visibility="collapsed",
            )
            if uploaded is not None:
                try:
                    payload = json.loads(uploaded.read().decode("utf-8"))
                except json.JSONDecodeError as e:
                    st.error(f"File is not valid JSON: {e}")
                else:
                    # Accept two shapes: a bare list of rows (the legacy
                    # Download-as-JSON output) or the full saved-portfolio
                    # envelope with metadata.
                    if isinstance(payload, dict) and "products" in payload:
                        rows_in   = payload.get("products")
                        name_in   = payload.get("name")
                        ccy_in    = payload.get("reference_currency")
                    else:
                        rows_in, name_in, ccy_in = payload, None, None

                    rows, err = _validate_uploaded_rows(rows_in)
                    if err:
                        st.error(err)
                    else:
                        ps.replace_user_rows(rows or [])
                        if name_in:
                            ps.set_name(name_in)
                        ps.set_reference_currency(
                            ccy_in or ps.DEFAULT_REFERENCE_CURRENCY
                        )
                        st.success(
                            f"Loaded {len(rows or [])} product(s). "
                            "Opening portfolio..."
                        )
                        _finish("user")

    st.markdown("")
    if st.button("← Back", key="onboard_back"):
        _go_to("start")


# ──────────────────────────────────────────────────────────────────────────
# Public entry
# ──────────────────────────────────────────────────────────────────────────

def _inject_card_alignment_css() -> None:
    """Make bordered onboarding cards flex columns so the action button
    (the last child of each card) sits at the bottom of the card,
    aligned horizontally across all cards regardless of how much copy
    sits above it.

    The selectors target Streamlit's stable test-ids for the bordered
    container (`stVerticalBlockBorderWrapper`) and the inner content
    block (`stVerticalBlock`).  CSS injected once per render — Streamlit
    deduplicates identical style blocks.
    """
    st.markdown(
        """
        <style>
        div[data-testid="stVerticalBlockBorderWrapper"] > div:first-child {
            height: 100%;
            display: flex;
            flex-direction: column;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] > div:first-child
            > div[data-testid="stVerticalBlock"] {
            flex: 1;
            display: flex;
            flex-direction: column;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] > div:first-child
            > div[data-testid="stVerticalBlock"] > div:last-child {
            margin-top: auto;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render() -> None:
    """Render whichever onboarding screen the user is currently on."""
    _inject_card_alignment_css()
    step = st.session_state.get(SESSION_STEP_KEY, "start")
    if step == "load_choose":
        _render_load_choose()
    else:
        _render_start()
