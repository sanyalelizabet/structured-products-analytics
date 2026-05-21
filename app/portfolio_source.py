"""Single source of truth for the active portfolio.

Every consumer that needs the live portfolio should import
:func:`get_active_portfolio` from this module rather than reading
``data.portfolio.portfolio`` directly.  That gives us one place to
dispatch between the bundled demo portfolio and a user-built one held
in ``st.session_state``.

Session state keys this module reads:

* ``portfolio_mode``       — ``"demo"`` or ``"user"``.  ``None`` (or
                              missing) means the onboarding gate hasn't
                              completed yet; callers should not invoke
                              this function in that state.
* ``user_portfolio_rows``  — list of row dicts, used when mode is
                              ``"user"``.  Empty list ⇒ empty DataFrame.

The function deliberately does not crash on a missing mode (it falls
back to the demo) so a sloppy import order during development doesn't
produce confusing errors.  Production code paths should always have
``portfolio_mode`` set before calling.
"""
from __future__ import annotations

from typing import Literal

import pandas as pd
import streamlit as st

from data.portfolio import portfolio as _demo_portfolio


PortfolioMode = Literal["demo", "user"]

SESSION_MODE_KEY     = "portfolio_mode"
SESSION_ROWS_KEY     = "user_portfolio_rows"
SESSION_NAME_KEY     = "portfolio_name"
SESSION_CURRENCY_KEY = "portfolio_currency"

# Display name to use when the user hasn't saved their portfolio yet.
UNSAVED_PORTFOLIO_NAME = "New portfolio"

# Reference-currency options surfaced in the UI.  The first entry is the
# default for newly-created portfolios.
SUPPORTED_CURRENCIES: tuple[str, ...] = ("CHF", "USD", "EUR", "GBP")
DEFAULT_REFERENCE_CURRENCY = "CHF"


def get_active_portfolio() -> pd.DataFrame:
    """Return the active portfolio as a DataFrame.

    Dispatches on ``st.session_state['portfolio_mode']``.  Unknown or
    missing modes fall back to the demo portfolio.
    """
    mode = st.session_state.get(SESSION_MODE_KEY)
    if mode == "user":
        rows = st.session_state.get(SESSION_ROWS_KEY, [])
        # An empty user portfolio is still a valid state — return an
        # empty DataFrame and let upstream views render an empty state.
        return pd.DataFrame(rows)
    # Default and explicit "demo" case both fall here.
    return _demo_portfolio


def get_mode() -> PortfolioMode | None:
    """Return the currently-selected portfolio mode (or None if unset)."""
    return st.session_state.get(SESSION_MODE_KEY)


def set_mode(mode: PortfolioMode) -> None:
    if mode not in ("demo", "user"):
        raise ValueError(f"Invalid portfolio mode: {mode!r}")
    st.session_state[SESSION_MODE_KEY] = mode


def clear_mode() -> None:
    """Reset mode + portfolio identity (sends the user back to the splash).

    The in-progress ``user_portfolio_rows`` are preserved so switching
    back to user mode restores them.  Name and currency reset to their
    defaults — picking up a different portfolio shouldn't keep stale
    metadata from the previous one.
    """
    st.session_state.pop(SESSION_MODE_KEY, None)
    clear_identity()


def append_user_row(row: dict) -> None:
    """Append a finalised row to the user portfolio."""
    if SESSION_ROWS_KEY not in st.session_state:
        st.session_state[SESSION_ROWS_KEY] = []
    st.session_state[SESSION_ROWS_KEY].append(row)


def replace_user_rows(rows: list[dict]) -> None:
    """Replace the user portfolio wholesale (used by JSON upload)."""
    st.session_state[SESSION_ROWS_KEY] = list(rows)


def clear_user_rows() -> None:
    st.session_state[SESSION_ROWS_KEY] = []


def portfolio_size() -> int:
    """Number of products in the active portfolio (any mode)."""
    return len(get_active_portfolio())


# ──────────────────────────────────────────────────────────────────────────
# Portfolio identity: name + reference currency
# ──────────────────────────────────────────────────────────────────────────
# These are properties of the *whole* portfolio (not individual products).
# They live in session state and are persisted into the saved JSON when
# the user names + saves their portfolio.

def get_name() -> str:
    """Display name of the active portfolio.

    Falls back to a friendly placeholder when nothing else is set
    (e.g. a freshly-created portfolio that hasn't been saved yet, or
    demo mode).
    """
    mode = get_mode()
    if mode == "demo":
        return "Demo portfolio"
    return st.session_state.get(SESSION_NAME_KEY, UNSAVED_PORTFOLIO_NAME)


def set_name(name: str) -> None:
    st.session_state[SESSION_NAME_KEY] = name


def clear_name() -> None:
    st.session_state.pop(SESSION_NAME_KEY, None)


def get_reference_currency() -> str:
    """Currency the portfolio is rolled up in for analytics displays."""
    return st.session_state.get(
        SESSION_CURRENCY_KEY, DEFAULT_REFERENCE_CURRENCY
    )


def set_reference_currency(ccy: str) -> None:
    if ccy not in SUPPORTED_CURRENCIES:
        raise ValueError(
            f"Unsupported reference currency {ccy!r}; "
            f"choose one of {SUPPORTED_CURRENCIES}."
        )
    st.session_state[SESSION_CURRENCY_KEY] = ccy


def clear_identity() -> None:
    """Reset name + currency to defaults (used on Switch portfolio)."""
    clear_name()
    st.session_state.pop(SESSION_CURRENCY_KEY, None)
