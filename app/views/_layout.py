"""Shared layout helpers for the Streamlit views."""
from __future__ import annotations


def fit_height(n_rows: int) -> int:
    """Streamlit dataframe height that fits ``n_rows`` rows + header without
    introducing an inner scrollbar.

    Streamlit's default row height is ~35 px and the header is ~38 px.  A
    small bottom padding keeps the last row from sitting flush against the
    table edge.
    """
    return 38 + 35 * max(int(n_rows), 1) + 4
