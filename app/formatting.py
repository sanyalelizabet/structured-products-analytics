"""Swiss-style numeric formatting for monetary values.

Swiss financial convention uses an apostrophe as the thousands separator
and a period as the decimal mark, e.g. ``1'234'567.00``.  Python's format
mini-language has no apostrophe grouping option, so amounts are first
formatted with the standard comma grouping and the separator is then
substituted.
"""

from __future__ import annotations

import math
from numbers import Real


def chf(value, decimals: int = 2, signed: bool = False) -> str:
    """Return ``value`` in Swiss thousands notation (``1'234'567``).

    Non-numeric or missing values are passed through as an em dash so the
    function can be applied uniformly to mixed columns.  ``signed=True``
    forces an explicit leading ``+``/``-``; ``decimals`` controls the number
    of fractional digits.
    """
    if value is None or not isinstance(value, Real) or (
        isinstance(value, float) and math.isnan(value)
    ):
        return "—"
    sign = "+" if signed else ""
    formatted = f"{value:{sign},.{decimals}f}"
    return formatted.replace(",", "'")
