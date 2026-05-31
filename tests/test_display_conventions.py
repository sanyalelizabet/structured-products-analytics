"""Display-convention contract tests.

The views label a number of columns and key metrics as percentages or as money.
The values behind those labels are produced one layer below, by
``PortfolioAnalytics.build_product_analytics`` plus the single percent-scaling
step in :func:`src.portfolio_analytics.scale_display_units`. These tests pin that
contract so a column headed "(%)" can never silently revert to carrying a raw
fraction (the regression that previously affected the product view's
"Total Return (%)").

The Streamlit rendering itself (literal header strings) is not exercised here —
that needs a full app harness — but every value that flows into those headers is.
"""
import numpy as np
import pandas as pd
import pytest

from src.portfolio.portfolio_analytics import (
    PERCENT_DISPLAY_COLUMNS,
    PortfolioAnalytics,
    scale_display_units,
)
from tests.conftest import make_portfolio


_FX_MAP = {
    ("CHF", "CHF"): 1.0,
    ("EUR", "CHF"): 1.0,
    ("USD", "CHF"): 1.0 / 1.1,
}
_FX_AS_OF = pd.Timestamp("2026-05-20")


def _raw_product_df():
    """A freshly built product frame in its native (fractional) units."""
    pa = PortfolioAnalytics(
        make_portfolio(), reference_currency="CHF",
        fx_rates=_FX_MAP, fx_as_of=_FX_AS_OF,
    )
    return pa.build_product_analytics()


# ──────────────────────────────────────────────────────────────────────────
# The percent-scaling contract
# ──────────────────────────────────────────────────────────────────────────

class TestPercentScalingContract:
    def test_declared_percent_columns_exist(self):
        """Every column the display layer promises to percent-ify is actually
        produced by the analytics frame — guards against a silent rename."""
        df = _raw_product_df()
        for column in PERCENT_DISPLAY_COLUMNS:
            assert column in df.columns, f"missing percent column {column!r}"

    def test_scaling_multiplies_declared_columns_by_100(self):
        df = _raw_product_df()
        raw = df.copy()
        scale_display_units(df)
        for column in PERCENT_DISPLAY_COLUMNS:
            before = raw[column].to_numpy(dtype=float)
            after = df[column].to_numpy(dtype=float)
            finite = np.isfinite(before) & np.isfinite(after)
            assert np.allclose(after[finite], before[finite] * 100.0), (
                f"{column} was not scaled by exactly 100"
            )

    def test_scaling_leaves_other_columns_untouched(self):
        df = _raw_product_df()
        raw = df.copy()
        scale_display_units(df)
        untouched = [
            "total_cost", "total_payoff", "pnl", "total_notional", "weight_pct",
        ]
        for column in untouched:
            if column not in df.columns:
                continue
            before = raw[column].to_numpy(dtype=float)
            after = df[column].to_numpy(dtype=float)
            finite = np.isfinite(before) & np.isfinite(after)
            assert np.array_equal(after[finite], before[finite]), (
                f"{column} must not be rescaled by scale_display_units"
            )

    def test_scaling_returns_same_object(self):
        """In-place semantics matter: callers hold ``analytics.product_df`` and
        expect to observe the percentages without re-fetching."""
        df = _raw_product_df()
        assert scale_display_units(df) is df

    def test_missing_columns_are_skipped(self):
        partial = pd.DataFrame({"return_pct": [0.05, -0.10], "pnl": [1.0, -2.0]})
        out = scale_display_units(partial)  # no ytm / distance columns present
        assert np.allclose(out["return_pct"], [5.0, -10.0])
        assert np.array_equal(out["pnl"], [1.0, -2.0])


# ──────────────────────────────────────────────────────────────────────────
# Economic identities behind the displayed values
# ──────────────────────────────────────────────────────────────────────────

class TestValueIdentities:
    def test_total_return_pct_matches_pnl_over_cost(self):
        """After scaling, "Total Return (%)" must equal pnl / cost in percent —
        the exact correspondence the product-view header asserts."""
        df = scale_display_units(_raw_product_df())
        nonzero = df["total_cost"] != 0
        expected = df.loc[nonzero, "pnl"] / df.loc[nonzero, "total_cost"] * 100.0
        np.testing.assert_allclose(
            df.loc[nonzero, "return_pct"].to_numpy(dtype=float),
            expected.to_numpy(dtype=float),
            rtol=1e-9, atol=1e-9,
        )

    def test_weight_pct_is_already_percent_and_sums_to_100(self):
        """weight_pct is emitted in percent by the analytics layer (hence it is
        deliberately *excluded* from the scaler) and must form a partition."""
        df = _raw_product_df()
        assert "weight_pct" not in PERCENT_DISPLAY_COLUMNS
        total = float(df["weight_pct"].sum())
        assert abs(total - 100.0) < 1e-6, f"weights sum to {total}, not 100%"

    def test_pnl_is_payoff_minus_cost(self):
        df = _raw_product_df()
        np.testing.assert_allclose(
            df["pnl"].to_numpy(dtype=float),
            (df["total_payoff"] - df["total_cost"]).to_numpy(dtype=float),
            rtol=1e-9, atol=1e-6,
        )
