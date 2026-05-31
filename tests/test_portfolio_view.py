import numpy as np
import pandas as pd

from app.views import portfolio


class _DummyExpander:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _DummyStreamlit:
    def __init__(self):
        self.dataframes = []
        self.messages = []

    def expander(self, *_args, **_kwargs):
        return _DummyExpander()

    def dataframe(self, value, **kwargs):
        self.dataframes.append((value, kwargs))

    def info(self, message):
        self.messages.append(("info", message))

    def warning(self, message):
        self.messages.append(("warning", message))


def test_render_correlation_matrix_handles_read_only_backing_array(monkeypatch):
    arr = np.array([[0.2, 0.4], [0.4, 0.3]], dtype=float)
    arr.setflags(write=False)
    corr_df = pd.DataFrame(arr, index=["AAA", "BBB"], columns=["AAA", "BBB"])
    underlying_df = pd.DataFrame(
        {
            "isin": ["AAA", "BBB"],
            "underlying": ["Alpha SA", "Beta AG"],
        }
    )
    dummy_st = _DummyStreamlit()
    monkeypatch.setattr(portfolio, "st", dummy_st)

    portfolio._render_correlation_matrix(None, corr_df, underlying_df)

    assert len(dummy_st.dataframes) == 1
    rendered = dummy_st.dataframes[0][0].data
    assert rendered.loc["Alpha SA (AAA)", "Alpha SA (AAA)"] == 1.0
    assert rendered.loc["Beta AG (BBB)", "Beta AG (BBB)"] == 1.0
    assert rendered.loc["Alpha SA (AAA)", "Beta AG (BBB)"] == 0.4
