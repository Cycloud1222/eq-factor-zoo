import numpy as np
import pandas as pd

from factor_zoo.data import A_SHARE_PATTERN, forward_returns


def test_a_share_pattern_excludes_indices_and_b_shares() -> None:
    assert A_SHARE_PATTERN.match("sh.600000")
    assert A_SHARE_PATTERN.match("sh.688001")
    assert A_SHARE_PATTERN.match("sz.000001")
    assert A_SHARE_PATTERN.match("sz.300001")
    assert not A_SHARE_PATTERN.match("sh.000001")
    assert not A_SHARE_PATTERN.match("sz.399001")
    assert not A_SHARE_PATTERN.match("sh.900901")


def test_forward_return_uses_next_open_and_next_rebalance_close() -> None:
    dates = pd.to_datetime(["2025-01-31", "2025-02-03", "2025-02-28"])
    panel = pd.DataFrame({
        "date": dates,
        "code": "sh.600000",
        "open": [10.0, 10.5, 11.0],
        "close": [10.2, 10.8, 12.0],
    })
    result = forward_returns(
        panel,
        [pd.Timestamp("2025-01-31"), pd.Timestamp("2025-02-28")],
        horizons=(1,),
    )
    # Entry is 2025-02-03 open; exit is 2025-02-28 close.
    assert len(result) == 1
    assert np.isclose(result.iloc[0]["forward_return"], 12.0 / 10.5 - 1)


def test_forward_return_does_not_use_signal_day_close() -> None:
    panel = pd.DataFrame({
        "date": pd.to_datetime(["2025-01-31", "2025-02-03", "2025-02-28"]),
        "code": "sh.600000",
        "open": [1.0, 100.0, 100.0],
        "close": [100.0, 100.0, 101.0],
    })
    result = forward_returns(
        panel,
        [pd.Timestamp("2025-01-31"), pd.Timestamp("2025-02-28")],
        horizons=(1,),
    )
    assert np.isclose(result.iloc[0]["forward_return"], 0.01)
