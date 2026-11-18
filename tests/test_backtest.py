import numpy as np
import pandas as pd

from factor_zoo.evaluate import backtest_long_only


def test_backtest_subtracts_turnover_cost() -> None:
    weights = pd.DataFrame({
        "date": [pd.Timestamp("2025-01-31")],
        "code": ["sh.600000"],
        "weight": [0.5],
    })
    labels = pd.DataFrame({
        "date": [pd.Timestamp("2025-01-31")],
        "code": ["sh.600000"],
        "forward_return": [0.1],
    })
    series, metrics = backtest_long_only(weights, labels, cost_bps=10)
    assert np.isclose(series.iloc[0]["gross_return"], 0.05)
    assert np.isclose(series.iloc[0]["cost"], 0.0005)
    assert np.isclose(series.iloc[0]["net_return"], 0.0495)
    assert not metrics.empty
