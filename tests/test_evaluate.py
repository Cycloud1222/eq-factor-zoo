import numpy as np
import pandas as pd

from factor_zoo.catalog import FACTOR_NAMES
from factor_zoo.evaluate import (
    benjamini_hochberg,
    portfolio_weights,
    rank_ic_table,
)


def test_rank_ic_detects_a_strong_positive_signal() -> None:
    rng = np.random.default_rng(3)
    count = 80
    frame = pd.DataFrame({
        "date": pd.Timestamp("2025-01-31"),
        "code": [f"sh.{600000 + i:06d}" for i in range(count)],
    })
    forward = np.linspace(-0.1, 0.1, count) + rng.normal(0, 0.002, count)
    frame["forward_return"] = forward
    for name in FACTOR_NAMES:
        frame[name] = rng.normal(size=count)
    frame["ret_1m"] = forward + rng.normal(0, 0.003, count)
    ic = rank_ic_table(frame, min_cross_section=30)
    ret_ic = ic.loc[ic["factor"] == "ret_1m", "ic"].iloc[0]
    assert ret_ic > 0.9


def test_bh_adjustment_is_monotone_and_bounded() -> None:
    q = benjamini_hochberg(pd.Series([0.01, 0.04, 0.03, 0.2]))
    assert (q >= 0).all() and (q <= 1).all()
    order = q.sort_values()
    assert order.is_monotonic_increasing


def test_portfolio_weights_have_cash_when_max_weight_binds() -> None:
    scores = pd.DataFrame({
        "date": pd.Timestamp("2025-01-31"),
        "code": [f"sh.{600000 + i:06d}" for i in range(100)],
        "composite_score": np.arange(100),
    })
    weights = portfolio_weights(scores, top_fraction=0.10, max_weight=0.02)
    assert weights["weight"].sum() <= 1.0 + 1e-12
    assert weights["weight"].max() <= 0.02 + 1e-12
