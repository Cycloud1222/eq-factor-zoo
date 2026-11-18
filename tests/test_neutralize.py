import numpy as np
import pandas as pd

from factor_zoo.catalog import FACTOR_NAMES
from factor_zoo.neutralize import neutralize_factors, winsorize_zscore


def _panel() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    count = 120
    size = np.exp(rng.normal(22, 1, count))
    frame = pd.DataFrame({
        "date": pd.Timestamp("2025-01-31"),
        "code": [f"sh.{600000 + i:06d}" for i in range(count)],
        "float_mcap_proxy": size,
        "industry": [f"I{i % 6}" for i in range(count)],
    })
    for index, name in enumerate(FACTOR_NAMES):
        frame[name] = np.log(size) * 0.4 + rng.normal(0, 0.1, count) + index * 0.01
    return frame


def test_winsorize_removes_extreme_outlier() -> None:
    series = pd.Series([0, 1, 2, 3, 1000])
    result = winsorize_zscore(series)
    assert result.max() < 5


def test_size_exposure_is_reduced_after_neutralization() -> None:
    raw = _panel()
    before = raw["ret_1m"].corr(np.log(raw["float_mcap_proxy"]), method="spearman")
    neutral = neutralize_factors(raw)
    after = neutral["ret_1m"].corr(np.log(neutral["float_mcap_proxy"]), method="spearman")
    assert abs(after) < abs(before) * 0.25
