import numpy as np
import pandas as pd

from factor_zoo.factors import compute_factors
from factor_zoo.synthetic import make_synthetic_panel


def test_factor_panel_has_all_catalog_columns_and_one_month_end_row() -> None:
    panel, _ = make_synthetic_panel(n_stocks=2, n_days=300)
    last_date = panel["date"].max()
    factors = compute_factors(panel, [last_date])
    assert len(factors) == 2
    assert factors["date"].nunique() == 1
    for column in (
        "ret_1m", "ret_3m", "ret_6m", "ret_12m_ex_1m", "ma_gap_20",
        "vol_20", "vol_60", "downvol_60", "max_ret_20", "skew_60",
        "beta_252", "idio_vol_60", "turn_20", "turn_chg_5_20", "amihud_20",
        "volume_trend_20_60", "pv_corr_20", "amount_vol_20", "range_20",
        "overnight_ret_20", "intraday_ret_20", "overnight_vol_share_20",
        "rsi_14", "dist_52w_high",
    ):
        assert column in factors.columns


def test_ret_1m_matches_price_change_over_21_rows() -> None:
    panel, _ = make_synthetic_panel(n_stocks=1, n_days=300)
    last_date = panel["date"].max()
    factors = compute_factors(panel, [last_date])
    expected = panel.sort_values("date")["close"].iloc[-1] / panel.sort_values("date")["close"].iloc[-22] - 1
    assert np.isclose(factors.iloc[0]["ret_1m"], expected, equal_nan=True)
