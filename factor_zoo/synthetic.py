"""Synthetic data for smoke tests and the demo command.

The generator is deliberately simple. It is not a market simulator; it only
creates valid OHLCV/universe shapes so the full data contract can be exercised
without downloading market data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_panel(
    n_stocks: int = 80,
    n_days: int = 520,
    seed: int = 7,
    start: str = "2016-01-04",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    industries = ["银行", "医药", "电子", "消费", "制造", "能源", "地产", "材料"]
    rows = []

    for index in range(n_stocks):
        code = f"sh.6{index:05d}" if index % 2 == 0 else f"sz.0{index:05d}"
        industry = industries[index % len(industries)]
        price = 10 + index * 0.4
        market_cap = 3e9 + index * 5e8
        returns = rng.normal(0.0002, 0.018, size=n_days)
        close = price * np.exp(np.cumsum(returns))
        overnight = rng.normal(0, 0.004, size=n_days)
        intraday = rng.normal(0, 0.012, size=n_days)
        open_ = np.r_[price, close[:-1]] * (1 + overnight)
        close = open_ * (1 + intraday)
        high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.02, n_days))
        low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.02, n_days))
        previous = np.r_[price, close[:-1]]
        turn = np.clip(rng.normal(1.0, 0.25, n_days), 0.05, 5.0)
        amount = market_cap * turn / 100.0
        volume = amount / np.maximum(close, 0.1)
        for day in range(n_days):
            rows.append({
                "date": dates[day],
                "code": code,
                "open": open_[day],
                "high": high[day],
                "low": low[day],
                "close": close[day],
                "preclose": previous[day],
                "volume": volume[day],
                "amount": amount[day],
                "adjustflag": 1,
                "turn": turn[day],
                "tradestatus": 1,
                "pctChg": (close[day] / previous[day] - 1) * 100,
                "isST": 0,
                "prev_close": previous[day],
                "ret_1d": close[day] / previous[day] - 1,
            })

    panel = pd.DataFrame(rows)
    month_ends = (
        panel[["date"]]
        .drop_duplicates()
        .assign(month=lambda frame: frame["date"].dt.to_period("M"))
        .groupby("month", as_index=False)["date"]
        .max()
    )["date"].tolist()
    industry_rows = []
    for date in month_ends:
        for index in range(n_stocks):
            code = f"sh.6{index:05d}" if index % 2 == 0 else f"sz.0{index:05d}"
            industry_rows.append({
                "date": date,
                "code": code,
                "industry": industries[index % len(industries)],
                "industry_classification": "测试行业",
                "update_date": date,
            })
    return panel, pd.DataFrame(industry_rows)
