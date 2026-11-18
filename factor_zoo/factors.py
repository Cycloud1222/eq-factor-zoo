"""Fixed price-volume factor formulas.

All functions operate on one stock's daily series and return monthly snapshots.
The factor catalog in ``catalog.py`` is the public definition of record.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .catalog import FACTOR_NAMES


def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0).rolling(window, min_periods=window).mean()
    losses = (-delta.clip(upper=0)).rolling(window, min_periods=window).mean()
    relative_strength = _safe_div(gains, losses)
    rsi = 100 - 100 / (1 + relative_strength)
    rsi = rsi.where(losses != 0, 100.0)
    return rsi.where(gains.notna() & losses.notna())


def _one_stock(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("date").copy()
    close = pd.to_numeric(group["close"], errors="coerce")
    open_ = pd.to_numeric(group["open"], errors="coerce")
    high = pd.to_numeric(group["high"], errors="coerce")
    low = pd.to_numeric(group["low"], errors="coerce")
    previous_close = pd.to_numeric(group["prev_close"], errors="coerce")
    volume = pd.to_numeric(group["volume"], errors="coerce")
    amount = pd.to_numeric(group["amount"], errors="coerce")
    turn = pd.to_numeric(group["turn"], errors="coerce")
    return_1d = pd.to_numeric(group["ret_1d"], errors="coerce")
    market_return = pd.to_numeric(group["market_ret"], errors="coerce")

    out = pd.DataFrame({"date": group["date"], "code": group["code"]})
    out["close"] = close
    out["amount"] = amount
    out["turn"] = turn
    out["float_mcap_proxy"] = np.where(
        turn > 0, amount / (turn / 100.0), np.nan
    )

    out["ret_1m"] = close / close.shift(21) - 1
    out["ret_3m"] = close / close.shift(63) - 1
    out["ret_6m"] = close / close.shift(126) - 1
    out["ret_12m_ex_1m"] = close.shift(21) / close.shift(252) - 1
    out["ma_gap_20"] = close / close.rolling(20, min_periods=20).mean() - 1

    out["vol_20"] = return_1d.rolling(20, min_periods=20).std() * np.sqrt(252)
    out["vol_60"] = return_1d.rolling(60, min_periods=40).std() * np.sqrt(252)
    out["downvol_60"] = (
        (np.minimum(return_1d, 0) ** 2).rolling(60, min_periods=40).mean().pow(0.5)
        * np.sqrt(252)
    )
    out["max_ret_20"] = return_1d.rolling(20, min_periods=20).max()
    out["skew_60"] = return_1d.rolling(60, min_periods=40).skew()

    market_var = market_return.rolling(252, min_periods=120).var()
    market_cov = return_1d.rolling(252, min_periods=120).cov(market_return)
    beta = _safe_div(market_cov, market_var)
    out["beta_252"] = beta
    residual = return_1d - beta * market_return
    out["idio_vol_60"] = residual.rolling(60, min_periods=40).std() * np.sqrt(252)

    turn_mean_20 = turn.rolling(20, min_periods=20).mean()
    out["turn_20"] = turn_mean_20
    out["turn_chg_5_20"] = (
        turn.rolling(5, min_periods=5).mean() / turn_mean_20 - 1
    )
    out["amihud_20"] = (
        (return_1d.abs() / amount.replace(0, np.nan))
        .rolling(20, min_periods=20)
        .mean()
        * 1e8
    )
    out["volume_trend_20_60"] = (
        volume.rolling(20, min_periods=20).mean()
        / volume.rolling(60, min_periods=40).mean()
        - 1
    )
    volume_change = volume.pct_change()
    out["pv_corr_20"] = return_1d.rolling(20, min_periods=20).corr(volume_change)
    out["amount_vol_20"] = amount.pct_change().rolling(20, min_periods=20).std()
    out["range_20"] = ((high - low) / close).rolling(20, min_periods=20).mean()

    overnight = open_ / previous_close - 1
    intraday = close / open_ - 1
    out["overnight_ret_20"] = overnight.rolling(20, min_periods=20).mean()
    out["intraday_ret_20"] = intraday.rolling(20, min_periods=20).mean()
    out["overnight_vol_share_20"] = _safe_div(
        overnight.rolling(20, min_periods=20).std(),
        return_1d.rolling(20, min_periods=20).std(),
    )

    out["rsi_14"] = _rsi(close, 14)
    out["dist_52w_high"] = close / close.rolling(252, min_periods=252).max() - 1
    return out


def compute_factors(
    panel: pd.DataFrame,
    dates: list[pd.Timestamp],
) -> pd.DataFrame:
    """Compute all catalog factors at the requested month-end dates."""
    required = {
        "date", "code", "open", "high", "low", "close", "prev_close",
        "volume", "amount", "turn", "ret_1d",
    }
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"面板缺少因子所需字段：{sorted(missing)}")
    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"])
    panel["market_ret"] = panel.groupby("date")["ret_1d"].transform("mean")
    dates = set(pd.Timestamp(date) for date in dates)
    frames = []
    for _, group in panel.groupby("code", sort=False):
        calculated = _one_stock(group)
        calculated = calculated.loc[calculated["date"].isin(dates)]
        if not calculated.empty:
            frames.append(calculated)
    if not frames:
        raise ValueError("没有生成任何因子行")
    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values(["date", "code"]).reset_index(drop=True)
    for name in FACTOR_NAMES:
        if name not in result.columns:
            raise ValueError(f"因子结果缺少列：{name}")
    return result


def save_factors(factors: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    factors.to_parquet(path, index=False)
