"""Factor metadata and the fixed research catalog.

The catalog is intentionally explicit: changing a formula requires changing its
versioned description here and adding/updating a regression test.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FactorSpec:
    name: str
    family: str
    description: str
    formula: str
    expected_sign: str
    lookback_days: int


FACTOR_CATALOG: tuple[FactorSpec, ...] = (
    FactorSpec("ret_1m", "momentum_reversal", "过去 21 日收益，短期反转候选", "close / close.shift(21) - 1", "negative", 21),
    FactorSpec("ret_3m", "momentum_reversal", "过去 63 日收益", "close / close.shift(63) - 1", "either", 63),
    FactorSpec("ret_6m", "momentum_reversal", "过去 126 日收益", "close / close.shift(126) - 1", "either", 126),
    FactorSpec("ret_12m_ex_1m", "momentum_reversal", "12 个月动量剔除最近 1 个月", "close.shift(21) / close.shift(252) - 1", "positive", 252),
    FactorSpec("ma_gap_20", "momentum_reversal", "收盘价相对 20 日均线偏离", "close / rolling_mean(close,20) - 1", "either", 20),
    FactorSpec("vol_20", "volatility_risk", "20 日年化已实现波动率", "std(ret_1d,20) * sqrt(252)", "negative", 20),
    FactorSpec("vol_60", "volatility_risk", "60 日年化已实现波动率", "std(ret_1d,60) * sqrt(252)", "negative", 60),
    FactorSpec("downvol_60", "volatility_risk", "60 日下行波动率", "sqrt(mean(min(ret_1d,0)^2,60)) * sqrt(252)", "negative", 60),
    FactorSpec("max_ret_20", "volatility_risk", "20 日最大单日收益，彩票型股票候选", "max(ret_1d,20)", "negative", 20),
    FactorSpec("skew_60", "volatility_risk", "60 日收益偏度", "skew(ret_1d,60)", "negative", 60),
    FactorSpec("beta_252", "volatility_risk", "252 日市场 beta", "cov(ret_1d, market_ret,252) / var(market_ret,252)", "either", 252),
    FactorSpec("idio_vol_60", "volatility_risk", "60 日市场残差波动率", "std(ret_1d - beta * market_ret,60)", "negative", 60),
    FactorSpec("turn_20", "liquidity_volume", "20 日平均换手率", "mean(turn,20)", "either", 20),
    FactorSpec("turn_chg_5_20", "liquidity_volume", "5 日/20 日换手率变化", "mean(turn,5) / mean(turn,20) - 1", "either", 20),
    FactorSpec("amihud_20", "liquidity_volume", "20 日 Amihud 非流动性", "mean(abs(ret_1d) / amount,20) * 1e8", "positive", 20),
    FactorSpec("volume_trend_20_60", "liquidity_volume", "20 日/60 日成交量趋势", "mean(volume,20) / mean(volume,60) - 1", "either", 60),
    FactorSpec("pv_corr_20", "liquidity_volume", "20 日价格与成交量变化相关", "corr(ret_1d, volume.pct_change(),20)", "either", 20),
    FactorSpec("amount_vol_20", "liquidity_volume", "20 日成交额波动", "std(amount.pct_change(),20)", "either", 20),
    FactorSpec("range_20", "liquidity_volume", "20 日高低价区间", "mean((high-low)/close,20)", "either", 20),
    FactorSpec("overnight_ret_20", "overnight_intraday", "20 日平均隔夜收益", "mean(open/prev_close - 1,20)", "either", 20),
    FactorSpec("intraday_ret_20", "overnight_intraday", "20 日平均日内收益", "mean(close/open - 1,20)", "either", 20),
    FactorSpec("overnight_vol_share_20", "overnight_intraday", "隔夜波动占总波动比例", "std(overnight_ret,20) / std(ret_1d,20)", "either", 20),
    FactorSpec("rsi_14", "technical_position", "14 日相对强弱指标", "100 - 100 / (1 + avg_gain/avg_loss)", "either", 14),
    FactorSpec("dist_52w_high", "technical_position", "距 252 日最高收盘价的距离", "close / rolling_max(close,252) - 1", "either", 252),
)

FACTOR_NAMES: tuple[str, ...] = tuple(spec.name for spec in FACTOR_CATALOG)
FACTOR_BY_NAME = {spec.name: spec for spec in FACTOR_CATALOG}
FAMILY_NAMES: tuple[str, ...] = tuple(dict.fromkeys(spec.family for spec in FACTOR_CATALOG))


def catalog_frame():
    """Return the catalog as a DataFrame without importing pandas at module import."""
    import pandas as pd

    return pd.DataFrame(
        [
            {
                "factor": spec.name,
                "family": spec.family,
                "description": spec.description,
                "formula": spec.formula,
                "expected_sign": spec.expected_sign,
                "lookback_days": spec.lookback_days,
            }
            for spec in FACTOR_CATALOG
        ]
    )
