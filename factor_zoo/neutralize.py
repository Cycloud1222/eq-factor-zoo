"""Cross-sectional winsorization and industry/size neutralization."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .catalog import FACTOR_NAMES


def winsorize_zscore(
    series: pd.Series,
    lower: float = 0.01,
    upper: float = 0.99,
) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if values.notna().sum() < 3:
        return values
    low, high = values.quantile([lower, upper])
    values = values.clip(low, high)
    std = values.std(ddof=0)
    if not np.isfinite(std) or std <= 1e-12:
        return values * 0
    return (values - values.mean()) / std


def _residualize(
    factor: pd.Series,
    log_size: pd.Series,
    industry: pd.Series,
) -> pd.Series:
    frame = pd.DataFrame({"factor": factor, "log_size": log_size, "industry": industry})
    frame = frame.dropna(subset=["factor", "log_size"])
    if len(frame) < 30:
        return factor
    design = pd.get_dummies(frame["industry"].fillna("Unknown"), drop_first=True)
    design = pd.concat(
        [pd.Series(1.0, index=frame.index, name="const"), frame["log_size"], design],
        axis=1,
    )
    matrix = design.to_numpy(dtype=float)
    target = frame["factor"].to_numpy(dtype=float)
    coefficients, *_ = np.linalg.lstsq(matrix, target, rcond=None)
    residual = target - matrix @ coefficients
    result = pd.Series(np.nan, index=factor.index, dtype=float)
    result.loc[frame.index] = residual
    return result


def neutralize_factors(
    factors: pd.DataFrame,
    min_cross_section: int = 30,
    lower: float = 0.01,
    upper: float = 0.99,
) -> pd.DataFrame:
    """Neutralize each factor by date against log float market cap and industry."""
    required = {"date", "code", "float_mcap_proxy", "industry"}
    missing = required - set(factors.columns)
    if missing:
        raise ValueError(f"因子表缺少中性化字段：{sorted(missing)}")
    result_frames = []
    for date, group in factors.groupby("date", sort=True):
        group = group.copy()
        group["log_size"] = np.log(
            pd.to_numeric(group["float_mcap_proxy"], errors="coerce").where(
                pd.to_numeric(group["float_mcap_proxy"], errors="coerce") > 0
            )
        )
        if group["log_size"].notna().sum() < min_cross_section:
            group["log_size"] = 0.0
        for name in FACTOR_NAMES:
            standardized = winsorize_zscore(group[name], lower, upper)
            group[name] = _residualize(standardized, group["log_size"], group["industry"])
            group[name] = winsorize_zscore(group[name], lower, upper)
        result_frames.append(group)
    if not result_frames:
        raise ValueError("没有可中性化的日期")
    result = pd.concat(result_frames, ignore_index=True)
    return result.sort_values(["date", "code"]).reset_index(drop=True)


def exposure_summary(factors: pd.DataFrame) -> pd.DataFrame:
    """Mean absolute correlation of each factor with log size and industry dummies."""
    rows = []
    for date, group in factors.groupby("date", sort=True):
        group = group.copy()
        group["log_size"] = np.log(
            pd.to_numeric(group["float_mcap_proxy"], errors="coerce").where(
                pd.to_numeric(group["float_mcap_proxy"], errors="coerce") > 0
            )
        )
        if group["log_size"].notna().sum() < 30:
            continue
        for name in FACTOR_NAMES:
            valid = group[[name, "log_size"]].dropna()
            if len(valid) < 30:
                continue
            rows.append({
                "date": date,
                "factor": name,
                "size_corr": valid[name].corr(valid["log_size"], method="spearman"),
            })
    if not rows:
        return pd.DataFrame(columns=["factor", "mean_abs_size_corr"])
    frame = pd.DataFrame(rows)
    return (
        frame.assign(abs_size_corr=frame["size_corr"].abs())
        .groupby("factor", as_index=False)["abs_size_corr"]
        .mean()
        .rename(columns={"abs_size_corr": "mean_abs_size_corr"})
    )
