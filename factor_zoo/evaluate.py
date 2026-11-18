"""Cross-sectional factor evaluation and portfolio construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_1samp

from .catalog import FACTOR_NAMES


def attach_forward(
    factors: pd.DataFrame,
    forward: pd.DataFrame,
    horizon: int = 1,
) -> pd.DataFrame:
    labels = forward.loc[forward["horizon"] == horizon, ["date", "code", "forward_return"]]
    data = factors.merge(labels, on=["date", "code"], how="inner", validate="one_to_one")
    return data.sort_values(["date", "code"]).reset_index(drop=True)


def _period_for(date: pd.Timestamp, design_end: pd.Timestamp, validation_end: pd.Timestamp) -> str:
    if date <= design_end:
        return "design"
    if date <= validation_end:
        return "validation"
    return "test"


def add_period(
    frame: pd.DataFrame,
    design_end: str | pd.Timestamp,
    validation_end: str | pd.Timestamp,
) -> pd.DataFrame:
    frame = frame.copy()
    frame["period"] = [
        _period_for(pd.Timestamp(date), pd.Timestamp(design_end), pd.Timestamp(validation_end))
        for date in frame["date"]
    ]
    return frame


def rank_ic_table(
    data: pd.DataFrame,
    min_cross_section: int = 30,
) -> pd.DataFrame:
    """Daily cross-sectional Spearman IC for each catalog factor."""
    rows = []
    for date, group in data.groupby("date", sort=True):
        if len(group) < min_cross_section:
            continue
        for factor in FACTOR_NAMES:
            valid = group[[factor, "forward_return"]].dropna()
            if len(valid) < min_cross_section or valid[factor].nunique() < 2:
                continue
            ic, _ = spearmanr(valid[factor], valid["forward_return"])
            if np.isfinite(ic):
                rows.append({"date": date, "factor": factor, "ic": float(ic), "n": len(valid)})
    return pd.DataFrame(rows)


def newey_west_t(series: pd.Series, maxlags: int = 3) -> float:
    """HAC t-statistic for the mean of a monthly IC series."""
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) < 5:
        return np.nan
    if np.std(values, ddof=1) <= 1e-14:
        return np.nan
    try:
        import statsmodels.api as sm

        model = sm.OLS(values, np.ones((len(values), 1))).fit(
            cov_type="HAC",
            cov_kwds={"maxlags": min(maxlags, len(values) - 1), "use_correction": False},
            use_t=False,
        )
        return float(model.tvalues[0])
    except Exception:
        return float(ttest_1samp(values, 0.0).statistic)


def ic_summary(ic: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for factor, group in ic.groupby("factor", sort=False):
        values = group["ic"].dropna()
        if values.empty:
            continue
        std = values.std(ddof=1)
        rows.append({
            "factor": factor,
            "n_dates": int(len(values)),
            "mean_ic": float(values.mean()),
            "std_ic": float(std),
            "icir": float(values.mean() / std) if std > 0 else np.nan,
            "t_hac": newey_west_t(values),
            "positive_rate": float((values > 0).mean()),
        })
    return pd.DataFrame(rows)


def ic_by_period(
    ic: pd.DataFrame,
    design_end: str | pd.Timestamp,
    validation_end: str | pd.Timestamp,
) -> pd.DataFrame:
    ic = add_period(ic, design_end, validation_end)
    rows = []
    for (factor, period), group in ic.groupby(["factor", "period"], sort=True):
        values = group["ic"].dropna()
        std = values.std(ddof=1) if len(values) >= 2 else np.nan
        rows.append({
            "factor": factor,
            "period": period,
            "n_dates": int(len(values)),
            "mean_ic": float(values.mean()) if len(values) else np.nan,
            "icir": float(values.mean() / std) if std and std > 0 else np.nan,
            "positive_rate": float((values > 0).mean()) if len(values) else np.nan,
        })
    return pd.DataFrame(rows)


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    """Return BH-adjusted q-values aligned to the input index."""
    p = pd.to_numeric(p_values, errors="coerce")
    result = pd.Series(np.nan, index=p.index, dtype=float)
    valid = p.dropna().sort_values()
    if valid.empty:
        return result
    n = len(valid)
    adjusted = valid * n / np.arange(1, n + 1)
    adjusted = adjusted[::-1].cummin()[::-1].clip(upper=1.0)
    result.loc[adjusted.index] = adjusted.to_numpy()
    return result


def factor_quantiles(
    data: pd.DataFrame,
    n_quantiles: int = 5,
    min_cross_section: int = 30,
) -> pd.DataFrame:
    """Daily quantile portfolio returns for every factor."""
    rows = []
    labels = [f"Q{i}" for i in range(1, n_quantiles + 1)]
    for date, group in data.groupby("date", sort=True):
        if len(group) < min_cross_section:
            continue
        benchmark = group["forward_return"].mean()
        for factor in FACTOR_NAMES:
            valid = group[[factor, "forward_return"]].dropna()
            if len(valid) < min_cross_section:
                continue
            try:
                groups = pd.qcut(valid[factor].rank(method="first"), n_quantiles, labels=labels)
            except ValueError:
                continue
            means = valid.groupby(groups, observed=False)["forward_return"].mean()
            if means.notna().sum() < n_quantiles:
                continue
            row = {"date": date, "factor": factor, "benchmark_return": benchmark}
            row.update({label: float(means[label]) for label in labels})
            row["long_short"] = row[labels[-1]] - row[labels[0]]
            row["long_excess"] = row[labels[-1]] - benchmark
            rows.append(row)
    return pd.DataFrame(rows)


def quantile_summary(quantiles: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for factor, group in quantiles.groupby("factor", sort=False):
        row = {"factor": factor, "n_dates": len(group)}
        for column in [column for column in group.columns if column.startswith("Q")]:
            row[f"{column}_mean"] = group[column].mean()
        row["long_short_mean"] = group["long_short"].mean()
        row["long_excess_mean"] = group["long_excess"].mean()
        row["long_short_t"] = newey_west_t(group["long_short"])
        row["long_excess_t"] = newey_west_t(group["long_excess"])
        rows.append(row)
    return pd.DataFrame(rows)


def decay_table(
    factors: pd.DataFrame,
    forward: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 3, 6, 12),
    min_cross_section: int = 30,
) -> pd.DataFrame:
    rows = []
    for horizon in horizons:
        data = attach_forward(factors, forward, horizon=horizon)
        ic = rank_ic_table(data, min_cross_section=min_cross_section)
        summary = ic_summary(ic)
        if summary.empty:
            continue
        summary["horizon"] = horizon
        rows.append(summary)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def factor_correlation(
    factors: pd.DataFrame,
    min_cross_section: int = 30,
) -> pd.DataFrame:
    """Average cross-sectional Spearman correlation matrix across dates."""
    matrices = []
    for _, group in factors.groupby("date", sort=True):
        if len(group) < min_cross_section:
            continue
        matrices.append(group[list(FACTOR_NAMES)].corr(method="spearman"))
    if not matrices:
        return pd.DataFrame(index=FACTOR_NAMES, columns=FACTOR_NAMES, dtype=float)
    return pd.concat(matrices).groupby(level=0).mean().reindex(
        index=FACTOR_NAMES, columns=FACTOR_NAMES
    )


@dataclass(frozen=True)
class SelectedFactor:
    name: str
    sign: float


def select_factors(
    period_ic: pd.DataFrame,
    correlation: pd.DataFrame,
    max_factors: int = 6,
    correlation_threshold: float = 0.70,
) -> list[SelectedFactor]:
    """Select stable, de-correlated factors without touching the test period."""
    pivot = period_ic.pivot(index="factor", columns="period", values="icir")
    mean_pivot = period_ic.pivot(index="factor", columns="period", values="mean_ic")
    candidates = []
    for factor in pivot.index:
        design_icir = pivot.at[factor, "design"] if "design" in pivot.columns else np.nan
        validation_icir = pivot.at[factor, "validation"] if "validation" in pivot.columns else np.nan
        design_mean = mean_pivot.at[factor, "design"] if "design" in mean_pivot.columns else np.nan
        validation_mean = (
            mean_pivot.at[factor, "validation"] if "validation" in mean_pivot.columns else np.nan
        )
        if not np.isfinite(design_icir) or not np.isfinite(validation_icir):
            continue
        if design_icir * validation_icir <= 0:
            continue
        candidates.append((factor, abs(validation_icir), float(np.sign(validation_mean))))
    candidates.sort(key=lambda item: item[1], reverse=True)

    chosen: list[SelectedFactor] = []
    for factor, _, sign in candidates:
        if len(chosen) >= max_factors:
            break
        if any(
            factor in correlation.index
            and selected.name in correlation.columns
            and abs(float(correlation.loc[factor, selected.name])) > correlation_threshold
            for selected in chosen
        ):
            continue
        chosen.append(SelectedFactor(factor, sign))
    return chosen


def composite_scores(
    factors: pd.DataFrame,
    selected: list[SelectedFactor],
) -> pd.DataFrame:
    if not selected:
        raise ValueError("没有可用的入选因子")
    frames = []
    names = [item.name for item in selected]
    for date, group in factors.groupby("date", sort=True):
        group = group.copy()
        score = pd.Series(0.0, index=group.index)
        valid_weight = pd.Series(0.0, index=group.index)
        for item in selected:
            rank = group[item.name].rank(pct=True)
            available = rank.notna()
            score.loc[available] += item.sign * rank.loc[available]
            valid_weight.loc[available] += 1.0
        group["composite_score"] = score / valid_weight.replace(0, np.nan)
        group = group.loc[group["composite_score"].notna()]
        frames.append(group[["date", "code", "composite_score"]])
    return pd.concat(frames, ignore_index=True).sort_values(["date", "code"])


def portfolio_weights(
    scores: pd.DataFrame,
    top_fraction: float = 0.10,
    max_weight: float = 0.02,
    min_names: int = 10,
) -> pd.DataFrame:
    frames = []
    for date, group in scores.groupby("date", sort=True):
        group = group.dropna(subset=["composite_score"]).copy()
        count = max(min_names, int(np.ceil(len(group) * top_fraction)))
        selected = group.nlargest(min(count, len(group)), "composite_score").copy()
        if selected.empty:
            continue
        weight = min(1.0 / len(selected), max_weight)
        selected["weight"] = weight
        total = selected["weight"].sum()
        if total > 1.0:
            selected["weight"] /= total
        frames.append(selected[["date", "code", "weight"]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["date", "code", "weight"]
    )


def benchmark_returns(labels: pd.DataFrame) -> pd.DataFrame:
    return (
        labels.groupby("date", as_index=False)["forward_return"]
        .mean()
        .rename(columns={"forward_return": "benchmark_return"})
    )


def performance_stats(returns: pd.Series, periods_per_year: int = 12) -> dict:
    values = pd.to_numeric(returns, errors="coerce").dropna()
    if values.empty:
        return {}
    cumulative = (1 + values).cumprod()
    annual_return = cumulative.iloc[-1] ** (periods_per_year / len(values)) - 1
    annual_vol = values.std(ddof=1) * np.sqrt(periods_per_year)
    drawdown = cumulative / cumulative.cummax() - 1
    sharpe = annual_return / annual_vol if annual_vol > 0 else np.nan
    return {
        "months": int(len(values)),
        "annual_return": float(annual_return),
        "annual_vol": float(annual_vol),
        "sharpe": float(sharpe) if np.isfinite(sharpe) else np.nan,
        "max_drawdown": float(drawdown.min()),
        "worst_month": float(values.min()),
        "best_month": float(values.max()),
        "hit_rate": float((values > 0).mean()),
    }


def backtest_long_only(
    weights: pd.DataFrame,
    labels: pd.DataFrame,
    cost_bps: float = 10.0,
    periods_per_year: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    weights = weights.copy().sort_values(["date", "code"])
    labels = labels[["date", "code", "forward_return"]].copy()
    merged = weights.merge(labels, on=["date", "code"], how="inner", validate="one_to_one")
    rows = []
    previous = pd.Series(dtype=float)
    for date, group in merged.groupby("date", sort=True):
        weights_today = group.set_index("code")["weight"]
        gross = float((group["weight"] * group["forward_return"]).sum())
        all_codes = previous.index.union(weights_today.index)
        turnover = float(
            weights_today.reindex(all_codes, fill_value=0.0)
            .sub(previous.reindex(all_codes, fill_value=0.0))
            .abs()
            .sum()
        )
        cost = turnover * cost_bps / 10000.0
        rows.append({
            "date": date,
            "gross_return": gross,
            "turnover": turnover,
            "cost": cost,
            "net_return": gross - cost,
        })
        previous = weights_today
    strategy = pd.DataFrame(rows)
    benchmark = (
        labels.groupby("date", as_index=False)["forward_return"]
        .mean()
        .rename(columns={"forward_return": "benchmark_return"})
    )
    result = strategy.merge(benchmark, on="date", how="left")
    result["excess_return"] = result["net_return"] - result["benchmark_return"]
    metrics = pd.DataFrame([
        {"series": "strategy", **performance_stats(result["net_return"], periods_per_year)},
        {"series": "benchmark", **performance_stats(result["benchmark_return"], periods_per_year)},
    ])
    return result, metrics


def save_table(frame: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
