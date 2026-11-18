"""End-to-end research pipeline shared by CLI and demo."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import data, evaluate, factors as factor_module, neutralize
from .config import Settings
from .synthetic import make_synthetic_panel


def _add_p_values(summary: pd.DataFrame) -> pd.DataFrame:
    summary = summary.copy()
    summary["p_value"] = 2 * norm.sf(np.abs(summary["t_hac"]))
    summary["q_value"] = evaluate.benjamini_hochberg(summary["p_value"])
    return summary


def run_research(
    settings: Settings,
    panel: pd.DataFrame,
    industry: pd.DataFrame,
    prefix: str = "",
) -> dict[str, pd.DataFrame]:
    """Run universe, factor, evaluation and backtest stages in memory."""
    settings.prepare_dirs()
    dates = data.rebalance_dates(panel)
    universe = data.build_universe(settings, panel=panel, industry=industry)
    raw_factors = factor_module.compute_factors(panel, dates)
    merge_cols = ["date", "code", "industry", "eligible"]
    raw_factors = raw_factors.merge(
        universe[merge_cols], on=["date", "code"], how="inner", validate="one_to_one"
    )
    neutral_factors = neutralize.neutralize_factors(
        raw_factors,
        min_cross_section=int(settings.factors["min_cross_section"]),
        lower=float(settings.factors["winsor_lower"]),
        upper=float(settings.factors["winsor_upper"]),
    )
    forward = data.forward_returns(panel, dates, horizons=(1, 3, 6, 12))
    data_1m = evaluate.attach_forward(neutral_factors, forward, horizon=1)
    ic = evaluate.rank_ic_table(
        data_1m, min_cross_section=int(settings.factors["min_cross_section"])
    )
    summary = _add_p_values(evaluate.ic_summary(ic))
    period_ic = evaluate.ic_by_period(
        ic,
        settings.walk_forward["design_end"],
        settings.walk_forward["validation_end"],
    )
    quantiles = evaluate.factor_quantiles(
        data_1m,
        n_quantiles=int(settings.portfolio["n_quantiles"]),
        min_cross_section=int(settings.factors["min_cross_section"]),
    )
    quantile_summary = evaluate.quantile_summary(quantiles)
    decay = evaluate.decay_table(
        neutral_factors,
        forward,
        horizons=(1, 3, 6, 12),
        min_cross_section=int(settings.factors["min_cross_section"]),
    )
    correlation = evaluate.factor_correlation(
        neutral_factors, min_cross_section=int(settings.factors["min_cross_section"])
    )
    selected = evaluate.select_factors(
        period_ic,
        correlation,
        max_factors=int(settings.portfolio["max_factors"]),
        correlation_threshold=float(settings.portfolio["correlation_threshold"]),
    )
    selected_frame = pd.DataFrame(
        [{"factor": item.name, "sign": item.sign} for item in selected]
    )
    scores = evaluate.composite_scores(neutral_factors, selected) if selected else pd.DataFrame()
    weights = evaluate.portfolio_weights(
        scores,
        top_fraction=float(settings.portfolio["top_fraction"]),
        max_weight=float(settings.portfolio["max_weight"]),
    ) if not scores.empty else pd.DataFrame(columns=["date", "code", "weight"])
    backtest, metrics = evaluate.backtest_long_only(
        weights,
        data_1m[["date", "code", "forward_return"]],
        cost_bps=float(settings.portfolio["cost_bps"]),
        periods_per_year=int(settings.portfolio["annualization"]),
    ) if not weights.empty else (
        pd.DataFrame(),
        pd.DataFrame(),
    )

    name = f"{prefix}_" if prefix else ""
    outputs = {
        "universe": universe,
        "factors_raw": raw_factors,
        "factors_neutral": neutral_factors,
        "forward_returns": forward,
        "ic_daily": ic,
        "ic_summary": summary,
        "ic_by_period": period_ic,
        "quantiles": quantiles,
        "quantile_summary": quantile_summary,
        "decay": decay,
        "correlation": correlation.reset_index().rename(columns={"index": "factor"}),
        "selected_factors": selected_frame,
        "weights": weights,
        "backtest_series": backtest,
        "backtest_metrics": metrics,
    }
    for key in (
        "universe", "factors_raw", "factors_neutral", "forward_returns", "ic_daily",
        "ic_summary", "ic_by_period", "quantiles", "quantile_summary", "decay",
        "correlation", "selected_factors", "weights", "backtest_series", "backtest_metrics",
    ):
        frame = outputs[key]
        if frame.empty:
            continue
        suffix = ".parquet" if key in {"universe", "factors_raw", "factors_neutral", "forward_returns"} else ".csv"
        path = settings.results_dir / f"{name}{key}{suffix}"
        path.parent.mkdir(parents=True, exist_ok=True)
        if suffix == ".parquet":
            frame.to_parquet(path, index=False)
        else:
            frame.to_csv(path, index=False, encoding="utf-8-sig")
    return outputs


def run_demo(settings: Settings, n_stocks: int = 80, n_days: int = 1500) -> dict[str, pd.DataFrame]:
    panel, industry = make_synthetic_panel(n_stocks=n_stocks, n_days=n_days)
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(settings.processed_dir / "daily_panel.parquet", index=False)
    industry.to_parquet(settings.processed_dir / "industry.parquet", index=False)
    return run_research(settings, panel, industry, prefix="demo")
