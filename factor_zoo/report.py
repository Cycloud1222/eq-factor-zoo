"""Small report generator for reproducible factor research artifacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import Settings


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig") if path.exists() else pd.DataFrame()


def write_report(settings: Settings, prefix: str = "") -> Path:
    name = f"{prefix}_" if prefix else ""
    results = settings.results_dir
    ic = _read_csv(results / f"{name}ic_summary.csv")
    quantiles = _read_csv(results / f"{name}quantile_summary.csv")
    metrics = _read_csv(results / f"{name}backtest_metrics.csv")
    series = _read_csv(results / f"{name}backtest_series.csv")
    selected = _read_csv(results / f"{name}selected_factors.csv")
    if ic.empty and metrics.empty:
        raise FileNotFoundError("没有可报告的结果，请先运行 research 或 demo")

    settings.figures_dir.mkdir(parents=True, exist_ok=True)
    docs_dir = settings.root / "docs" / "reports"
    docs_dir.mkdir(parents=True, exist_ok=True)

    figure_lines = []
    if not ic.empty:
        ranked = ic.sort_values("mean_ic", ascending=False).head(12).iloc[::-1]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(ranked["factor"], ranked["mean_ic"], color="#376996")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Mean RankIC")
        ax.set_title("Top factor mean RankIC")
        fig.tight_layout()
        path = settings.figures_dir / f"{name}ic_bar.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        figure_lines.append(f"![因子 IC]({path.relative_to(settings.root).as_posix()})")

    if not series.empty and "net_return" in series.columns:
        series = series.copy()
        series["date"] = pd.to_datetime(series["date"])
        series["strategy_cum"] = (1 + series["net_return"]).cumprod()
        series["benchmark_cum"] = (1 + series["benchmark_return"]).cumprod()
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(series["date"], series["strategy_cum"], label="Strategy")
        ax.plot(series["date"], series["benchmark_cum"], label="EW benchmark")
        ax.set_title("Long-only factor portfolio")
        ax.set_ylabel("Cumulative NAV")
        ax.legend()
        fig.tight_layout()
        path = settings.figures_dir / f"{name}equity_curve.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        figure_lines.append(f"![组合净值]({path.relative_to(settings.root).as_posix()})")

    lines = [
        "# EQ-FACTOR-01 因子研究自动报告",
        "",
        "> 本报告由 `factor-zoo report` 生成。若使用的是 demo 数据，所有数值只用于流程验证，不是市场结论。",
        "",
        "## 运行摘要",
        "",
        f"- 因子数量：{len(ic) if not ic.empty else 'NA'}",
        f"- 入选因子：{', '.join(selected['factor'].tolist()) if not selected.empty else '无'}",
        "",
    ]
    if not ic.empty:
        lines += ["## 因子 IC 摘要", "", ic.to_markdown(index=False, floatfmt=".4f"), ""]
    if not quantiles.empty:
        lines += ["## 分组摘要", "", quantiles.to_markdown(index=False, floatfmt=".4f"), ""]
    if not metrics.empty:
        lines += ["## 回测指标", "", metrics.to_markdown(index=False, floatfmt=".4f"), ""]
    lines += ["## 图表", "", *[line + "\n" for line in figure_lines]]
    lines += [
        "## 解释边界",
        "",
        "- 当前组合是研究性 long-only 组合，不等同于可实盘策略。",
        "- 需要使用真实数据、滚动样本外和交易成本复核后才能写入简历或报告结论。",
        "- 因子筛选只能使用设计和验证期；2023–2025 不得参与筛选。",
    ]
    destination = docs_dir / f"{'demo_' if prefix else ''}factor_zoo_report.md"
    destination.write_text("\n".join(lines), encoding="utf-8")
    return destination
