"""Command line entry point for EQ-FACTOR-01."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import data
from .config import load_settings
from .pipeline import run_demo, run_research
from .report import write_report


def _load_industry(settings):
    path = settings.processed_dir / "industry.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"找不到行业表：{path}。真实数据请先运行 download-industry；demo 会自动生成。"
        )
    return pd.read_parquet(path)


def _research(settings, prefix: str = "") -> None:
    panel = data.build_panel(settings)
    industry = _load_industry(settings)
    outputs = run_research(settings, panel, industry, prefix=prefix)
    selected = outputs["selected_factors"]
    if selected.empty:
        print("没有因子通过设计和验证期筛选；请检查数据覆盖与因子分布。")
    else:
        print("入选因子：")
        print(selected.to_string(index=False))
    print(f"结果已写入：{settings.results_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EQ-FACTOR-01 A 股价量因子库")
    parser.add_argument("--config", default="configs/base.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("download-stock-lists", help="下载历史股票列表并生成月度并集")

    daily = sub.add_parser("download-daily", help="下载 BaoStock 日线")
    daily.add_argument("--limit", type=int, default=0)
    daily.add_argument("--offset", type=int, default=0)
    daily.add_argument("--force", action="store_true")
    daily.add_argument("--retries", type=int, default=3)
    daily.add_argument("--sleep", type=float, default=0.05)

    sub.add_parser("download-industry", help="下载月度历史行业快照")

    panel_parser = sub.add_parser("build-panel", help="合并 raw parquet 为日线面板")
    panel_parser.add_argument("--force", action="store_true")

    universe_parser = sub.add_parser("build-universe", help="生成月度动态股票池")
    universe_parser.add_argument("--force", action="store_true")

    factors_parser = sub.add_parser("build-factors", help="构建因子并进行完整研究")
    factors_parser.add_argument("--prefix", default="")

    eval_parser = sub.add_parser("evaluate", help="运行因子评价与样本外组合")
    eval_parser.add_argument("--prefix", default="")

    backtest_parser = sub.add_parser("backtest", help="运行组合回测（同 research）")
    backtest_parser.add_argument("--prefix", default="")

    research_parser = sub.add_parser("research", help="运行完整研究流程")
    research_parser.add_argument("--prefix", default="")

    report_parser = sub.add_parser("report", help="从结果表生成报告")
    report_parser.add_argument("--prefix", default="")

    demo_parser = sub.add_parser("demo", help="用合成数据跑通完整流程")
    demo_parser.add_argument("--stocks", type=int, default=80)
    demo_parser.add_argument("--days", type=int, default=1500)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    settings = load_settings(args.config)
    settings.prepare_dirs()

    if args.command == "download-stock-lists":
        data.download_stock_lists(settings)
    elif args.command == "download-daily":
        data.download_daily(
            settings,
            limit=args.limit,
            offset=args.offset,
            force=args.force,
            retries=args.retries,
            sleep_seconds=args.sleep,
        )
    elif args.command == "download-industry":
        data.download_industry(settings)
    elif args.command == "build-panel":
        data.build_panel(settings, force=args.force)
    elif args.command == "build-universe":
        panel = data.build_panel(settings, force=args.force)
        industry = _load_industry(settings)
        data.build_universe(settings, panel=panel, industry=industry)
    elif args.command in {"build-factors", "evaluate", "backtest", "research"}:
        _research(settings, prefix=args.prefix)
    elif args.command == "report":
        path = write_report(settings, prefix=args.prefix)
        print(f"报告已生成：{path}")
    elif args.command == "demo":
        outputs = run_demo(settings, n_stocks=args.stocks, n_days=args.days)
        print("Demo 选中因子：")
        print(outputs["selected_factors"].to_string(index=False))
        path = write_report(settings, prefix="demo")
        print(f"Demo 报告：{path}")
    else:
        raise ValueError(f"未知命令：{args.command}")


if __name__ == "__main__":
    main()
