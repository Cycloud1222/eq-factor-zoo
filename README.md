# EQ-FACTOR-01：A 股价格量因子库

目标是用全 A 股动态股票池研究 24 个价量因子：它们在中性化后是否仍有横截面预测力，哪些因子只是换手、风格暴露或多重检验造成的假象，以及一个去相关组合在真实成本下是否还能存活。

这不是“找一个高收益因子”的项目。主结果是因子评价表、因子衰减、相关聚类和可复现的样本外组合；如果最终没有稳定 alpha，也会如实交付零结果。

## 当前状态

- 数据管道、24 个因子、行业/规模中性化、RankIC、分组、衰减、相关性、BH 多重检验、组合回测和报告代码已实现。
- 合成数据 demo 已能端到端运行；当前已用 1,262 个可用历史行情代码完成首轮真实样本审计，结果见 [首轮样本报告](docs/reports/sample_results_2026-11-18.md)。
- 历史列表并集已扩展到 **5,411 个 A 股代码**，但日线下载只完成了 1,309 个；BaoStock/AKShare 的并发限制使全量补数暂时中断。当前结果不能外推为全市场结论。
- 当前报告中的 demo 数值只用于验证流程，不是市场结论。

## 快速开始

```powershell
python -m pip install -e ".[dev]"

# 用合成数据验证全部流程
factor-zoo demo --stocks 80 --days 1500

# 真实数据：先验证小批量，再全量
factor-zoo download-stock-lists
factor-zoo download-daily --limit 20 --offset 0
factor-zoo download-industry
factor-zoo build-panel
factor-zoo build-universe
factor-zoo research
factor-zoo report
```

当前已有的真实样本运行使用 `configs/sample_2020.yaml`：

```powershell
factor-zoo --config configs/sample_2020.yaml build-panel --force
factor-zoo --config configs/sample_2020.yaml build-universe --force
factor-zoo --config configs/sample_2020.yaml research --prefix sample
factor-zoo --config configs/sample_2020.yaml report --prefix sample
```

也可以直接调用模块：

```powershell
python -m factor_zoo.cli demo
```

## 数据

- 主数据源：BaoStock，前复权关闭，使用后复权日线；BaoStock 的 `turn` 和 `amount` 用于构造流通市值代理。
- 股票列表：每月末的历史股票列表并集，避免只使用当前股票池。
- 行业：BaoStock 月度历史行业快照；同一股票只使用决策日之前最近一次快照，并记录快照年龄。
- 规模代理：`float_mcap_proxy = amount / (turn / 100)`；使用前应与 AKShare 或 Wind 的流通市值抽样核对。

原始数据和缓存不进入 Git。详见 [研究设计](docs/RESEARCH_DESIGN.md) 和 [因子目录](docs/FACTOR_CATALOG.md)。

当前真实样本的详细表格和图表见 [首轮样本报告](docs/reports/sample_results_2026-11-18.md) 与 [自动因子报告](docs/reports/sample_factor_zoo_report.md)。

## 研究口径

- 样本：2016-01-01 至 2025-12-31；2016–2019 设计，2020–2022 验证，2023–2025 最终样本外。
- 股票池：每月末上市满 120 个交易日、非 ST、正常交易、过去 20 日成交额中位数不低于 2,000 万元。
- 因子：24 个价量因子，先 winsorize 再对 log 规模和行业虚拟变量回归取残差。
- 评价：RankIC、ICIR、Newey–West t 值、Q1–Q5、Q5−Q1、long-only 超额、1/3/6/12 个月衰减、因子相关性聚类、Benjamini–Hochberg 校正。
- 组合：训练/验证期筛选、相关性去重、最多 6 个因子等权 rank 合成；主组合为 long-only top 10%，同时报告研究性多空组合。
- 交易：月末收盘生成信号，下一交易日开盘建仓；单边成本基准 10 bps，并报告 20/30 bps 敏感性。

## 目录

```text
configs/                 研究参数
factor_zoo/              数据、因子、评价、组合、报告和 CLI
data/raw/                下载的原始数据（不提交）
data/processed/          面板、股票池、因子（不提交）
results/tables/          IC、分组、衰减、回测结果
results/figures/         报告图表
docs/                    研究设计、因子目录和报告
tests/                   合成数据单元测试
```

## 重要限制

1. 先完成数据闸门再解释结果：全历史股票列表、足够的行业快照、月度覆盖和规模代理校验。当前 1,262 代码样本只是阶段性结果。
2. 2023–2025 不得参与因子筛选；如果改代码后重新选择因子，必须把测试集重新定义。
3. 当前是日频研究和月频调仓，不包含分钟级、订单簿或实盘执行。
4. 规模代理是从换手率反推的近似值；没有通过抽样校验前不要把规模中性化结果当作最终结论。
5. 101 因子不是本项目的主线；后续如加入，只能作为独立扩展并重新预注册。

全量数据补数状态见 [数据扩展状态](docs/reports/data_expansion_status_2026-11-18.md)。

## 验证

```powershell
python -m pytest tests -q
```

CI 使用同一套合成数据测试，不会下载外部行情。
