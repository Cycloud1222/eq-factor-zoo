# 数据扩展状态（2026-11-18）

## 已完成

- 历史股票列表缓存：76 个月份，覆盖 2016-01-29 至 2022-03-31，并补入 2025-12-31 当前列表。
- 过滤指数、B 股和无效代码后，股票代码并集为 **5,411 个 A 股代码**。
- 现有原始日线文件：**1,309 个代码**。
- 行业快照：2019–2025 年度快照，已经能支撑 2020 年起的 point-in-time 合并。
- 2020–2025 年 1,262 代码的因子样本已跑通，见 [首轮样本结果](sample_results_2026-11-18.md)。

## 未完成

完整日线下载没有完成。当前网络环境下：

- BaoStock 的股票列表和单股票历史接口延迟很高；
- 并发会话会触发 `用户未登录` 或 `黑名单用户`，不适合用高并发硬刷；
- AKShare 备用源在 4 线程下可工作，但补齐数千只股票的耗时仍较长，且在部分并发请求下会被代理断开。

因此当前仓库保留的是一个**可复现的 1,262 代码样本**，而不是完整全 A 股结果。

## 恢复后的执行顺序

```powershell
# 若使用 BaoStock，先小批量确认接口恢复
factor-zoo --config configs/full_2020.yaml download-daily --workers 4 --limit 20

# 或用 AKShare 补缺失代码，分批运行
factor-zoo --config configs/full_2020.yaml download-akshare-missing --workers 4

# 全部代码补齐后
factor-zoo --config configs/full_2020.yaml build-panel --force
factor-zoo --config configs/sample_2020.yaml build-universe --force
factor-zoo --config configs/full_2020.yaml research --prefix full
factor-zoo --config configs/full_2020.yaml report --prefix full
```

不要在数据覆盖不完整时把 `full` 结果写成全市场结论；完整结果至少要求：

1. 5,411 个代码的日线文件覆盖率达到 90%；
2. 每月动态股票池有至少 300 只可交易股票；
3. 2024–2025 仍保持未参与因子筛选。
