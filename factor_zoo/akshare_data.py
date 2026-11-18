"""Optional AKShare fallback for stocks missing from the BaoStock cache.

AKShare and BaoStock expose different column names. This module normalizes the
AKShare daily-history response into the same contract consumed by
``data.build_panel``. It deliberately writes into the same raw directory only
for missing codes; existing BaoStock files are never overwritten.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd

from .config import Settings


def _symbol(code: str) -> str:
    return code.split(".", 1)[-1]


def normalize_akshare(frame: pd.DataFrame, code: str) -> pd.DataFrame:
    required = {"日期", "开盘", "收盘", "最高", "最低", "成交量", "成交额", "换手率"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{code} AKShare 缺少字段：{sorted(missing)}")
    output = pd.DataFrame({
        "date": pd.to_datetime(frame["日期"], errors="coerce").dt.strftime("%Y-%m-%d"),
        "code": code,
        "open": pd.to_numeric(frame["开盘"], errors="coerce"),
        "high": pd.to_numeric(frame["最高"], errors="coerce"),
        "low": pd.to_numeric(frame["最低"], errors="coerce"),
        "close": pd.to_numeric(frame["收盘"], errors="coerce"),
        "volume": pd.to_numeric(frame["成交量"], errors="coerce"),
        "amount": pd.to_numeric(frame["成交额"], errors="coerce"),
        "turn": pd.to_numeric(frame["换手率"], errors="coerce"),
    })
    output = output.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    output["preclose"] = output["close"].shift(1)
    if "涨跌幅" in frame.columns:
        output["pctChg"] = pd.to_numeric(frame["涨跌幅"], errors="coerce")
    else:
        output["pctChg"] = (output["close"] / output["preclose"] - 1) * 100
    output["adjustflag"] = 1
    output["tradestatus"] = 1
    output["isST"] = 0
    output["source"] = "akshare-stock_zh_a_hist"
    return output[
        [
            "date", "code", "open", "high", "low", "close", "preclose",
            "volume", "amount", "adjustflag", "turn", "tradestatus",
            "pctChg", "isST", "source",
        ]
    ]


def download_missing_akshare(
    settings: Settings,
    workers: int = 4,
    retries: int = 2,
    limit: int = 0,
) -> pd.DataFrame:
    """Download only codes that do not have a raw parquet file."""
    try:
        import akshare as ak
    except ImportError as error:
        raise RuntimeError(
            "需要 AKShare 回退下载：python -m pip install akshare"
        ) from error

    union_path = settings.raw_dir / "stock_list_union.parquet"
    if not union_path.exists():
        raise FileNotFoundError("请先运行 build-stock-union 或 download-stock-lists")
    raw_dir = settings.raw_dir / "baostock"
    raw_dir.mkdir(parents=True, exist_ok=True)
    codes = sorted(
        code
        for code in pd.read_parquet(union_path)["code"].astype(str).unique()
        if not (raw_dir / f"{code}.parquet").exists()
    )
    if limit:
        codes = codes[:limit]
    if not codes:
        return pd.DataFrame(columns=["code", "status", "rows", "error", "source"])

    def fetch(code: str) -> dict:
        last_error = ""
        for attempt in range(1, retries + 1):
            try:
                raw = ak.stock_zh_a_hist(
                    symbol=_symbol(code),
                    period="daily",
                    start_date=settings.start_date.strftime("%Y%m%d"),
                    end_date=settings.end_date.strftime("%Y%m%d"),
                    adjust="hfq",
                )
                if raw is None or raw.empty:
                    raise ValueError("空数据")
                normalized = normalize_akshare(raw, code)
                normalized.to_parquet(raw_dir / f"{code}.parquet", index=False)
                return {
                    "code": code,
                    "status": "success",
                    "rows": len(normalized),
                    "error": "",
                    "source": "akshare",
                }
            except Exception as error:
                last_error = str(error)
                if attempt < retries:
                    import time

                    time.sleep(2)
        return {
            "code": code,
            "status": "failed",
            "rows": 0,
            "error": last_error,
            "source": "akshare",
        }

    records = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(fetch, code): code for code in codes}
        for index, future in enumerate(as_completed(futures), start=1):
            record = future.result()
            records.append(record)
            if index % 25 == 0 or index == len(codes):
                print(f"[{index}/{len(codes)}] AKShare missing-code download")

    manifest = pd.DataFrame(records)
    manifest_path = settings.raw_dir / "akshare_manifest.csv"
    manifest["downloaded_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    return manifest
