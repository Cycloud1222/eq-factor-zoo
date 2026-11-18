"""Data acquisition and point-in-time A-share universe construction."""

from __future__ import annotations

import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Settings


# Excludes Shanghai/Shenzhen indices such as sz.399001 and sh.000001.
A_SHARE_PATTERN = re.compile(
    r"^(?:sh\.(?:60[0135]|68[89])\d{3}|sz\.(?:00[0-3]|30[0-2])\d{3})$"
)
DAILY_FIELDS = (
    "date,code,open,high,low,close,preclose,"
    "volume,amount,adjustflag,turn,tradestatus,pctChg,isST"
)
DAILY_COLUMNS = DAILY_FIELDS.split(",")


def _fetch_result(result, table_name: str) -> pd.DataFrame:
    if result.error_code != "0":
        raise RuntimeError(f"{table_name} 查询失败：{result.error_msg}")
    rows = []
    while result.next():
        rows.append(result.get_row_data())
    return pd.DataFrame(rows, columns=result.fields)


def _login():
    import baostock as bs

    result = bs.login()
    if result.error_code != "0":
        raise RuntimeError(f"BaoStock 登录失败：{result.error_msg}")
    return bs


def _logout(bs) -> None:
    try:
        bs.logout()
    except Exception:
        pass


def trade_calendar(settings: Settings) -> pd.DataFrame:
    """Return the BaoStock trading calendar over the configured sample."""
    bs = _login()
    try:
        frame = _fetch_result(
            bs.query_trade_dates(
                start_date=settings.start_date.strftime("%Y-%m-%d"),
                end_date=settings.end_date.strftime("%Y-%m-%d"),
            ),
            "交易日历",
        )
    finally:
        _logout(bs)
    frame["calendar_date"] = pd.to_datetime(frame["calendar_date"], errors="raise")
    frame["is_trading_day"] = frame["is_trading_day"].astype(str)
    return frame.loc[frame["is_trading_day"] == "1", ["calendar_date"]].reset_index(drop=True)


def month_end_dates(settings: Settings, calendar: pd.DataFrame | None = None) -> list[pd.Timestamp]:
    """Last trading day of each month in the configured sample."""
    calendar = trade_calendar(settings) if calendar is None else calendar
    dates = pd.to_datetime(calendar["calendar_date"]).sort_values()
    periods = dates.dt.to_period("M")
    return dates.groupby(periods).max().sort_values().tolist()


def snapshot_dates(settings: Settings, frequency: str = "monthly") -> list[pd.Timestamp]:
    """Select month-end dates at monthly, quarterly or annual resolution."""
    dates = month_end_dates(settings)
    series = pd.Series(dates)
    if frequency == "monthly":
        return series.tolist()
    if frequency == "quarterly":
        periods = series.dt.to_period("Q")
    elif frequency == "annual":
        periods = series.dt.to_period("Y")
    else:
        raise ValueError(f"不支持的行业快照频率：{frequency}")
    return series.groupby(periods).max().sort_values().tolist()


def _stock_list_path(settings: Settings, day: pd.Timestamp) -> Path:
    return settings.raw_dir / "stock_lists" / f"{day:%Y-%m-%d}.parquet"


def download_stock_lists(settings: Settings, force: bool = False) -> pd.DataFrame:
    """Download monthly stock lists and persist their union.

    Monthly snapshots are used to include names that later left the market;
    using only the current stock list would reintroduce survivorship bias.
    """
    settings.prepare_dirs()
    dates = month_end_dates(settings)
    bs = _login()
    frames: list[pd.DataFrame] = []
    try:
        for day in dates:
            path = _stock_list_path(settings, day)
            if path.exists() and not force:
                raw = pd.read_parquet(path)
            else:
                raw = _fetch_result(bs.query_all_stock(day=f"{day:%Y-%m-%d}"), "股票列表")
                path.parent.mkdir(parents=True, exist_ok=True)
                raw.to_parquet(path, index=False)
            if "code" not in raw.columns:
                raise ValueError(f"{day} 股票列表缺少 code 字段：{raw.columns.tolist()}")
            raw = raw.copy()
            raw["code"] = raw["code"].astype("string").str.strip()
            raw = raw.loc[raw["code"].str.match(A_SHARE_PATTERN, na=False)].copy()
            raw["date"] = day
            frames.append(raw[["date", "code"]])
    finally:
        _logout(bs)

    if not frames:
        raise RuntimeError("没有取得任何历史股票列表")
    union = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(["date", "code"])
        .sort_values(["date", "code"])
        .reset_index(drop=True)
    )
    union.to_parquet(settings.raw_dir / "stock_list_union.parquet", index=False)
    union.to_csv(
        settings.raw_dir / "stock_list_union.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return union


def load_stock_codes(settings: Settings, limit: int = 0, offset: int = 0) -> list[str]:
    path = settings.raw_dir / "stock_list_union.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"找不到历史股票列表：{path}。请先运行 download-stock-lists。"
        )
    codes = sorted(
        pd.read_parquet(path)["code"].astype(str).drop_duplicates().tolist()
    )
    codes = codes[offset:]
    return codes[:limit] if limit else codes


def download_daily(
    settings: Settings,
    codes: list[str] | None = None,
    force: bool = False,
    limit: int = 0,
    offset: int = 0,
    retries: int = 3,
    sleep_seconds: float = 0.05,
) -> pd.DataFrame:
    """Download daily bars for the union of historical A-share codes."""
    settings.prepare_dirs()
    codes = codes or load_stock_codes(settings, limit=limit, offset=offset)
    raw_dir = settings.raw_dir / "baostock"
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = settings.raw_dir / "download_manifest.csv"
    manifest = (
        pd.read_csv(manifest_path, encoding="utf-8-sig")
        if manifest_path.exists()
        else pd.DataFrame(
            columns=["date", "code", "status", "rows", "error", "downloaded_at"]
        )
    )
    existing = set()
    if not force:
        existing = {
            str(code)
            for code in manifest.loc[manifest["status"] == "success", "code"].astype(str)
            if (raw_dir / f"{code}.parquet").exists()
        }

    bs = _login()
    records = []
    try:
        for index, code in enumerate(codes, start=1):
            path = raw_dir / f"{code}.parquet"
            if code in existing and path.exists():
                continue
            last_error = ""
            frame = pd.DataFrame()
            for attempt in range(1, max(1, retries) + 1):
                try:
                    frame = _fetch_result(
                        bs.query_history_k_data_plus(
                            code,
                            DAILY_FIELDS,
                            start_date=settings.start_date.strftime("%Y-%m-%d"),
                            end_date=settings.end_date.strftime("%Y-%m-%d"),
                            frequency="d",
                            adjustflag="1",
                        ),
                        f"{code} 日线",
                    )
                    last_error = ""
                    break
                except Exception as error:
                    last_error = str(error)
                    if attempt < retries:
                        time.sleep(2)

            if last_error or frame.empty:
                status = "empty" if not last_error else "failed"
                records.append({
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "code": code,
                    "status": status,
                    "rows": 0,
                    "error": last_error,
                    "downloaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                })
                print(f"[{index}/{len(codes)}] {code}: {status}")
            else:
                frame.to_parquet(path, index=False)
                records.append({
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "code": code,
                    "status": "success",
                    "rows": len(frame),
                    "error": "",
                    "downloaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                })
                print(f"[{index}/{len(codes)}] {code}: {len(frame)} rows")
            if sleep_seconds:
                time.sleep(sleep_seconds)
    finally:
        _logout(bs)

    if records:
        manifest = pd.concat([manifest, pd.DataFrame(records)], ignore_index=True)
        manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    return manifest


def download_industry(
    settings: Settings,
    force: bool = False,
    frequency: str = "monthly",
) -> pd.DataFrame:
    """Download BaoStock industry snapshots at the requested resolution.

    Annual snapshots are a practical fallback when API latency makes monthly
    snapshots prohibitively slow. The effective date and age are still kept
    point-in-time; the resolution used is recorded in the output.
    """
    settings.prepare_dirs()
    dates = snapshot_dates(settings, frequency=frequency)
    bs = _login()
    frames = []
    try:
        for day in dates:
            path = settings.raw_dir / "industry" / f"{day:%Y-%m-%d}.parquet"
            if path.exists() and not force:
                raw = pd.read_parquet(path)
            else:
                raw = _fetch_result(
                    bs.query_stock_industry(date=f"{day:%Y-%m-%d}"),
                    "行业分类",
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                raw.to_parquet(path, index=False)
            if raw.empty:
                continue
            frame = raw.copy()
            frame.columns = [str(column).strip() for column in frame.columns]
            frame["date"] = day
            frames.append(frame)
    finally:
        _logout(bs)

    if not frames:
        raise RuntimeError("没有取得行业快照")
    industry = pd.concat(frames, ignore_index=True)
    rename = {
        "code": "code",
        "industry": "industry",
        "industryClassification": "industry_classification",
        "updateDate": "update_date",
    }
    for old, new in rename.items():
        if old in industry.columns and old != new:
            industry = industry.rename(columns={old: new})
    needed = ["date", "code", "industry", "industry_classification", "update_date"]
    missing = [column for column in needed if column not in industry.columns]
    if missing:
        raise ValueError(f"行业表缺少字段：{missing}")
    industry = industry[needed].copy()
    industry["code"] = industry["code"].astype("string").str.strip()
    industry = industry.loc[
        industry["code"].str.match(A_SHARE_PATTERN, na=False)
    ].copy()
    industry["date"] = pd.to_datetime(industry["date"])
    industry["industry"] = industry["industry"].fillna("Unknown").astype("string")
    industry["snapshot_frequency"] = frequency
    industry.to_parquet(settings.processed_dir / "industry.parquet", index=False)
    return industry


def build_panel(settings: Settings, force: bool = False) -> pd.DataFrame:
    """Merge per-stock parquet files into one point-in-time daily panel."""
    destination = settings.processed_dir / "daily_panel.parquet"
    if destination.exists() and not force:
        return pd.read_parquet(destination)
    files = sorted((settings.raw_dir / "baostock").glob("*.parquet"))
    if not files:
        raise FileNotFoundError("未找到 raw/baostock/*.parquet，请先下载行情")

    frames = []
    for path in files:
        frame = pd.read_parquet(path)
        if "code" not in frame.columns:
            raise ValueError(f"{path.name} 缺少 code")
        frames.append(frame)
    panel = pd.concat(frames, ignore_index=True)
    missing = [column for column in DAILY_COLUMNS if column not in panel.columns]
    if missing:
        raise ValueError(f"面板缺少字段：{missing}")
    panel = panel[DAILY_COLUMNS].copy()
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
    panel["code"] = panel["code"].astype("string")
    for column in ("open", "high", "low", "close", "preclose", "volume", "amount", "turn", "pctChg"):
        panel[column] = pd.to_numeric(panel[column], errors="coerce")
    for column in ("adjustflag", "tradestatus", "isST"):
        panel[column] = pd.to_numeric(panel[column], errors="coerce").astype("Int64")
    panel = panel.loc[panel["adjustflag"] == 1].copy()
    panel = panel.dropna(subset=["date", "code", "close"]).copy()
    panel = panel.drop_duplicates(["date", "code"]).sort_values(["code", "date"])
    panel["prev_close"] = panel.groupby("code", sort=False)["close"].shift(1)
    panel["ret_1d"] = panel["close"] / panel["prev_close"] - 1
    panel = panel.reset_index(drop=True)
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(destination, index=False)
    return panel


def rebalance_dates(panel: pd.DataFrame) -> list[pd.Timestamp]:
    """Last trading day of each month present in the panel."""
    dates = pd.Series(pd.to_datetime(panel["date"].dropna().unique()))
    periods = dates.dt.to_period("M")
    return dates.groupby(periods).max().sort_values().tolist()


def attach_industry(
    universe: pd.DataFrame,
    industry: pd.DataFrame,
) -> pd.DataFrame:
    """Attach the latest industry snapshot available at each universe date."""
    left = universe.sort_values(["date", "code"]).copy()
    right = industry.rename(columns={"date": "industry_date"}).sort_values(
        ["industry_date", "code"]
    )
    merged = pd.merge_asof(
        left,
        right,
        left_on="date",
        right_on="industry_date",
        by="code",
        direction="backward",
    )
    merged["industry"] = merged["industry"].fillna("Unknown")
    merged["industry_age_days"] = (
        merged["date"] - merged["industry_date"]
    ).dt.days
    return merged


def build_universe(
    settings: Settings,
    panel: pd.DataFrame | None = None,
    industry: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the month-end eligible universe and record rejection reasons."""
    panel = build_panel(settings) if panel is None else panel
    industry = (
        pd.read_parquet(settings.processed_dir / "industry.parquet")
        if industry is None and (settings.processed_dir / "industry.parquet").exists()
        else industry
    )
    panel = panel.sort_values(["code", "date"]).copy()
    panel["listed_days"] = panel.groupby("code", sort=False).cumcount() + 1
    panel["amount_median_20"] = panel.groupby("code", sort=False)["amount"].transform(
        lambda series: series.rolling(20, min_periods=20).median()
    )
    panel["turn"] = pd.to_numeric(panel["turn"], errors="coerce")
    panel["float_mcap_proxy"] = np.where(
        panel["turn"] > 0,
        panel["amount"] / (panel["turn"] / 100.0),
        np.nan,
    )
    dates = set(rebalance_dates(panel))
    universe = panel.loc[panel["date"].isin(dates)].copy()
    universe = universe.rename(columns={"date": "date"})

    if industry is not None and not industry.empty:
        universe = attach_industry(universe, industry)
    else:
        universe["industry"] = "Unknown"
        universe["industry_age_days"] = np.nan

    min_days = int(settings.universe["min_listed_days"])
    min_amount = float(settings.universe["min_median_amount_20d"])
    exclude_st = bool(settings.universe["exclude_st"])
    exclude_suspended = bool(settings.universe["exclude_suspended"])

    reasons = []
    for row in universe.itertuples():
        reason = ""
        if pd.isna(row.close) or row.close <= 0:
            reason = "invalid_price"
        elif row.listed_days < min_days:
            reason = "too_new"
        elif exclude_st and (pd.isna(row.isST) or row.isST == 1):
            reason = "st_or_unknown"
        elif exclude_suspended and (pd.isna(row.tradestatus) or row.tradestatus != 1):
            reason = "suspended"
        elif pd.isna(row.amount_median_20) or row.amount_median_20 < min_amount:
            reason = "illiquid"
        elif pd.isna(row.float_mcap_proxy) or row.float_mcap_proxy <= 0:
            reason = "missing_size"
        reasons.append(reason)
    universe["eligible"] = [reason == "" for reason in reasons]
    universe["reject_reason"] = reasons
    universe = universe.loc[universe["eligible"]].copy()
    universe.to_parquet(settings.processed_dir / "universe.parquet", index=False)
    return universe


def forward_returns(
    panel: pd.DataFrame,
    dates: list[pd.Timestamp],
    horizons: tuple[int, ...] = (1, 3, 6, 12),
) -> pd.DataFrame:
    """Open-to-next-rebalance-close forward returns.

    Signal date t is a month-end close. Entry is the next trading day's open.
    Exit is the close of the t+h-th rebalance date. This avoids using the
    signal-day close for execution and keeps the label independent of the
    current-day cross-section.
    """
    dates = sorted(pd.Timestamp(date) for date in dates)
    output = []
    for code, group in panel.sort_values(["code", "date"]).groupby("code", sort=False):
        group = group.sort_values("date").reset_index(drop=True)
        trading_dates = group["date"].to_numpy()
        open_values = group["open"].to_numpy(dtype=float)
        close_values = group["close"].to_numpy(dtype=float)
        for index, date in enumerate(dates):
            entry_position = np.searchsorted(trading_dates, np.datetime64(date), side="right")
            if entry_position >= len(group) or not np.isfinite(open_values[entry_position]):
                continue
            for horizon in horizons:
                target_index = index + horizon
                if target_index >= len(dates):
                    continue
                target_date = dates[target_index]
                exit_position = np.searchsorted(
                    trading_dates, np.datetime64(target_date), side="right"
                ) - 1
                if exit_position < entry_position or not np.isfinite(close_values[exit_position]):
                    continue
                output.append({
                    "date": date,
                    "code": code,
                    "horizon": horizon,
                    "forward_return": close_values[exit_position] / open_values[entry_position] - 1,
                })
    return pd.DataFrame(output)
