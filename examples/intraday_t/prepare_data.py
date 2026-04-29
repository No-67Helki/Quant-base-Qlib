"""
一键将本地 CSV 数据转换并导入 Qlib bin 格式
==============================================
用法（在 examples/intraday_t/ 目录下执行）：

  python prepare_data.py convert_day
  python prepare_data.py convert_all   # 日频 + 分钟频

原始 CSV 要求：
  data/301536/day_data/301536_daily_qfq.csv
  data/301536/min_data/<任意名>.csv      （分钟数据，可选）

转换后数据写入：
  data/cn_data/          ← 日频 Qlib bin
  data/cn_data_1min/     ← 分钟频 Qlib bin（如有分钟CSV）
"""
from __future__ import annotations

import sys
import subprocess
from pathlib import Path

import fire
import pandas as pd

# ------------------------------------------------------------------ paths ---
PROJ_ROOT = Path(__file__).resolve().parents[2]   # qlib_helki/
DATA_ROOT  = PROJ_ROOT / "data"
DAY_CSV_DIR = DATA_ROOT / "301536" / "day_data"
MIN_CSV_DIR = DATA_ROOT / "301536" / "min_data"
QLIB_DAY_DIR = DATA_ROOT / "cn_data"
QLIB_MIN_DIR = DATA_ROOT / "cn_data_1min"
DUMP_SCRIPT  = PROJ_ROOT / "scripts" / "dump_bin.py"

# 正确的 Qlib 交易所前缀（深圳 = SZ, 上海 = SH）
EXCHANGE = "SZ"
STOCK_CODE = "301536"          # 与 CSV 文件名一致
SYMBOL = f"{EXCHANGE.lower()}{STOCK_CODE}"   # sz301536

# 日频列名映射（中文 → 英文）
DAY_COL_MAP = {
    "日期":   "date",
    "股票代码": "symbol",
    "开盘":   "open",
    "收盘":   "close",
    "最高":   "high",
    "最低":   "low",
    "成交量":  "volume",
    "成交额":  "amount",
}

# 分钟数据列名映射（如果你的分钟CSV用的其他名称，在这里改）
MIN_COL_MAP = {
    "时间":    "date",
    "date":    "date",
    "datetime":"date",
    "代码":    "symbol",
    "股票代码": "symbol",
    "开盘":    "open",
    "开盘价":   "open",
    "open":    "open",
    "收盘":    "close",
    "收盘价":   "close",
    "close":   "close",
    "最高":    "high",
    "最高价":   "high",
    "high":    "high",
    "最低":    "low",
    "最低价":   "low",
    "low":     "low",
    "成交量":   "volume",
    "volume":  "volume",
    "成交额":   "amount",
    "amount":  "amount",
}


def _prepare_day_csv(out_dir: Path) -> Path:
    """将中文日频 CSV 转换为 Qlib dump_bin 所需的标准格式。"""
    # 找到源文件
    csvs = list(DAY_CSV_DIR.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"未找到日频 CSV: {DAY_CSV_DIR}")
    src = csvs[0]
    print(f"[日频] 读取: {src}")

    df = pd.read_csv(src, encoding="utf-8-sig")
    # 只保留并重命名需要的列
    df = df.rename(columns=DAY_COL_MAP)

    # 确保必要列存在
    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV 缺少列: {missing}，现有列: {list(df.columns)}")

    # 补充 Qlib 必需的 factor（已前复权时 factor=1.0）
    if "factor" not in df.columns:
        df["factor"] = 1.0

    # 补充 vwap（估算 = 成交额/成交量）
    if "vwap" not in df.columns and "amount" in df.columns:
        df["vwap"] = df["amount"] / df["volume"].replace(0, float("nan"))
    elif "vwap" not in df.columns:
        df["vwap"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4

    # 保证日期格式一致
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    # dump_bin.py 用文件名识别股票，不需要 symbol 列
    # 只保留数值型列，避免 "could not convert string to float" 报错
    keep = ["date", "open", "high", "low", "close", "volume",
            "amount", "vwap", "factor"]
    df = df[[c for c in keep if c in df.columns]]

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{SYMBOL}.csv"
    df.to_csv(out_path, index=False)
    print(f"[日频] 转换完成 → {out_path}  ({len(df)} 行)")
    return out_path


def _prepare_min_csv(out_dir: Path) -> Path | None:
    """将分钟 CSV 转换为 Qlib 格式（若存在）。支持多文件合并、GBK/UTF-8 自动识别。"""
    csvs = sorted(MIN_CSV_DIR.glob("*.csv"))
    if not csvs:
        print("[分钟] 未找到分钟数据 CSV，跳过。")
        return None

    frames = []
    for src in csvs:
        # 自动尝试 GBK / UTF-8 编码
        df = None
        for enc in ("gbk", "utf-8-sig", "utf-8"):
            try:
                df = pd.read_csv(src, encoding=enc)
                break
            except UnicodeDecodeError:
                continue
        if df is None:
            raise RuntimeError(f"无法以 GBK/UTF-8 读取 {src}")
        print(f"[分钟] 读取: {src.name}  ({len(df)} 行)")
        df = df.rename(columns=MIN_COL_MAP)
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)

    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"分钟 CSV 缺少列: {missing}，现有列: {list(df.columns)}")

    if "factor" not in df.columns:
        df["factor"] = 1.0
    if "vwap" not in df.columns and "amount" in df.columns:
        df["vwap"] = df["amount"] / df["volume"].replace(0, float("nan"))
    elif "vwap" not in df.columns:
        df["vwap"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4

    df["symbol"] = SYMBOL
    # 兼容混合日期格式（如 "2024-03-28 09:30:00" 与 "2026/01/05 09:30"）
    df["date"] = pd.to_datetime(df["date"], format="mixed", errors="coerce")
    bad = df["date"].isna().sum()
    if bad:
        print(f"[分钟] 警告：{bad} 行日期解析失败，已丢弃")
        df = df.dropna(subset=["date"])
    # 去重并按时间排序
    df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    df["date"] = df["date"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # dump_bin.py 用文件名识别股票，不需要 symbol 列
    keep = ["date", "open", "high", "low", "close", "volume",
            "amount", "vwap", "factor"]
    df = df[[c for c in keep if c in df.columns]]

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{SYMBOL}.csv"
    df.to_csv(out_path, index=False)
    print(f"[分钟] 合并转换完成 → {out_path}  ({len(df)} 行)")
    return out_path


def _run_dump(csv_dir: Path, qlib_dir: Path, freq: str = "day") -> None:
    """调用 Qlib scripts/dump_bin.py dump_all。"""
    qlib_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(DUMP_SCRIPT), "dump_all",
        "--data_path", str(csv_dir),
        "--qlib_dir",  str(qlib_dir),
        "--freq",      freq,
        "--date_field_name",   "date",
        "--symbol_field_name", "symbol",
        "--max_workers", "1",
    ]
    print(f"\n[dump] 运行: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"dump_bin.py 失败，exit code={result.returncode}")
    print(f"[dump] 完成 → {qlib_dir}")


def convert_day():
    """只转换并导入日频数据。"""
    tmp = DATA_ROOT / "_tmp_day"
    _prepare_day_csv(tmp)
    _run_dump(tmp, QLIB_DAY_DIR, freq="day")
    # 清理临时目录
    import shutil; shutil.rmtree(tmp, ignore_errors=True)
    print("\n✅ 日频数据导入完成！")
    print(f"   目标目录: {QLIB_DAY_DIR}")


def convert_min():
    """只转换并导入分钟数据（需 min_data 目录下有 CSV）。"""
    tmp = DATA_ROOT / "_tmp_min"
    out = _prepare_min_csv(tmp)
    if out is None:
        print("无分钟数据，退出。")
        return
    _run_dump(tmp, QLIB_MIN_DIR, freq="1min")
    import shutil; shutil.rmtree(tmp, ignore_errors=True)
    print("\n✅ 分钟数据导入完成！")
    print(f"   目标目录: {QLIB_MIN_DIR}")


def convert_all():
    """转换并导入日频 + 分钟数据。"""
    convert_day()
    convert_min()


if __name__ == "__main__":
    fire.Fire({"convert_day": convert_day, "convert_min": convert_min, "convert_all": convert_all})
