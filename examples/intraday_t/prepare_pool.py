"""
从全 A 股日线 CSV 库中筛选出与 SZ301536 同质的训练股票池
============================================================
用法：
  python prepare_pool.py build           # 完整流程：筛选 → 转换 → dump bin

筛选规则（创业板 + 科创板 小盘成长风格）：
  * 板块：30xxxx（创业板）+ 68xxxx（科创板）
  * 上市时长：首日 ≤ 2023-04-29（保证 ≥ 3 年历史，可以撑过训练窗口）
  * 数据完整度：交易日数 ≥ 600 行
  * 流动性：最近 252 个交易日日均成交额 ≥ 5000 万
  * 健康度：剔除最近 60 日内停牌天数 > 20% 的标的
  * 必须包含目标股票 SZ301536（即使它本身上市不足 3 年，也强制保留）

输出：
  data/cn_data_pool/                 ← Qlib bin 训练池
  data/_pool_csv/                    ← 中间临时 CSV（dump 完可清理）
  data/_pool_selected.csv            ← 入选股票清单 + 元信息
"""
from __future__ import annotations

import sys
import shutil
import subprocess
from pathlib import Path

import fire
import pandas as pd

# ---------------- paths ----------------
PROJ_ROOT = Path(__file__).resolve().parents[2]   # qlib_helki/
RAW_DIR    = PROJ_ROOT / "data" / "A_Stock_daily_qfq"
TMP_CSV    = PROJ_ROOT / "data" / "_pool_csv"
QLIB_DIR   = PROJ_ROOT / "data" / "cn_data_pool"
LIST_PATH  = PROJ_ROOT / "data" / "_pool_selected.csv"
DUMP_SCRIPT = PROJ_ROOT / "scripts" / "dump_bin.py"

# 目标股票（始终强制入选）
TARGET_CODE = "301536"

# 筛选阈值
EARLIEST_LISTED = "2023-04-29"   # 首日不得晚于此日 → 历史 ≥ 3 年
MIN_ROWS         = 600            # 至少 600 个交易日
MIN_AVG_AMOUNT   = 5e7            # 最近 252 日日均成交额 ≥ 5000 万
RECENT_WINDOW    = 252
SUSPEND_WINDOW   = 60             # 检查最近 60 日
SUSPEND_RATIO    = 0.20           # 停牌天数 > 20% 则剔除
BOARD_PREFIXES   = ("30", "68")   # 创业板 + 科创板

# CSV 列映射（同 prepare_data.py 一致）
COL_MAP = {
    "日期":   "date",
    "股票代码": "symbol",
    "开盘":   "open",
    "收盘":   "close",
    "最高":   "high",
    "最低":   "low",
    "成交量":  "volume",
    "成交额":  "amount",
}


def _code_to_symbol(code: str) -> str:
    """301536 → sz301536; 600519 → sh600519; 688981 → sh688981."""
    if code.startswith(("60", "68")):
        return f"sh{code}"
    return f"sz{code}"


def _read_one(path: Path) -> pd.DataFrame | None:
    """读单只股票 CSV，标准化列。"""
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        try:
            df = pd.read_csv(path, encoding="gbk")
        except Exception:
            return None
    df = df.rename(columns=COL_MAP)
    need = ["date", "open", "high", "low", "close", "volume", "amount"]
    if not all(c in df.columns for c in need):
        return None
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


def _passes_filter(code: str, df: pd.DataFrame) -> tuple[bool, dict]:
    """返回 (是否入选, 元信息)。目标股票永远入选。"""
    info = {
        "code": code,
        "rows": len(df),
        "first_date": df["date"].iloc[0] if len(df) else None,
        "last_date":  df["date"].iloc[-1] if len(df) else None,
        "avg_amount_252": float("nan"),
        "suspend_ratio_60": float("nan"),
        "selected": False,
        "reason": "",
    }

    # 目标股票强制入选（不应用任何过滤）
    if code == TARGET_CODE:
        info["selected"] = True
        info["reason"] = "target"
        # 仍计算流动性指标供观察
        if len(df) >= RECENT_WINDOW:
            info["avg_amount_252"] = df["amount"].tail(RECENT_WINDOW).mean()
        return True, info

    # 板块过滤
    if not code.startswith(BOARD_PREFIXES):
        info["reason"] = "off_board"
        return False, info

    # 数据量
    if len(df) < MIN_ROWS:
        info["reason"] = f"rows<{MIN_ROWS}"
        return False, info

    # 上市时间
    if df["date"].iloc[0] > pd.Timestamp(EARLIEST_LISTED):
        info["reason"] = f"listed_after_{EARLIEST_LISTED}"
        return False, info

    # 流动性（最近 252 日日均成交额）
    avg_amount = df["amount"].tail(RECENT_WINDOW).mean()
    info["avg_amount_252"] = avg_amount
    if avg_amount < MIN_AVG_AMOUNT:
        info["reason"] = f"low_liquidity({avg_amount/1e8:.2f}亿)"
        return False, info

    # 停牌检测：最近 60 日里 volume==0 的天数
    recent = df.tail(SUSPEND_WINDOW)
    suspend_ratio = (recent["volume"] == 0).mean()
    info["suspend_ratio_60"] = suspend_ratio
    if suspend_ratio > SUSPEND_RATIO:
        info["reason"] = f"too_many_suspends({suspend_ratio:.2%})"
        return False, info

    info["selected"] = True
    info["reason"] = "ok"
    return True, info


def _to_qlib_csv(code: str, df: pd.DataFrame, out_dir: Path) -> Path:
    """转换为 dump_bin.py 期望的格式。"""
    df = df.copy()
    df["factor"] = 1.0
    if "amount" in df.columns and "volume" in df.columns:
        df["vwap"] = df["amount"] / df["volume"].replace(0, float("nan"))
    else:
        df["vwap"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    keep = ["date", "open", "high", "low", "close", "volume",
            "amount", "vwap", "factor"]
    df = df[keep]
    symbol = _code_to_symbol(code)
    out = out_dir / f"{symbol}.csv"
    df.to_csv(out, index=False)
    return out


def _run_dump(csv_dir: Path, qlib_dir: Path) -> None:
    qlib_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(DUMP_SCRIPT), "dump_all",
        "--data_path", str(csv_dir),
        "--qlib_dir",  str(qlib_dir),
        "--freq",      "day",
        "--date_field_name",   "date",
        "--symbol_field_name", "symbol",  # dump_bin 用文件名，但仍要传参
        "--max_workers", "4",
    ]
    print(f"[dump] {' '.join(cmd)}")
    res = subprocess.run(cmd, check=False)
    if res.returncode != 0:
        raise RuntimeError(f"dump_bin failed exit={res.returncode}")


def build():
    """完整构建流程。"""
    if not RAW_DIR.exists():
        raise FileNotFoundError(RAW_DIR)

    # 1) 列出候选文件（创业板 + 科创板，强制包含目标）
    all_csvs = list(RAW_DIR.glob("*_daily_qfq.csv"))
    cand_csvs = []
    for p in all_csvs:
        code = p.stem.split("_")[0]
        if code.startswith(BOARD_PREFIXES) or code == TARGET_CODE:
            cand_csvs.append(p)
    print(f"[1/4] 候选文件数: {len(cand_csvs)} / 总数 {len(all_csvs)}")

    # 2) 逐个筛选
    print(f"[2/4] 筛选中...")
    rows = []
    selected_dfs: dict[str, pd.DataFrame] = {}
    for i, p in enumerate(cand_csvs):
        if i and i % 200 == 0:
            print(f"   ...processed {i}/{len(cand_csvs)}")
        code = p.stem.split("_")[0]
        df = _read_one(p)
        if df is None or len(df) == 0:
            rows.append({"code": code, "selected": False, "reason": "read_fail"})
            continue
        ok, info = _passes_filter(code, df)
        rows.append(info)
        if ok:
            selected_dfs[code] = df

    info_df = pd.DataFrame(rows)
    info_df.to_csv(LIST_PATH, index=False, encoding="utf-8-sig")

    n_sel = info_df["selected"].sum()
    print(f"\n[筛选汇总] 入选 {n_sel} / 候选 {len(cand_csvs)}")
    print(f"   清单: {LIST_PATH}")
    print(info_df[info_df["selected"]].head(10).to_string(index=False))

    # 拒绝原因 top
    rej = info_df[~info_df["selected"]]["reason"].value_counts()
    print(f"\n[拒绝原因 Top]")
    print(rej.head(10).to_string())

    if TARGET_CODE not in selected_dfs:
        raise RuntimeError(f"目标股票 {TARGET_CODE} 未入选，请检查源数据")

    # 3) 转换为 Qlib CSV
    print(f"\n[3/4] 写入临时 CSV → {TMP_CSV}")
    if TMP_CSV.exists():
        shutil.rmtree(TMP_CSV)
    TMP_CSV.mkdir(parents=True)
    for code, df in selected_dfs.items():
        _to_qlib_csv(code, df, TMP_CSV)

    # 4) dump_bin
    print(f"\n[4/4] dump_bin → {QLIB_DIR}")
    if QLIB_DIR.exists():
        shutil.rmtree(QLIB_DIR)
    _run_dump(TMP_CSV, QLIB_DIR)

    # 清理临时
    shutil.rmtree(TMP_CSV, ignore_errors=True)

    # 写一个 instruments/pool.txt 与 all.txt 一致（方便配置直接引用）
    inst_dir = QLIB_DIR / "instruments"
    if (inst_dir / "all.txt").exists():
        shutil.copy(inst_dir / "all.txt", inst_dir / "pool.txt")

    print(f"\n[OK] 训练池构建完成")
    print(f"  bin 路径: {QLIB_DIR}")
    print(f"  入选数:   {n_sel}")
    print(f"  使用方式: market: pool, 在 yaml 的 provider_uri.day 指向 {QLIB_DIR}")


if __name__ == "__main__":
    fire.Fire({"build": build})
