"""
诊断 Alpha158HF 特征 & DEnsembleModel 收敛性
==============================================
用法：
  python diagnose.py
"""
from __future__ import annotations
import sys
from pathlib import Path

DIRNAME = Path(__file__).resolve().parent
sys.path.insert(0, str(DIRNAME))

import numpy as np
import pandas as pd
import qlib
from qlib.utils import init_instance_by_config
from qlib.config import REG_CN

# ---- 初始化 ----
qlib.init(
    provider_uri={
        "day":  str(DIRNAME.parents[1] / "data" / "cn_data"),
        "1min": str(DIRNAME.parents[1] / "data" / "cn_data_1min"),
    },
    region=REG_CN,
)

# ---- 构造 dataset ----
DH_CFG = dict(
    start_time="2024-03-28",
    end_time="2026-04-28",
    fit_start_time="2024-03-28",
    fit_end_time="2025-09-30",
    instruments=["SZ301536"],
    infer_processors=[
        {"class": "RobustZScoreNorm",
         "kwargs": {"fields_group": "feature", "clip_outlier": True}},
        {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
    ],
    learn_processors=[{"class": "DropnaLabel"}],
    label=["Ref($close, -2) / Ref($close, -1) - 1"],
)

dataset = init_instance_by_config({
    "class": "DatasetH",
    "module_path": "qlib.data.dataset",
    "kwargs": {
        "handler": {
            "class": "Alpha158HF",
            "module_path": "handlers.alpha158_hf",
            "kwargs": DH_CFG,
        },
        "segments": {
            "train": ["2024-03-28", "2025-06-30"],
            "valid": ["2025-07-01", "2025-12-31"],
            "test":  ["2026-01-01", "2026-04-28"],
        },
    },
})

# ---- 1. 检查 train/valid/test 三段特征 ----
print("=" * 80)
print("[1] 特征矩阵基本检查")
print("=" * 80)

df_train = dataset.prepare("train", col_set=["feature", "label"])
df_valid = dataset.prepare("valid", col_set=["feature", "label"])
df_test  = dataset.prepare("test",  col_set=["feature", "label"])

for name, df in [("train", df_train), ("valid", df_valid), ("test", df_test)]:
    feat = df["feature"]
    lab = df["label"]
    print(f"\n[{name}] shape={feat.shape}, label shape={lab.shape}")
    print(f"  样本时间范围: {feat.index.get_level_values('datetime').min()} → "
          f"{feat.index.get_level_values('datetime').max()}")
    print(f"  特征列数: {feat.shape[1]}, 列名前5: {list(feat.columns[:5])} ... 后5: {list(feat.columns[-5:])}")
    nan_ratio = feat.isna().mean().mean()
    print(f"  整体NaN比例: {nan_ratio:.4f}")
    # 找出方差为0或接近0的特征
    var = feat.var(axis=0)
    zero_var = var[var.abs() < 1e-10]
    print(f"  方差≈0的特征数: {len(zero_var)} / {feat.shape[1]}")
    if len(zero_var):
        print(f"    示例: {list(zero_var.index[:10])}")
    # label 状况
    print(f"  label NaN 比例: {lab.isna().mean().values}")
    print(f"  label 描述: mean={lab.mean().values}, std={lab.std().values}, "
          f"min={lab.min().values}, max={lab.max().values}")

# ---- 2. 重点：HF 5 个因子在 test 段的分布 ----
print("\n" + "=" * 80)
print("[2] 5 个 HF 因子分布 (test 段)")
print("=" * 80)
hf_names = ["OVN_GAP", "MOM_TAIL5", "RNG_OPEN", "INTRA_VOL5", "PV_DIV5"]
test_feat = df_test["feature"]
present = [n for n in hf_names if n in test_feat.columns]
print(f"  HF 因子在列中存在: {present}")
if present:
    print(test_feat[present].describe().T[["mean", "std", "min", "max"]])

# ---- 3. label 与每个特征的 IC（Spearman） ----
print("\n" + "=" * 80)
print("[3] train 段 单特征 IC (Spearman) Top10 / Bottom10")
print("=" * 80)
feat_tr = df_train["feature"]
lab_tr = df_train["label"].iloc[:, 0]
# 由于是单股票时序，按 datetime 维度排序计算 Spearman
mask = ~lab_tr.isna()
ic_list = []
for col in feat_tr.columns:
    s = feat_tr[col][mask]
    if s.std() < 1e-10:
        continue
    ic = s.corr(lab_tr[mask], method="spearman")
    ic_list.append((col, ic))
ic_df = pd.DataFrame(ic_list, columns=["feature", "rank_ic"]).dropna()
ic_df["abs"] = ic_df["rank_ic"].abs()
ic_df = ic_df.sort_values("abs", ascending=False)
print("Top10 |rank IC|:")
print(ic_df.head(10).to_string(index=False))
print("HF 因子 IC:")
print(ic_df[ic_df["feature"].isin(hf_names)].to_string(index=False))

# ---- 4. 训练 DoubleEnsemble 并检查预测分布 ----
print("\n" + "=" * 80)
print("[4] 训练 DEnsembleModel 并诊断预测")
print("=" * 80)
model = init_instance_by_config({
    "class": "DEnsembleModel",
    "module_path": "qlib.contrib.model.double_ensemble",
    "kwargs": {
        "base_model": "gbm",
        "loss": "mse",
        "num_models": 3,
        "enable_sr": True,
        "enable_fs": True,
        "alpha1": 1.0, "alpha2": 1.0,
        "bins_sr": 10, "bins_fs": 5, "decay": 0.5,
        "sample_ratios": [0.8, 0.7, 0.6, 0.5, 0.4],
        "sub_weights": [1, 1, 1],
        "epochs": 100,
        "colsample_bytree": 0.8,
        "learning_rate": 0.03,
        "subsample": 0.85,
        "lambda_l1": 0.5, "lambda_l2": 1.0,
        "max_depth": 4, "min_child_samples": 5,
        "num_leaves": 16, "num_threads": 4,
        "verbosity": -1,
    },
})

model.fit(dataset)
pred_test = model.predict(dataset, segment="test")
print(f"\n  预测 shape: {pred_test.shape}")
print(f"  pred describe:\n{pred_test.describe()}")
print(f"  unique values count: {pred_test.nunique()}")
print(f"  std: {pred_test.std():.10f}")
print(f"  前10行预测:\n{pred_test.head(10)}")
print(f"  后10行预测:\n{pred_test.tail(10)}")

# 训练段预测——看是否能学到训练数据本身
pred_tr = model.predict(dataset, segment="train")
print(f"\n  train 预测 std: {pred_tr.std():.10f}, unique={pred_tr.nunique()}")
print(f"  train label std:  {lab_tr.std():.10f}")

# 集成预测的 test rank IC （放在 sub_model 之前以避免 Feature Selection 列数不一致中断）
ytest = df_test["label"].iloc[:, 0].values
pred_arr = pred_test.values
rank_ic_test = pd.Series(pred_arr).corr(pd.Series(ytest), method="spearman")
pearson_ic_test = pd.Series(pred_arr).corr(pd.Series(ytest), method="pearson")
print(f"\n  [4c] 集成模型在 test 段: rank_IC={rank_ic_test:.4f}, pearson_IC={pearson_ic_test:.4f}")
print(f"  [4d] 集成模型在 train 段: rank_IC="
      f"{pd.Series(pred_tr.values).corr(pd.Series(lab_tr.values), method='spearman'):.4f}")
pred_va = model.predict(dataset, segment="valid")
yva = df_valid["label"].iloc[:, 0].values
print(f"  [4e] 集成模型在 valid 段: rank_IC="
      f"{pd.Series(pred_va.values).corr(pd.Series(yva), method='spearman'):.4f}")

# 对子模型分别看预测
print("\n  [4b] 各子模型在 test 段独立预测的 std (注：DoubleEnsemble 内部做了 feature selection):")
xtest = df_test["feature"].values
ens = model.ensemble
if isinstance(ens, dict):
    iter_subs = list(ens.values())
else:
    iter_subs = list(ens)
for i, sub in enumerate(iter_subs):
    try:
        p = sub.predict(xtest)
        rank_ic = pd.Series(p).corr(pd.Series(ytest), method="spearman")
        print(f"    sub_model_{i}: std={np.std(p):.6f}, unique={len(np.unique(p))}, test_rank_IC={rank_ic:.4f}")
    except Exception as e:
        print(f"    sub_model_{i}: 预测失败（feature selection 列数不一致），跳过：{e}")

