# Copyright (c) Qlib_helki intraday-T example.
# Licensed under the MIT License.
"""
单股票日内做T策略 - 单次训练 + 嵌套TWAP回测入口

用法：
    python workflow.py run_once
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import fire
from ruamel.yaml import YAML

import qlib
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.workflow.record_temp import PortAnaRecord, SigAnaRecord, SignalRecord
from model_serving import export_from_model

DIRNAME = Path(__file__).absolute().resolve().parent
# 把当前目录加入 sys.path 以便 IntradayTStrategy 的 module_path 能解析
if str(DIRNAME) not in sys.path:
    sys.path.insert(0, str(DIRNAME))


def _load_yaml(path: Path) -> dict:
    yaml = YAML(typ="safe", pure=True)
    with path.open("r", encoding="utf-8") as f:
        return yaml.load(f)


def run_once(
    config: str = str(DIRNAME / "config_doubleensemble.yaml"),
    exp_name: str = "intraday_t_doubleensemble",
):
    """单次训练 + 回测"""
    cfg = _load_yaml(Path(config))

    # ---- 初始化 Qlib ----
    qlib_init = cfg["qlib_init"]
    qlib.init(**qlib_init)

    task = cfg["task"]

    # ---- 训练模型 ----
    with R.start(experiment_name=exp_name):
        model = init_instance_by_config(task["model"])
        dataset = init_instance_by_config(task["dataset"])
        R.log_params(**flatten_dict(task))
        model.fit(dataset)
        R.save_objects(**{"params.pkl": model})

        recorder = R.get_recorder()
        # ---- 信号 / IC ----
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()
        sar = SigAnaRecord(recorder)
        sar.generate()

        # ---- 嵌套TWAP执行回测 ----
        par = PortAnaRecord(recorder, cfg["port_analysis_config"], "day")
        par.generate()

        # ---- 导出模型供实盘推理 ----
        export_path = Path("models") / f"{exp_name}.pkl"
        try:
            export_from_model(model, dataset, str(export_path))
            print(f"[export] 模型已导出至 {export_path}")
        except Exception as e:
            print(f"[export] 模型导出失败 (非致命): {e}")

    print(f"[OK] 实验完成。可用 `mlflow ui` 查看 {exp_name} 结果。")


if __name__ == "__main__":
    fire.Fire({"run_once": run_once})
