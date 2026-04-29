# Copyright (c) Qlib_helki intraday-T example.
# Licensed under the MIT License.
"""
单股票日内做T策略 - 滚动训练驱动脚本

每 step 个交易日重训一次模型，将各窗口预测拼接后用嵌套TWAP回测。

用法：
    python rolling_train.py run                          # 默认 step=20, horizon=1
    python rolling_train.py run --step 10 --horizon 1    # 自定义
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import fire

import qlib
from qlib.contrib.rolling.base import Rolling
from qlib.utils import init_instance_by_config

DIRNAME = Path(__file__).absolute().resolve().parent
if str(DIRNAME) not in sys.path:
    sys.path.insert(0, str(DIRNAME))


class IntradayTRolling(Rolling):
    """单股票滚动训练；继承自 Qlib 的通用 Rolling 模块"""

    DEFAULT_CONF = DIRNAME / "config_doubleensemble.yaml"

    def __init__(
        self,
        conf_path: str = str(DEFAULT_CONF),
        horizon: int = 1,
        step: int = 20,
        exp_name: str = "intraday_t_rolling",
        export_models: bool = False,
        **kwargs,
    ):
        super().__init__(
            conf_path=conf_path,
            horizon=horizon,
            step=step,
            exp_name=exp_name,
            **kwargs,
        )
        self.export_models = export_models

    def run(self):
        super().run()
        if self.export_models:
            self._export_latest_model()

    def _export_latest_model(self):
        """Export the most recent rolling model for live serving."""
        try:
            from model_serving import ModelExporter
            import mlflow
            from qlib.workflow import R

            exp = R.get_exp(experiment_name=self.exp_name)
            rec = exp.list_recorders()[-1] if exp.list_recorders() else None
            if rec is None:
                print("[export] No recorders found in rolling experiment.")
                return
            model = rec.load_object("params.pkl")
            if model is None:
                print("[export] No params.pkl found in latest recorder.")
                return

            cfg = self._raw_conf()
            handler_cfg = cfg["task"]["dataset"]["kwargs"]["handler"]
            handler = init_instance_by_config(handler_cfg)
            _, feature_names = handler.get_feature_config()

            export_path = Path("models") / f"{self.exp_name}_latest.pkl"
            exporter = ModelExporter(model, list(feature_names))
            exporter.export(str(export_path))
            print(f"[export] Rolling model exported to {export_path}")
        except Exception as e:
            print(f"[export] Rolling model export failed (non-fatal): {e}")


def run(
    conf_path: str = str(DIRNAME / "config_doubleensemble.yaml"),
    step: int = 20,
    horizon: int = 1,
    exp_name: str = "intraday_t_rolling",
):
    from ruamel.yaml import YAML

    with open(conf_path, "r", encoding="utf-8") as f:
        cfg = YAML(typ="safe", pure=True).load(f)
    qlib.init(**cfg["qlib_init"])

    rolling = IntradayTRolling(
        conf_path=conf_path,
        horizon=horizon,
        step=step,
        exp_name=exp_name,
    )
    rolling.run()
    print(f"[OK] 滚动训练完成。实验名: {exp_name}")


if __name__ == "__main__":
    fire.Fire({"run": run})
