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
        **kwargs,
    ):
        super().__init__(
            conf_path=conf_path,
            horizon=horizon,
            step=step,
            exp_name=exp_name,
            **kwargs,
        )


def run(
    conf_path: str = str(DIRNAME / "config_doubleensemble.yaml"),
    step: int = 20,
    horizon: int = 1,
    exp_name: str = "intraday_t_rolling",
):
    # 提前 init qlib（Rolling 内部使用 qlib.config 中的 provider_uri）
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
