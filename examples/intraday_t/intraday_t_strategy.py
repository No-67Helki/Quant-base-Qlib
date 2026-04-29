# Copyright (c) Qlib_helki intraday-T example.
# Licensed under the MIT License.
"""
IntradayTStrategy
-----------------
单股票日内做T策略（日级别决策，由内层 TWAP 在分钟级拆单执行）。

核心逻辑：
    score = signal[今日]
    if  score > buy_thresh  : 当日买入 trade_amount
    elif score < sell_thresh: 当日卖出 trade_amount（允许首笔即卖：消耗已有持仓）
    else                    : 不交易

设计要点：
    1. 不做裸卖空。卖出量被夹紧到当前持有量，避免Position._sell_stock报错。
    2. 第一笔可为卖出：依赖 backtest.account 在初始化时已注入指定股票持仓。
    3. 仅生成日级 TradeDecision；具体撮合由内层 NestedExecutor + TWAPStrategy 完成。
"""
from __future__ import annotations

from typing import Dict, List, Text, Tuple, Union

import numpy as np
import pandas as pd

from qlib.backtest.decision import Order, OrderDir, TradeDecisionWO
from qlib.contrib.strategy.signal_strategy import BaseSignalStrategy
from qlib.log import get_module_logger


class IntradayTStrategy(BaseSignalStrategy):
    """单标的日内做T策略

    Parameters
    ----------
    signal : 模型预测信号（典型为 (model, dataset)）
    stock_id : str
        交易标的代码（与初始持仓 / 数据 instrument 一致）
    buy_thresh : float
        预测分值高于该阈值则触发买入
    sell_thresh : float
        预测分值低于该阈值则触发卖出（通常为负数）
    trade_unit : int
        最小交易单位（A股=100）
    position_pct : float
        单笔下单使用账户总价值的比例 (0,1]
    allow_first_sell : bool
        是否允许首笔为卖出（仅作为显式开关，撮合层面依赖初始持仓）
    """

    def __init__(
        self,
        *,
        signal=None,
        stock_id: str = "",
        buy_thresh: float = 0.0,
        sell_thresh: float = 0.0,
        trade_unit: int = 100,
        position_pct: float = 0.3,
        allow_first_sell: bool = True,
        **kwargs,
    ):
        super().__init__(signal=signal, **kwargs)  # type: ignore[arg-type]
        if not stock_id:
            raise ValueError("`stock_id` 必须显式指定")
        if sell_thresh > buy_thresh:
            raise ValueError("sell_thresh 必须 <= buy_thresh")

        self.stock_id = stock_id
        self.buy_thresh = float(buy_thresh)
        self.sell_thresh = float(sell_thresh)
        self.trade_unit = int(trade_unit)
        self.position_pct = float(position_pct)
        self.allow_first_sell = bool(allow_first_sell)
        self.logger = get_module_logger("IntradayTStrategy")

    # --------------------------------------------------------------- #
    # 工具方法
    # --------------------------------------------------------------- #
    def _round_unit(self, amount: float) -> float:
        """按trade_unit向下取整"""
        if amount <= 0:
            return 0.0
        return float(int(amount // self.trade_unit) * self.trade_unit)

    def _get_score(self, pred_start_time, pred_end_time) -> float:
        """从信号中取出当前交易日的预测分值"""
        try:
            pred = self.signal.get_signal(start_time=pred_start_time, end_time=pred_end_time)
        except Exception as e:
            self.logger.warning(f"signal.get_signal 失败: {e}")
            return np.nan
        if pred is None:
            return np.nan
        if isinstance(pred, pd.Series):
            # MultiIndex 或单层 Index 都尝试
            if self.stock_id in pred.index.get_level_values(-1):
                try:
                    return float(pred.xs(self.stock_id, level=-1).iloc[0])
                except Exception:
                    return float(pred.iloc[0])
            if len(pred) >= 1:
                return float(pred.iloc[0])
        if isinstance(pred, pd.DataFrame):
            if self.stock_id in pred.index.get_level_values(-1):
                try:
                    return float(pred.xs(self.stock_id, level=-1).to_numpy().flat[0])  # type: ignore[arg-type]
                except Exception:
                    return float(pred.to_numpy().flat[0])  # type: ignore[arg-type]
            if not pred.empty:
                return float(pred.to_numpy().flat[0])  # type: ignore[arg-type]
        return np.nan

    # --------------------------------------------------------------- #
    # 核心：每个外层（日级）trade_step 触发一次
    # --------------------------------------------------------------- #
    def generate_trade_decision(self, execute_result=None):
        # 当前日级别交易步
        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
        # 信号对齐：使用上一交易日的预测分值
        pred_start_time, pred_end_time = self.trade_calendar.get_step_time(trade_step, shift=1)

        score = self._get_score(pred_start_time, pred_end_time)

        if not np.isfinite(score):
            return TradeDecisionWO(order_list=[], strategy=self)

        # ------- 资金与持仓状态 -------
        current_temp = self.trade_position
        try:
            total_value = current_temp.calculate_value()
        except Exception:
            total_value = current_temp.get_cash() if current_temp else 0.0
        cash = current_temp.get_cash() if current_temp else 0.0
        try:
            held_amount = current_temp.get_stock_amount(self.stock_id)
        except Exception:
            held_amount = 0.0

        # ------- 决策 -------
        order_list: List[Order] = []
        if score > self.buy_thresh:
            # ---- BUY ----
            # 用账户价值×position_pct 估算下单金额，用最近close估算单价
            try:
                price_ref = self.trade_exchange.get_close(
                    stock_id=self.stock_id,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                )
            except Exception:
                price_ref = None
            price_ref_f: float = float(price_ref) if price_ref is not None else float("nan")  # type: ignore[arg-type]
            if not np.isfinite(price_ref_f) or price_ref_f <= 0:
                self.logger.warning(f"{trade_start_time} 无法取得参考价，跳过")
                return TradeDecisionWO(order_list=[], strategy=self)
            budget = min(total_value * self.position_pct, cash)
            target_amount = self._round_unit(budget / price_ref_f)
            if target_amount > 0:
                order_list.append(
                    Order(
                        stock_id=self.stock_id,
                        amount=target_amount,
                        start_time=trade_start_time,
                        end_time=trade_end_time,
                        direction=OrderDir.BUY,
                    )
                )

        elif score < self.sell_thresh:
            # ---- SELL（可作为首笔，消耗已有持仓） ----
            if held_amount <= 0:
                # 没有持仓，跳过（不开做空）
                return TradeDecisionWO(order_list=[], strategy=self)
            # 每笔卖出 position_pct 比例的当前持仓
            sell_amount = self._round_unit(held_amount * self.position_pct)
            sell_amount = min(sell_amount, held_amount)
            if sell_amount > 0:
                order_list.append(
                    Order(
                        stock_id=self.stock_id,
                        amount=sell_amount,
                        start_time=trade_start_time,
                        end_time=trade_end_time,
                        direction=OrderDir.SELL,
                    )
                )

        return TradeDecisionWO(order_list=order_list, strategy=self)
