# Copyright (c) Qlib_helki intraday-T example.
# Licensed under the MIT License.
"""
IntradayTStrategy (production-hardened)
---------------------------------------
Single-stock intraday T+0 strategy with:

    - Fixed signal format extraction (MultiIndex (instrument, datetime) Series)
    - Separate buy_pct / sell_pct sizing
    - Adaptive threshold calibration (opt-in)
    - Hard stop-loss (opt-in)
    - Prediction quality guard (NaN, zero-variance, out-of-range)
    - Risk manager integration (circuit breakers, daily limits, duplicate prevention)

Integration:
    Daily decision: IntradayTStrategy generates BUY/SELL TradeDecisionWO
    Intraday execution: NestedExecutor → SimulatorExecutor(1min) → TWAPStrategy
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from qlib.backtest.decision import Order, OrderDir, TradeDecisionWO
from qlib.contrib.strategy.signal_strategy import BaseSignalStrategy
from qlib.log import get_module_logger


class IntradayTStrategy(BaseSignalStrategy):
    """Single-stock intraday T+0 strategy.

    Parameters
    ----------
    signal : Signal
        Model prediction signal. Expected format: MultiIndex (instrument, datetime)
        Series or DataFrame with a single column.
    stock_id : str
        Trading target instrument code (e.g. "SZ301536").
    buy_thresh : float
        Static buy threshold when adaptive is disabled.
    sell_thresh : float
        Static sell threshold when adaptive is disabled.
    trade_unit : int
        Minimum trade unit (100 for A-shares).
    buy_pct : float
        Fraction of account value to allocate per buy order (0..1).
    sell_pct : float
        Fraction of current holdings to sell per sell order (0..1).
    allow_first_sell : bool
        Allow selling before first buy (requires initial position in account).

    -- Risk controls (opt-in) --
    risk_limits : RiskLimits or dict, optional
        RiskLimits instance or dict to construct one.

    -- Stop-loss (opt-in) --
    enable_stop_loss : bool
    stop_loss_pct : float
        Fractional loss from entry price that triggers force-sell.
        e.g. 0.03 = sell when down 3% from entry.

    -- Adaptive thresholds (opt-in) --
    enable_adaptive_thresh : bool
    adaptive_thresh_lookback : int
        Number of steps between recalibration.
    adaptive_thresh_percentile : float
        Fraction of distribution used for threshold (e.g. 0.20 → top/bottom 20%).

    -- Signal guard (opt-in) --
    enable_signal_guard : bool
        Reject NaN, inf, zero-variance, and out-of-range signals.
    """

    def __init__(
        self,
        *,
        signal=None,
        stock_id: str = "",
        buy_thresh: float = 0.0,
        sell_thresh: float = 0.0,
        trade_unit: int = 100,
        position_pct: Optional[float] = None,       # deprecated
        buy_pct: float = 0.30,
        sell_pct: float = 0.50,
        allow_first_sell: bool = True,
        # risk controls
        risk_limits=None,
        # stop-loss
        enable_stop_loss: bool = False,
        stop_loss_pct: float = 0.03,
        # adaptive thresholds
        enable_adaptive_thresh: bool = False,
        adaptive_thresh_lookback: int = 60,
        adaptive_thresh_percentile: float = 0.20,
        # signal guard
        enable_signal_guard: bool = True,
        **kwargs,
    ):
        super().__init__(signal=signal, **kwargs)

        if not stock_id:
            raise ValueError("`stock_id` must be explicitly specified")
        if sell_thresh > buy_thresh:
            raise ValueError("sell_thresh must be <= buy_thresh")

        self.stock_id = stock_id
        self.buy_thresh = float(buy_thresh)
        self.sell_thresh = float(sell_thresh)
        self.trade_unit = int(trade_unit)
        self.allow_first_sell = bool(allow_first_sell)

        # buy_pct / sell_pct
        if position_pct is not None:
            warnings.warn(
                "`position_pct` is deprecated; use `buy_pct` and `sell_pct`.",
                DeprecationWarning,
            )
            self.buy_pct = float(position_pct)
            self.sell_pct = float(position_pct)
        else:
            self.buy_pct = float(buy_pct)
            self.sell_pct = float(sell_pct)
        if not 0 < self.buy_pct <= 1:
            raise ValueError(f"buy_pct must be in (0, 1], got {self.buy_pct}")
        if not 0 < self.sell_pct <= 1:
            raise ValueError(f"sell_pct must be in (0, 1], got {self.sell_pct}")

        # risk manager
        self.risk_manager = None
        if risk_limits is not None:
            from .risk_manager import RiskLimits, RiskManager
            if isinstance(risk_limits, dict):
                risk_limits = RiskLimits(**risk_limits)
            self.risk_manager = RiskManager(risk_limits, initial_capital=0.0)

        # stop-loss
        self.enable_stop_loss = bool(enable_stop_loss)
        self.stop_loss_pct = float(stop_loss_pct)
        self._entry_prices: Dict[str, float] = {}
        self._last_held_amount: float = 0.0

        # adaptive thresholds
        self.enable_adaptive_thresh = bool(enable_adaptive_thresh)
        self.adaptive_thresh_lookback = int(adaptive_thresh_lookback)
        self.adaptive_thresh_percentile = float(adaptive_thresh_percentile)
        self._pred_buffer: List[float] = []
        self._adaptive_buy_thresh = self.buy_thresh
        self._adaptive_sell_thresh = self.sell_thresh
        self._adaptive_recal_counter = 0

        # signal guard
        self.enable_signal_guard = bool(enable_signal_guard)
        self._signal_history: List[float] = []
        self._signal_max_history = 30

        self.logger = get_module_logger("IntradayTStrategy")

    # ==================================================================
    # Utilities
    # ==================================================================

    def _round_unit(self, amount: float) -> float:
        if amount <= 0:
            return 0.0
        return float(int(amount // self.trade_unit) * self.trade_unit)

    # ==================================================================
    # Signal extraction (standardized)
    # ==================================================================

    def _get_score(self, pred_start_time, pred_end_time) -> float:
        """Extract prediction for `self.stock_id` from the signal.

        Expects signal in the standardized Qlib format:
        MultiIndex (instrument, datetime) Series or single-column DataFrame.
        """
        try:
            pred = self.signal.get_signal(
                start_time=pred_start_time, end_time=pred_end_time
            )
        except Exception as e:
            self.logger.warning(f"signal.get_signal failed: {e}")
            return np.nan

        if pred is None:
            return np.nan

        if isinstance(pred, pd.DataFrame):
            pred = pred.iloc[:, 0]

        if isinstance(pred, pd.Series):
            if isinstance(pred.index, pd.MultiIndex):
                if self.stock_id in pred.index.get_level_values(0):
                    matching = pred.xs(self.stock_id, level=0)
                    if len(matching) > 0:
                        return float(matching.iloc[-1])
                return np.nan
            else:
                if len(pred) >= 1:
                    return float(pred.iloc[-1])

        return np.nan

    # ==================================================================
    # Adaptive thresholds
    # ==================================================================

    def _update_adaptive_thresholds(self, current_score: float) -> None:
        if not np.isfinite(current_score):
            return
        self._pred_buffer.append(current_score)
        max_buf = max(self.adaptive_thresh_lookback * 2, 60)
        if len(self._pred_buffer) > max_buf:
            self._pred_buffer.pop(0)

        self._adaptive_recal_counter += 1
        if self._adaptive_recal_counter < self.adaptive_thresh_lookback:
            return
        self._adaptive_recal_counter = 0

        if len(self._pred_buffer) < self.adaptive_thresh_lookback:
            return

        arr = np.array(self._pred_buffer)
        upper = np.percentile(arr, 100 * (1 - self.adaptive_thresh_percentile))
        lower = np.percentile(arr, 100 * self.adaptive_thresh_percentile)
        if upper - lower > 1e-10:
            self._adaptive_buy_thresh = upper
            self._adaptive_sell_thresh = lower
            self.logger.debug(
                f"Adaptive thresh updated: buy={upper:.6f}, sell={lower:.6f}"
            )

    # ==================================================================
    # Signal quality guard
    # ==================================================================

    def _validate_signal(self, score: float) -> bool:
        if not np.isfinite(score):
            self.logger.warning("Signal guard: score is NaN or inf, skipping.")
            return False
        if abs(score) > 0.25:
            self.logger.warning(
                f"Signal guard: score {score:.4f} out of range [-0.25, 0.25]."
            )
            return False
        self._signal_history.append(score)
        if len(self._signal_history) > self._signal_max_history:
            self._signal_history.pop(0)
        if len(self._signal_history) >= 10:
            if np.std(self._signal_history) < 1e-10:
                self.logger.warning(
                    "Signal guard: near-zero variance in recent signals."
                )
                return False
        return True

    # ==================================================================
    # Stop-loss entry price tracking
    # ==================================================================

    def _track_stop_loss_entry(self, trade_start_time, trade_end_time) -> None:
        """Update entry price when position changes."""
        pos = self.trade_position
        if pos is None:
            return
        current_amount = pos.get_stock_amount(self.stock_id)
        prev_amount = self._last_held_amount

        if current_amount > prev_amount + 0.5:
            try:
                entry_price = self.trade_exchange.get_close(
                    stock_id=self.stock_id,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                )
            except Exception:
                entry_price = None
            if entry_price is not None and np.isfinite(entry_price) and entry_price > 0:
                old_price = self._entry_prices.get(self.stock_id, entry_price)
                new_bought = current_amount - prev_amount
                self._entry_prices[self.stock_id] = (
                    prev_amount * old_price + new_bought * entry_price
                ) / current_amount
        elif current_amount < 0.5:
            self._entry_prices.pop(self.stock_id, None)

        self._last_held_amount = current_amount

    def _check_stop_loss(
        self, held_amount: float, trade_start_time, trade_end_time
    ) -> Optional[Order]:
        if not self.enable_stop_loss or held_amount <= 0:
            return None
        entry_price = self._entry_prices.get(self.stock_id)
        if entry_price is None or entry_price <= 0:
            return None
        try:
            current_price = self.trade_exchange.get_close(
                stock_id=self.stock_id,
                start_time=trade_start_time,
                end_time=trade_end_time,
            )
        except Exception:
            return None
        if current_price is None or not np.isfinite(current_price) or current_price <= 0:
            return None

        pnl_pct = (current_price - entry_price) / entry_price
        if pnl_pct <= -self.stop_loss_pct:
            self.logger.warning(
                f"Stop-loss: {self.stock_id} entry={entry_price:.2f} "
                f"current={current_price:.2f} loss={-pnl_pct:.2%}"
            )
            return Order(
                stock_id=self.stock_id,
                amount=held_amount,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=OrderDir.SELL,
            )
        return None

    # ==================================================================
    # Core decision logic
    # ==================================================================

    def generate_trade_decision(self, execute_result=None):
        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(
            trade_step
        )
        pred_start_time, pred_end_time = self.trade_calendar.get_step_time(
            trade_step, shift=1
        )

        # -- position state --
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

        order_list: List[Order] = []

        # -- risk manager checks --
        if self.risk_manager is not None:
            current_date = str(trade_start_time.date())
            self.risk_manager.on_new_day(current_date)

            self.risk_manager.update_equity_curve(total_value)
            broken, reason = self.risk_manager.check_circuit_breakers(total_value)
            if broken:
                self.logger.warning(f"Circuit breaker active: {reason}")
                return TradeDecisionWO(order_list=[], strategy=self)

            if not self.risk_manager.check_trade_frequency():
                self.logger.debug("Trade frequency limit reached today.")
                return TradeDecisionWO(order_list=[], strategy=self)

        # -- stop-loss check (before signal logic) --
        if self.enable_stop_loss:
            stop_order = self._check_stop_loss(
                held_amount, trade_start_time, trade_end_time
            )
            if stop_order is not None:
                if self.risk_manager is not None:
                    if not self.risk_manager.check_duplicate_order(
                        self.stock_id, OrderDir.SELL, str(trade_start_time.date()),
                        stop_order.amount,
                    ):
                        self.logger.debug("Duplicate stop-loss order blocked.")
                    else:
                        self.risk_manager.record_trade()
                        order_list.append(stop_order)
                else:
                    order_list.append(stop_order)
                return TradeDecisionWO(order_list=order_list, strategy=self)

        # -- track entry price for stop-loss --
        self._track_stop_loss_entry(trade_start_time, trade_end_time)

        # -- get signal --
        score = self._get_score(pred_start_time, pred_end_time)

        # -- signal quality guard --
        if self.enable_signal_guard and not self._validate_signal(score):
            return TradeDecisionWO(order_list=[], strategy=self)

        if not np.isfinite(score):
            return TradeDecisionWO(order_list=[], strategy=self)

        # -- determine thresholds --
        if self.enable_adaptive_thresh:
            buy_thresh = self._adaptive_buy_thresh
            sell_thresh = self._adaptive_sell_thresh
            self._update_adaptive_thresholds(score)
        else:
            buy_thresh = self.buy_thresh
            sell_thresh = self.sell_thresh

        # -- BUY --
        if score > buy_thresh:
            try:
                price_ref = self.trade_exchange.get_close(
                    stock_id=self.stock_id,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                )
            except Exception:
                price_ref = None
            price_ref_f = float(price_ref) if price_ref is not None else float("nan")
            if not np.isfinite(price_ref_f) or price_ref_f <= 0:
                self.logger.warning(
                    f"{trade_start_time}: no valid reference price, skipping BUY."
                )
                return TradeDecisionWO(order_list=[], strategy=self)

            budget = min(total_value * self.buy_pct, cash)
            target_amount = self._round_unit(budget / price_ref_f)
            if target_amount <= 0:
                return TradeDecisionWO(order_list=[], strategy=self)

            # position concentration check
            if self.risk_manager is not None:
                projected_val = (held_amount + target_amount) * price_ref_f
                if not self.risk_manager.check_position_concentration(
                    projected_val, total_value
                ):
                    self.logger.debug("Position concentration limit reached.")
                    return TradeDecisionWO(order_list=[], strategy=self)

            # duplicate check
            if self.risk_manager is not None:
                if not self.risk_manager.check_duplicate_order(
                    self.stock_id, OrderDir.BUY, str(trade_start_time.date()),
                    target_amount,
                ):
                    self.logger.debug("Duplicate buy order blocked.")
                    return TradeDecisionWO(order_list=[], strategy=self)
                self.risk_manager.record_trade()

            order_list.append(
                Order(
                    stock_id=self.stock_id,
                    amount=float(target_amount),
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                    direction=OrderDir.BUY,
                )
            )

        # -- SELL --
        elif score < sell_thresh:
            if held_amount <= 0:
                return TradeDecisionWO(order_list=[], strategy=self)

            sell_amount = self._round_unit(held_amount * self.sell_pct)
            sell_amount = min(sell_amount, held_amount)
            if sell_amount <= 0:
                return TradeDecisionWO(order_list=[], strategy=self)

            if self.risk_manager is not None:
                if not self.risk_manager.check_duplicate_order(
                    self.stock_id, OrderDir.SELL, str(trade_start_time.date()),
                    sell_amount,
                ):
                    self.logger.debug("Duplicate sell order blocked.")
                    return TradeDecisionWO(order_list=[], strategy=self)
                self.risk_manager.record_trade()

            order_list.append(
                Order(
                    stock_id=self.stock_id,
                    amount=float(sell_amount),
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                    direction=OrderDir.SELL,
                )
            )

        return TradeDecisionWO(order_list=order_list, strategy=self)

    # ==================================================================
    # Hooks
    # ==================================================================

    def post_exe_step(self, execute_result: Optional[list]) -> None:
        """Track P&L for risk manager after each execution step."""
        super().post_exe_step(execute_result)
        if self.risk_manager is not None and execute_result is not None:
            for order, trade_val, trade_cost, trade_price in execute_result:
                if trade_val > 1e-5 and np.isfinite(trade_price):
                    _pnl = (
                        -trade_cost
                        if order.direction == OrderDir.BUY
                        else trade_val - trade_cost
                    )
                    self.risk_manager.update_daily_pnl(float(_pnl))
        if self.enable_stop_loss or self.risk_manager is not None:
            try:
                self._track_stop_loss_entry(
                    *self.trade_calendar.get_step_time(self.trade_calendar.get_trade_step())
                )
            except (IndexError, KeyError):
                pass
