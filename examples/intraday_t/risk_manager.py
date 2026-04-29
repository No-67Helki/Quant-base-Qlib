# =============================================================================
# Risk Manager — pre-trade checks, circuit breakers, daily limits
# =============================================================================
"""
Risk management layer for intraday T+0 strategy.

Provides:
    - Daily loss limit (stop trading when daily P&L exceeds threshold)
    - Max cumulative drawdown circuit breaker
    - Position concentration limit
    - Trade frequency limit (max trades per day)
    - Duplicate order prevention (idempotency)
    - State export/import for checkpointing

Usage:
    limits = RiskLimits(daily_loss_limit_pct=0.03, max_drawdown_pct=0.15)
    rm = RiskManager(limits, initial_capital=500000)
    rm.on_new_day("2026-04-28")
    ok, reason = rm.check_circuit_breakers(current_equity=490000)
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, Optional, Set, Tuple


@dataclass
class RiskLimits:
    """Risk control thresholds. Set to None to disable a check."""

    daily_loss_limit_pct: Optional[float] = None
    max_drawdown_pct: Optional[float] = None
    max_position_conc_pct: Optional[float] = None
    max_trades_per_day: Optional[int] = None
    max_order_amount: Optional[int] = None


class RiskManager:
    """Pre-trade risk gate. Called before any order is issued."""

    def __init__(self, limits: RiskLimits, initial_capital: float = 0.0):
        self.limits = limits
        self.initial_capital = initial_capital
        self._peak_equity = initial_capital
        self._daily_pnl = 0.0
        self._daily_trade_count = 0
        self._last_date: Optional[str] = None
        self._circuit_broken = False
        self._circuit_reason = ""
        self._submitted_order_keys: Set[str] = set()

    # -- daily reset ----------------------------------------------------------
    def on_new_day(self, current_date: str) -> None:
        if self._last_date != current_date:
            self._daily_pnl = 0.0
            self._daily_trade_count = 0
            self._submitted_order_keys.clear()
            self._last_date = current_date

    # -- circuit breakers -----------------------------------------------------
    def update_daily_pnl(self, pnl_change: float) -> None:
        self._daily_pnl += pnl_change

    def update_equity_curve(self, current_equity: float) -> None:
        if current_equity > self._peak_equity:
            self._peak_equity = current_equity

    def check_circuit_breakers(self, current_equity: float) -> Tuple[bool, str]:
        """Returns (is_broken, reason)."""
        if self._circuit_broken:
            return True, self._circuit_reason

        if self.limits.daily_loss_limit_pct is not None and self.initial_capital > 0:
            daily_loss_ratio = -self._daily_pnl / self.initial_capital
            if daily_loss_ratio >= self.limits.daily_loss_limit_pct:
                self._circuit_broken = True
                self._circuit_reason = f"daily_loss_limit({self.limits.daily_loss_limit_pct:.1%})"
                return True, self._circuit_reason

        if self.limits.max_drawdown_pct is not None and self._peak_equity > 0:
            drawdown = (self._peak_equity - current_equity) / self._peak_equity
            if drawdown >= self.limits.max_drawdown_pct:
                self._circuit_broken = True
                self._circuit_reason = f"max_drawdown({self.limits.max_drawdown_pct:.1%})"
                return True, self._circuit_reason

        return False, ""

    def reset_circuit(self) -> None:
        """Manually reset the circuit breaker (e.g. after manual review)."""
        self._circuit_broken = False
        self._circuit_reason = ""
        self._peak_equity = self.initial_capital
        self._daily_pnl = 0.0

    # -- pre-trade checks -----------------------------------------------------
    def check_position_concentration(self, stock_value: float, total_value: float) -> bool:
        if self.limits.max_position_conc_pct is None:
            return True
        if total_value <= 0:
            return True
        return stock_value / total_value <= self.limits.max_position_conc_pct

    def check_trade_frequency(self) -> bool:
        if self.limits.max_trades_per_day is None:
            return True
        return self._daily_trade_count < self.limits.max_trades_per_day

    def check_duplicate_order(
        self,
        stock_id: str,
        direction: int,
        date: str,
        amount: float,
    ) -> bool:
        """Returns True if this is NOT a duplicate (can proceed)."""
        key = hashlib.md5(
            f"{stock_id}:{direction}:{date}:{amount:.0f}".encode()
        ).hexdigest()
        if key in self._submitted_order_keys:
            return False
        self._submitted_order_keys.add(key)
        return True

    def record_trade(self) -> None:
        self._daily_trade_count += 1

    # -- state persistence ----------------------------------------------------
    def get_state(self) -> dict:
        return {
            "peak_equity": self._peak_equity,
            "daily_pnl": self._daily_pnl,
            "daily_trade_count": self._daily_trade_count,
            "last_date": self._last_date,
            "circuit_broken": self._circuit_broken,
            "circuit_reason": self._circuit_reason,
        }

    def restore_state(self, state: dict) -> None:
        for k, v in state.items():
            attr = f"_{k}"
            if hasattr(self, attr):
                setattr(self, attr, v)
