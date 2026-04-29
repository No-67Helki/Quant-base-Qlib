"""
JSON-based strategy state persistence for checkpointing and crash recovery.

Usage:
    sm = StateManager("state/checkpoint.json")
    sm.snapshot(strategy, risk_manager)
    ...
    sm.restore(strategy, risk_manager)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


@dataclass
class StrategyState:
    timestamp: str = ""
    positions: Dict[str, float] = None         # stock_id → amount
    cash: float = 0.0
    total_value: float = 0.0
    entry_prices: Dict[str, float] = None      # stop-loss entry prices
    adaptive_buy_thresh: float = 0.0
    adaptive_sell_thresh: float = 0.0
    pred_buffer: list = None
    signal_history: list = None
    risk_state: dict = None

    def __post_init__(self):
        if self.positions is None:
            self.positions = {}
        if self.entry_prices is None:
            self.entry_prices = {}
        if self.pred_buffer is None:
            self.pred_buffer = []
        if self.signal_history is None:
            self.signal_history = []
        if self.risk_state is None:
            self.risk_state = {}


class StateManager:
    def __init__(self, path: str, auto_save: bool = False):
        self.path = Path(path)
        self.auto_save = auto_save

    def snapshot(self, strategy, risk_manager=None) -> StrategyState:
        state = StrategyState(timestamp=datetime.now().isoformat())

        pos = strategy.trade_position
        if pos is not None:
            try:
                state.cash = pos.get_cash()
                state.total_value = pos.calculate_value()
                state.positions = {str(k): float(v) for k, v in pos.get_stock_amount_dict().items()}
            except Exception:
                pass

        state.entry_prices = dict(getattr(strategy, "_entry_prices", {}))
        state.adaptive_buy_thresh = float(getattr(strategy, "_adaptive_buy_thresh", 0))
        state.adaptive_sell_thresh = float(getattr(strategy, "_adaptive_sell_thresh", 0))
        state.pred_buffer = list(getattr(strategy, "_pred_buffer", []))
        state.signal_history = list(getattr(strategy, "_signal_history", []))

        if risk_manager is not None:
            state.risk_state = risk_manager.get_state()

        if self.auto_save:
            self.save(state)
        return state

    def restore(self, strategy, risk_manager=None) -> bool:
        if not self.path.exists():
            return False
        state = self.load()
        if state is None:
            return False

        strategy._entry_prices = state.entry_prices
        strategy._adaptive_buy_thresh = state.adaptive_buy_thresh
        strategy._adaptive_sell_thresh = state.adaptive_sell_thresh
        strategy._pred_buffer = state.pred_buffer
        strategy._signal_history = state.signal_history

        if risk_manager is not None and state.risk_state:
            risk_manager.restore_state(state.risk_state)
        return True

    def save(self, state: StrategyState) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(asdict(state), f, indent=2, ensure_ascii=False)

    def load(self) -> Optional[StrategyState]:
        if not self.path.exists():
            return None
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return StrategyState(**data)
