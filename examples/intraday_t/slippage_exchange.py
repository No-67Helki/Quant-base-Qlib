"""
Exchange subclass adding bid-ask spread noise for conservative backtest estimates.

Usage (in YAML config):
    exchange_kwargs:
        class: SlippageExchange
        module_path: slippage_exchange
        kwargs:
            slippage_std: 0.0005
            freq: 1min
            ...
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from qlib.backtest.exchange import Exchange


class SlippageExchange(Exchange):
    """Exchange with configurable Gaussian slippage noise per fill.

    Adds N(0, slippage_std) × price as slippage cost per trade, simulating
    bid-ask bounce and adverse selection in a conservative manner.
    """

    def __init__(
        self,
        slippage_std: float = 0.0005,
        slippage_seed: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.slippage_std = slippage_std
        self._rng = np.random.RandomState(slippage_seed)

    def _apply_slippage(self, deal_price: float, direction: int) -> float:
        """Apply symmetric slippage: BUY pays more, SELL receives less."""
        if self.slippage_std <= 0:
            return deal_price
        noise = self._rng.normal(0, self.slippage_std)
        slippage_factor = 1 + abs(noise)
        if direction == 0:    # BUY
            return deal_price * slippage_factor
        else:                  # SELL
            return deal_price / slippage_factor

    def deal_order(self, order, trade_account=None, position=None):
        result = super().deal_order(order, trade_account=trade_account, position=position)
        if result is None or len(result) < 4:
            return result
        trade_val, trade_cost, trade_price = result[1], result[2], result[3]
        if trade_val > 1e-5 and trade_price is not None and np.isfinite(trade_price):
            adj_price = self._apply_slippage(trade_price, order.direction)
            adj_val = trade_val * (adj_price / trade_price) if trade_price > 0 else trade_val
            adj_cost = trade_cost + abs(trade_val - adj_val)
            return (result[0], adj_val, adj_cost, adj_price)
        return result
