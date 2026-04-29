"""
Live TWAP executor bridge for miniQMT integration.

Architecture:
    Daily decision: IntradayTStrategy generates BUY/SELL TradeDecisionWO
    Intraday execution: LiveTWAPExecutor slices orders into 1min TWAP child orders
    Broker: BrokerAdapter wraps miniQMT API for order submission / position query

Usage (conceptual — requires miniQMT runtime):
    broker = MiniQMTBrokerAdapter(account="123456")
    executor = LiveTWAPExecutor(
        broker=broker,
        stock_id="301536",
        twap_minutes=240,
    )
    executor.submit_daily_target(buy_target=3000)
    executor.run()  # blocking loop, fires orders each minute
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple


class OrderState(Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass
class TrackedOrder:
    order_id: str
    stock_id: str
    direction: str  # BUY / SELL
    total_amount: int
    filled_amount: int = 0
    avg_price: float = 0.0
    state: OrderState = OrderState.PENDING
    submitted_at: str = ""
    last_update: str = ""


class BrokerAdapter:
    """Abstract interface for broker API.

    Subclass this for miniQMT or other broker integrations.
    """

    def query_positions(self) -> Dict[str, int]:
        raise NotImplementedError

    def query_cash(self) -> float:
        raise NotImplementedError

    def query_position(self, stock_id: str) -> int:
        raise NotImplementedError

    def submit_order(
        self, stock_id: str, direction: str, amount: int, price: float = 0.0
    ) -> str:
        """Submit an order, return order_id."""
        raise NotImplementedError

    def cancel_order(self, order_id: str) -> bool:
        raise NotImplementedError

    def query_order(self, order_id: str) -> dict:
        raise NotImplementedError

    def get_realtime_quote(self, stock_id: str) -> dict:
        raise NotImplementedError


class LiveTWAPExecutor:
    """1-minute TWAP execution loop for live trading.

    Takes a daily target order, slices it into equal child orders over
    `twap_minutes` minutes, and submits them on a clock-driven loop.
    """

    def __init__(
        self,
        broker: BrokerAdapter,
        stock_id: str,
        twap_minutes: int = 240,
        trade_unit: int = 100,
        max_slippage_pct: float = 0.02,
        on_fill: Optional[Callable] = None,
    ):
        self.broker = broker
        self.stock_id = stock_id
        self.twap_minutes = twap_minutes
        self.trade_unit = trade_unit
        self.max_slippage_pct = max_slippage_pct
        self.on_fill = on_fill

        self._daily_buy_target = 0
        self._daily_sell_target = 0
        self._child_orders: List[TrackedOrder] = []
        self._slice_index = 0
        self._running = False

    def submit_daily_target(self, buy_target: int = 0, sell_target: int = 0) -> None:
        self._daily_buy_target = max(0, int(buy_target // self.trade_unit) * self.trade_unit)
        self._daily_sell_target = max(0, int(sell_target // self.trade_unit) * self.trade_unit)
        self._slice_index = 0
        self._child_orders.clear()

    @property
    def remaining_buy(self) -> int:
        filled = sum(o.filled_amount for o in self._child_orders if o.direction == "BUY")
        return max(0, self._daily_buy_target - filled)

    @property
    def remaining_sell(self) -> int:
        filled = sum(o.filled_amount for o in self._child_orders if o.direction == "SELL")
        return max(0, self._daily_sell_target - filled)

    def _round_trade_unit(self, amount: int) -> int:
        return max(0, int(amount // self.trade_unit) * self.trade_unit)

    def _next_slice_amount(self, remaining: int) -> int:
        slices_left = max(1, self.twap_minutes - self._slice_index)
        raw = remaining / slices_left
        return self._round_trade_unit(int(raw))

    def execute_slice(self) -> Tuple[int, int]:
        """Submit one TWAP slice. Returns (buy_submitted, sell_submitted)."""
        buy_amt = 0
        sell_amt = 0
        quote = self.broker.get_realtime_quote(self.stock_id)
        current_price = quote.get("last_price", 0)

        if self.remaining_buy > 0:
            amt = self._next_slice_amount(self.remaining_buy)
            if amt > 0 and current_price > 0:
                oid = self.broker.submit_order(self.stock_id, "BUY", amt, current_price)
                self._child_orders.append(TrackedOrder(
                    order_id=oid, stock_id=self.stock_id, direction="BUY",
                    total_amount=amt, submitted_at=datetime.now().isoformat(),
                ))
                buy_amt = amt

        if self.remaining_sell > 0:
            amt = self._next_slice_amount(self.remaining_sell)
            held = self.broker.query_position(self.stock_id)
            amt = min(amt, held)
            if amt > 0 and current_price > 0:
                oid = self.broker.submit_order(self.stock_id, "SELL", amt, current_price)
                self._child_orders.append(TrackedOrder(
                    order_id=oid, stock_id=self.stock_id, direction="SELL",
                    total_amount=amt, submitted_at=datetime.now().isoformat(),
                ))
                sell_amt = amt

        self._slice_index += 1
        return buy_amt, sell_amt

    def sync_orders(self) -> List[TrackedOrder]:
        """Poll broker for order status updates. Returns newly filled orders."""
        newly_filled: List[TrackedOrder] = []
        for o in self._child_orders:
            if o.state in (OrderState.FILLED, OrderState.CANCELLED, OrderState.REJECTED):
                continue
            status = self.broker.query_order(o.order_id)
            old_filled = o.filled_amount
            o.filled_amount = status.get("filled_amount", o.filled_amount)
            o.avg_price = status.get("avg_price", o.avg_price)
            o.state = OrderState(status.get("state", o.state.value))
            o.last_update = datetime.now().isoformat()
            if o.filled_amount > old_filled:
                newly_filled.append(o)
                if self.on_fill:
                    self.on_fill(o)
        return newly_filled

    def run(self, end_time: Optional[datetime] = None) -> None:
        """Blocking TWAP loop. Submits a slice every 60 seconds until
        targets are filled or end_time is reached."""
        self._running = True
        while self._running:
            now = datetime.now()
            if end_time and now >= end_time:
                break
            if self.remaining_buy <= 0 and self.remaining_sell <= 0:
                break

            self._slice_index += 1
            self.execute_slice()
            self.sync_orders()

            # Sleep until next minute boundary
            next_minute = (now + timedelta(minutes=1)).replace(second=0, microsecond=0)
            wait = (next_minute - datetime.now()).total_seconds()
            if wait > 0:
                time.sleep(min(wait, 60))

    def stop(self) -> None:
        self._running = False

    def cancel_all(self) -> int:
        cancelled = 0
        for o in self._child_orders:
            if o.state in (OrderState.PENDING, OrderState.SUBMITTED, OrderState.PARTIAL):
                if self.broker.cancel_order(o.order_id):
                    o.state = OrderState.CANCELLED
                    cancelled += 1
        return cancelled
