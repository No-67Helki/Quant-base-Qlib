"""
Pydantic config validation for intraday-T strategy.
Validates the YAML config before it reaches Qlib's init_instance_by_config.
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class StrategyKwargs(BaseModel):
    signal: str = "<PRED>"
    stock_id: str
    buy_thresh: float = 0.0
    sell_thresh: float = 0.0
    trade_unit: int = 100
    buy_pct: float = Field(default=0.30, gt=0, le=1)
    sell_pct: float = Field(default=0.50, gt=0, le=1)
    allow_first_sell: bool = True
    enable_stop_loss: bool = False
    stop_loss_pct: float = Field(default=0.03, gt=0, lt=1)
    enable_adaptive_thresh: bool = False
    adaptive_thresh_lookback: int = Field(default=60, ge=10)
    adaptive_thresh_percentile: float = Field(default=0.20, gt=0, lt=0.5)
    enable_signal_guard: bool = True

    @field_validator("sell_thresh")
    @classmethod
    def sell_le_buy(cls, v, info):
        if "buy_thresh" in info.data and v > info.data["buy_thresh"]:
            raise ValueError("sell_thresh must be <= buy_thresh")
        return v


class RiskLimitsConfig(BaseModel):
    daily_loss_limit_pct: Optional[float] = Field(default=None, ge=0, le=1)
    max_drawdown_pct: Optional[float] = Field(default=None, ge=0, le=1)
    max_position_conc_pct: Optional[float] = Field(default=None, ge=0, le=1)
    max_trades_per_day: Optional[int] = Field(default=None, ge=1)
    max_order_amount: Optional[int] = Field(default=None, ge=0)


class ExchangeKwargs(BaseModel):
    codes: Optional[List[str]] = None
    freq: str = "1min"
    limit_threshold: float = 0.195
    deal_price: str = "$vwap"
    open_cost: float = 0.0005
    close_cost: float = 0.0015
    min_cost: float = 5.0
    impact_cost: float = Field(default=0.0005, ge=0)
    trade_unit: int = 100


class PortAnalysisConfig(BaseModel):
    strategy: dict
    backtest: dict
    executor: dict


def validate_strategy_kwargs(kwargs: dict) -> StrategyKwargs:
    return StrategyKwargs(**kwargs)


def validate_risk_limits(limits: Optional[dict]) -> Optional[RiskLimitsConfig]:
    if limits is None:
        return None
    return RiskLimitsConfig(**limits)


def validate_exchange_kwargs(kwargs: dict) -> ExchangeKwargs:
    return ExchangeKwargs(**kwargs)
