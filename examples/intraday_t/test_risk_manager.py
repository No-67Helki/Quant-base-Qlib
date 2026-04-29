"""Unit tests for risk_manager.py"""
import pytest
from risk_manager import RiskLimits, RiskManager


class TestRiskManager:
    def test_initial_state(self):
        rm = RiskManager(RiskLimits(), initial_capital=100000)
        rm.on_new_day("2026-04-29")
        ok, reason = rm.check_circuit_breakers(100000)
        assert ok is False
        assert reason == ""

    def test_daily_loss_breaker(self):
        limits = RiskLimits(daily_loss_limit_pct=0.03)
        rm = RiskManager(limits, initial_capital=100000)
        rm.on_new_day("2026-04-29")
        # Simulate 4k loss (4% > 3%)
        rm.update_daily_pnl(-4000)
        broken, reason = rm.check_circuit_breakers(96000)
        assert broken is True
        assert "daily_loss_limit" in reason

    def test_daily_loss_not_triggered_when_within_limit(self):
        limits = RiskLimits(daily_loss_limit_pct=0.03)
        rm = RiskManager(limits, initial_capital=100000)
        rm.on_new_day("2026-04-29")
        rm.update_daily_pnl(-2000)  # 2% loss
        broken, _ = rm.check_circuit_breakers(98000)
        assert broken is False

    def test_max_drawdown_breaker(self):
        limits = RiskLimits(max_drawdown_pct=0.10)
        rm = RiskManager(limits, initial_capital=100000)
        rm.on_new_day("2026-04-29")
        rm.update_equity_curve(100000)
        broken, reason = rm.check_circuit_breakers(85000)  # 15% drawdown
        assert broken is True
        assert "max_drawdown" in reason

    def test_drawdown_not_triggered_after_peak_update(self):
        limits = RiskLimits(max_drawdown_pct=0.10)
        rm = RiskManager(limits, initial_capital=100000)
        rm.on_new_day("2026-04-29")
        rm.update_equity_curve(100000)
        rm.update_equity_curve(110000)  # new peak
        broken, _ = rm.check_circuit_breakers(100100)  # ~9% from new peak
        assert broken is False

    def test_daily_reset(self):
        limits = RiskLimits(daily_loss_limit_pct=0.03)
        rm = RiskManager(limits, initial_capital=100000)
        rm.on_new_day("2026-04-29")
        rm.update_daily_pnl(-4000)
        assert rm._daily_pnl < -3000
        # New day resets
        rm.on_new_day("2026-04-30")
        assert rm._daily_pnl == 0.0
        assert rm._daily_trade_count == 0
        # Circuit breaker should be cleared by new day
        broken, _ = rm.check_circuit_breakers(100000)
        assert broken is False

    def test_trade_frequency(self):
        limits = RiskLimits(max_trades_per_day=3)
        rm = RiskManager(limits)
        rm.on_new_day("2026-04-29")
        assert rm.check_trade_frequency() is True
        rm.record_trade()
        rm.record_trade()
        rm.record_trade()
        assert rm.check_trade_frequency() is False

    def test_trade_frequency_no_limit(self):
        rm = RiskManager(RiskLimits())  # no limit set
        rm.on_new_day("2026-04-29")
        for _ in range(100):
            rm.record_trade()
        assert rm.check_trade_frequency() is True

    def test_position_concentration(self):
        limits = RiskLimits(max_position_conc_pct=0.50)
        rm = RiskManager(limits)
        assert rm.check_position_concentration(50000, 100000) is True   # 50%
        assert rm.check_position_concentration(51000, 100000) is False  # 51%

    def test_position_concentration_no_limit(self):
        rm = RiskManager(RiskLimits())
        assert rm.check_position_concentration(100000, 100000) is True  # 100%

    def test_duplicate_order_blocks(self):
        rm = RiskManager(RiskLimits())
        rm.on_new_day("2026-04-29")
        # First order OK
        assert rm.check_duplicate_order("SZ301536", 0, "2026-04-29", 1000) is True
        # Same order blocked
        assert rm.check_duplicate_order("SZ301536", 0, "2026-04-29", 1000) is False

    def test_duplicate_different_direction_allowed(self):
        rm = RiskManager(RiskLimits())
        rm.on_new_day("2026-04-29")
        rm.check_duplicate_order("SZ301536", 0, "2026-04-29", 1000)  # BUY
        # SELL with same amount is OK
        assert rm.check_duplicate_order("SZ301536", 1, "2026-04-29", 1000) is True

    def test_duplicate_cleared_on_new_day(self):
        rm = RiskManager(RiskLimits())
        rm.on_new_day("2026-04-29")
        rm.check_duplicate_order("SZ301536", 0, "2026-04-29", 1000)
        rm.on_new_day("2026-04-30")
        assert rm.check_duplicate_order("SZ301536", 0, "2026-04-30", 1000) is True

    def test_reset_circuit(self):
        limits = RiskLimits(daily_loss_limit_pct=0.03)
        rm = RiskManager(limits, initial_capital=100000)
        rm.on_new_day("2026-04-29")
        rm.update_daily_pnl(-4000)
        broken, _ = rm.check_circuit_breakers(96000)
        assert broken is True
        rm.reset_circuit()
        assert rm._circuit_broken is False
        assert rm._daily_pnl == 0.0

    def test_state_export_import(self):
        limits = RiskLimits(daily_loss_limit_pct=0.03, max_drawdown_pct=0.15)
        rm = RiskManager(limits, initial_capital=200000)
        rm.on_new_day("2026-04-29")
        rm.update_daily_pnl(-1000)
        rm.update_equity_curve(200000)
        rm.record_trade()

        state = rm.get_state()
        assert state["daily_pnl"] == -1000
        assert state["daily_trade_count"] == 1

        # Restore to a fresh manager
        rm2 = RiskManager(limits, initial_capital=200000)
        rm2.restore_state(state)
        assert rm2._daily_pnl == -1000
        assert rm2._peak_equity == 200000

    def test_disabled_limits_never_break(self):
        rm = RiskManager(RiskLimits(), initial_capital=100000)
        rm.on_new_day("2026-04-29")
        rm.update_daily_pnl(-50000)  # 50% loss
        rm.update_equity_curve(100000)
        broken, _ = rm.check_circuit_breakers(40000)  # 60% drawdown
        assert broken is False
