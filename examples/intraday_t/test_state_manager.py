"""Unit tests for state_manager.py"""
import json
import os
import tempfile
from unittest.mock import Mock

from state_manager import StateManager, StrategyState


class TestStrategyState:
    def test_defaults(self):
        s = StrategyState()
        assert s.positions == {}
        assert s.entry_prices == {}
        assert s.pred_buffer == []
        assert s.signal_history == []
        assert s.risk_state == {}
        assert s.cash == 0.0
        assert s.total_value == 0.0

    def test_serialization_roundtrip(self):
        s = StrategyState(
            timestamp="2026-04-29T10:00:00",
            positions={"SZ301536": 2000},
            cash=450000.0,
            total_value=580000.0,
            entry_prices={"SZ301536": 65.0},
            adaptive_buy_thresh=0.001,
            adaptive_sell_thresh=-0.0008,
            pred_buffer=[0.001, -0.002, 0.003],
            signal_history=[0.001, -0.002],
            risk_state={"peak_equity": 600000, "daily_pnl": 1500},
        )
        data = json.loads(json.dumps(s.__dict__))
        restored = StrategyState(**data)
        assert restored.positions == {"SZ301536": 2000}
        assert restored.cash == 450000.0
        assert restored.pred_buffer == [0.001, -0.002, 0.003]
        assert restored.risk_state["peak_equity"] == 600000


class TestStateManager:
    def test_save_and_load(self):
        path = os.path.join(tempfile.gettempdir(), "test_sm_save.json")
        sm = StateManager(path)
        state = StrategyState(
            timestamp="2026-04-29",
            positions={"SZ301536": 1000},
            cash=400000.0,
            adaptive_buy_thresh=0.0015,
        )
        sm.save(state)
        assert os.path.exists(path)
        loaded = sm.load()
        assert loaded is not None
        assert loaded.positions == {"SZ301536": 1000}
        assert loaded.cash == 400000.0
        assert loaded.adaptive_buy_thresh == 0.0015
        os.remove(path)

    def test_load_missing_file(self):
        path = os.path.join(tempfile.gettempdir(), "nonexistent_sm.json")
        sm = StateManager(path)
        assert sm.load() is None
        assert sm.restore(None, None) is False

    def test_auto_save(self):
        path = os.path.join(tempfile.gettempdir(), "test_autosave_sm.json")
        sm = StateManager(path, auto_save=True)

        # Create a mock strategy object
        strategy = Mock()
        strategy.trade_position = None
        strategy._entry_prices = {"SZ301536": 65.0}
        strategy._adaptive_buy_thresh = 0.001
        strategy._adaptive_sell_thresh = -0.0005
        strategy._pred_buffer = [0.001]
        strategy._signal_history = [0.001, 0.002]

        state = sm.snapshot(strategy, None)
        assert state.entry_prices == {"SZ301536": 65.0}
        assert state.adaptive_buy_thresh == 0.001
        assert os.path.exists(path)

        loaded = sm.load()
        assert loaded is not None
        assert loaded.entry_prices == {"SZ301536": 65.0}
        os.remove(path)

    def test_restore(self):
        path = os.path.join(tempfile.gettempdir(), "test_restore_sm.json")
        sm = StateManager(path)
        state = StrategyState(
            timestamp="2026-04-29",
            entry_prices={"SZ301536": 70.0},
            adaptive_buy_thresh=0.002,
            adaptive_sell_thresh=-0.001,
            pred_buffer=[0.001, 0.002, -0.001],
            signal_history=[0.001],
            risk_state={"peak_equity": 500000, "daily_pnl": 1000},
        )
        sm.save(state)

        strategy = Mock()
        risk_manager = Mock()
        sm.restore(strategy, risk_manager)
        assert strategy._entry_prices == {"SZ301536": 70.0}
        assert strategy._adaptive_buy_thresh == 0.002
        assert strategy._pred_buffer == [0.001, 0.002, -0.001]
        risk_manager.restore_state.assert_called_once()
        os.remove(path)
