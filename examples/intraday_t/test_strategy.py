"""Unit tests for intraday_t_strategy pure-logic methods.

Tests do NOT require Qlib initialization — they mock out the signal and
BaseSignalStrategy dependencies to test the strategy's internal methods.
"""
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _make_strategy(**kwargs):
    """Create an IntradayTStrategy with a mocked signal, bypassing Qlib init."""
    from intraday_t_strategy import IntradayTStrategy

    defaults = dict(stock_id="SZ301536")
    defaults.update(kwargs)

    with patch("qlib.contrib.strategy.signal_strategy.create_signal_from",
               return_value=Mock()):
        return IntradayTStrategy(**defaults)


class TestSignalGuard:
    def setup_method(self):
        self.s = _make_strategy(enable_signal_guard=True)

    def test_rejects_nan(self):
        assert self.s._validate_signal(np.nan) is False

    def test_rejects_inf(self):
        assert self.s._validate_signal(float("inf")) is False
        assert self.s._validate_signal(float("-inf")) is False

    def test_rejects_out_of_range(self):
        assert self.s._validate_signal(0.30) is False
        assert self.s._validate_signal(-0.30) is False

    def test_accepts_valid(self):
        assert self.s._validate_signal(0.001) is True
        assert self.s._validate_signal(-0.002) is True
        assert self.s._validate_signal(0.0) is True

    def test_boundary_values(self):
        assert self.s._validate_signal(0.25) is True
        assert self.s._validate_signal(-0.25) is True
        assert self.s._validate_signal(0.250001) is False

    def test_zero_variance_detection(self):
        for _ in range(15):
            self.s._validate_signal(0.001)
        assert len(self.s._signal_history) >= 10
        assert np.std(self.s._signal_history) < 1e-10
        assert self.s._validate_signal(0.001) is False

    def test_history_capped(self):
        for i in range(50):
            self.s._validate_signal(np.sin(i) * 0.01)
        assert len(self.s._signal_history) <= self.s._signal_max_history


class TestRoundUnit:
    def test_basic_rounding(self):
        s = _make_strategy(trade_unit=100)
        assert s._round_unit(350) == 300
        assert s._round_unit(100) == 100
        assert s._round_unit(99) == 0
        assert s._round_unit(0) == 0.0
        assert s._round_unit(-10) == 0.0


class TestAdaptiveThresholds:
    def test_buffer_limits(self):
        s = _make_strategy(
            enable_adaptive_thresh=True, adaptive_thresh_lookback=10
        )
        for _ in range(100):
            s._update_adaptive_thresholds(np.random.randn() * 0.01)
        # max_buf = max(lookback*2, 60) = 60
        assert len(s._pred_buffer) <= 60

    def test_rejects_nan(self):
        s = _make_strategy(enable_adaptive_thresh=True)
        for _ in range(30):
            s._update_adaptive_thresholds(np.random.randn() * 0.01)
        initial_buy = s._adaptive_buy_thresh
        s._update_adaptive_thresholds(np.nan)
        assert s._adaptive_buy_thresh == initial_buy

    def test_no_update_with_few_samples(self):
        s = _make_strategy(
            enable_adaptive_thresh=True, adaptive_thresh_lookback=100
        )
        for _ in range(20):
            s._update_adaptive_thresholds(np.random.randn() * 0.01)
        assert s._adaptive_buy_thresh == s.buy_thresh
        assert s._adaptive_sell_thresh == s.sell_thresh

    def test_thresholds_diverge_from_static(self):
        s = _make_strategy(
            enable_adaptive_thresh=True, adaptive_thresh_lookback=5,
            adaptive_thresh_percentile=0.3,
            buy_thresh=0.0, sell_thresh=0.0,
        )
        # Feed deterministic values: half positive, half negative
        for i in range(30):
            s._update_adaptive_thresholds(0.01 if i % 2 == 0 else -0.01)
        # With mixed signs and 30% percentile, thresholds should move away from zero
        assert s._adaptive_buy_thresh > 0.0
        assert s._adaptive_sell_thresh < 0.0


class TestStopLoss:
    def test_no_stop_loss_when_disabled(self):
        s = _make_strategy(enable_stop_loss=False)
        assert s._check_stop_loss(1000, None, None) is None

    def test_no_stop_loss_without_entry_price(self):
        s = _make_strategy(enable_stop_loss=True)
        s._entry_prices = {}
        assert s._check_stop_loss(1000, None, None) is None

    def test_no_stop_loss_without_holding(self):
        s = _make_strategy(enable_stop_loss=True)
        assert s._check_stop_loss(0, None, None) is None


class TestConstructor:
    def test_rejects_empty_stock_id(self):
        with pytest.raises(ValueError, match="stock_id"):
            _make_strategy(stock_id="")

    def test_rejects_inverted_thresholds(self):
        with pytest.raises(ValueError):
            _make_strategy(stock_id="SZ301536", buy_thresh=0.0, sell_thresh=0.01)

    def test_valid_thresholds(self):
        s = _make_strategy(buy_thresh=0.01, sell_thresh=0.01)
        assert s.buy_thresh == 0.01

    def test_rejects_invalid_buy_pct(self):
        with pytest.raises(ValueError):
            _make_strategy(buy_pct=1.5)

    def test_rejects_invalid_sell_pct(self):
        with pytest.raises(ValueError):
            _make_strategy(sell_pct=0.0)

    def test_deprecated_position_pct(self):
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            s = _make_strategy(position_pct=0.5)
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert s.buy_pct == 0.5
            assert s.sell_pct == 0.5

    def test_default_values(self):
        s = _make_strategy()
        assert s.buy_pct == 0.30
        assert s.sell_pct == 0.50
        assert s.enable_signal_guard is True
        assert s.enable_stop_loss is False
        assert s.enable_adaptive_thresh is False
        assert s.risk_manager is None
