"""
Structured logging, prediction quality tracking, and anomaly detection.

Usage:
    monitor = StrategyMonitor(log_path="logs/trades.jsonl")
    monitor.log_trade(...)
    monitor.track_prediction(score, actual_return)
    monitor.check_anomalies()  # returns list of warning strings
"""
from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, List, Optional

import numpy as np


@dataclass
class TradeRecord:
    timestamp: str
    direction: str  # BUY / SELL
    stock_id: str
    amount: float
    price: float
    cost: float
    reason: str = ""


class PredictionTracker:
    def __init__(self, window: int = 252):
        self.window = window
        self._scores: Deque[float] = deque(maxlen=window)
        self._actuals: Deque[float] = deque(maxlen=window)
        self._timestamps: Deque[str] = deque(maxlen=window)

    def add(self, score: float, actual_return: float, timestamp: str = "") -> None:
        self._scores.append(float(score))
        self._actuals.append(float(actual_return))
        if timestamp:
            self._timestamps.append(timestamp)

    @property
    def rolling_ic(self) -> float:
        if len(self._scores) < 10:
            return 0.0
        s = np.array(self._scores)
        a = np.array(self._actuals)
        mask = np.isfinite(s) & np.isfinite(a)
        if mask.sum() < 10:
            return 0.0
        try:
            return float(np.corrcoef(s[mask], a[mask])[0, 1])
        except Exception:
            return 0.0

    @property
    def mean_score(self) -> float:
        if not self._scores:
            return 0.0
        return float(np.mean(list(self._scores)))

    @property
    def std_score(self) -> float:
        if len(self._scores) < 2:
            return 0.0
        return float(np.std(list(self._scores)))


class AnomalyDetector:
    def __init__(self, zero_trade_streak_thresh: int = 10, pred_collapse_thresh: float = 1e-6):
        self.zero_trade_streak_thresh = zero_trade_streak_thresh
        self.pred_collapse_thresh = pred_collapse_thresh
        self._zero_trade_streak = 0
        self._pred_buffer: Deque[float] = deque(maxlen=30)

    def check(self, tracker: PredictionTracker) -> List[str]:
        warnings: List[str] = []
        if len(tracker._scores) >= 10 and tracker.std_score < self.pred_collapse_thresh:
            warnings.append(f"Prediction collapse: std={tracker.std_score:.2e} over last {len(tracker._scores)} steps")
        if len(tracker._scores) >= 20 and abs(tracker.rolling_ic) < 0.01:
            warnings.append(f"Rolling IC near zero: {tracker.rolling_ic:.4f}")
        return warnings

    def track_trade(self, had_trade: bool) -> List[str]:
        if had_trade:
            self._zero_trade_streak = 0
        else:
            self._zero_trade_streak += 1
        warnings: List[str] = []
        if self._zero_trade_streak >= self.zero_trade_streak_thresh:
            warnings.append(f"Zero-trade streak: {self._zero_trade_streak} steps")
        return warnings

    def track_prediction(self, score: float) -> None:
        self._pred_buffer.append(float(score))


class StrategyMonitor:
    def __init__(self, log_path: str = "logs/trades.jsonl"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.pred_tracker = PredictionTracker()
        self.anomaly_detector = AnomalyDetector()
        self._daily_trade_count = 0
        self._last_date = ""

    def log_trade(self, record: TradeRecord) -> None:
        with open(self.log_path, "a", encoding="utf-8") as f:
            json.dump(asdict(record), f, ensure_ascii=False)
            f.write("\n")
        self._daily_trade_count += 1

    def track_prediction(self, score: float, actual_return: float = 0.0, timestamp: str = "") -> None:
        self.pred_tracker.add(score, actual_return, timestamp)
        self.anomaly_detector.track_prediction(score)

    def check_anomalies(self, had_trade: bool) -> List[str]:
        trade_warnings = self.anomaly_detector.track_trade(had_trade)
        pred_warnings = self.anomaly_detector.check(self.pred_tracker)
        return trade_warnings + pred_warnings

    def on_new_day(self, date_str: str) -> None:
        if date_str != self._last_date:
            self._daily_trade_count = 0
            self._last_date = date_str
