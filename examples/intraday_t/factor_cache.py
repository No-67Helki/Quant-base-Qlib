"""
Incremental Alpha158HF factor calculator for live trading.

Maintains rolling OHLCV DataFrames and computes all 163 features (158 Alpha158
+ 5 HF) using only pandas/numpy — zero Qlib dependency at inference time.

Feature layout (Alpha158 default config):
    9 Kbar + 4 Price + 29 rolling operators x 5 windows [5,10,20,30,60] + 5 HF = 163

Usage:
    cache = FactorCache(max_window=200)
    cache.update(bar)                         # bar = dict with OHLCV for one bar
    features = cache.compute()                # Dict[str, float] with 163 features
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


class FactorCache:
    """Rolling OHLCV cache with complete Alpha158HF feature computation."""

    _WINDOWS = [5, 10, 20, 30, 60]

    def __init__(self, max_window: int = 200):
        self.max_window = max_window
        self._df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------

    def update(self, bar: dict) -> None:
        """Ingest one OHLCV bar (minute or daily)."""
        row = pd.DataFrame([{
            "open": float(bar["open"]),
            "high": float(bar["high"]),
            "low": float(bar["low"]),
            "close": float(bar["close"]),
            "volume": float(bar.get("volume", 0)),
            "amount": float(bar.get("amount", 0)),
            "vwap": float(bar.get("vwap", bar["close"])),
        }])
        if self._df is None:
            self._df = row
        else:
            self._df = pd.concat([self._df, row], ignore_index=True)
        if len(self._df) > self.max_window:
            self._df = self._df.iloc[-self.max_window:].reset_index(drop=True)

    # ------------------------------------------------------------------
    # Rolling helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _slope(x: np.ndarray, y: np.ndarray) -> float:
        """Slope of linear regression y ~ x."""
        n = len(x)
        if n < 2 or np.std(x) < 1e-12:
            return 0.0
        return float(np.cov(x, y)[0, 1] / np.var(x))

    @staticmethod
    def _rsquare(x: np.ndarray, y: np.ndarray) -> float:
        """R-squared of linear regression y ~ x."""
        if len(x) < 2 or np.std(y) < 1e-12:
            return 0.0
        corr = np.corrcoef(x, y)[0, 1]
        return float(corr ** 2)

    @staticmethod
    def _resi(x: np.ndarray, y: np.ndarray) -> float:
        """Last residual from linear regression y ~ x."""
        n = len(x)
        if n < 2 or np.std(x) < 1e-12:
            return 0.0
        slope = np.cov(x, y)[0, 1] / np.var(x)
        intercept = np.mean(y) - slope * np.mean(x)
        return float((y[-1] - (slope * x[-1] + intercept)) / y[-1]) if abs(y[-1]) > 1e-12 else 0.0

    @staticmethod
    def _idxmax(x: np.ndarray) -> float:
        """Days since max value (0=most recent, 1=oldest)."""
        if len(x) == 0:
            return 0.0
        return float(np.argmax(x[::-1]))

    @staticmethod
    def _idxmin(x: np.ndarray) -> float:
        """Days since min value (0=most recent, 1=oldest)."""
        if len(x) == 0:
            return 0.0
        return float(np.argmin(x[::-1]))

    # ------------------------------------------------------------------
    # Feature computation
    # ------------------------------------------------------------------

    def compute(self) -> Dict[str, float]:
        """Compute all 163 features from current rolling window.

        Returns a dict mapping feature name -> scalar value (latest timestep).
        """
        if self._df is None or len(self._df) < 5:
            return {}

        df = self._df
        c = df["close"]
        o = df["open"]
        h = df["high"]
        l = df["low"]
        v = df["volume"]
        vwap = df["vwap"]

        feats: Dict[str, float] = {}

        # Pre-compute intermediate series
        ret = c.pct_change()                 # daily return
        vchg = v.pct_change()                # volume change
        logv = np.log(v + 1)                 # log volume
        logv_chg = np.log(v / v.shift(1).replace(0, np.nan).fillna(1) + 1e-12)
        up_mask = c > c.shift(1)
        dn_mask = c < c.shift(1)
        vol_up = v > v.shift(1)
        vol_dn = v < v.shift(1)
        abs_ret = ret.abs()
        abs_vchg = vchg.abs()
        pos_ret = ret.clip(lower=0)
        neg_ret = (-ret).clip(lower=0)
        pos_vchg = vchg.clip(lower=0)
        neg_vchg = (-vchg).clip(lower=0)
        close_div = c.iloc[-1]
        eps = 1e-12

        # ================================================================
        # Section 1: Kbar features (9)
        # ================================================================
        feats["KMID"]  = float((c.iloc[-1] - o.iloc[-1]) / o.iloc[-1]) if o.iloc[-1] else 0.0
        feats["KLEN"]  = float((h.iloc[-1] - l.iloc[-1]) / o.iloc[-1]) if o.iloc[-1] else 0.0
        hl_range = h.iloc[-1] - l.iloc[-1] + eps
        feats["KMID2"] = float((c.iloc[-1] - o.iloc[-1]) / hl_range)
        feats["KUP"]   = float((h.iloc[-1] - max(o.iloc[-1], c.iloc[-1])) / o.iloc[-1]) if o.iloc[-1] else 0.0
        feats["KUP2"]  = float((h.iloc[-1] - max(o.iloc[-1], c.iloc[-1])) / hl_range)
        feats["KLOW"]  = float((min(o.iloc[-1], c.iloc[-1]) - l.iloc[-1]) / o.iloc[-1]) if o.iloc[-1] else 0.0
        feats["KLOW2"] = float((min(o.iloc[-1], c.iloc[-1]) - l.iloc[-1]) / hl_range)
        feats["KSFT"]  = float((2*c.iloc[-1] - h.iloc[-1] - l.iloc[-1]) / o.iloc[-1]) if o.iloc[-1] else 0.0
        feats["KSFT2"] = float((2*c.iloc[-1] - h.iloc[-1] - l.iloc[-1]) / hl_range)

        # ================================================================
        # Section 2: Price features (4) — field/close
        # ================================================================
        feats["OPEN0"] = float(o.iloc[-1] / close_div) if close_div else 0.0
        feats["HIGH0"] = float(h.iloc[-1] / close_div) if close_div else 0.0
        feats["LOW0"]  = float(l.iloc[-1] / close_div) if close_div else 0.0
        feats["VWAP0"] = float(vwap.iloc[-1] / close_div) if close_div else 0.0

        # ================================================================
        # Section 3: Rolling operators (29 operators x 5 windows = 145)
        # ================================================================
        for d in self._WINDOWS:
            if len(df) < d:
                continue
            tail_c = c.tail(d)
            tail_h = h.tail(d)
            tail_l = l.tail(d)
            tail_v = v.tail(d)
            tail_ret = ret.tail(d)
            tail_vchg = vchg.tail(d)

            t = np.arange(d)
            cy = tail_c.values
            hy = tail_h.values
            ly = tail_l.values
            vy = tail_v.values

            # ROC: Ref($close, d)/$close
            feats[f"ROC{d}"] = float(tail_c.iloc[0] / close_div) if close_div else 0.0

            # MA: Mean($close, d)/$close
            feats[f"MA{d}"] = float(cy.mean() / close_div) if close_div else 0.0

            # STD: Std($close, d)/$close
            feats[f"STD{d}"] = float(cy.std() / close_div) if close_div and len(cy) >= 2 else 0.0

            # BETA: Slope($close, d)/$close
            feats[f"BETA{d}"] = float(self._slope(t, cy) / close_div) if close_div else 0.0

            # RSQR: Rsquare($close, d)
            feats[f"RSQR{d}"] = self._rsquare(t, cy)

            # RESI: Resi($close, d)/$close
            feats[f"RESI{d}"] = self._resi(t, cy)

            # MAX: Max($high, d)/$close
            feats[f"MAX{d}"] = float(hy.max() / close_div) if close_div else 0.0

            # MIN: Min($low, d)/$close
            feats[f"MIN{d}"] = float(ly.min() / close_div) if close_div else 0.0

            # QTLU: Quantile($close, d, 0.8)/$close
            feats[f"QTLU{d}"] = float(np.percentile(cy, 80) / close_div) if close_div else 0.0

            # QTLD: Quantile($close, d, 0.2)/$close
            feats[f"QTLD{d}"] = float(np.percentile(cy, 20) / close_div) if close_div else 0.0

            # RANK: Rank($close, d) — percentile rank of last value
            feats[f"RANK{d}"] = float((cy < cy[-1]).sum() / d)

            # RSV: ($close - Min($low,d)) / (Max($high,d) - Min($low,d))
            hi_max, lo_min = hy.max(), ly.min()
            feats[f"RSV{d}"] = float((cy[-1] - lo_min) / (hi_max - lo_min + eps))

            # IMAX: IdxMax($high, d)/d
            feats[f"IMAX{d}"] = self._idxmax(hy) / d

            # IMIN: IdxMin($low, d)/d
            feats[f"IMIN{d}"] = self._idxmin(ly) / d

            # IMXD: (IdxMax - IdxMin)/d
            feats[f"IMXD{d}"] = (self._idxmax(hy) - self._idxmin(ly)) / d

            # CORR: Corr($close, Log($volume+1), d)
            logv_tail = logv.tail(d).values
            feats[f"CORR{d}"] = float(np.corrcoef(cy, logv_tail)[0, 1]) if np.std(logv_tail) > eps else 0.0

            # CORD: Corr(close/Ref(close,1), Log(volume/Ref(volume,1)+1), d)
            r_tail = tail_ret.dropna().values
            lv_tail = logv_chg.tail(d).dropna().values
            min_len = min(len(r_tail), len(lv_tail))
            if min_len >= 3:
                feats[f"CORD{d}"] = float(np.corrcoef(r_tail[-min_len:], lv_tail[-min_len:])[0, 1])
            else:
                feats[f"CORD{d}"] = 0.0

            # CNTP: Mean($close > Ref($close,1), d)
            feats[f"CNTP{d}"] = float(up_mask.tail(d).mean())

            # CNTN: Mean($close < Ref($close,1), d)
            feats[f"CNTN{d}"] = float(dn_mask.tail(d).mean())

            # CNTD: CNTP - CNTN
            feats[f"CNTD{d}"] = feats[f"CNTP{d}"] - feats[f"CNTN{d}"]

            # SUMP / SUMN / SUMD
            p = pos_ret.tail(d).sum()
            n = neg_ret.tail(d).sum()
            a = abs_ret.tail(d).sum() + eps
            feats[f"SUMP{d}"] = float(p / a)
            feats[f"SUMN{d}"] = float(n / a)
            feats[f"SUMD{d}"] = float((p - n) / a)

            # VMA: Mean($volume, d)/($volume + eps)
            feats[f"VMA{d}"] = float(vy.mean() / (vy[-1] + eps))

            # VSTD: Std($volume, d)/($volume + eps)
            feats[f"VSTD{d}"] = float(vy.std() / (vy[-1] + eps)) if len(vy) >= 2 else 0.0

            # WVMA: Std(Abs(ret)*vol, d) / Mean(Abs(ret)*vol, d)
            wv = (abs_ret * v).tail(d)
            wv_mean = wv.mean()
            feats[f"WVMA{d}"] = float(wv.std() / (wv_mean + eps)) if wv_mean > eps and len(wv) >= 2 else 0.0

            # VSUMP / VSUMN / VSUMD
            vp = pos_vchg.tail(d).sum()
            vn = neg_vchg.tail(d).sum()
            va = abs_vchg.tail(d).sum() + eps
            feats[f"VSUMP{d}"] = float(vp / va)
            feats[f"VSUMN{d}"] = float(vn / va)
            feats[f"VSUMD{d}"] = float((vp - vn) / va)

        # ================================================================
        # Section 4: High-frequency factors (5)
        # ================================================================
        feats["OVN_GAP"] = float(o.iloc[-1] / c.iloc[-2] - 1) if len(df) >= 2 and c.iloc[-2] else 0.0

        if len(df) >= 5:
            intra_ret = ((c - o) / o.replace(0, np.nan)).tail(5)
            feats["MOM_TAIL5"] = float(intra_ret.mean())

            ret_1 = c / c.shift(1) - 1
            feats["INTRA_VOL5"] = float(ret_1.tail(5).std() if len(ret_1.tail(5)) >= 2 else 0.0)

            pchg = (c / c.shift(1) - 1).tail(5)
            vchg_log = np.log(v / v.shift(1).replace(0, np.nan) + 1e-12).tail(5)
            if vchg_log.std() > eps:
                feats["PV_DIV5"] = float(pchg.corr(vchg_log))
            else:
                feats["PV_DIV5"] = 0.0
        else:
            feats["MOM_TAIL5"] = 0.0
            feats["INTRA_VOL5"] = 0.0
            feats["PV_DIV5"] = 0.0

        feats["RNG_OPEN"] = float((h.iloc[-1] - l.iloc[-1]) / o.iloc[-1]) if o.iloc[-1] > 0 else 0.0

        return feats

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def ready(self) -> bool:
        return self._df is not None and len(self._df) >= 60

    @property
    def feature_count(self) -> int:
        """Total feature count when compute() has all windows."""
        if self._df is None or len(self._df) < 5:
            return 0
        return 9 + 4 + len([d for d in self._WINDOWS if len(self._df) >= d]) * 29 + 5
