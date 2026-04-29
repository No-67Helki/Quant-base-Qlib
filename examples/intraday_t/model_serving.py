"""
Export DEnsembleModel to a standalone format with zero Qlib Dataset dependency.

Usage:
    # Export after training
    exporter = ModelExporter(model, feature_names)
    exporter.export("models/intraday_t.pkl")

    # Load and predict
    serving = ModelServing.load("models/intraday_t.pkl")
    prediction = serving.predict(features_dict)  # Dict[str, np.ndarray]
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


class ModelServing:
    """Standalone inference wrapper for exported DEnsembleModel.

    Accepts raw feature arrays — no Qlib Dataset, no handlers, no calendar.
    """

    def __init__(
        self,
        boosters: list,
        sub_features: List[List[str]],
        sub_weights: List[float],
        feature_names: List[str],
    ):
        self.boosters = boosters
        self.sub_features = sub_features
        self.sub_weights = sub_weights
        self.feature_names = feature_names

    def predict(self, features: Dict[str, np.ndarray]) -> float:
        """Predict a single scalar score from a feature dict.

        Parameters
        ----------
        features : dict
            Maps feature name → scalar or single-element array.
            Must include all columns referenced by sub_features.

        Returns
        -------
        float
            Ensemble prediction (weighted average of sub-models).
        """
        preds = []
        for booster, sf, w in zip(self.boosters, self.sub_features, self.sub_weights):
            x = np.array([[features.get(f, 0.0) for f in sf]], dtype=np.float64)
            p = float(booster.predict(x)[0])
            preds.append(p * w)
        total_w = sum(self.sub_weights)
        return float(sum(preds) / total_w) if total_w > 0 else 0.0

    def predict_batch(self, feature_matrix: np.ndarray) -> np.ndarray:
        """Predict from a (N, D) array ordered by self.feature_names."""
        preds = np.zeros(len(feature_matrix), dtype=np.float64)
        for booster, sf, w in zip(self.boosters, self.sub_features, self.sub_weights):
            col_idx = [self.feature_names.index(f) for f in sf if f in self.feature_names]
            if not col_idx:
                continue
            x = feature_matrix[:, col_idx]
            preds += booster.predict(x) * w
        total_w = sum(self.sub_weights)
        if total_w > 0:
            preds /= total_w
        return preds

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "ModelServing":
        with open(path, "rb") as f:
            return pickle.load(f)


class ModelExporter:
    """Extract trained DEnsembleModel internals for standalone serving."""

    def __init__(self, model, feature_names: List[str]):
        self.model = model
        self.feature_names = list(feature_names)

    def export(self, path: str) -> ModelServing:
        ens = self.model.ensemble
        if isinstance(ens, dict):
            boosters = list(ens.values())
        else:
            boosters = list(ens)

        sub_features = []
        if hasattr(self.model, "sub_features") and self.model.sub_features:
            for sf in self.model.sub_features:
                sub_features.append(list(sf) if hasattr(sf, "tolist") else list(sf))
        if len(sub_features) != len(boosters):
            sub_features = [self.feature_names] * len(boosters)

        sub_weights = []
        if hasattr(self.model, "sub_weights") and self.model.sub_weights:
            sub_weights = list(self.model.sub_weights)
        if len(sub_weights) != len(boosters):
            sub_weights = [1.0] * len(boosters)

        serving = ModelServing(boosters, sub_features, sub_weights, self.feature_names)
        serving.save(path)
        return serving


def export_from_model(model, dataset, path: str) -> ModelServing:
    """Convenience: export a trained model after fitting.

    Extracts feature names from dataset if available, otherwise uses
    the handler's get_feature_config.
    """
    try:
        df = dataset.prepare("train", col_set=["feature"])
        feature_names = list(df["feature"].columns)
    except Exception:
        try:
            handler = dataset.handler
            _, feature_names = handler.get_feature_config()
        except Exception:
            raise RuntimeError("Cannot determine feature names from model or dataset.")

    exporter = ModelExporter(model, feature_names)
    return exporter.export(path)
