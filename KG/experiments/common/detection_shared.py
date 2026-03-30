from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Dict, List, Sequence, Tuple

import numpy as np


ATTACK_TYPES: Tuple[str, ...] = (
    "phantom_congestion",
    "signal_spoofing",
    "false_closure",
    "position_spoofing",
)
CLASS_LABELS: Tuple[str, ...] = ("benign",) + ATTACK_TYPES
CANDIDATE_CLAIM_TYPES = {"congestion_alert", "closure_alert", "signal_report", "vehicle_position"}
FEATURE_NAMES = [
    "Congestion alert",
    "Closure alert",
    "Signal report",
    "Position claim",
    "Sender is RSU",
    "Density gap",
    "Observed RSU density",
    "Sender near segment",
    "Closure flag",
    "Adjacent flow",
    "Controller mismatch",
    "Majority mismatch",
    "Adjacency violation",
    "RSU region change",
]


@dataclass
class MulticlassStats:
    labels: Sequence[str]
    confusion: Dict[str, Dict[str, int]] = field(default_factory=dict)
    latencies: List[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.confusion:
            self.confusion = {
                true_label: {pred_label: 0 for pred_label in self.labels}
                for true_label in self.labels
            }

    def add(self, true_label: str, pred_label: str) -> None:
        self.confusion[true_label][pred_label] += 1

    def support(self, label: str) -> int:
        return sum(self.confusion[label].values())

    def predicted(self, label: str) -> int:
        return sum(self.confusion[true][label] for true in self.labels)

    def true_positive(self, label: str) -> int:
        return self.confusion[label][label]

    def precision(self, label: str) -> float:
        denom = self.predicted(label)
        return self.true_positive(label) / denom if denom else 0.0

    def recall(self, label: str) -> float:
        denom = self.support(label)
        return self.true_positive(label) / denom if denom else 0.0

    def f1(self, label: str) -> float:
        p = self.precision(label)
        r = self.recall(label)
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def accuracy(self) -> float:
        total = sum(self.support(label) for label in self.labels)
        correct = sum(self.true_positive(label) for label in self.labels)
        return correct / total if total else 0.0

    def macro_precision(self) -> float:
        return sum(self.precision(label) for label in self.labels) / len(self.labels)

    def macro_recall(self) -> float:
        return sum(self.recall(label) for label in self.labels) / len(self.labels)

    def macro_f1(self) -> float:
        return sum(self.f1(label) for label in self.labels) / len(self.labels)

    def average_latency(self) -> float:
        return mean(self.latencies) if self.latencies else 0.0


class MulticlassLogisticRegressionModel:
    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.weights: np.ndarray | None = None
        self.labels: Tuple[str, ...] = CLASS_LABELS

    def fit(self, x_rows: List[List[float]], y_rows: List[int], lr: float = 0.1, epochs: int = 1800, reg: float = 1e-3) -> None:
        x = np.asarray(x_rows, dtype=float)
        y = np.asarray(y_rows, dtype=int)
        self.mean = x.mean(axis=0)
        self.scale = x.std(axis=0)
        self.scale[self.scale == 0] = 1.0
        x_scaled = (x - self.mean) / self.scale
        x_aug = np.hstack([np.ones((x_scaled.shape[0], 1)), x_scaled])
        num_classes = len(self.labels)
        y_onehot = np.eye(num_classes)[y]
        self.weights = np.zeros((x_aug.shape[1], num_classes), dtype=float)
        for _ in range(epochs):
            logits = x_aug @ self.weights
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(np.clip(logits, -30, 30))
            preds = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            gradient = (x_aug.T @ (preds - y_onehot)) / len(y)
            gradient[1:, :] += reg * self.weights[1:, :]
            self.weights -= lr * gradient

    def predict_proba(self, features: List[float]) -> np.ndarray:
        assert self.mean is not None and self.scale is not None and self.weights is not None
        x = (np.asarray(features, dtype=float) - self.mean) / self.scale
        x_aug = np.hstack([1.0, x])
        logits = x_aug @ self.weights
        logits -= logits.max()
        exp_logits = np.exp(np.clip(logits, -30, 30))
        return exp_logits / exp_logits.sum()

    def predict_label(self, features: List[float]) -> str:
        probabilities = self.predict_proba(features)
        return self.labels[int(np.argmax(probabilities))]

    def feature_importance(self) -> List[Tuple[str, float]]:
        assert self.weights is not None
        coeffs = np.abs(self.weights[1:, :]).sum(axis=1)
        return sorted(zip(FEATURE_NAMES, coeffs.tolist()), key=lambda item: item[1], reverse=True)

    def class_feature_importance(self) -> List[Dict[str, float | str]]:
        assert self.weights is not None
        rows: List[Dict[str, float | str]] = []
        for class_index, class_name in enumerate(self.labels):
            for feature_index, feature_name in enumerate(FEATURE_NAMES):
                coefficient = float(self.weights[feature_index + 1, class_index])
                rows.append(
                    {
                        "class": class_name,
                        "feature": feature_name,
                        "coefficient": coefficient,
                        "abs_coefficient": abs(coefficient),
                    }
                )
        return rows
