from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Dict, List, Optional, Sequence, Tuple

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

    def balanced_accuracy(self) -> float:
        return self.macro_recall()

    def class_supports(self) -> Dict[str, int]:
        return {label: self.support(label) for label in self.labels}

    def average_latency(self) -> float:
        return mean(self.latencies) if self.latencies else 0.0


class MulticlassLogisticRegressionModel:
    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.weights: np.ndarray | None = None
        self.labels: Tuple[str, ...] = CLASS_LABELS

    def fit(
        self,
        x_rows: List[List[float]],
        y_rows: List[int],
        lr: float = 0.1,
        epochs: int = 1800,
        reg: float = 1e-3,
        class_weight_mode: Optional[str] = None,
    ) -> None:
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
        sample_weights = np.ones(len(y), dtype=float)
        if class_weight_mode == "balanced":
            class_counts = np.bincount(y, minlength=num_classes)
            class_weights = np.ones(num_classes, dtype=float)
            nonzero = class_counts > 0
            class_weights[nonzero] = len(y) / (num_classes * class_counts[nonzero])
            sample_weights = class_weights[y]
        normalizer = float(sample_weights.sum()) if sample_weights.size else 1.0
        for _ in range(epochs):
            logits = x_aug @ self.weights
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(np.clip(logits, -30, 30))
            preds = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            residual = (preds - y_onehot) * sample_weights[:, None]
            gradient = (x_aug.T @ residual) / normalizer
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


class GaussianNaiveBayesModel:
    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.class_means: np.ndarray | None = None
        self.class_vars: np.ndarray | None = None
        self.log_priors: np.ndarray | None = None
        self.labels: Tuple[str, ...] = CLASS_LABELS

    def fit(self, x_rows: List[List[float]], y_rows: List[int]) -> None:
        x = np.asarray(x_rows, dtype=float)
        y = np.asarray(y_rows, dtype=int)
        self.mean = x.mean(axis=0)
        self.scale = x.std(axis=0)
        self.scale[self.scale == 0] = 1.0
        x_scaled = (x - self.mean) / self.scale
        class_means = []
        class_vars = []
        log_priors = []
        for class_index in range(len(self.labels)):
            class_rows = x_scaled[y == class_index]
            if len(class_rows) == 0:
                class_means.append(np.zeros(x_scaled.shape[1], dtype=float))
                class_vars.append(np.ones(x_scaled.shape[1], dtype=float))
                log_priors.append(np.log(1e-12))
                continue
            class_means.append(class_rows.mean(axis=0))
            class_vars.append(class_rows.var(axis=0) + 1e-6)
            log_priors.append(np.log(len(class_rows) / len(y)))
        self.class_means = np.asarray(class_means, dtype=float)
        self.class_vars = np.asarray(class_vars, dtype=float)
        self.log_priors = np.asarray(log_priors, dtype=float)

    def predict_proba(self, features: List[float]) -> np.ndarray:
        assert self.mean is not None and self.scale is not None
        assert self.class_means is not None and self.class_vars is not None and self.log_priors is not None
        x = (np.asarray(features, dtype=float) - self.mean) / self.scale
        log_probs = self.log_priors.copy()
        log_probs -= 0.5 * np.sum(np.log(2.0 * np.pi * self.class_vars), axis=1)
        log_probs -= 0.5 * np.sum(((x - self.class_means) ** 2) / self.class_vars, axis=1)
        log_probs -= float(np.max(log_probs))
        probs = np.exp(np.clip(log_probs, -60.0, 60.0))
        total = float(probs.sum())
        if total <= 0.0:
            probs = np.ones(len(self.labels), dtype=float)
            total = float(probs.sum())
        return probs / total

    def predict_label(self, features: List[float]) -> str:
        probabilities = self.predict_proba(features)
        return self.labels[int(np.argmax(probabilities))]
