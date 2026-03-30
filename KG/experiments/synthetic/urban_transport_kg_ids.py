from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Sequence, Tuple
import csv
import json
import random
import matplotlib
matplotlib.use("Agg")
import sys
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from experiments.common.project_paths import SYNTHETIC_RESULTS_DIR, ensure_pipeline_directories
from rendering.scripts.paper_asset_renderers import render_synthetic_assets

BASELINE_COLOR = "#b8c4d6"
LEARNING_COLOR = "#4c956c"
KG_COLOR = "#1f5a91"
ABLATION_COLORS = ["#1f5a91", "#6c8ebf", "#95b8d1", "#c6d8e6"]
GRID_COLOR = "#d6dde6"
TEXT_COLOR = "#18324a"


SEGMENT_TO_RSU = {
    "S1": "R1",
    "S2": "R1",
    "S3": "R2",
    "S4": "R2",
    "S5": "R3",
    "S6": "R3",
}

SEGMENT_TO_INTERSECTION = {
    "S1": "I1",
    "S2": "I1",
    "S3": "I2",
    "S4": "I2",
    "S5": "I3",
    "S6": "I3",
}

ADJACENT_SEGMENTS = {
    "S1": {"S2", "S3"},
    "S2": {"S1", "S4"},
    "S3": {"S1", "S5"},
    "S4": {"S2", "S6"},
    "S5": {"S3", "S6"},
    "S6": {"S4", "S5"},
}

ROUTES = [
    ["S1", "S3", "S5"],
    ["S2", "S4", "S6"],
    ["S1", "S2", "S4"],
    ["S3", "S5", "S6"],
    ["S2", "S1", "S3"],
    ["S4", "S6", "S5"],
]

SIGNAL_CYCLE = ("green", "green", "yellow", "red", "red", "red")
ATTACK_TYPES = ("phantom_congestion", "signal_spoofing", "false_closure", "position_spoofing")
CLASS_LABELS = ("benign",) + ATTACK_TYPES
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
class Vehicle:
    vehicle_id: str
    route: List[str]
    route_index: int = 0

    @property
    def current_segment(self) -> str:
        return self.route[self.route_index]

    def move(self) -> None:
        if self.route_index < len(self.route) - 1:
            self.route_index += 1


@dataclass
class Message:
    message_id: str
    tick: int
    sender_id: str
    sender_type: str
    claim_type: str
    malicious: bool = False
    attack_type: Optional[str] = None
    segment: Optional[str] = None
    intersection: Optional[str] = None
    signal_state: Optional[str] = None
    density: Optional[int] = None
    closed: Optional[bool] = None
    claimed_vehicle: Optional[str] = None


@dataclass
class DetectionStats:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    latencies: List[int] = field(default_factory=list)

    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    def f1(self) -> float:
        p = self.precision()
        r = self.recall()
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def false_positive_rate(self) -> float:
        denom = self.fp + self.tn
        return self.fp / denom if denom else 0.0

    def average_latency(self) -> float:
        return mean(self.latencies) if self.latencies else 0.0


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


def message_label(message: Message) -> Optional[str]:
    if message.claim_type not in CANDIDATE_CLAIM_TYPES:
        return None
    return message.attack_type if message.malicious and message.attack_type else "benign"


class BaselineDetector:
    def __init__(self) -> None:
        self.rsu_density: Dict[str, int] = {}
        self.controller_signal: Dict[str, str] = {}
        self.vehicle_position: Dict[str, str] = {}

    def predict_label(self, message: Message) -> Optional[str]:
        if message.claim_type == "rsu_density":
            self.rsu_density[message.segment or ""] = int(message.density or 0)
            return None
        if message.claim_type == "controller_signal":
            self.controller_signal[message.intersection or ""] = message.signal_state or "unknown"
            return None

        if message.claim_type == "congestion_alert":
            density = self.rsu_density.get(message.segment or "", 0)
            return "phantom_congestion" if (message.density or 0) - density >= 5 else "benign"

        if message.claim_type == "closure_alert":
            density = self.rsu_density.get(message.segment or "", 0)
            return "false_closure" if (message.closed and density >= 4) else "benign"

        if message.claim_type == "signal_report":
            if message.sender_type == "vehicle":
                controller = self.controller_signal.get(message.intersection or "", "unknown")
                return "signal_spoofing" if controller != "unknown" and controller != message.signal_state else "benign"
            return "benign"

        if message.claim_type == "vehicle_position":
            claimed_vehicle = message.claimed_vehicle or message.sender_id
            previous = self.vehicle_position.get(claimed_vehicle)
            current = message.segment or ""
            if previous and current not in ADJACENT_SEGMENTS.get(previous, set()) and current != previous:
                prediction = "position_spoofing"
            else:
                prediction = "benign"
            if not message.malicious:
                self.vehicle_position[message.sender_id] = current
            else:
                self.vehicle_position[claimed_vehicle] = current
            return prediction

        return None

    def observe(self, message: Message) -> bool:
        prediction = self.predict_label(message)
        return prediction not in (None, "benign")


class GraphBackedKGState:
    def __init__(self) -> None:
        self.graph = nx.DiGraph()
        self._build_static_graph()

    def _build_static_graph(self) -> None:
        for segment, rsu in SEGMENT_TO_RSU.items():
            intersection = SEGMENT_TO_INTERSECTION[segment]
            controller = f"C_{intersection}"
            self.graph.add_node(segment, kind="segment", current_density=0)
            self.graph.add_node(rsu, kind="rsu")
            self.graph.add_node(intersection, kind="intersection", controller_signal="unknown", signal_counts={})
            self.graph.add_node(controller, kind="controller")
            self.graph.add_edge(segment, rsu, relation="monitored_by")
            self.graph.add_edge(intersection, controller, relation="controlled_by")

        for segment, neighbors in ADJACENT_SEGMENTS.items():
            for neighbor in neighbors:
                self.graph.add_edge(segment, neighbor, relation="adjacent_to")

    def _ensure_vehicle(self, vehicle_id: str) -> None:
        if vehicle_id and vehicle_id not in self.graph:
            self.graph.add_node(vehicle_id, kind="vehicle")

    def update_rsu_density(self, segment: str, density: int) -> None:
        if segment in self.graph:
            self.graph.nodes[segment]["current_density"] = int(density)

    def update_controller_signal(self, intersection: str, signal_state: str) -> None:
        if intersection in self.graph:
            self.graph.nodes[intersection]["controller_signal"] = signal_state

    def update_vehicle_position(self, vehicle_id: str, segment: str) -> None:
        self._ensure_vehicle(vehicle_id)
        if vehicle_id in self.graph and segment in self.graph:
            self.graph.nodes[vehicle_id]["current_segment"] = segment
            stale_edges = [
                (src, dst)
                for src, dst, data in self.graph.out_edges(vehicle_id, data=True)
                if data.get("relation") == "located_on"
            ]
            self.graph.remove_edges_from(stale_edges)
            self.graph.add_edge(vehicle_id, segment, relation="located_on")

    def update_signal_observation(self, vehicle_id: str, intersection: str, signal_state: str) -> None:
        self._ensure_vehicle(vehicle_id)
        if not intersection:
            return
        counts = dict(self.graph.nodes[intersection].get("signal_counts", {}))
        counts[signal_state] = counts.get(signal_state, 0) + 1
        self.graph.nodes[intersection]["signal_counts"] = counts
        stale_edges = [
            (src, dst)
            for src, dst, data in self.graph.out_edges(vehicle_id, data=True)
            if data.get("relation") == "observed_signal" and dst == intersection
        ]
        self.graph.remove_edges_from(stale_edges)
        self.graph.add_edge(vehicle_id, intersection, relation="observed_signal", signal_state=signal_state)

    def segment_density(self, segment: str) -> int:
        return int(self.graph.nodes.get(segment, {}).get("current_density", 0))

    def controller_signal(self, intersection: str) -> Optional[str]:
        value = self.graph.nodes.get(intersection, {}).get("controller_signal")
        return value if value != "unknown" else None

    def vehicle_segment(self, vehicle_id: str) -> Optional[str]:
        return self.graph.nodes.get(vehicle_id, {}).get("current_segment")

    def adjacent_segments(self, segment: str) -> set[str]:
        if segment not in self.graph:
            return set()
        return {
            neighbor
            for neighbor in self.graph.successors(segment)
            if self.graph.edges[segment, neighbor].get("relation") == "adjacent_to"
        }

    def majority_signal(self, intersection: str) -> Optional[str]:
        counts = self.graph.nodes.get(intersection, {}).get("signal_counts", {})
        if not counts:
            return None
        return max(counts.items(), key=lambda item: item[1])[0]


class KnowledgeGraphDetector:
    def __init__(self, use_topology: bool = True, use_rsu_context: bool = True, use_crowd_context: bool = True) -> None:
        self.use_topology = use_topology
        self.use_rsu_context = use_rsu_context
        self.use_crowd_context = use_crowd_context
        self.state = GraphBackedKGState()

    def predict_label(self, message: Message) -> Optional[str]:
        if message.claim_type == "rsu_density":
            self.state.update_rsu_density(message.segment or "", int(message.density or 0))
            return None

        if message.claim_type == "controller_signal":
            self.state.update_controller_signal(message.intersection or "", message.signal_state or "unknown")
            return None

        if message.claim_type == "vehicle_position" and not message.malicious:
            self.state.update_vehicle_position(message.sender_id, message.segment or "")
            return "benign"

        if message.claim_type == "signal_observation":
            self.state.update_signal_observation(message.sender_id, message.intersection or "", message.signal_state or "unknown")
            return None

        if message.claim_type == "congestion_alert":
            return "phantom_congestion" if self._detect_phantom_congestion(message) else "benign"

        if message.claim_type == "closure_alert":
            return "false_closure" if self._detect_false_closure(message) else "benign"

        if message.claim_type == "signal_report":
            return "signal_spoofing" if self._detect_signal_spoofing(message) else "benign"

        if message.claim_type == "vehicle_position":
            prediction = "position_spoofing" if self._detect_position_spoofing(message) else "benign"
            claimed_vehicle = message.claimed_vehicle or message.sender_id
            self.state.update_vehicle_position(claimed_vehicle, message.segment or "")
            return prediction

        return None

    def observe(self, message: Message) -> bool:
        prediction = self.predict_label(message)
        return prediction not in (None, "benign")

    def _majority_signal(self, intersection: str) -> Optional[str]:
        return self.state.majority_signal(intersection)

    def _detect_phantom_congestion(self, message: Message) -> bool:
        score = 0
        target_segment = message.segment or ""
        rsu_density = self.state.segment_density(target_segment)
        sender_segment = self.state.vehicle_segment(message.sender_id)

        if self.use_rsu_context and (message.density or 0) - rsu_density >= 2:
            score += 1
        if self.use_topology and sender_segment not in {target_segment, *self.state.adjacent_segments(target_segment)}:
            score += 1
        if self.use_rsu_context and rsu_density <= 2:
            score += 1
        return score >= 2

    def _detect_false_closure(self, message: Message) -> bool:
        score = 0
        target_segment = message.segment or ""
        rsu_density = self.state.segment_density(target_segment)
        adjacent_flow = any(self.state.segment_density(seg) > 0 for seg in self.state.adjacent_segments(target_segment))

        if message.closed:
            score += 1
        if self.use_rsu_context and rsu_density > 0:
            score += 1
        if self.use_topology and self.use_rsu_context and adjacent_flow:
            score += 1
        return score >= 2

    def _detect_signal_spoofing(self, message: Message) -> bool:
        score = 0
        intersection = message.intersection or ""
        controller = self.state.controller_signal(intersection)
        majority = self._majority_signal(intersection)

        if controller and controller != message.signal_state:
            score += 1
        if self.use_crowd_context and majority and majority != message.signal_state:
            score += 1
        if message.sender_type == "rsu":
            score += 1
        return score >= 2

    def _detect_position_spoofing(self, message: Message) -> bool:
        claimed_vehicle = message.claimed_vehicle or ""
        previous = self.state.vehicle_segment(claimed_vehicle)
        current = message.segment or ""
        score = 0

        if self.use_topology and previous and current not in self.state.adjacent_segments(previous) and current != previous:
            score += 1
        if self.use_rsu_context and SEGMENT_TO_RSU.get(previous or "", "") != SEGMENT_TO_RSU.get(current, ""):
            score += 1
        return score >= 2

class FlatFeatureContext:
    def __init__(self) -> None:
        self.rsu_density: Dict[str, int] = {}
        self.controller_signal: Dict[str, str] = {}
        self.vehicle_position: Dict[str, str] = {}
        self.vehicle_signal_observations: Dict[str, Dict[str, int]] = {}

    def _majority_signal(self, intersection: str) -> Optional[str]:
        counts = self.vehicle_signal_observations.get(intersection, {})
        if not counts:
            return None
        return max(counts.items(), key=lambda item: item[1])[0]

    def features_for(self, message: Message) -> Optional[List[float]]:
        if message.claim_type not in {"congestion_alert", "closure_alert", "signal_report", "vehicle_position"}:
            return None

        target_segment = message.segment or ""
        rsu_density = float(self.rsu_density.get(target_segment, 0))
        sender_segment = self.vehicle_position.get(message.sender_id)
        neighborhood = {target_segment, *ADJACENT_SEGMENTS.get(target_segment, set())}
        sender_in_neighborhood = 1.0 if sender_segment in neighborhood else 0.0
        density_gap = float((message.density or 0) - rsu_density)
        adjacent_flow = 1.0 if any(self.rsu_density.get(seg, 0) > 0 for seg in ADJACENT_SEGMENTS.get(target_segment, set())) else 0.0
        controller_mismatch = 0.0
        majority_mismatch = 0.0
        if message.intersection:
            controller = self.controller_signal.get(message.intersection, "unknown")
            majority = self._majority_signal(message.intersection)
            controller_mismatch = 1.0 if controller != "unknown" and controller != message.signal_state else 0.0
            majority_mismatch = 1.0 if majority and majority != message.signal_state else 0.0
        previous = self.vehicle_position.get(message.claimed_vehicle or "")
        adjacency_violation = 1.0 if previous and target_segment not in ADJACENT_SEGMENTS.get(previous, set()) and target_segment != previous else 0.0
        rsu_region_change = 1.0 if SEGMENT_TO_RSU.get(previous or "", "") != SEGMENT_TO_RSU.get(target_segment, "") else 0.0
        return [
            1.0 if message.claim_type == "congestion_alert" else 0.0,
            1.0 if message.claim_type == "closure_alert" else 0.0,
            1.0 if message.claim_type == "signal_report" else 0.0,
            1.0 if message.claim_type == "vehicle_position" else 0.0,
            1.0 if message.sender_type == "rsu" else 0.0,
            density_gap,
            rsu_density,
            sender_in_neighborhood,
            float(message.closed or 0),
            adjacent_flow,
            controller_mismatch,
            majority_mismatch,
            adjacency_violation,
            rsu_region_change,
        ]

    def update(self, message: Message) -> None:
        if message.claim_type == "rsu_density":
            self.rsu_density[message.segment or ""] = int(message.density or 0)
        elif message.claim_type == "controller_signal":
            self.controller_signal[message.intersection or ""] = message.signal_state or "unknown"
        elif message.claim_type == "vehicle_position" and not message.malicious:
            self.vehicle_position[message.sender_id] = message.segment or ""
        elif message.claim_type == "signal_observation":
            intersection = message.intersection or ""
            observed = self.vehicle_signal_observations.setdefault(intersection, {})
            state = message.signal_state or "unknown"
            observed[state] = observed.get(state, 0) + 1


class MulticlassLogisticRegressionModel:
    def __init__(self) -> None:
        self.mean: Optional[np.ndarray] = None
        self.scale: Optional[np.ndarray] = None
        self.weights: Optional[np.ndarray] = None
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
        ranking = sorted(zip(FEATURE_NAMES, coeffs.tolist()), key=lambda item: item[1], reverse=True)
        return ranking

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


class FlatFeatureLogisticDetector:
    def __init__(self, model: MulticlassLogisticRegressionModel) -> None:
        self.model = model
        self.context = FlatFeatureContext()

    def predict_label(self, message: Message) -> Optional[str]:
        features = self.context.features_for(message)
        prediction: Optional[str] = None
        if features is not None:
            prediction = self.model.predict_label(features)
        self.context.update(message)
        return prediction

    def observe(self, message: Message) -> bool:
        prediction = self.predict_label(message)
        return prediction not in (None, "benign")


def split_episodes(episodes: Sequence[Tuple[List[Message], Optional[str], int]], train_ratio: float = 0.7):
    grouped: Dict[Optional[str], List[Tuple[List[Message], Optional[str], int]]] = {}
    for item in episodes:
        grouped.setdefault(item[1], []).append(item)
    train, test = [], []
    for attack_type, items in grouped.items():
        cutoff = max(1, int(len(items) * train_ratio))
        train.extend(items[:cutoff])
        test.extend(items[cutoff:])
    return train, test


def build_learning_dataset(episodes: Sequence[Tuple[List[Message], Optional[str], int]]) -> Tuple[List[List[float]], List[int]]:
    x_rows: List[List[float]] = []
    y_rows: List[int] = []
    for messages, _, _ in episodes:
        context = FlatFeatureContext()
        for message in messages:
            features = context.features_for(message)
            label = message_label(message)
            if features is not None and label is not None:
                # Learn on flat message features without explicit graph structure.
                x_rows.append(features)
                y_rows.append(CLASS_LABELS.index(label))
            context.update(message)
    return x_rows, y_rows


class UrbanTransportExperiment:
    def __init__(self, seed: int) -> None:
        self.random = random.Random(seed)
        self.message_counter = 0

    def _next_message_id(self) -> str:
        self.message_counter += 1
        return f"m{self.message_counter:06d}"

    def _signal_state(self, tick: int, intersection: str) -> str:
        offset = {"I1": 0, "I2": 1, "I3": 2}[intersection]
        return SIGNAL_CYCLE[(tick + offset) % len(SIGNAL_CYCLE)]

    def _spawn_vehicles(self) -> List[Vehicle]:
        vehicles: List[Vehicle] = []
        for idx in range(12):
            route = list(self.random.choice(ROUTES))
            vehicles.append(Vehicle(vehicle_id=f"V{idx + 1}", route=route))
        return vehicles

    def _legitimate_messages(self, tick: int, vehicles: Sequence[Vehicle], occupancies: Dict[str, int]) -> List[Message]:
        messages: List[Message] = []

        for segment, density in occupancies.items():
            messages.append(
                Message(
                    message_id=self._next_message_id(),
                    tick=tick,
                    sender_id=SEGMENT_TO_RSU[segment],
                    sender_type="rsu",
                    claim_type="rsu_density",
                    segment=segment,
                    density=density,
                )
            )

        for intersection in ("I1", "I2", "I3"):
            messages.append(
                Message(
                    message_id=self._next_message_id(),
                    tick=tick,
                    sender_id=f"C_{intersection}",
                    sender_type="controller",
                    claim_type="controller_signal",
                    intersection=intersection,
                    signal_state=self._signal_state(tick, intersection),
                )
            )

        for vehicle in vehicles:
            segment = vehicle.current_segment
            intersection = SEGMENT_TO_INTERSECTION[segment]
            signal_state = self._signal_state(tick, intersection)
            messages.append(
                Message(
                    message_id=self._next_message_id(),
                    tick=tick,
                    sender_id=vehicle.vehicle_id,
                    sender_type="vehicle",
                    claim_type="vehicle_position",
                    segment=segment,
                )
            )
            messages.append(
                Message(
                    message_id=self._next_message_id(),
                    tick=tick,
                    sender_id=vehicle.vehicle_id,
                    sender_type="vehicle",
                    claim_type="signal_observation",
                    intersection=intersection,
                    signal_state=signal_state,
                )
            )

        return messages

    def _attack_message(self, tick: int, vehicles: Sequence[Vehicle], occupancies: Dict[str, int], attack_type: str) -> Message:
        attacker = vehicles[0]
        low_density_segments = [seg for seg, density in occupancies.items() if density <= 2]
        target_segment = low_density_segments[0] if low_density_segments else "S6"

        if attack_type == "phantom_congestion":
            return Message(
                message_id=self._next_message_id(),
                tick=tick,
                sender_id=attacker.vehicle_id,
                sender_type="vehicle",
                claim_type="congestion_alert",
                malicious=True,
                attack_type=attack_type,
                segment=target_segment,
                density=4,
            )

        if attack_type == "signal_spoofing":
            target_intersection = "I2"
            controller = self._signal_state(tick, target_intersection)
            forged = "red" if controller == "green" else "green"
            return Message(
                message_id=self._next_message_id(),
                tick=tick,
                sender_id="R2",
                sender_type="rsu",
                claim_type="signal_report",
                malicious=True,
                attack_type=attack_type,
                intersection=target_intersection,
                signal_state=forged,
            )

        if attack_type == "false_closure":
            return Message(
                message_id=self._next_message_id(),
                tick=tick,
                sender_id=attacker.vehicle_id,
                sender_type="vehicle",
                claim_type="closure_alert",
                malicious=True,
                attack_type=attack_type,
                segment="S5",
                closed=True,
            )

        spoof_segment = "S6" if attacker.current_segment != "S6" else "S1"
        return Message(
            message_id=self._next_message_id(),
            tick=tick,
            sender_id=attacker.vehicle_id,
            sender_type="vehicle",
            claim_type="vehicle_position",
            malicious=True,
            attack_type=attack_type,
            segment=spoof_segment,
            claimed_vehicle=attacker.vehicle_id,
        )

    def run_episode(self, attack_type: Optional[str], ticks: int = 18) -> Tuple[List[Message], int]:
        vehicles = self._spawn_vehicles()
        attack_start = 6
        all_messages: List[Message] = []

        for tick in range(ticks):
            if tick and tick % 3 == 0:
                for vehicle in vehicles:
                    if self.random.random() < 0.65:
                        vehicle.move()

            occupancies = {segment: 0 for segment in SEGMENT_TO_RSU}
            for vehicle in vehicles:
                occupancies[vehicle.current_segment] += 1

            all_messages.extend(self._legitimate_messages(tick, vehicles, occupancies))

            if attack_type and attack_start <= tick < attack_start + 4:
                all_messages.append(self._attack_message(tick, vehicles, occupancies, attack_type))

        return all_messages, attack_start


def evaluate_detector(detector_name: str, detector_factory, episodes: Sequence[Tuple[List[Message], Optional[str], int]]) -> Dict[str, object]:
    stats = MulticlassStats(CLASS_LABELS)
    per_attack: Dict[str, MulticlassStats] = {attack: MulticlassStats(CLASS_LABELS) for attack in ATTACK_TYPES}

    for messages, attack_type, attack_start in episodes:
        detector = detector_factory()
        first_detection_tick: Optional[int] = None

        for message in messages:
            prediction = detector.predict_label(message)
            true_label = message_label(message)
            if true_label is None or prediction is None:
                continue
            stats.add(true_label, prediction)
            if attack_type:
                per_attack[attack_type].add(true_label, prediction)
            if message.malicious and prediction == true_label and first_detection_tick is None:
                first_detection_tick = message.tick

        if attack_type and first_detection_tick is not None:
            latency = first_detection_tick - attack_start
            stats.latencies.append(latency)
            per_attack[attack_type].latencies.append(latency)

    summary = {
        "detector": detector_name,
        "accuracy": round(stats.accuracy(), 4),
        "macro_precision": round(stats.macro_precision(), 4),
        "macro_recall": round(stats.macro_recall(), 4),
        "macro_f1": round(stats.macro_f1(), 4),
        "average_latency": round(stats.average_latency(), 4),
        "confusion_matrix": stats.confusion,
        "per_attack": {},
    }

    for attack, attack_stats in per_attack.items():
        summary["per_attack"][attack] = {
            "precision": round(attack_stats.precision(attack), 4),
            "recall": round(attack_stats.recall(attack), 4),
            "f1": round(attack_stats.f1(attack), 4),
            "average_latency": round(attack_stats.average_latency(), 4),
        }

    return summary


def run_full_experiment(seed: int = 42, episodes_per_case: int = 25) -> Dict[str, object]:
    simulator = UrbanTransportExperiment(seed=seed)
    episodes: List[Tuple[List[Message], Optional[str], int]] = []

    for attack_type in ATTACK_TYPES:
        for _ in range(episodes_per_case):
            messages, attack_start = simulator.run_episode(attack_type)
            episodes.append((messages, attack_type, attack_start))
        for _ in range(max(5, episodes_per_case // 5)):
            messages, attack_start = simulator.run_episode(None)
            episodes.append((messages, None, attack_start))

    train_episodes, test_episodes = split_episodes(episodes)
    x_train, y_train = build_learning_dataset(train_episodes)
    learning_model = MulticlassLogisticRegressionModel()
    learning_model.fit(x_train, y_train)

    return {
        "seed": seed,
        "episodes_per_case": episodes_per_case,
        "train_episodes": len(train_episodes),
        "test_episodes": len(test_episodes),
        "baseline": evaluate_detector("baseline", BaselineDetector, test_episodes),
        "flat_feature_logistic": evaluate_detector(
            "flat_feature_logistic",
            lambda: FlatFeatureLogisticDetector(learning_model),
            test_episodes,
        ),
        "knowledge_graph": evaluate_detector("knowledge_graph", KnowledgeGraphDetector, test_episodes),
        "ablations": {
            "full_kg": evaluate_detector("full_kg", KnowledgeGraphDetector, test_episodes),
            "kg_no_topology": evaluate_detector(
                "kg_no_topology",
                lambda: KnowledgeGraphDetector(use_topology=False),
                test_episodes,
            ),
            "kg_no_rsu": evaluate_detector(
                "kg_no_rsu",
                lambda: KnowledgeGraphDetector(use_rsu_context=False),
                test_episodes,
            ),
            "kg_no_crowd": evaluate_detector(
                "kg_no_crowd",
                lambda: KnowledgeGraphDetector(use_crowd_context=False),
                test_episodes,
            ),
        },
        "logistic_feature_importance": [
            {"feature": feature, "importance": round(value, 6)}
            for feature, value in learning_model.feature_importance()
        ],
    }


def write_outputs(results: Dict[str, object], output_dir: Path) -> None:
    ensure_pipeline_directories()
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    summary_rows = []
    for detector_name in ("baseline", "flat_feature_logistic", "knowledge_graph"):
        detector = results[detector_name]
        summary_rows.append(
            {
                "detector": detector["detector"],
                "accuracy": detector["accuracy"],
                "macro_precision": detector["macro_precision"],
                "macro_recall": detector["macro_recall"],
                "macro_f1": detector["macro_f1"],
                "average_latency": detector["average_latency"],
            }
        )

    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    confusion_rows = []
    for detector_name in ("baseline", "flat_feature_logistic", "knowledge_graph"):
        matrix = results[detector_name]["confusion_matrix"]
        for true_label in CLASS_LABELS:
            for pred_label in CLASS_LABELS:
                confusion_rows.append(
                    {
                        "detector": detector_name,
                        "true_label": true_label,
                        "pred_label": pred_label,
                        "count": matrix[true_label][pred_label],
                    }
                )
    with (output_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(confusion_rows[0].keys()))
        writer.writeheader()
        writer.writerows(confusion_rows)

    scenario_rows = []
    for detector_name in ("baseline", "flat_feature_logistic", "knowledge_graph"):
        per_attack = results[detector_name]["per_attack"]
        for attack_type, metrics in per_attack.items():
            scenario_rows.append(
                {
                    "detector": detector_name,
                    "attack_type": attack_type,
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "average_latency": metrics["average_latency"],
                }
            )

    with (output_dir / "scenario_breakdown.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scenario_rows[0].keys()))
        writer.writeheader()
        writer.writerows(scenario_rows)

    with (output_dir / "logistic_feature_importance.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["feature", "importance"])
        writer.writeheader()
        writer.writerows(results["logistic_feature_importance"])

    full_kg_f1 = results["ablations"]["full_kg"]["macro_f1"]
    ablation_rows = []
    for key, metrics in results["ablations"].items():
        relative_degradation = 0.0
        if key != "full_kg" and full_kg_f1:
            relative_degradation = max(0.0, (full_kg_f1 - metrics["macro_f1"]) / full_kg_f1 * 100.0)
        ablation_rows.append(
            {
                "variant": key,
                "macro_recall": metrics["macro_recall"],
                "macro_f1": metrics["macro_f1"],
                "relative_f1_degradation_pct": round(relative_degradation, 2),
                "latency": metrics["average_latency"],
            }
        )
    with (output_dir / "ablation_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ablation_rows[0].keys()))
        writer.writeheader()
        writer.writerows(ablation_rows)

    render_synthetic_assets(PROJECT_ROOT)


def _style_axes(ax) -> None:
    ax.set_facecolor("white")
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9eb0c3")
    ax.spines["bottom"].set_color("#9eb0c3")
    ax.tick_params(colors=TEXT_COLOR)


def _plot_overall_performance(summary_rows: List[Dict[str, object]], output_path: Path) -> None:
    metrics = ["precision", "recall", "f1", "false_positive_rate", "average_latency"]
    labels = ["Precision", "Recall", "F1", "FPR", "Latency"]
    detectors = [
        ("Baseline", BASELINE_COLOR, next(row for row in summary_rows if row["detector"] == "baseline")),
        ("Flat-feature logistic", LEARNING_COLOR, next(row for row in summary_rows if row["detector"] == "flat_feature_logistic")),
        ("KG-based", KG_COLOR, next(row for row in summary_rows if row["detector"] == "knowledge_graph")),
    ]

    x = range(len(metrics))
    width = 0.24
    fig, ax = plt.subplots(figsize=(8.6, 4.8), facecolor="white")
    _style_axes(ax)
    offsets = [-width, 0.0, width]
    for offset, (label, color, row) in zip(offsets, detectors):
        ax.bar([i + offset for i in x], [row[m] for m in metrics], width=width, color=color, edgecolor="white", label=label)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, color=TEXT_COLOR)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Metric value", color=TEXT_COLOR)
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.02), borderaxespad=0.0)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_per_attack_recall(results: Dict[str, object], output_path: Path) -> None:
    attack_order = list(ATTACK_TYPES)
    attack_labels = ["Phantom\ncongestion", "Signal\nspoofing", "False\nclosure", "Position\nspoofing"]
    baseline = results["baseline"]["per_attack"]
    learning = results["flat_feature_logistic"]["per_attack"]
    kg = results["knowledge_graph"]["per_attack"]

    x = range(len(attack_order))
    width = 0.24
    fig, ax = plt.subplots(figsize=(8.8, 4.9), facecolor="white")
    _style_axes(ax)
    ax.bar([i - width for i in x], [baseline[a]["recall"] for a in attack_order], width=width, color=BASELINE_COLOR, edgecolor="white", label="Baseline")
    ax.bar([i for i in x], [learning[a]["recall"] for a in attack_order], width=width, color=LEARNING_COLOR, edgecolor="white", label="Flat-feature logistic")
    ax.bar([i + width for i in x], [kg[a]["recall"] for a in attack_order], width=width, color=KG_COLOR, edgecolor="white", label="KG-based")
    ax.set_xticks(list(x))
    ax.set_xticklabels(attack_labels, color=TEXT_COLOR)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Recall", color=TEXT_COLOR)
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.02), borderaxespad=0.0)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_logistic_feature_importance(feature_rows: List[Dict[str, float]], output_path: Path) -> None:
    top_rows = feature_rows[:8]
    labels = [row["feature"] for row in reversed(top_rows)]
    values = [row["importance"] for row in reversed(top_rows)]
    fig, ax = plt.subplots(figsize=(8.0, 4.8), facecolor="white")
    _style_axes(ax)
    ax.barh(range(len(labels)), values, color=LEARNING_COLOR, edgecolor="white")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, color=TEXT_COLOR)
    ax.set_xlabel(r"Absolute coefficient magnitude $|w_j|$", color=TEXT_COLOR)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_ablation(ablations: Dict[str, Dict[str, object]], output_path: Path) -> None:
    labels = ["No topology", "No RSU", "No crowd"]
    keys = ["kg_no_topology", "kg_no_rsu", "kg_no_crowd"]
    full_f1 = ablations["full_kg"]["f1"]
    values = [max(0.0, (full_f1 - ablations[key]["f1"]) / full_f1 * 100.0) for key in keys]
    fig, ax = plt.subplots(figsize=(7.4, 4.6), facecolor="white")
    _style_axes(ax)
    ax.bar(range(len(labels)), values, color=ABLATION_COLORS, edgecolor="white")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, color=TEXT_COLOR)
    ax.set_ylim(0, max(values) * 1.2 if values else 1.0)
    ax.set_ylabel("Relative F1 degradation (%)", color=TEXT_COLOR)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
