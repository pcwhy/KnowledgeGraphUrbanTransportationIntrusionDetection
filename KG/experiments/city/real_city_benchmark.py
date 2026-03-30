from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Sequence, Tuple
import csv
import json
import math
import random
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests

import networkx as nx
import osmnx as ox

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from experiments.common.detection_shared import (
    ATTACK_TYPES,
    CANDIDATE_CLAIM_TYPES,
    CLASS_LABELS,
    MulticlassLogisticRegressionModel,
    MulticlassStats,
)
from experiments.common.project_paths import CITY_CACHE_DIR, CITY_RESULTS_DIR, PAPER_FIGURES_DIR, ensure_pipeline_directories
from experiments.city.real_city_case_study import annotate_edges_with_aadt, download_txdot_aadt
from rendering.scripts.paper_asset_renderers import render_city_assets


BASELINE_COLOR = "#b8c4d6"
KG_COLOR = "#1f5a91"
GRID_COLOR = "#d6dde6"
TEXT_COLOR = "#18324a"
GRID_SIZE = 10
ROUTE_LIBRARY_CAP = 800
EPISODE_VEHICLES = 500
EPISODES_PER_ATTACK = 8
TRAIN_EPISODES_PER_ATTACK = 5
ATTACK_START_TICK = 6
ATTACK_WINDOW_TICKS = 4
ATTACKER_SHARE = 0.02

ATTACK_GROUP = {
    "phantom_congestion": "intent_hiding",
    "false_closure": "intent_hiding",
    "signal_spoofing": "spoofing",
    "position_spoofing": "spoofing",
}


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
class DetectionStats:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    latencies: List[int] = field(default_factory=list)

    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    def f1(self) -> float:
        p = self.precision()
        r = self.recall()
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def false_positive_rate(self) -> float:
        return self.fp / (self.fp + self.tn) if (self.fp + self.tn) else 0.0

    def average_latency(self) -> float:
        return mean(self.latencies) if self.latencies else 0.0


def message_label(message: Message) -> Optional[str]:
    if message.claim_type not in CANDIDATE_CLAIM_TYPES:
        return None
    return message.attack_type if message.malicious and message.attack_type else "benign"


class BaselineDetector:
    def __init__(self, adjacency: Dict[str, set[str]]) -> None:
        self.adjacency = adjacency
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
            if previous and current not in self.adjacency.get(previous, set()) and current != previous:
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
    def __init__(self, adjacency: Dict[str, set[str]], segment_to_rsu: Dict[str, str], segment_to_intersection: Dict[str, str]) -> None:
        self.graph = nx.DiGraph()
        self._build_static_graph(adjacency, segment_to_rsu, segment_to_intersection)

    def _build_static_graph(
        self,
        adjacency: Dict[str, set[str]],
        segment_to_rsu: Dict[str, str],
        segment_to_intersection: Dict[str, str],
    ) -> None:
        for segment, rsu in segment_to_rsu.items():
            intersection = segment_to_intersection.get(segment, "")
            controller = f"C_{intersection}" if intersection else ""
            self.graph.add_node(segment, kind="segment", current_density=0)
            self.graph.add_node(rsu, kind="rsu")
            self.graph.add_edge(segment, rsu, relation="monitored_by")
            if intersection:
                self.graph.add_node(intersection, kind="intersection", controller_signal="unknown", signal_counts={})
                self.graph.add_node(controller, kind="controller")
                self.graph.add_edge(intersection, controller, relation="controlled_by")

        for segment, neighbors in adjacency.items():
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
        if not intersection or intersection not in self.graph:
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


class KGDetector:
    def __init__(
        self,
        adjacency: Dict[str, set[str]],
        segment_to_rsu: Dict[str, str],
        segment_to_intersection: Dict[str, str],
    ) -> None:
        self.segment_to_rsu = segment_to_rsu
        self.state = GraphBackedKGState(adjacency, segment_to_rsu, segment_to_intersection)

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
            return "phantom_congestion" if self._detect_congestion(message) else "benign"
        if message.claim_type == "closure_alert":
            return "false_closure" if self._detect_closure(message) else "benign"
        if message.claim_type == "signal_report":
            return "signal_spoofing" if self._detect_signal(message) else "benign"
        if message.claim_type == "vehicle_position":
            prediction = "position_spoofing" if self._detect_position(message) else "benign"
            claimed_vehicle = message.claimed_vehicle or message.sender_id
            self.state.update_vehicle_position(claimed_vehicle, message.segment or "")
            return prediction
        return None

    def observe(self, message: Message) -> bool:
        prediction = self.predict_label(message)
        return prediction not in (None, "benign")

    def _majority_signal(self, intersection: str) -> Optional[str]:
        return self.state.majority_signal(intersection)

    def _detect_congestion(self, message: Message) -> bool:
        target = message.segment or ""
        rsu_density = self.state.segment_density(target)
        sender_segment = self.state.vehicle_segment(message.sender_id)
        neighborhood = {target, *self.state.adjacent_segments(target)}
        score = 0
        if (message.density or 0) - rsu_density >= 2:
            score += 1
        if sender_segment not in neighborhood:
            score += 1
        if rsu_density <= 2:
            score += 1
        return score >= 2

    def _detect_closure(self, message: Message) -> bool:
        target = message.segment or ""
        rsu_density = self.state.segment_density(target)
        adjacent_flow = any(self.state.segment_density(seg) > 0 for seg in self.state.adjacent_segments(target))
        score = 0
        if message.closed:
            score += 1
        if rsu_density > 0:
            score += 1
        if adjacent_flow:
            score += 1
        return score >= 2

    def _detect_signal(self, message: Message) -> bool:
        intersection = message.intersection or ""
        controller = self.state.controller_signal(intersection)
        majority = self._majority_signal(intersection)
        score = 0
        if controller and controller != message.signal_state:
            score += 1
        if majority and majority != message.signal_state:
            score += 1
        if message.sender_type == "rsu":
            score += 1
        return score >= 2

    def _detect_position(self, message: Message) -> bool:
        claimed_vehicle = message.claimed_vehicle or ""
        previous = self.state.vehicle_segment(claimed_vehicle)
        current = message.segment or ""
        score = 0
        if previous and current not in self.state.adjacent_segments(previous) and current != previous:
            score += 1
        if self.segment_to_rsu.get(previous or "") != self.segment_to_rsu.get(current):
            score += 1
        return score >= 2


class FlatFeatureContext:
    def __init__(self, adjacency: Dict[str, set[str]], segment_to_rsu: Dict[str, str]) -> None:
        self.adjacency = adjacency
        self.segment_to_rsu = segment_to_rsu
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
        neighborhood = {target_segment, *self.adjacency.get(target_segment, set())}
        sender_in_neighborhood = 1.0 if sender_segment in neighborhood else 0.0
        density_gap = float((message.density or 0) - rsu_density)
        adjacent_flow = 1.0 if any(self.rsu_density.get(seg, 0) > 0 for seg in self.adjacency.get(target_segment, set())) else 0.0
        controller_mismatch = 0.0
        majority_mismatch = 0.0
        if message.intersection:
            controller = self.controller_signal.get(message.intersection, "unknown")
            majority = self._majority_signal(message.intersection)
            controller_mismatch = 1.0 if controller != "unknown" and controller != message.signal_state else 0.0
            majority_mismatch = 1.0 if majority and majority != message.signal_state else 0.0
        previous = self.vehicle_position.get(message.claimed_vehicle or "")
        adjacency_violation = 1.0 if previous and target_segment not in self.adjacency.get(previous, set()) and target_segment != previous else 0.0
        rsu_region_change = 1.0 if self.segment_to_rsu.get(previous or "", "") != self.segment_to_rsu.get(target_segment, "") else 0.0
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


class FlatFeatureLogisticDetector:
    def __init__(self, model: MulticlassLogisticRegressionModel, adjacency: Dict[str, set[str]], segment_to_rsu: Dict[str, str]) -> None:
        self.model = model
        self.context = FlatFeatureContext(adjacency, segment_to_rsu)

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


def split_episodes(
    episodes: Sequence[Tuple[List[Message], Optional[str], int]],
    train_ratio: float = 0.7,
    train_count_per_group: Optional[int] = None,
):
    grouped: Dict[Optional[str], List[Tuple[List[Message], Optional[str], int]]] = {}
    for item in episodes:
        grouped.setdefault(item[1], []).append(item)
    train, test = [], []
    for items in grouped.values():
        if train_count_per_group is not None:
            cutoff = max(1, min(len(items) - 1, train_count_per_group))
        else:
            cutoff = max(1, int(len(items) * train_ratio))
        train.extend(items[:cutoff])
        test.extend(items[cutoff:])
    return train, test


def build_learning_dataset(episodes, adjacency: Dict[str, set[str]], segment_to_rsu: Dict[str, str]) -> Tuple[List[List[float]], List[int]]:
    x_rows: List[List[float]] = []
    y_rows: List[int] = []
    for messages, _, _ in episodes:
        context = FlatFeatureContext(adjacency, segment_to_rsu)
        for message in messages:
            features = context.features_for(message)
            label = message_label(message)
            if features is not None and label is not None:
                x_rows.append(features)
                y_rows.append(CLASS_LABELS.index(label))
            context.update(message)
    return x_rows, y_rows


def segment_id(u: int, v: int, key: int) -> str:
    return f"{u}|{v}|{key}"


def load_city_graph(city_key: str) -> nx.MultiDiGraph:
    return ox.load_graphml(CITY_CACHE_DIR / f"{city_key}_drive.graphml")


def load_city_edges(city_key: str, graph: nx.MultiDiGraph):
    import geopandas as gpd

    cache = CITY_CACHE_DIR / f"{city_key}_txdot_aadt.geojson"
    if cache.exists():
        aadt_gdf = gpd.read_file(cache)
    else:
        _, edges = ox.graph_to_gdfs(graph)
        aadt_gdf = download_txdot_aadt(tuple(edges.total_bounds), cache)
    return annotate_edges_with_aadt(graph, aadt_gdf)


def download_txdot_permanent_count_stations(bounds, cache_path: Path):
    import geopandas as gpd

    if cache_path.exists():
        return gpd.read_file(cache_path)

    minx, miny, maxx, maxy = bounds
    url = (
        "https://services.arcgis.com/KTcxiTD9dsQw4r7Z/ArcGIS/rest/services/"
        "TxDOT_Permanent_Count_Stations/FeatureServer/0/query"
    )
    params = {
        "where": "1=1",
        "geometry": f"{minx},{miny},{maxx},{maxy}",
        "geometryType": "esriGeometryEnvelope",
        "inSR": 4326,
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "*",
        "returnGeometry": "true",
        "outSR": 4326,
        "f": "geojson",
    }
    response = requests.get(url, params=params, timeout=120)
    response.raise_for_status()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(response.text, encoding="utf-8")
    return gpd.read_file(cache_path)


def load_city_count_stations(city_key: str, graph: nx.MultiDiGraph):
    import geopandas as gpd

    cache = CITY_CACHE_DIR / f"{city_key}_txdot_permanent_count_stations.geojson"
    if cache.exists():
        return gpd.read_file(cache)
    _, edges = ox.graph_to_gdfs(graph)
    return download_txdot_permanent_count_stations(tuple(edges.total_bounds), cache)


def _coerce_aadt(value) -> float:
    try:
        if value is None:
            return 1.0
        numeric = float(value)
        return numeric if math.isfinite(numeric) and numeric > 0 else 1.0
    except Exception:
        return 1.0


def _expand_segments(seed_segments: Sequence[str], adjacency: Dict[str, set[str]], hops: int) -> set[str]:
    frontier = set(seed_segments)
    visited = set(seed_segments)
    for _ in range(max(0, hops)):
        next_frontier: set[str] = set()
        for seg in frontier:
            next_frontier.update(adjacency.get(seg, set()))
        next_frontier -= visited
        visited.update(next_frontier)
        frontier = next_frontier
        if not frontier:
            break
    return visited


def build_taz_context(graph: nx.MultiDiGraph, grid_size: int = 4):
    nodes, edges = ox.graph_to_gdfs(graph)
    west, south, east, north = edges.total_bounds
    dx = (east - west) / grid_size
    dy = (north - south) / grid_size

    def zone_of_point(x: float, y: float) -> str:
        ix = min(grid_size - 1, max(0, int((x - west) / dx))) if dx > 0 else 0
        iy = min(grid_size - 1, max(0, int((y - south) / dy))) if dy > 0 else 0
        return f"Z{ix}_{iy}"

    node_zone = {int(node_id): zone_of_point(float(row.x), float(row.y)) for node_id, row in nodes.iterrows()}
    return node_zone


def build_real_city_context(city_key: str):
    graph = load_city_graph(city_key)
    edges_gdf = load_city_edges(city_key, graph)
    count_stations = load_city_count_stations(city_key, graph)
    node_zone = build_taz_context(graph, grid_size=GRID_SIZE)

    zone_production: Dict[str, float] = {}
    zone_attraction: Dict[str, float] = {}
    zone_nodes: Dict[str, List[int]] = {f"Z{ix}_{iy}": [] for ix in range(GRID_SIZE) for iy in range(GRID_SIZE)}
    segment_to_zone: Dict[str, str] = {}
    segment_to_intersection: Dict[str, str] = {}
    segment_coords: Dict[str, Tuple[float, float]] = {}
    adjacency: Dict[str, set[str]] = {}
    segment_aadt: Dict[str, float] = {}

    for node_id, zone in node_zone.items():
        zone_nodes.setdefault(zone, []).append(node_id)

    for row in edges_gdf.itertuples():
        seg = segment_id(int(row.u), int(row.v), int(row.key))
        zone_u = node_zone[int(row.u)]
        zone_v = node_zone[int(row.v)]
        aadt = _coerce_aadt(getattr(row, "AADT_CUR", None))
        zone_production[zone_u] = zone_production.get(zone_u, 0.0) + aadt
        zone_attraction[zone_v] = zone_attraction.get(zone_v, 0.0) + aadt
        segment_to_zone[seg] = zone_u
        segment_to_intersection[seg] = f"I_{row.v}"
        segment_aadt[seg] = aadt
        geom = getattr(row, "geometry", None)
        if geom is not None:
            mid = geom.interpolate(0.5, normalized=True)
            segment_coords[seg] = (mid.x, mid.y)
        else:
            x1, y1 = float(graph.nodes[int(row.u)]["x"]), float(graph.nodes[int(row.u)]["y"])
            x2, y2 = float(graph.nodes[int(row.v)]["x"]), float(graph.nodes[int(row.v)]["y"])
            segment_coords[seg] = ((x1 + x2) / 2, (y1 + y2) / 2)
        adjacency.setdefault(seg, set())

    # Build adjacency from shared nodes.
    by_endpoint: Dict[int, List[str]] = {}
    for seg in segment_coords:
        u, v, _ = seg.split("|")
        by_endpoint.setdefault(int(u), []).append(seg)
        by_endpoint.setdefault(int(v), []).append(seg)
    for connected in by_endpoint.values():
        for seg_a in connected:
            adjacency.setdefault(seg_a, set()).update(seg_b for seg_b in connected if seg_b != seg_a)

    unique_segments = list(segment_coords.keys())

    station_segments: List[str] = []
    if count_stations is not None and not count_stations.empty:
        xs = count_stations.geometry.x.tolist()
        ys = count_stations.geometry.y.tolist()
        nearest = ox.distance.nearest_edges(graph, xs, ys)
        for edge in zip(*nearest) if isinstance(nearest, tuple) else nearest:
            if len(edge) == 3:
                station_segments.append(segment_id(int(edge[0]), int(edge[1]), int(edge[2])))

    if not station_segments:
        sorted_segments = sorted(unique_segments, key=lambda s: segment_coords[s][0])
        anchor_count = 4
        station_segments = [
            sorted_segments[int(i * max(1, len(sorted_segments) - 1) / max(1, anchor_count - 1))]
            for i in range(anchor_count)
        ]

    observed_segments = _expand_segments(station_segments, adjacency, hops=1)
    monitored_intersections = {
        segment_to_intersection[seg]
        for seg in observed_segments
        if seg in segment_to_intersection
    }

    def nearest_station(seg: str) -> str:
        x, y = segment_coords[seg]
        idx = min(
            range(len(station_segments)),
            key=lambda i: (segment_coords[station_segments[i]][0] - x) ** 2 + (segment_coords[station_segments[i]][1] - y) ** 2,
        )
        return f"R{idx + 1}"

    segment_to_rsu = {seg: nearest_station(seg) for seg in unique_segments}

    aadt_values = sorted(segment_aadt.values())
    median_aadt = aadt_values[len(aadt_values) // 2] if aadt_values else 1.0
    max_aadt = max(aadt_values) if aadt_values else 1.0
    observed_fraction = len(observed_segments) / max(1, len(unique_segments))
    monitored_fraction = len(monitored_intersections) / max(1, len(set(segment_to_intersection.values())))
    message_drop_rate = max(0.02, min(0.30, 0.22 - 0.35 * observed_fraction))
    density_noise_std = max(0.15, min(1.10, 0.90 - 0.90 * observed_fraction))
    vehicle_observation_drop = max(0.0, min(0.15, 0.10 - 0.08 * monitored_fraction))
    controller_delay = 1 if monitored_fraction < 0.12 else 0
    return {
        "graph": graph,
        "edges_gdf": edges_gdf,
        "count_stations": count_stations,
        "zone_production": zone_production,
        "zone_attraction": zone_attraction,
        "zone_nodes": zone_nodes,
        "segment_to_zone": segment_to_zone,
        "segment_to_intersection": segment_to_intersection,
        "segment_to_rsu": segment_to_rsu,
        "segment_coords": segment_coords,
        "adjacency": adjacency,
        "segments": unique_segments,
        "segment_aadt": segment_aadt,
        "observed_segments": observed_segments,
        "monitored_intersections": monitored_intersections,
        "observation_profile": {
            "station_count": len(station_segments),
            "observed_fraction": observed_fraction,
            "monitored_fraction": monitored_fraction,
            "message_drop_rate": message_drop_rate,
            "density_noise_std": density_noise_std,
            "vehicle_observation_drop": vehicle_observation_drop,
            "controller_delay": controller_delay,
            "median_aadt": median_aadt,
            "max_aadt": max_aadt,
        },
    }


def build_trip_library(context, seed: int, num_trips: int) -> List[List[str]]:
    rng = random.Random(seed)
    trips: List[List[str]] = []
    progress_step = max(1, num_trips // 5)
    for idx in range(num_trips):
        route = sample_route_from_od(context, rng)
        if route:
            trips.append(route)
        if (idx + 1) % progress_step == 0 or idx + 1 == num_trips:
            print(
                f"[{context['city'].title()}] route library progress: {idx + 1}/{num_trips} candidates, {len(trips)} valid",
                flush=True,
            )
    return trips


def sample_od_pair(context, rng: random.Random) -> Tuple[str, str]:
    origins = list(context["zone_production"].keys())
    origin_weights = [context["zone_production"][z] for z in origins]
    origin_zone = rng.choices(origins, weights=origin_weights, k=1)[0]

    dests = list(context["zone_attraction"].keys())
    dest_weights = [context["zone_attraction"][z] for z in dests]
    for _ in range(30):
        dest_zone = rng.choices(dests, weights=dest_weights, k=1)[0]
        if dest_zone != origin_zone:
            return origin_zone, dest_zone
    return origin_zone, dest_zone


def sample_route_from_od(context, rng: random.Random) -> List[str]:
    graph = context["graph"]
    for _ in range(80):
        origin_zone, dest_zone = sample_od_pair(context, rng)
        origin_candidates = context["zone_nodes"].get(origin_zone, [])
        dest_candidates = context["zone_nodes"].get(dest_zone, [])
        if not origin_candidates or not dest_candidates:
            continue
        origin = rng.choice(origin_candidates)
        destination = rng.choice(dest_candidates)
        if origin == destination:
            continue
        try:
            node_path = nx.shortest_path(graph, origin, destination, weight="travel_time")
        except Exception:
            continue
        route_segments: List[str] = []
        for u, v in zip(node_path[:-1], node_path[1:]):
            bundle = graph.get_edge_data(u, v)
            if not bundle:
                continue
            best_key = min(bundle, key=lambda k: bundle[k].get("travel_time", bundle[k].get("length", 1.0)))
            route_segments.append(segment_id(int(u), int(v), int(best_key)))
        if len(route_segments) >= 4:
            return route_segments
    return []


def signal_state(tick: int, intersection_id: str) -> str:
    cycle = ("green", "green", "yellow", "red", "red", "red")
    offset = sum(ord(ch) for ch in intersection_id) % len(cycle)
    return cycle[(tick + offset) % len(cycle)]


def density_multiplier(context, segment: str) -> float:
    profile = context["observation_profile"]
    aadt = context["segment_aadt"].get(segment, profile["median_aadt"])
    max_aadt = max(profile["max_aadt"], profile["median_aadt"], 1.0)
    return 1.0 + 3.0 * min(1.0, aadt / max_aadt)


def measured_density(context, segment: str, occupancy: int, rng: random.Random) -> int:
    profile = context["observation_profile"]
    scaled = occupancy * density_multiplier(context, segment)
    noisy = scaled + rng.gauss(0.0, profile["density_noise_std"])
    return max(0, int(round(noisy)))


def run_episode(context, attack_type: Optional[str], seed: int, num_vehicles: int, ticks: int = 18) -> Tuple[List[Message], int]:
    rng = random.Random(seed)
    vehicles: List[Vehicle] = []
    for idx in range(num_vehicles):
        route = list(rng.choice(context["trip_library"]))
        if route:
            vehicles.append(Vehicle(vehicle_id=f"V{idx+1}", route=route))
    attack_start = ATTACK_START_TICK
    maintenance_segment = rng.choice(sorted(context["observed_segments"] or context["active_segments"]))
    maintenance_window = range(8, 11)
    messages: List[Message] = []
    msg_counter = 0
    intersections = sorted(set(context["segment_to_intersection"].values()))
    attacker_count = max(1, int(round(num_vehicles * ATTACKER_SHARE)))
    attacker_pool = vehicles[:attacker_count] if vehicles else []

    def next_id() -> str:
        nonlocal msg_counter
        msg_counter += 1
        return f"m{msg_counter:06d}"

    for tick in range(ticks):
        if tick and tick % 3 == 0:
            for vehicle in vehicles:
                if rng.random() < 0.65:
                    vehicle.move()

        occupancies = {seg: 0 for seg in context["active_segments"]}
        for vehicle in vehicles:
            occupancies[vehicle.current_segment] += 1

        for seg, density in occupancies.items():
            if seg not in context["observed_segments"]:
                continue
            if rng.random() < context["observation_profile"]["message_drop_rate"]:
                continue
            messages.append(
                Message(
                    next_id(),
                    tick,
                    context["segment_to_rsu"][seg],
                    "rsu",
                    "rsu_density",
                    segment=seg,
                    density=measured_density(context, seg, density, rng),
                )
            )

        for intersection in intersections:
            if intersection not in context["monitored_intersections"]:
                continue
            if rng.random() < context["observation_profile"]["message_drop_rate"]:
                continue
            delayed_tick = max(0, tick - context["observation_profile"]["controller_delay"])
            messages.append(
                Message(
                    next_id(),
                    tick,
                    f"C_{intersection}",
                    "controller",
                    "controller_signal",
                    intersection=intersection,
                    signal_state=signal_state(delayed_tick, intersection),
                )
            )

        for vehicle in vehicles:
            seg = vehicle.current_segment
            intersection = context["segment_to_intersection"][seg]
            messages.append(Message(next_id(), tick, vehicle.vehicle_id, "vehicle", "vehicle_position", segment=seg))
            if rng.random() >= context["observation_profile"]["vehicle_observation_drop"]:
                messages.append(
                    Message(
                        next_id(),
                        tick,
                        vehicle.vehicle_id,
                        "vehicle",
                        "signal_observation",
                        intersection=intersection,
                        signal_state=signal_state(tick, intersection),
                    )
                )

        measured_occupancies = {
            seg: measured_density(context, seg, occupancies.get(seg, 0), rng)
            for seg in context["active_segments"]
        }

        high_density_segments = [seg for seg, density in measured_occupancies.items() if density >= 4]
        if high_density_segments:
            legit_seg = max(high_density_segments, key=lambda seg: measured_occupancies[seg])
            legit_senders = [vehicle for vehicle in vehicles if vehicle.current_segment == legit_seg]
            legit_sender = legit_senders[0].vehicle_id if legit_senders else context["segment_to_rsu"][legit_seg]
            legit_sender_type = "vehicle" if legit_senders else "rsu"
            messages.append(
                Message(
                    next_id(),
                    tick,
                    legit_sender,
                    legit_sender_type,
                    "congestion_alert",
                    False,
                    None,
                    segment=legit_seg,
                    density=measured_occupancies[legit_seg],
                )
            )

        if tick in maintenance_window:
            messages.append(
                Message(
                    next_id(),
                    tick,
                    context["segment_to_rsu"][maintenance_segment],
                    "rsu",
                    "closure_alert",
                    False,
                    None,
                    segment=maintenance_segment,
                    closed=True,
                )
            )

        legit_signal_seg = rng.choice(sorted(context["observed_segments"] or context["active_segments"]))
        legit_intersection = context["segment_to_intersection"][legit_signal_seg]
        messages.append(
            Message(
                next_id(),
                tick,
                context["segment_to_rsu"][legit_signal_seg],
                "rsu",
                "signal_report",
                False,
                None,
                intersection=legit_intersection,
                signal_state=signal_state(tick, legit_intersection),
            )
        )

        if attack_type and attack_start <= tick < attack_start + ATTACK_WINDOW_TICKS and attacker_pool:
            attacker = attacker_pool[(tick - attack_start) % len(attacker_pool)]
            current_seg = attacker.current_segment
            low_density = [seg for seg, d in occupancies.items() if d <= 2]
            candidate_seg = low_density[0] if low_density else current_seg
            if attack_type == "phantom_congestion":
                claimed_density = measured_occupancies.get(candidate_seg, 0) + max(
                    3,
                    int(round(2 + density_multiplier(context, candidate_seg))),
                )
                messages.append(
                    Message(
                        next_id(),
                        tick,
                        attacker.vehicle_id,
                        "vehicle",
                        "congestion_alert",
                        True,
                        attack_type,
                        segment=candidate_seg,
                        density=claimed_density,
                    )
                )
            elif attack_type == "signal_spoofing":
                target_intersection = context["segment_to_intersection"][current_seg]
                actual = signal_state(tick, target_intersection)
                forged = "red" if actual == "green" else "green"
                messages.append(Message(next_id(), tick, context["segment_to_rsu"][current_seg], "rsu", "signal_report", True, attack_type, intersection=target_intersection, signal_state=forged))
            elif attack_type == "false_closure":
                messages.append(Message(next_id(), tick, attacker.vehicle_id, "vehicle", "closure_alert", True, attack_type, segment=current_seg, closed=True))
            elif attack_type == "position_spoofing":
                non_adjacent = [seg for seg in context["active_segments"] if seg not in context["adjacency"].get(current_seg, set()) and seg != current_seg]
                spoof_seg = non_adjacent[0] if non_adjacent else current_seg
                messages.append(Message(next_id(), tick, attacker.vehicle_id, "vehicle", "vehicle_position", True, attack_type, segment=spoof_seg, claimed_vehicle=attacker.vehicle_id))

    return messages, attack_start


def evaluate_detector(detector_factory, episodes, detector_name: str, city_key: str):
    stats = MulticlassStats(CLASS_LABELS)
    per_attack = {attack: MulticlassStats(CLASS_LABELS) for attack in ATTACK_TYPES}
    total_episodes = len(episodes)
    progress_step = max(1, total_episodes // 4)
    for idx, (messages, attack_type, attack_start) in enumerate(episodes, start=1):
        detector = detector_factory()
        first_detection_tick = None
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
        if idx % progress_step == 0 or idx == total_episodes:
            print(
                f"[{city_key.title()}] evaluated {detector_name}: {idx}/{total_episodes} episodes",
                flush=True,
            )
    return {
        "accuracy": round(stats.accuracy(), 4),
        "macro_precision": round(stats.macro_precision(), 4),
        "macro_recall": round(stats.macro_recall(), 4),
        "macro_f1": round(stats.macro_f1(), 4),
        "average_latency": round(stats.average_latency(), 4),
        "confusion_matrix": stats.confusion,
        "per_attack": {
            attack: {
                "recall": round(per_attack[attack].recall(attack), 4),
                "f1": round(per_attack[attack].f1(attack), 4),
                "average_latency": round(per_attack[attack].average_latency(), 4),
            }
            for attack in ATTACK_TYPES
        },
    }


def run_city_benchmark(city_key: str, seed: int = 42, episodes_per_case: int = EPISODES_PER_ATTACK):
    print(f"[{city_key.title()}] building real-city context", flush=True)
    context = build_real_city_context(city_key)
    context["city"] = city_key

    num_vehicles = EPISODE_VEHICLES
    trip_library_size = ROUTE_LIBRARY_CAP
    print(
        f"[{city_key.title()}] configuration: {len(context['zone_nodes'])} zones, {trip_library_size} routes, {num_vehicles} vehicles, attacker share {ATTACKER_SHARE:.1%}",
        flush=True,
    )
    context["trip_library"] = build_trip_library(context, seed=seed, num_trips=trip_library_size)
    context["active_segments"] = sorted({seg for trip in context["trip_library"] for seg in trip})
    print(
        f"[{city_key.title()}] active network after route sampling: {len(context['active_segments'])} segments",
        flush=True,
    )

    episodes = []
    counter = 0
    for attack in ATTACK_TYPES:
        print(f"[{city_key.title()}] generating episodes for attack: {attack}", flush=True)
        for _ in range(episodes_per_case):
            messages, attack_start = run_episode(context, attack, seed + counter, num_vehicles=num_vehicles)
            episodes.append((messages, attack, attack_start))
            counter += 1
        print(
            f"[{city_key.title()}] generated {len(episodes)}/{episodes_per_case * len(ATTACK_TYPES)} attack episodes",
            flush=True,
        )
    benign_episodes = 0

    train_episodes, test_episodes = split_episodes(
        episodes,
        train_count_per_group=TRAIN_EPISODES_PER_ATTACK,
    )
    print(
        f"[{city_key.title()}] split episodes into {len(train_episodes)} train and {len(test_episodes)} test",
        flush=True,
    )
    x_train, y_train = build_learning_dataset(train_episodes, context["adjacency"], context["segment_to_rsu"])
    learning_model = MulticlassLogisticRegressionModel()
    print(f"[{city_key.title()}] training flat-feature logistic detector on {len(x_train)} samples", flush=True)
    learning_model.fit(x_train, y_train)

    baseline = evaluate_detector(lambda: BaselineDetector(context["adjacency"]), test_episodes, "baseline", city_key)
    learning = evaluate_detector(
        lambda: FlatFeatureLogisticDetector(learning_model, context["adjacency"], context["segment_to_rsu"]),
        test_episodes,
        "flat-feature logistic",
        city_key,
    )
    kg = evaluate_detector(
        lambda: KGDetector(context["adjacency"], context["segment_to_rsu"], context["segment_to_intersection"]),
        test_episodes,
        "knowledge graph",
        city_key,
    )
    print(
        f"[{city_key.title()}] completed benchmark: baseline macro-F1={baseline['macro_f1']}, logistic macro-F1={learning['macro_f1']}, KG macro-F1={kg['macro_f1']}",
        flush=True,
    )
    return {
        "city": city_key,
        "segments": len(context["segments"]),
        "zones": len(context["zone_nodes"]),
        "num_vehicles": num_vehicles,
        "episodes_per_attack": episodes_per_case,
        "train_episodes": len(train_episodes),
        "test_episodes": len(test_episodes),
        "attacker_share": ATTACKER_SHARE,
        "attack_window": f"Ticks {ATTACK_START_TICK}-{ATTACK_START_TICK + ATTACK_WINDOW_TICKS - 1}",
        "attack_mix": "50% spoofing / 50% intent hiding",
        "active_segments": len(context["active_segments"]),
        "observed_segments_count": len(context["observed_segments"]),
        "monitored_intersections_count": len(context["monitored_intersections"]),
        "context": context,
        "train_data": train_episodes,
        "test_data": test_episodes,
        "learning_model": learning_model,
        "baseline": baseline,
        "flat_feature_logistic": learning,
        "knowledge_graph": kg,
    }


def evaluate_transfer(results: List[dict]) -> List[dict]:
    transfer_rows: List[dict] = []
    for source in results:
        source_model = source["learning_model"]
        for target in results:
            print(
                f"[transfer] evaluating {source['city'].title()} model on {target['city'].title()} test data",
                flush=True,
            )
            logistic = evaluate_detector(
                lambda sm=source_model, tc=target["context"]: FlatFeatureLogisticDetector(sm, tc["adjacency"], tc["segment_to_rsu"]),
                target["test_data"],
                f"{source['city']}-to-{target['city']} logistic",
                target["city"],
            )
            baseline = evaluate_detector(
                lambda tc=target["context"]: BaselineDetector(tc["adjacency"]),
                target["test_data"],
                f"{source['city']}-to-{target['city']} baseline",
                target["city"],
            )
            kg = evaluate_detector(
                lambda tc=target["context"]: KGDetector(
                    tc["adjacency"],
                    tc["segment_to_rsu"],
                    tc["segment_to_intersection"],
                ),
                target["test_data"],
                f"{source['city']}-to-{target['city']} knowledge graph",
                target["city"],
            )
            transfer_rows.append(
                {
                    "source_city": source["city"],
                    "target_city": target["city"],
                    "baseline_macro_f1": baseline["macro_f1"],
                    "logistic_macro_f1": logistic["macro_f1"],
                    "kg_macro_f1": kg["macro_f1"],
                    "delta_baseline_macro_f1": round(baseline["macro_f1"] - target["baseline"]["macro_f1"], 4),
                    "delta_logistic_macro_f1": round(logistic["macro_f1"] - target["flat_feature_logistic"]["macro_f1"], 4),
                    "delta_kg_macro_f1": round(kg["macro_f1"] - target["knowledge_graph"]["macro_f1"], 4),
                }
            )
    return transfer_rows


def write_outputs(results: List[dict], transfer_rows: List[dict]) -> None:
    ensure_pipeline_directories()
    outdir = CITY_RESULTS_DIR
    outdir.mkdir(parents=True, exist_ok=True)
    serializable_results = []
    for item in results:
        serializable_results.append(
            {
                "city": item["city"],
                "segments": item["segments"],
                "zones": item["zones"],
                "num_vehicles": item["num_vehicles"],
                "train_episodes": item["train_episodes"],
                "test_episodes": item["test_episodes"],
                "observation_profile": item["context"]["observation_profile"],
                "logistic_feature_importance": [
                    {"feature": feature, "value": value}
                    for feature, value in item["learning_model"].feature_importance()
                ],
                "baseline": item["baseline"],
                "flat_feature_logistic": item["flat_feature_logistic"],
                "knowledge_graph": item["knowledge_graph"],
            }
        )
    with (outdir / "real_city_benchmark.json").open("w", encoding="utf-8") as handle:
        json.dump({"city_results": serializable_results, "transfer": transfer_rows}, handle, indent=2)
    print(f"[output] wrote {(outdir / 'real_city_benchmark.json').name}", flush=True)

    rows = []
    config_rows = []
    for item in results:
        config_rows.append(
            {
                "city": item["city"],
                "grid_zoning": f"{GRID_SIZE} x {GRID_SIZE}",
                "route_library_cap": ROUTE_LIBRARY_CAP,
                "vehicles_per_episode": item["num_vehicles"],
                "episodes_per_scenario": item["episodes_per_attack"],
                "training_episodes": item["train_episodes"],
                "test_episodes": item["test_episodes"],
                "attack_window": item["attack_window"],
                "test_attacker_share_pct": round(item["attacker_share"] * 100.0, 1),
                "test_attack_mix": item["attack_mix"],
                "active_segments": item["active_segments"],
                "observed_segments": item["observed_segments_count"],
                "monitored_intersections": item["monitored_intersections_count"],
            }
        )
        for detector_name in ("baseline", "flat_feature_logistic", "knowledge_graph"):
            det = item[detector_name]
            rows.append({
                "city": item["city"],
                "detector": detector_name,
                "zones": item["zones"],
                "vehicles": item["num_vehicles"],
                "accuracy": det["accuracy"],
                "macro_precision": det["macro_precision"],
                "macro_recall": det["macro_recall"],
                "macro_f1": det["macro_f1"],
                "latency": det["average_latency"],
            })

    with (outdir / "real_city_benchmark.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[output] wrote {(outdir / 'real_city_benchmark.csv').name}", flush=True)

    with (outdir / "real_city_configuration.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(config_rows[0].keys()))
        writer.writeheader()
        writer.writerows(config_rows)
    print(f"[output] wrote {(outdir / 'real_city_configuration.csv').name}", flush=True)

    with (outdir / "city_transfer.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(transfer_rows[0].keys()))
        writer.writeheader()
        writer.writerows(transfer_rows)
    print(f"[output] wrote {(outdir / 'city_transfer.csv').name}", flush=True)

    feature_rows = []
    for item in results:
        for row in item["learning_model"].class_feature_importance():
            feature_rows.append(
                {
                    "city": item["city"],
                    "class": row["class"],
                    "feature": row["feature"],
                    "coefficient": round(float(row["coefficient"]), 6),
                    "abs_coefficient": round(float(row["abs_coefficient"]), 6),
                }
            )
    with (outdir / "city_logistic_feature_importance.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(feature_rows[0].keys()))
        writer.writeheader()
        writer.writerows(feature_rows)
    print(f"[output] wrote {(outdir / 'city_logistic_feature_importance.csv').name}", flush=True)

    render_city_assets(PROJECT_ROOT)


def plot_city_logistic_feature_importance(results: List[dict]) -> None:
    figures_dir = PAPER_FIGURES_DIR
    figures_dir.mkdir(parents=True, exist_ok=True)

    class_labels = {
        "benign": "Benign",
        "phantom_congestion": "Phantom",
        "signal_spoofing": "Signal",
        "false_closure": "Closure",
        "position_spoofing": "Position",
    }
    feature_aliases = {
        "Congestion alert": "Cong.",
        "Closure alert": "Close",
        "Signal report": "Signal",
        "Position claim": "Pos.",
        "Sender is RSU": "RSU",
        "Density gap": "DensGap",
        "Observed RSU density": "ObsDens",
        "Sender near segment": "NearSeg",
        "Closure flag": "ClsFlag",
        "Adjacent flow": "AdjFlow",
        "Controller mismatch": "CtrlMis",
        "Majority mismatch": "MajMis",
        "Adjacency violation": "AdjViol",
        "RSU region change": "RSUReg",
    }
    text_color = "#18324a"
    grid_color = "#d6dde6"
    class_order = ["benign", "phantom_congestion", "signal_spoofing", "false_closure", "position_spoofing"]
    feature_order = [feature for feature, _ in results[0]["learning_model"].feature_importance()]
    averaged = {class_name: [] for class_name in class_order}
    for class_name in class_order:
        for feature_name in feature_order:
            values = []
            for item in results:
                for row in item["learning_model"].class_feature_importance():
                    if row["class"] == class_name and row["feature"] == feature_name:
                        values.append(float(row["abs_coefficient"]))
                        break
            averaged[class_name].append(sum(values) / len(values))

    matrix = np.asarray([averaged[class_name] for class_name in class_order], dtype=float)

    fig, ax = plt.subplots(figsize=(3.35, 2.8))
    image = ax.imshow(matrix, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(feature_order)))
    ax.set_xticklabels([feature_aliases.get(name, name) for name in feature_order], rotation=55, ha="right", fontsize=6.2, color=text_color)
    ax.set_yticks(range(len(class_order)))
    ax.set_yticklabels([class_labels[name] for name in class_order], fontsize=7.2, color=text_color)
    ax.set_xlabel("Feature", fontsize=8, color=text_color)
    ax.set_ylabel("Class", fontsize=8, color=text_color)
    ax.set_facecolor("white")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(grid_color)
    ax.spines["bottom"].set_color(grid_color)
    ax.tick_params(colors=text_color)
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean absolute coefficient", color=text_color, fontsize=7.5)
    cbar.ax.tick_params(labelsize=6.5, colors=text_color)
    fig.patch.set_facecolor("white")
    plt.tight_layout(pad=0.4)
    plt.savefig(figures_dir / "city_logistic_feature_importance.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[output] wrote {(figures_dir / 'city_logistic_feature_importance.png').name}", flush=True)

def main() -> None:
    ensure_pipeline_directories()
    print("[main] starting real-city benchmark", flush=True)
    results = [run_city_benchmark(city_key, seed=42 + idx) for idx, city_key in enumerate(("austin", "houston", "dallas"))]
    print("[main] evaluating cross-city transfer", flush=True)
    transfer_rows = evaluate_transfer(results)
    print("[main] writing outputs", flush=True)
    write_outputs(results, transfer_rows)
    for item in results:
        print(
            item["city"],
            item["zones"],
            item["num_vehicles"],
            item["baseline"]["macro_f1"],
            item["flat_feature_logistic"]["macro_f1"],
            item["knowledge_graph"]["macro_f1"],
        )


if __name__ == "__main__":
    main()
