from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple
import csv
import json
import random

from sequential_city_support import build_real_city_context, build_trip_library, signal_state


CITY_KEY = "austin"
EPISODE_VEHICLES = 500
GRID_SIZE = 10
ROUTE_LIBRARY_CAP = 800
ATTACK_START = 6
ATTACK_DURATION = 4
TRAIN_RATIO = 0.7
DEFAULT_ATTACKER_FRACTION = 1.0 / EPISODE_VEHICLES
TEST_ATTACKER_FRACTION = 0.02
SCENARIOS = (
    "benign_nominal",
    "regional_gps_failure",
    "position_spoofing",
    "intent_hiding",
)
CLASS_LABELS = SCENARIOS
BENIGN_LABEL = "benign_nominal"
UNRESOLVED_LABEL = "unresolved"
INTRUSION_POSITIVE_LABELS = {"position_spoofing", "intent_hiding"}


@dataclass
class Message:
    message_id: str
    tick: int
    sender_id: str
    sender_type: str
    claim_type: str
    malicious: bool = False
    scenario: str = "benign_nominal"
    label: str = BENIGN_LABEL
    segment: Optional[str] = None
    claimed_vehicle: Optional[str] = None
    target_vehicle: Optional[str] = None
    witness_segment: Optional[str] = None
    intersection: Optional[str] = None
    signal_state: Optional[str] = None
    rsu_region: Optional[str] = None
    gps_quality_ok: Optional[bool] = None


@dataclass
class PendingReport:
    message: Message
    previous_trusted: Optional[str]
    previous_reported: Optional[str]
    claimed_segment: str
    region: str
    reported_step_consistent: bool
    same_region_as_previous: bool


@dataclass
class SuspiciousTrajectoryReport:
    message: Message
    claimed_segment: str
    route_inconsistent: bool
    witness_mismatch: bool
    reported_step_consistent: bool


@dataclass
class Vehicle:
    vehicle_id: str
    route: List[str]
    route_index: int = 0
    last_claimed_segment: Optional[str] = None

    @property
    def current_segment(self) -> str:
        return self.route[self.route_index]

    def move(self) -> None:
        if self.route_index < len(self.route) - 1:
            self.route_index += 1


@dataclass
class MulticlassStats:
    labels: Sequence[str]
    confusion: Dict[str, Dict[str, int]] = field(init=False)
    latencies: Dict[str, List[int]] = field(init=False)

    def __post_init__(self) -> None:
        self.confusion = {
            true_label: {predicted_label: 0 for predicted_label in self.labels}
            for true_label in self.labels
        }
        self.latencies = {label: [] for label in self.labels if label != BENIGN_LABEL}

    def record(self, true_label: str, predicted_label: str) -> None:
        self.confusion[true_label][predicted_label] += 1

    def total(self) -> int:
        return sum(sum(row.values()) for row in self.confusion.values())

    def accuracy(self) -> float:
        correct = sum(self.confusion[label][label] for label in self.labels)
        total = self.total()
        return correct / total if total else 0.0

    def per_class_metrics(self, label: str) -> Dict[str, float]:
        tp = self.confusion[label][label]
        fp = sum(self.confusion[other][label] for other in self.labels if other != label)
        fn = sum(self.confusion[label][other] for other in self.labels if other != label)
        tn = self.total() - tp - fp - fn
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        false_positive_rate = fp / (fp + tn) if (fp + tn) else 0.0
        latency_values = self.latencies.get(label, [])
        average_latency = mean(latency_values) if latency_values else 0.0
        support = sum(self.confusion[label].values())
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "false_positive_rate": false_positive_rate,
            "average_latency": average_latency,
            "support": support,
        }

    def macro_precision(self) -> float:
        return mean(self.per_class_metrics(label)["precision"] for label in self.labels)

    def macro_recall(self) -> float:
        return mean(self.per_class_metrics(label)["recall"] for label in self.labels)

    def macro_f1(self) -> float:
        return mean(self.per_class_metrics(label)["f1"] for label in self.labels)

    def average_latency(self) -> float:
        values = [latency for label in self.latencies for latency in self.latencies[label]]
        return mean(values) if values else 0.0


@dataclass
class BinaryDetectionStats:
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
        precision = self.precision()
        recall = self.recall()
        return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    def false_positive_rate(self) -> float:
        denom = self.fp + self.tn
        return self.fp / denom if denom else 0.0

    def average_latency(self) -> float:
        return mean(self.latencies) if self.latencies else 0.0


class OriginalPositionKG:
    """Binary position-consistency rule from the original KG detector."""

    def __init__(self, adjacency: Dict[str, set[str]], segment_to_rsu: Dict[str, str]) -> None:
        self.adjacency = adjacency
        self.segment_to_rsu = segment_to_rsu
        self.last_reported_segment: Dict[str, str] = {}
        self.delay_ticks = 0

    def observe(self, message: Message) -> Optional[bool]:
        if message.claim_type != "vehicle_position":
            return None

        vehicle_id = message.claimed_vehicle or message.sender_id
        reported_segment = message.segment or ""
        previous = self.last_reported_segment.get(vehicle_id)
        adjacency_violation = bool(
            previous
            and reported_segment not in self.adjacency.get(previous, set())
            and reported_segment != previous
        )
        region_change = bool(
            previous
            and self.segment_to_rsu.get(previous, "") != self.segment_to_rsu.get(reported_segment, "")
        )
        flagged = adjacency_violation and region_change
        self.last_reported_segment[vehicle_id] = reported_segment
        return flagged


class SequentialRobustKG:
    """Sequential KG extension with RSU and peer witness evidence plus regional reliability tracking."""

    def __init__(self, adjacency: Dict[str, set[str]], segment_to_rsu: Dict[str, str]) -> None:
        self.adjacency = adjacency
        self.segment_to_rsu = segment_to_rsu
        self.trusted_segment: Dict[str, str] = {}
        self.last_reported_segment: Dict[str, str] = {}
        self.vehicle_suspicion: Dict[str, float] = defaultdict(float)
        self.region_events: Dict[str, Deque[Tuple[int, str, bool]]] = defaultdict(deque)
        self.region_gps_quality: Dict[str, Tuple[int, bool]] = {}
        self.tick_witnesses: Dict[Tuple[int, str], List[str]] = defaultdict(list)
        self.pending_reports: Dict[str, Deque[PendingReport]] = defaultdict(deque)
        self.spoofing_suspect_reports: Dict[str, Deque[SuspiciousTrajectoryReport]] = defaultdict(deque)
        self.finalized_reports: Deque[Tuple[Message, str]] = deque()
        self.window = 3
        self.min_region_vehicles = 5
        self.suspicious_trajectory_min_reports = 2

    def observe(self, message: Message) -> Optional[str]:
        if message.claim_type == "rsu_beacon_observation":
            key = (message.tick, message.target_vehicle or "")
            if message.witness_segment:
                self.tick_witnesses[key].append(message.witness_segment)
            return None

        if message.claim_type == "peer_beacon_observation":
            key = (message.tick, message.target_vehicle or "")
            if message.witness_segment:
                self.tick_witnesses[key].append(message.witness_segment)
            return None

        if message.claim_type == "rsu_gps_quality":
            if message.rsu_region and message.gps_quality_ok is not None:
                self.region_gps_quality[message.rsu_region] = (message.tick, message.gps_quality_ok)
            return None

        if message.claim_type != "vehicle_position":
            return None

        vehicle_id = message.claimed_vehicle or message.sender_id
        claimed_segment = message.segment or ""
        previous = self.trusted_segment.get(vehicle_id)
        previous_reported = self.last_reported_segment.get(vehicle_id)
        witnesses = self.tick_witnesses.pop((message.tick, vehicle_id), [])
        supported_segment = self._supported_segment(witnesses)

        route_inconsistent = bool(
            previous
            and claimed_segment not in self.adjacency.get(previous, set())
            and claimed_segment != previous
        )
        witness_mismatch = bool(supported_segment and claimed_segment != supported_segment)
        witness_inconsistent = self._witness_inconsistent(claimed_segment, witnesses)
        same_region_as_previous = self.segment_to_rsu.get(previous or "", "") == self.segment_to_rsu.get(claimed_segment, "")
        reported_step_consistent = bool(
            previous_reported
            and (
                claimed_segment == previous_reported
                or claimed_segment in self.adjacency.get(previous_reported, set())
            )
        )
        region_anchor = supported_segment or claimed_segment
        region = self.segment_to_rsu.get(region_anchor, "unknown")
        regional_signal = witness_mismatch or route_inconsistent
        self._record_region_event(region, message.tick, vehicle_id, regional_signal)
        degraded_region = self._region_is_degraded(region, message.tick)
        pending_label = self._resolve_pending_reports(
            vehicle_id=vehicle_id,
            claimed_segment=claimed_segment,
            supported_segment=supported_segment,
            witnesses=witnesses,
            route_inconsistent=route_inconsistent,
            witness_inconsistent=witness_inconsistent,
            degraded_region=degraded_region,
            regional_signal=regional_signal,
            reported_step_consistent=reported_step_consistent,
        )
        should_buffer_ungrounded = self._should_buffer_report(
            claimed_segment=claimed_segment,
            witnesses=witnesses,
            route_inconsistent=route_inconsistent,
            degraded_region=degraded_region,
            reported_step_consistent=reported_step_consistent,
            same_region_as_previous=same_region_as_previous,
        )
        if should_buffer_ungrounded:
            self.pending_reports[vehicle_id].append(
                PendingReport(
                    message=message,
                    previous_trusted=previous,
                    previous_reported=previous_reported,
                    claimed_segment=claimed_segment,
                    region=region,
                    reported_step_consistent=reported_step_consistent,
                    same_region_as_previous=same_region_as_previous,
                )
            )
            self.last_reported_segment[vehicle_id] = claimed_segment
            return None

        suspicious_candidate = bool(witnesses) and not degraded_region and (
            witness_mismatch
            or (route_inconsistent and not reported_step_consistent)
            or (reported_step_consistent and witness_inconsistent)
        )
        suspicious_action = self._update_spoofing_suspect_reports(
            vehicle_id=vehicle_id,
            message=message,
            claimed_segment=claimed_segment,
            route_inconsistent=route_inconsistent,
            witness_mismatch=witness_mismatch,
            reported_step_consistent=reported_step_consistent,
            suspicious_candidate=suspicious_candidate,
        )
        if suspicious_action == "buffered":
            self.last_reported_segment[vehicle_id] = claimed_segment
            return None
        if suspicious_action == "resolved_with_current":
            self.last_reported_segment[vehicle_id] = claimed_segment
            return None

        suspicion_delta = 0.0
        if witness_inconsistent and not degraded_region:
            suspicion_delta += 1.0
        if reported_step_consistent and witness_inconsistent and not degraded_region:
            suspicion_delta += 0.75
        if route_inconsistent and (witnesses or not reported_step_consistent) and not degraded_region:
            suspicion_delta += 1.0
        if witness_inconsistent and route_inconsistent and not degraded_region:
            suspicion_delta += 0.5
        if degraded_region and witness_inconsistent:
            suspicion_delta -= 0.75

        suspicion = max(0.0, self.vehicle_suspicion.get(vehicle_id, 0.0) + suspicion_delta - 0.25)
        self.vehicle_suspicion[vehicle_id] = suspicion

        predicted_label = BENIGN_LABEL
        if pending_label in {"intent_hiding", "position_spoofing", "regional_gps_failure"}:
            predicted_label = pending_label
        elif degraded_region and regional_signal:
            predicted_label = "regional_gps_failure"
        elif reported_step_consistent and witness_mismatch and suspicion >= 0.25:
            predicted_label = "intent_hiding"
        elif reported_step_consistent and witness_inconsistent and suspicion >= 0.75:
            predicted_label = "intent_hiding"
        elif reported_step_consistent and suspicion >= 1.0:
            predicted_label = "intent_hiding"
        elif witness_mismatch:
            predicted_label = "position_spoofing"
        elif route_inconsistent and not same_region_as_previous and not reported_step_consistent:
            predicted_label = "position_spoofing"
        elif suspicion >= 2.0 and same_region_as_previous:
            predicted_label = "intent_hiding"

        if witnesses and (predicted_label in {BENIGN_LABEL, "regional_gps_failure"}):
            self.trusted_segment[vehicle_id] = supported_segment
        elif predicted_label != BENIGN_LABEL and previous:
            self.trusted_segment[vehicle_id] = previous
        elif witnesses and supported_segment:
            self.trusted_segment[vehicle_id] = supported_segment
        else:
            self.trusted_segment[vehicle_id] = claimed_segment
        self.last_reported_segment[vehicle_id] = claimed_segment
        return predicted_label

    def drain_finalized(self) -> List[Tuple[Message, str]]:
        finalized = list(self.finalized_reports)
        self.finalized_reports.clear()
        return finalized

    def flush(self) -> List[Tuple[Message, str]]:
        for vehicle_id, pending in list(self.pending_reports.items()):
            while pending:
                item = pending.popleft()
                self.finalized_reports.append((item.message, UNRESOLVED_LABEL))
            self.pending_reports.pop(vehicle_id, None)
        for vehicle_id, pending in list(self.spoofing_suspect_reports.items()):
            label = self._classify_suspicious_trajectory(list(pending))
            while pending:
                item = pending.popleft()
                self.finalized_reports.append((item.message, label))
            self.spoofing_suspect_reports.pop(vehicle_id, None)
        return self.drain_finalized()

    def _expanded_neighborhood(self, segments: Iterable[str]) -> set[str]:
        expanded = set()
        for segment in segments:
            expanded.add(segment)
            expanded.update(self.adjacency.get(segment, set()))
        return expanded

    def _witness_inconsistent(self, claimed_segment: str, witnesses: Sequence[str]) -> bool:
        if not witnesses:
            return False
        supported = self._expanded_neighborhood(witnesses)
        return claimed_segment not in supported

    def _supported_segment(self, witnesses: Sequence[str]) -> str:
        if not witnesses:
            return ""
        return Counter(witnesses).most_common(1)[0][0]

    def _should_buffer_report(
        self,
        *,
        claimed_segment: str,
        witnesses: Sequence[str],
        route_inconsistent: bool,
        degraded_region: bool,
        reported_step_consistent: bool,
        same_region_as_previous: bool,
    ) -> bool:
        return (
            not witnesses
            and not degraded_region
            and reported_step_consistent
            and same_region_as_previous
            and bool(claimed_segment)
        )

    def _resolve_pending_reports(
        self,
        *,
        vehicle_id: str,
        claimed_segment: str,
        supported_segment: str,
        witnesses: Sequence[str],
        route_inconsistent: bool,
        witness_inconsistent: bool,
        degraded_region: bool,
        regional_signal: bool,
        reported_step_consistent: bool,
    ) -> Optional[str]:
        pending = self.pending_reports.get(vehicle_id)
        if not pending:
            return None

        label: Optional[str] = None
        if degraded_region and regional_signal:
            label = "regional_gps_failure"
        elif reported_step_consistent and supported_segment and claimed_segment != supported_segment:
            label = "intent_hiding"
        elif witness_inconsistent:
            label = "intent_hiding" if reported_step_consistent else "position_spoofing"
        elif witnesses and supported_segment:
            label = BENIGN_LABEL

        if label is None:
            return None

        while pending:
            item = pending.popleft()
            self.finalized_reports.append((item.message, label))
        self.pending_reports.pop(vehicle_id, None)

        if label in {BENIGN_LABEL, "regional_gps_failure"} and supported_segment:
            self.trusted_segment[vehicle_id] = supported_segment
        return label

    def _segments_consecutive_or_stationary(self, previous_segment: str, next_segment: str) -> bool:
        return (
            previous_segment == next_segment
            or next_segment in self.adjacency.get(previous_segment, set())
        )

    def _classify_suspicious_trajectory(
        self,
        reports: Sequence[SuspiciousTrajectoryReport],
    ) -> str:
        if not reports:
            return "position_spoofing"
        if len(reports) < self.suspicious_trajectory_min_reports:
            return "position_spoofing"
        claims = [item.claimed_segment for item in reports]
        trajectory_like = all(
            self._segments_consecutive_or_stationary(prev, curr)
            for prev, curr in zip(claims, claims[1:])
        )
        if trajectory_like:
            return "intent_hiding"
        return "position_spoofing"

    def _update_spoofing_suspect_reports(
        self,
        *,
        vehicle_id: str,
        message: Message,
        claimed_segment: str,
        route_inconsistent: bool,
        witness_mismatch: bool,
        reported_step_consistent: bool,
        suspicious_candidate: bool,
    ) -> str:
        pending = self.spoofing_suspect_reports.get(vehicle_id)
        if not pending and not suspicious_candidate:
            return "noop"
        if not pending and suspicious_candidate:
            self.spoofing_suspect_reports[vehicle_id].append(
                SuspiciousTrajectoryReport(
                    message=message,
                    claimed_segment=claimed_segment,
                    route_inconsistent=route_inconsistent,
                    witness_mismatch=witness_mismatch,
                    reported_step_consistent=reported_step_consistent,
                )
            )
            return "buffered"

        if pending and suspicious_candidate:
            pending.append(
                SuspiciousTrajectoryReport(
                    message=message,
                    claimed_segment=claimed_segment,
                    route_inconsistent=route_inconsistent,
                    witness_mismatch=witness_mismatch,
                    reported_step_consistent=reported_step_consistent,
                )
            )
            label = self._classify_suspicious_trajectory(list(pending))
            if label == "intent_hiding" or len(pending) >= self.suspicious_trajectory_min_reports:
                while pending:
                    item = pending.popleft()
                    self.finalized_reports.append((item.message, label))
                self.spoofing_suspect_reports.pop(vehicle_id, None)
                return "resolved_with_current"
            return "buffered"

        label = self._classify_suspicious_trajectory(list(pending))
        while pending:
            item = pending.popleft()
            self.finalized_reports.append((item.message, label))
        self.spoofing_suspect_reports.pop(vehicle_id, None)
        return "resolved_previous_only"

    def _record_region_event(self, region: str, tick: int, vehicle_id: str, mismatch: bool) -> None:
        events = self.region_events[region]
        events.append((tick, vehicle_id, mismatch))
        while events and tick - events[0][0] > self.window:
            events.popleft()

    def _region_is_degraded(self, region: str, tick: int) -> bool:
        events = self.region_events.get(region, deque())
        recent = [(vehicle_id, mismatch) for event_tick, vehicle_id, mismatch in events if tick - event_tick <= self.window]
        distinct_vehicles = {vehicle_id for vehicle_id, _ in recent}
        impacted_vehicles = {vehicle_id for vehicle_id, mismatch in recent if mismatch}
        mismatch_count = sum(1 for _, mismatch in recent if mismatch)
        tracking_trigger = (
            len(distinct_vehicles) >= self.min_region_vehicles
            and len(impacted_vehicles) >= 3
            and len(recent) > 0
            and mismatch_count / len(recent) >= 0.55
        )
        gps_state = self._rsu_gps_health_state(region, tick)
        if gps_state == "poor":
            return True
        if gps_state == "good":
            return False
        return tracking_trigger

    def _rsu_gps_health_state(self, region: str, tick: int) -> str:
        last = self.region_gps_quality.get(region)
        if not last:
            return "unknown"
        quality_tick, gps_quality_ok = last
        if tick - quality_tick > self.window:
            return "unknown"
        return "good" if gps_quality_ok else "poor"


class ProtectedPositionKG:
    """Trusted-state wrapper that protects the original KG position rule."""

    def __init__(self, adjacency: Dict[str, set[str]], segment_to_rsu: Dict[str, str]) -> None:
        self.adjacency = adjacency
        self.segment_to_rsu = segment_to_rsu
        self.sequential = SequentialRobustKG(adjacency, segment_to_rsu)
        self.delay_ticks = 1

    def observe(self, message: Message) -> Optional[bool]:
        if message.claim_type != "vehicle_position":
            self.sequential.observe(message)
            return None

        vehicle_id = message.claimed_vehicle or message.sender_id
        reported_segment = message.segment or ""
        previous_trusted = self.sequential.trusted_segment.get(vehicle_id)
        witnesses = list(self.sequential.tick_witnesses.get((message.tick, vehicle_id), []))
        has_witnesses = bool(witnesses)
        witness_inconsistent = self.sequential._witness_inconsistent(reported_segment, witnesses)
        predicted_label = self.sequential.observe(message)
        finalized = self.sequential.drain_finalized()
        finalized_current_label = next(
            (label for finalized_message, label in finalized if finalized_message.message_id == message.message_id),
            None,
        )
        predicted_label = finalized_current_label or predicted_label or BENIGN_LABEL
        if predicted_label == UNRESOLVED_LABEL:
            return None
        has_pending_buffer = bool(self.sequential.pending_reports.get(vehicle_id))
        has_trajectory_buffer = bool(self.sequential.spoofing_suspect_reports.get(vehicle_id))
        if predicted_label == BENIGN_LABEL and has_pending_buffer:
            return False

        adjacency_violation = bool(
            previous_trusted
            and reported_segment not in self.adjacency.get(previous_trusted, set())
            and reported_segment != previous_trusted
        )
        region_change = bool(
            previous_trusted
            and self.segment_to_rsu.get(previous_trusted, "") != self.segment_to_rsu.get(reported_segment, "")
        )
        baseline_flag = adjacency_violation and region_change

        if predicted_label == BENIGN_LABEL and has_trajectory_buffer:
            return baseline_flag

        if predicted_label == "regional_gps_failure":
            return False
        return (
            baseline_flag
            and predicted_label in INTRUSION_POSITIVE_LABELS
            and (
                witness_inconsistent
                or (not has_witnesses and predicted_label == "position_spoofing")
            )
        )


def collect_unresolved_report_ids(detector_factory, episodes) -> List[set[str]]:
    unresolved_by_episode: List[set[str]] = []
    for messages, _, _ in episodes:
        detector = detector_factory()
        unresolved_ids: set[str] = set()
        for message in messages:
            predicted_label = detector.observe(message)
            for finalized_message, finalized_label in detector.drain_finalized():
                if finalized_label == UNRESOLVED_LABEL:
                    unresolved_ids.add(finalized_message.message_id)
            if message.claim_type == "vehicle_position" and predicted_label == UNRESOLVED_LABEL:
                unresolved_ids.add(message.message_id)
        for finalized_message, finalized_label in detector.flush():
            if finalized_label == UNRESOLVED_LABEL:
                unresolved_ids.add(finalized_message.message_id)
        unresolved_by_episode.append(unresolved_ids)
    return unresolved_by_episode

def segment_hops(seed_segments: Sequence[str], adjacency: Dict[str, set[str]], depth: int) -> set[str]:
    visited = set(seed_segments)
    frontier = set(seed_segments)
    for _ in range(max(0, depth)):
        next_frontier: set[str] = set()
        for segment in frontier:
            next_frontier.update(adjacency.get(segment, set()))
        next_frontier -= visited
        visited.update(next_frontier)
        frontier = next_frontier
        if not frontier:
            break
    return visited


def build_focus_context(city_key: str = CITY_KEY, seed: int = 42) -> dict:
    context = build_real_city_context(city_key, grid_size=GRID_SIZE)
    context["city"] = city_key
    context["trip_library"] = build_trip_library(context, seed=seed, num_trips=ROUTE_LIBRARY_CAP)
    context["active_segments"] = sorted({segment for route in context["trip_library"] for segment in route})
    return context


def _primary_region_segments(context: dict) -> Dict[str, List[str]]:
    by_region: Dict[str, List[str]] = defaultdict(list)
    for segment in context["active_segments"]:
        by_region[context["segment_to_rsu"][segment]].append(segment)
    return by_region


def _pick_drift_region(context: dict) -> Tuple[str, set[str]]:
    region_segments = _primary_region_segments(context)
    region, segments = max(region_segments.items(), key=lambda item: len(item[1]))
    return region, set(segments)


def _gps_failed_claim(context: dict, true_segment: str, candidate_segments: Sequence[str], rng: random.Random) -> str:
    region = context["segment_to_rsu"][true_segment]
    same_region = [segment for segment in candidate_segments if context["segment_to_rsu"].get(segment) == region and segment != true_segment]
    pool = same_region or [segment for segment in candidate_segments if segment != true_segment]
    return rng.choice(pool) if pool else true_segment


def _position_spoofing_claim(
    context: dict,
    true_segment: str,
    previous_claim: Optional[str],
    rng: random.Random,
) -> str:
    pool = [
        segment
        for segment in context["active_segments"]
        if segment not in context["adjacency"].get(true_segment, set())
        and segment != true_segment
        and segment != previous_claim
    ]
    if not pool:
        pool = [
            segment
            for segment in context["active_segments"]
            if segment not in context["adjacency"].get(true_segment, set()) and segment != true_segment
        ]
    return rng.choice(pool) if pool else true_segment


def _intent_hiding_claim(context: dict, vehicle: Vehicle) -> str:
    previous_claim = vehicle.last_claimed_segment or vehicle.current_segment
    true_segment = vehicle.current_segment
    candidates = [
        segment
        for segment in context["adjacency"].get(previous_claim, set())
        if segment != true_segment
    ]
    same_region = [
        segment
        for segment in candidates
        if context["segment_to_rsu"].get(segment) == context["segment_to_rsu"].get(previous_claim, "")
    ]
    pool = same_region or candidates
    return pool[0] if pool else true_segment


def _peer_witness_segments(
    vehicle: Vehicle,
    vehicles: Sequence[Vehicle],
    adjacency: Dict[str, set[str]],
    segment_to_rsu: Dict[str, str],
) -> List[str]:
    witnesses: List[str] = []
    neighborhood = {vehicle.current_segment, *adjacency.get(vehicle.current_segment, set())}
    governing_region = segment_to_rsu.get(vehicle.current_segment, "")
    for other in vehicles:
        if other.vehicle_id == vehicle.vehicle_id:
            continue
        if (
            other.current_segment in neighborhood
            and segment_to_rsu.get(other.current_segment, "") == governing_region
        ):
            witnesses.append(vehicle.current_segment)
        if len(witnesses) >= 2:
            break
    return witnesses


def _select_attackers(
    vehicles: Sequence[Vehicle],
    *,
    rng: random.Random,
    scenario: str,
    attacker_fraction: float,
) -> Dict[str, str]:
    if scenario not in {"position_spoofing", "intent_hiding"} or not vehicles:
        return {}
    attacker_count = max(1, int(round(len(vehicles) * attacker_fraction)))
    chosen = rng.sample(list(vehicles), min(attacker_count, len(vehicles)))
    return {vehicle.vehicle_id: scenario for vehicle in chosen}


def generate_episode(
    context: dict,
    scenario: str,
    seed: int,
    ticks: int = 18,
    attacker_fraction: float = DEFAULT_ATTACKER_FRACTION,
) -> Tuple[List[Message], Optional[str], int]:
    rng = random.Random(seed)
    vehicles: List[Vehicle] = []
    for index in range(EPISODE_VEHICLES):
        route = list(rng.choice(context["trip_library"]))
        if route:
            vehicles.append(Vehicle(vehicle_id=f"V{index + 1}", route=route))

    attackers = _select_attackers(
        vehicles,
        rng=rng,
        scenario=scenario,
        attacker_fraction=attacker_fraction,
    )
    drift_region, degraded_segments = _pick_drift_region(context)
    messages: List[Message] = []
    counter = 0

    def next_id() -> str:
        nonlocal counter
        counter += 1
        return f"m{counter:06d}"

    for tick in range(ticks):
        if tick and tick % 3 == 0:
            for vehicle in vehicles:
                if rng.random() < 0.65:
                    vehicle.move()

        present_regions = sorted({context["segment_to_rsu"][vehicle.current_segment] for vehicle in vehicles})
        attack_active = scenario == "regional_gps_failure" and ATTACK_START <= tick < ATTACK_START + ATTACK_DURATION
        for region in present_regions:
            messages.append(
                Message(
                    message_id=next_id(),
                    tick=tick,
                    sender_id=f"rsu_quality_{region}",
                    sender_type="rsu",
                    claim_type="rsu_gps_quality",
                    scenario=scenario,
                    rsu_region=region,
                    gps_quality_ok=not (attack_active and region == drift_region),
                )
            )

        for vehicle in vehicles:
            true_segment = vehicle.current_segment
            claimed_segment = true_segment
            malicious = False
            label = BENIGN_LABEL

            in_attack_window = ATTACK_START <= tick < ATTACK_START + ATTACK_DURATION
            if scenario == "regional_gps_failure" and in_attack_window and true_segment in degraded_segments:
                candidate_segments = list(degraded_segments)
                claimed_segment = _gps_failed_claim(context, true_segment, candidate_segments, rng)
                malicious = claimed_segment != true_segment
                if malicious:
                    label = "regional_gps_failure"
            elif attackers.get(vehicle.vehicle_id) == "position_spoofing" and in_attack_window:
                claimed_segment = _position_spoofing_claim(context, true_segment, vehicle.last_claimed_segment, rng)
                malicious = True
                label = "position_spoofing"
            elif attackers.get(vehicle.vehicle_id) == "intent_hiding" and in_attack_window:
                claimed_segment = _intent_hiding_claim(context, vehicle)
                malicious = claimed_segment != true_segment
                if malicious:
                    label = "intent_hiding"

            if true_segment in context["observed_segments"]:
                messages.append(
                    Message(
                        message_id=next_id(),
                        tick=tick,
                        sender_id=context["segment_to_rsu"][true_segment],
                        sender_type="rsu",
                        claim_type="rsu_beacon_observation",
                        scenario=scenario,
                        target_vehicle=vehicle.vehicle_id,
                        witness_segment=true_segment,
                    )
                )

            for witness_segment in _peer_witness_segments(
                vehicle,
                vehicles,
                context["adjacency"],
                context["segment_to_rsu"],
            ):
                messages.append(
                    Message(
                        message_id=next_id(),
                        tick=tick,
                        sender_id=f"peer_{vehicle.vehicle_id}",
                        sender_type="vehicle",
                        claim_type="peer_beacon_observation",
                        scenario=scenario,
                        target_vehicle=vehicle.vehicle_id,
                        witness_segment=witness_segment,
                    )
                )

            messages.append(
                Message(
                    message_id=next_id(),
                    tick=tick,
                    sender_id=vehicle.vehicle_id,
                    sender_type="vehicle",
                    claim_type="vehicle_position",
                    malicious=malicious,
                    scenario=scenario,
                    label=label,
                    segment=claimed_segment,
                    claimed_vehicle=vehicle.vehicle_id,
                )
            )
            vehicle.last_claimed_segment = claimed_segment

        for vehicle in vehicles[:6]:
            intersection = context["segment_to_intersection"][vehicle.current_segment]
            messages.append(
                Message(
                    message_id=next_id(),
                    tick=tick,
                    sender_id=vehicle.vehicle_id,
                    sender_type="vehicle",
                    claim_type="signal_observation",
                    scenario=scenario,
                    intersection=intersection,
                    signal_state=signal_state(tick, intersection),
                )
            )

    attack_label = scenario if scenario in {"regional_gps_failure", "position_spoofing", "intent_hiding"} else None
    return messages, attack_label, ATTACK_START


def evaluate_detector(detector_factory, episodes, detector_name: str, excluded_message_ids: Optional[List[set[str]]] = None) -> Dict[str, object]:
    stats = MulticlassStats(CLASS_LABELS)

    excluded_message_ids = excluded_message_ids or [set() for _ in episodes]

    for episode_index, (messages, _, attack_start) in enumerate(episodes):
        detector = detector_factory()
        first_correct_detection: Dict[str, int] = {}
        excluded_ids = excluded_message_ids[episode_index]

        def record_prediction(source_message: Message, predicted: str) -> None:
            if source_message.message_id in excluded_ids or predicted == UNRESOLVED_LABEL:
                return
            true_label = source_message.label
            stats.record(true_label, predicted)
            if (
                true_label != BENIGN_LABEL
                and predicted == true_label
                and true_label not in first_correct_detection
            ):
                first_correct_detection[true_label] = source_message.tick

        for message in messages:
            predicted_label = detector.observe(message)
            for finalized_message, finalized_label in detector.drain_finalized():
                record_prediction(finalized_message, finalized_label)
            if message.claim_type != "vehicle_position":
                continue
            if predicted_label is not None:
                record_prediction(message, predicted_label)

        for finalized_message, finalized_label in detector.flush():
            record_prediction(finalized_message, finalized_label)

        for label, tick in first_correct_detection.items():
            stats.latencies[label].append(tick - attack_start)

    per_class = {
        label: {
            key: round(value, 4) if isinstance(value, float) else value
            for key, value in stats.per_class_metrics(label).items()
        }
        for label in CLASS_LABELS
    }
    confusion_matrix = {
        true_label: {predicted_label: stats.confusion[true_label][predicted_label] for predicted_label in CLASS_LABELS}
        for true_label in CLASS_LABELS
    }

    return {
        "detector": detector_name,
        "accuracy": round(stats.accuracy(), 4),
        "macro_precision": round(stats.macro_precision(), 4),
        "macro_recall": round(stats.macro_recall(), 4),
        "macro_f1": round(stats.macro_f1(), 4),
        "average_latency": round(stats.average_latency(), 4),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix,
    }


def evaluate_binary_detector(detector_factory, episodes, detector_name: str, excluded_message_ids: Optional[List[set[str]]] = None) -> Dict[str, object]:
    stats = BinaryDetectionStats()
    per_scenario: Dict[str, BinaryDetectionStats] = {scenario: BinaryDetectionStats() for scenario in SCENARIOS}

    excluded_message_ids = excluded_message_ids or [set() for _ in episodes]

    for episode_index, (messages, _, attack_start) in enumerate(episodes):
        detector = detector_factory()
        delay_ticks = getattr(detector, "delay_ticks", 0)
        first_detection_tick: Dict[str, int] = {}
        excluded_ids = excluded_message_ids[episode_index]

        for message in messages:
            flagged = detector.observe(message)
            if message.claim_type != "vehicle_position":
                continue
            if message.message_id in excluded_ids or flagged is None:
                continue
            scenario = message.label
            intrusion_positive = scenario in INTRUSION_POSITIVE_LABELS
            predicted_positive = bool(flagged)

            bucket = per_scenario[scenario]
            if intrusion_positive and predicted_positive:
                stats.tp += 1
                bucket.tp += 1
                if scenario not in first_detection_tick:
                    first_detection_tick[scenario] = message.tick + delay_ticks
            elif intrusion_positive and not predicted_positive:
                stats.fn += 1
                bucket.fn += 1
            elif not intrusion_positive and predicted_positive:
                stats.fp += 1
                bucket.fp += 1
            else:
                stats.tn += 1
                bucket.tn += 1

        for scenario, tick in first_detection_tick.items():
            stats.latencies.append(tick - attack_start)
            per_scenario[scenario].latencies.append(tick - attack_start)

    return {
        "detector": detector_name,
        "precision": round(stats.precision(), 4),
        "recall": round(stats.recall(), 4),
        "f1": round(stats.f1(), 4),
        "false_positive_rate": round(stats.false_positive_rate(), 4),
        "average_latency": round(stats.average_latency(), 4),
        "per_scenario": {
            scenario: {
                "precision": round(bucket.precision(), 4),
                "recall": round(bucket.recall(), 4),
                "f1": round(bucket.f1(), 4),
                "false_positive_rate": round(bucket.false_positive_rate(), 4),
                "average_latency": round(bucket.average_latency(), 4),
            }
            for scenario, bucket in per_scenario.items()
        },
    }


def _build_episode_splits(context: dict, seed: int, episodes_per_scenario: int) -> Tuple[List[Tuple[List[Message], Optional[str], int]], List[Tuple[List[Message], Optional[str], int]]]:
    train_episodes: List[Tuple[List[Message], Optional[str], int]] = []
    test_episodes: List[Tuple[List[Message], Optional[str], int]] = []
    train_per_scenario = max(1, int(episodes_per_scenario * TRAIN_RATIO))
    test_per_scenario = max(1, episodes_per_scenario - train_per_scenario)
    counter = 0
    for scenario in SCENARIOS:
        print(f"[focus] generating train episodes for {scenario}", flush=True)
        for _ in range(train_per_scenario):
            train_episodes.append(
                generate_episode(
                    context,
                    scenario,
                    seed + counter,
                    attacker_fraction=DEFAULT_ATTACKER_FRACTION,
                )
            )
            counter += 1
        print(f"[focus] generating test episodes for {scenario}", flush=True)
        for _ in range(test_per_scenario):
            attacker_fraction = TEST_ATTACKER_FRACTION if scenario in {"position_spoofing", "intent_hiding"} else DEFAULT_ATTACKER_FRACTION
            test_episodes.append(
                generate_episode(
                    context,
                    scenario,
                    seed + counter,
                    attacker_fraction=attacker_fraction,
                )
            )
            counter += 1
    return train_episodes, test_episodes


def run_experiment(seed: int = 42, episodes_per_scenario: int = 8) -> Dict[str, object]:
    context = build_focus_context(seed=seed)
    print(f"[focus] built {context['city'].title()} context with {len(context['active_segments'])} active segments", flush=True)

    train_episodes, test_episodes = _build_episode_splits(context, seed, episodes_per_scenario)
    print(f"[focus] evaluating on {len(test_episodes)} held-out episodes", flush=True)
    unresolved_ids = collect_unresolved_report_ids(
        lambda: SequentialRobustKG(context["adjacency"], context["segment_to_rsu"]),
        test_episodes,
    )

    robust = evaluate_detector(
        lambda: SequentialRobustKG(context["adjacency"], context["segment_to_rsu"]),
        test_episodes,
        "sequential_robust_kg",
        excluded_message_ids=unresolved_ids,
    )
    baseline_binary = evaluate_binary_detector(
        lambda: OriginalPositionKG(context["adjacency"], context["segment_to_rsu"]),
        test_episodes,
        "baseline_kg_raw",
        excluded_message_ids=unresolved_ids,
    )
    protected_binary = evaluate_binary_detector(
        lambda: ProtectedPositionKG(context["adjacency"], context["segment_to_rsu"]),
        test_episodes,
        "baseline_kg_protected",
        excluded_message_ids=unresolved_ids,
    )

    return {
        "city": context["city"],
        "grid_size": GRID_SIZE,
        "route_library_cap": ROUTE_LIBRARY_CAP,
        "episode_vehicles": EPISODE_VEHICLES,
        "episodes_per_scenario": episodes_per_scenario,
        "train_episodes": len(train_episodes),
        "test_episodes": len(test_episodes),
        "attack_start_tick": ATTACK_START,
        "attack_duration_ticks": ATTACK_DURATION,
        "active_segments": len(context["active_segments"]),
        "observed_segments": len(context["observed_segments"]),
        "monitored_intersections": len(context["monitored_intersections"]),
        "observation_profile": context["observation_profile"],
        "class_labels": list(CLASS_LABELS),
        "peer_witness_policy": "same_rsu_region_only",
        "ungrounded_position_policy": "buffer_and_reidentify",
        "trajectory_buffer_policy": "buffer_spoofing_suspects_and_escalate_adjacent_or_stationary_sequences",
        "evaluation_exclusion_policy": "exclude_unresolved_ungrounded_reports_from_comparative_metrics",
        "train_attacker_fraction": round(DEFAULT_ATTACKER_FRACTION, 4),
        "test_attacker_fraction": TEST_ATTACKER_FRACTION,
        "test_attack_mix": {
            "position_spoofing_share": 0.5,
            "intent_hiding_share": 0.5,
        },
        "track_a_state_discovery": {
            "class_labels": list(CLASS_LABELS),
            "sequential_robust_kg": robust,
        },
        "track_b_baseline_protection": {
            "positive_operational_states": sorted(INTRUSION_POSITIVE_LABELS),
            "negative_operational_states": [BENIGN_LABEL, "regional_gps_failure"],
            "baseline_kg_raw": baseline_binary,
            "baseline_kg_protected": protected_binary,
        },
    }


def write_outputs(results: Dict[str, object]) -> None:
    root = Path(__file__).resolve().parents[1]
    outdir = root / "results" / "paper_seq"
    outdir.mkdir(parents=True, exist_ok=True)

    with (outdir / "kg_sequential_focus_results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    rows = []
    for detector_key in ("baseline_kg_raw", "baseline_kg_protected"):
        detector = results["track_b_baseline_protection"][detector_key]
        rows.append(
            {
                "detector": detector["detector"],
                "precision": detector["precision"],
                "recall": detector["recall"],
                "f1": detector["f1"],
                "false_positive_rate": detector["false_positive_rate"],
                "average_latency": detector["average_latency"],
            }
        )

    with (outdir / "kg_sequential_focus_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    scenario_rows = []
    sequential = results["track_a_state_discovery"]["sequential_robust_kg"]
    for label, metrics in sequential["per_class"].items():
        scenario_rows.append(
            {
                "detector": sequential["detector"],
                "scenario": label,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "false_positive_rate": metrics["false_positive_rate"],
                "average_latency": metrics["average_latency"],
                "support": metrics["support"],
            }
        )

    with (outdir / "kg_sequential_focus_scenarios.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scenario_rows[0].keys()))
        writer.writeheader()
        writer.writerows(scenario_rows)

    binary_scenario_rows = []
    for detector_key in ("baseline_kg_raw", "baseline_kg_protected"):
        detector = results["track_b_baseline_protection"][detector_key]
        for scenario, metrics in detector["per_scenario"].items():
            binary_scenario_rows.append(
                {
                    "detector": detector["detector"],
                    "scenario": scenario,
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "false_positive_rate": metrics["false_positive_rate"],
                    "average_latency": metrics["average_latency"],
                }
            )
    with (outdir / "kg_sequential_focus_binary_scenarios.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(binary_scenario_rows[0].keys()))
        writer.writeheader()
        writer.writerows(binary_scenario_rows)
    confusion_rows = []
    detector = results["track_a_state_discovery"]["sequential_robust_kg"]
    for true_label, predicted_counts in detector["confusion_matrix"].items():
        for predicted_label, count in predicted_counts.items():
            confusion_rows.append(
                {
                    "detector": detector["detector"],
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "count": count,
                }
            )
    with (outdir / "kg_sequential_focus_confusion.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(confusion_rows[0].keys()))
        writer.writeheader()
        writer.writerows(confusion_rows)
    metadata = {
        "city": results["city"],
        "grid_size": results["grid_size"],
        "route_library_cap": results["route_library_cap"],
        "episode_vehicles": results["episode_vehicles"],
        "episodes_per_scenario": results["episodes_per_scenario"],
        "train_episodes": results["train_episodes"],
        "test_episodes": results["test_episodes"],
        "attack_start_tick": results["attack_start_tick"],
        "attack_duration_ticks": results["attack_duration_ticks"],
        "active_segments": results["active_segments"],
        "observed_segments": results["observed_segments"],
        "monitored_intersections": results["monitored_intersections"],
        "observation_profile": results["observation_profile"],
        "class_labels": results["track_a_state_discovery"]["class_labels"],
        "peer_witness_policy": results["peer_witness_policy"],
        "ungrounded_position_policy": results["ungrounded_position_policy"],
        "trajectory_buffer_policy": results["trajectory_buffer_policy"],
        "evaluation_exclusion_policy": results["evaluation_exclusion_policy"],
        "train_attacker_fraction": results["train_attacker_fraction"],
        "test_attacker_fraction": results["test_attacker_fraction"],
        "test_attack_mix": results["test_attack_mix"],
        "detectors": [row["detector"] for row in rows],
        "scenarios": list(SCENARIOS),
        "track_a_detector": sequential["detector"],
        "track_b_detectors": [row["detector"] for row in rows],
        "provenance": {
            "generator": "experiments/kg_sequential_focus.py",
        },
    }
    with (outdir / "kg_sequential_focus_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def main() -> None:
    results = run_experiment()
    write_outputs(results)
    print("Sequential KG focus experiment completed. Results written to results/paper_seq.", flush=True)


if __name__ == "__main__":
    main()
