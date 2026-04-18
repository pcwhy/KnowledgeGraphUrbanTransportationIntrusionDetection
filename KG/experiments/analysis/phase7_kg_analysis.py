from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import csv
import json
import sys

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from experiments.common.project_paths import ANALYSIS_RESULTS_DIR, CITY_RESULTS_DIR, ensure_pipeline_directories
from experiments.synthetic.urban_transport_kg_ids import (
    ATTACK_TYPES,
    ADJACENT_SEGMENTS,
    SEGMENT_TO_RSU,
    UrbanTransportExperiment,
    KnowledgeGraphDetector,
    message_label,
    split_episodes,
)
from rendering.scripts.paper_asset_renderers import render_phase7_assets


EPISODES_PER_ATTACK = 25
INTERPRETABILITY_ATTACKS: Sequence[str] = (
    "phantom_congestion",
    "signal_spoofing",
    "position_spoofing",
)


@dataclass
class ExampleRow:
    attack_family: str
    message_id: str
    tick: int
    representative_claim: str
    evidence_path: str
    violated_checks: str
    predicted_label: str


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows available for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[{timestamp()}] wrote {path}", flush=True)


def build_synthetic_test_episodes(seed: int = 42) -> Sequence[Tuple[List[object], Optional[str], int]]:
    experiment = UrbanTransportExperiment(seed)
    episodes = []
    for attack_type in ATTACK_TYPES:
        for _ in range(EPISODES_PER_ATTACK):
            messages, attack_start = experiment.run_episode(attack_type)
            episodes.append((messages, attack_type, attack_start))
    for _ in range(max(5, EPISODES_PER_ATTACK // 5)):
        messages, attack_start = experiment.run_episode(None)
        episodes.append((messages, None, attack_start))
    _, test_episodes = split_episodes(episodes)
    return test_episodes


def _format_signal(value: Optional[str]) -> str:
    return value if value else "unknown"


def explain_synthetic_message(detector: KnowledgeGraphDetector, message) -> ExampleRow:
    if message.claim_type == "congestion_alert":
        target_segment = message.segment or ""
        rsu_density = detector.state.segment_density(target_segment)
        sender_segment = detector.state.vehicle_segment(message.sender_id)
        neighborhood = {target_segment, *detector.state.adjacent_segments(target_segment)}
        checks: List[str] = []
        if (message.density or 0) - rsu_density >= 2:
            checks.append(f"claimed density {int(message.density or 0)} exceeds RSU density {rsu_density}")
        if sender_segment not in neighborhood:
            checks.append(f"sender location {sender_segment or 'unknown'} is outside {target_segment} and its neighbors")
        if rsu_density <= 2:
            checks.append(f"RSU still reports low occupancy ({rsu_density}) on {target_segment}")
        claim = f"{message.sender_id} reports congestion density {int(message.density or 0)} on {target_segment}"
        evidence_path = f"{message.sender_id} -> claimed segment {target_segment} -> RSU density; {message.sender_id} -> previous segment -> adjacent segments"
        predicted = "phantom_congestion" if len(checks) >= detector.alert_threshold else "benign"
        return ExampleRow("Phantom congestion", message.message_id, message.tick, claim, evidence_path, "; ".join(checks), predicted)

    if message.claim_type == "signal_report":
        intersection = message.intersection or ""
        controller = detector.state.controller_signal(intersection)
        majority = detector.state.majority_signal(intersection)
        checks = []
        if controller and controller != message.signal_state:
            checks.append(f"controller at {intersection} is {_format_signal(controller)} while report says {_format_signal(message.signal_state)}")
        if majority and majority != message.signal_state:
            checks.append(f"vehicle majority observation is {_format_signal(majority)}")
        if message.sender_type == "rsu":
            checks.append("sender type is RSU, so an inconsistent signal report is infrastructure-level evidence")
        claim = f"{message.sender_id} reports {_format_signal(message.signal_state)} at {intersection}"
        evidence_path = f"{intersection} -> controller state; vehicles -> observed_signal -> {intersection}; sender -> sender type"
        predicted = "signal_spoofing" if len(checks) >= detector.alert_threshold else "benign"
        return ExampleRow("Signal spoofing", message.message_id, message.tick, claim, evidence_path, "; ".join(checks), predicted)

    if message.claim_type == "vehicle_position":
        claimed_vehicle = message.claimed_vehicle or message.sender_id
        previous = detector.state.vehicle_segment(claimed_vehicle)
        current = message.segment or ""
        checks = []
        if previous and current not in detector.state.adjacent_segments(previous) and current != previous:
            checks.append(f"claimed segment {current} is nonadjacent to previous segment {previous}")
        if SEGMENT_TO_RSU.get(previous or "", "") != SEGMENT_TO_RSU.get(current, ""):
            checks.append(
                f"claim crosses RSU regions {SEGMENT_TO_RSU.get(previous or '', 'unknown')} -> {SEGMENT_TO_RSU.get(current, 'unknown')}"
            )
        claim = f"{claimed_vehicle} claims position on {current}"
        evidence_path = f"{claimed_vehicle} -> previous segment -> adjacent segments; previous segment -> RSU; claimed segment -> RSU"
        predicted = "position_spoofing" if len(checks) >= detector.alert_threshold else "benign"
        return ExampleRow("Position spoofing", message.message_id, message.tick, claim, evidence_path, "; ".join(checks), predicted)

    raise ValueError(f"Unsupported claim type for interpretability example: {message.claim_type}")


def collect_interpretability_examples() -> List[Dict[str, object]]:
    detector = KnowledgeGraphDetector(alert_threshold=2)
    captured: Dict[str, ExampleRow] = {}

    for messages, _, _ in build_synthetic_test_episodes():
        detector = KnowledgeGraphDetector(alert_threshold=2)
        for message in messages:
            true_label = message_label(message)
            if true_label is None:
                detector.predict_label(message)
                continue
            if (
                message.malicious
                and true_label in INTERPRETABILITY_ATTACKS
                and true_label not in captured
            ):
                example = explain_synthetic_message(detector, message)
                prediction = detector.predict_label(message)
                if prediction == true_label:
                    example.predicted_label = prediction
                    captured[true_label] = example
                    continue
            detector.predict_label(message)
        if len(captured) == len(INTERPRETABILITY_ATTACKS):
            break

    if len(captured) != len(INTERPRETABILITY_ATTACKS):
        missing = sorted(set(INTERPRETABILITY_ATTACKS) - set(captured))
        raise RuntimeError(f"Missing interpretability examples for attacks: {missing}")

    order = {
        "phantom_congestion": 0,
        "signal_spoofing": 1,
        "position_spoofing": 2,
    }
    return [
        {
            "attack_family": captured[key].attack_family,
            "message_id": captured[key].message_id,
            "tick": captured[key].tick,
            "representative_claim": captured[key].representative_claim,
            "evidence_path": captured[key].evidence_path,
            "violated_checks": captured[key].violated_checks,
            "predicted_label": captured[key].predicted_label,
        }
        for key in sorted(captured, key=lambda item: order[item])
    ]


def collect_city_failure_summary() -> List[Dict[str, object]]:
    city_results_path = CITY_RESULTS_DIR / "real_city_benchmark.json"
    payload = json.loads(city_results_path.read_text(encoding="utf-8"))
    rows: List[Dict[str, object]] = []
    for item in payload["city_results"]:
        kg = item["knowledge_graph"]
        confusion = kg["confusion_matrix"]
        benign_total = sum(confusion["benign"].values())
        benign_false_alarms = benign_total - confusion["benign"]["benign"]
        benign_false_alarm_rate_pct = 100.0 * benign_false_alarms / max(1, benign_total)
        rows.append(
            {
                "city": item["city"],
                "observed_fraction_pct": round(100.0 * float(item["observation_profile"]["observed_fraction"]), 3),
                "monitored_fraction_pct": round(100.0 * float(item["observation_profile"]["monitored_fraction"]), 3),
                "benign_false_alarm_rate_pct": round(benign_false_alarm_rate_pct, 3),
                "kg_macro_f1": float(kg["macro_f1"]),
                "phantom_recall": float(kg["per_attack"]["phantom_congestion"]["recall"]),
                "signal_recall": float(kg["per_attack"]["signal_spoofing"]["recall"]),
                "closure_recall": float(kg["per_attack"]["false_closure"]["recall"]),
                "position_recall": float(kg["per_attack"]["position_spoofing"]["recall"]),
            }
        )
    return rows


def main() -> None:
    ensure_pipeline_directories()
    ANALYSIS_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    interpretability_rows = collect_interpretability_examples()
    city_failure_rows = collect_city_failure_summary()

    write_csv(ANALYSIS_RESULTS_DIR / "kg_interpretability_examples.csv", interpretability_rows)
    write_csv(ANALYSIS_RESULTS_DIR / "kg_city_failure_summary.csv", city_failure_rows)

    render_phase7_assets(PROJECT_ROOT)


if __name__ == "__main__":
    main()
