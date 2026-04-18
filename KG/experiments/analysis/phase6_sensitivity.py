from __future__ import annotations

from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Sequence, Tuple
import csv
import math
import sys

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from experiments.common.detection_shared import GaussianNaiveBayesModel, MulticlassLogisticRegressionModel
from experiments.common.project_paths import CITY_RESULTS_DIR, SENSITIVITY_RESULTS_DIR, ensure_pipeline_directories
from experiments.synthetic.urban_transport_kg_ids import run_full_experiment
from experiments.city.real_city_benchmark import (
    ATTACK_TYPES,
    DETECTOR_ORDER,
    EPISODE_VEHICLES,
    EPISODES_PER_ATTACK,
    ROUTE_LIBRARY_CAP,
    TRAIN_EPISODES_PER_ATTACK,
    BaselineDetector,
    FlatFeatureLogisticDetector,
    FlatFeatureNaiveBayesDetector,
    KGDetector,
    build_learning_dataset,
    build_real_city_context,
    build_trip_library,
    evaluate_detector,
    run_episode,
    split_episodes,
)
from rendering.scripts.paper_asset_renderers import render_phase6_assets


THRESHOLD_SEEDS: Sequence[int] = (42, 43, 44, 45, 46, 47, 48)
CITY_ORDER: Sequence[str] = ("austin", "houston", "dallas")
CITY_BASE_SEEDS: Dict[str, int] = {
    "austin": 42,
    "houston": 43,
    "dallas": 44,
}
OBSERVATION_LEVELS: Sequence[Tuple[str, float]] = (
    ("Reduced", 0.5),
    ("Sparse", 0.25),
)


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[{timestamp()}] wrote {path}", flush=True)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def clone_city_context(base_context: Dict[str, object]) -> Dict[str, object]:
    cloned = dict(base_context)
    cloned["zone_nodes"] = {key: list(value) for key, value in base_context["zone_nodes"].items()}
    cloned["segment_to_zone"] = dict(base_context["segment_to_zone"])
    cloned["segment_to_intersection"] = dict(base_context["segment_to_intersection"])
    cloned["segment_to_rsu"] = dict(base_context["segment_to_rsu"])
    cloned["segment_coords"] = dict(base_context["segment_coords"])
    cloned["adjacency"] = {key: set(value) for key, value in base_context["adjacency"].items()}
    cloned["segments"] = list(base_context["segments"])
    cloned["segment_aadt"] = dict(base_context["segment_aadt"])
    cloned["observed_segments"] = set(base_context["observed_segments"])
    cloned["monitored_intersections"] = set(base_context["monitored_intersections"])
    cloned["observation_profile"] = dict(base_context["observation_profile"])
    return cloned


def apply_observation_scale(base_context: Dict[str, object], coverage_scale: float) -> Dict[str, object]:
    context = clone_city_context(base_context)
    observed_segments = sorted(
        context["observed_segments"],
        key=lambda segment: (-float(context["segment_aadt"].get(segment, 0.0)), segment),
    )
    observed_count = max(1, int(math.ceil(len(observed_segments) * coverage_scale)))
    selected_segments = set(observed_segments[:observed_count])
    monitored_intersections = {
        context["segment_to_intersection"][segment]
        for segment in selected_segments
        if segment in context["segment_to_intersection"]
    }
    observed_fraction = len(selected_segments) / max(1, len(context["segments"]))
    monitored_fraction = len(monitored_intersections) / max(
        1, len(set(context["segment_to_intersection"].values()))
    )
    drop_boost = 0.0
    noise_boost = 0.0
    vehicle_observation_boost = 0.0
    delay_boost = 0
    if coverage_scale <= 0.25:
        drop_boost = 0.18
        noise_boost = 0.45
        vehicle_observation_boost = 0.10
        delay_boost = 2
    elif coverage_scale <= 0.5:
        drop_boost = 0.10
        noise_boost = 0.25
        vehicle_observation_boost = 0.05
        delay_boost = 1
    profile = dict(context["observation_profile"])
    profile.update(
        {
            "station_count": len({context["segment_to_rsu"][segment] for segment in selected_segments}),
            "observed_fraction": observed_fraction,
            "monitored_fraction": monitored_fraction,
            "message_drop_rate": max(0.02, min(0.45, 0.22 - 0.35 * observed_fraction + drop_boost)),
            "density_noise_std": max(0.15, min(1.35, 0.90 - 0.90 * observed_fraction + noise_boost)),
            "vehicle_observation_drop": max(
                0.0,
                min(0.30, 0.10 - 0.08 * monitored_fraction + vehicle_observation_boost),
            ),
            "controller_delay": (1 if monitored_fraction < 0.12 else 0) + delay_boost,
        }
    )
    context["observed_segments"] = selected_segments
    context["monitored_intersections"] = monitored_intersections
    context["observation_profile"] = profile
    return context


def run_city_with_context(context: Dict[str, object], city_key: str, seed: int) -> Dict[str, object]:
    context["city"] = city_key
    context["trip_library"] = build_trip_library(context, seed=seed, num_trips=ROUTE_LIBRARY_CAP)
    context["active_segments"] = sorted({segment for trip in context["trip_library"] for segment in trip})

    episodes = []
    counter = 0
    for attack in ATTACK_TYPES:
        for _ in range(EPISODES_PER_ATTACK):
            messages, attack_start = run_episode(context, attack, seed + counter, num_vehicles=EPISODE_VEHICLES)
            episodes.append((messages, attack, attack_start))
            counter += 1

    train_episodes, test_episodes = split_episodes(
        episodes,
        train_count_per_group=TRAIN_EPISODES_PER_ATTACK,
    )
    x_train, y_train = build_learning_dataset(
        train_episodes,
        adjacency=context["adjacency"],
        segment_to_rsu=context["segment_to_rsu"],
    )
    learning_model = MulticlassLogisticRegressionModel()
    learning_model.fit(x_train, y_train)
    weighted_learning_model = MulticlassLogisticRegressionModel()
    weighted_learning_model.fit(x_train, y_train, class_weight_mode="balanced")
    gaussian_nb_model = GaussianNaiveBayesModel()
    gaussian_nb_model.fit(x_train, y_train)

    return {
        "baseline": evaluate_detector(
            lambda: BaselineDetector(context["adjacency"]),
            test_episodes,
            "baseline",
            city_key,
        ),
        "flat_feature_logistic": evaluate_detector(
            lambda: FlatFeatureLogisticDetector(learning_model, context["adjacency"], context["segment_to_rsu"]),
            test_episodes,
            "flat-feature logistic",
            city_key,
        ),
        "weighted_logistic": evaluate_detector(
            lambda: FlatFeatureLogisticDetector(weighted_learning_model, context["adjacency"], context["segment_to_rsu"]),
            test_episodes,
            "weighted logistic",
            city_key,
        ),
        "gaussian_naive_bayes": evaluate_detector(
            lambda: FlatFeatureNaiveBayesDetector(gaussian_nb_model, context["adjacency"], context["segment_to_rsu"]),
            test_episodes,
            "Gaussian naive Bayes",
            city_key,
        ),
        "knowledge_graph": evaluate_detector(
            lambda: KGDetector(context["adjacency"], context["segment_to_rsu"], context["segment_to_intersection"]),
            test_episodes,
            "knowledge graph",
            city_key,
        ),
    }


def run_threshold_sweep() -> tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    raw_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []

    for threshold in (1, 2, 3):
        threshold_rows = []
        for seed in THRESHOLD_SEEDS:
            print(f"[{timestamp()}] threshold sweep: threshold={threshold}, seed={seed}", flush=True)
            result = run_full_experiment(seed=seed, episodes_per_case=25, kg_alert_threshold=threshold)
            kg = result["knowledge_graph"]
            threshold_rows.append(
                {
                    "threshold": threshold,
                    "seed": seed,
                    "macro_recall": kg["macro_recall"],
                    "macro_f1": kg["macro_f1"],
                    "position_recall": kg["per_attack"]["position_spoofing"]["recall"],
                }
            )
        raw_rows.extend(threshold_rows)
        summary_rows.append(
            {
                "threshold": threshold,
                "mean_macro_recall": round(mean(float(row["macro_recall"]) for row in threshold_rows), 6),
                "std_macro_recall": round(pstdev(float(row["macro_recall"]) for row in threshold_rows), 6),
                "mean_macro_f1": round(mean(float(row["macro_f1"]) for row in threshold_rows), 6),
                "std_macro_f1": round(pstdev(float(row["macro_f1"]) for row in threshold_rows), 6),
                "mean_position_recall": round(mean(float(row["position_recall"]) for row in threshold_rows), 6),
                "std_position_recall": round(pstdev(float(row["position_recall"]) for row in threshold_rows), 6),
                "num_seeds": len(threshold_rows),
            }
        )

    return raw_rows, summary_rows


def run_observation_sweep() -> tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    raw_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    base_contexts = {city: build_real_city_context(city) for city in CITY_ORDER}

    nominal_rows = read_csv(CITY_RESULTS_DIR / "real_city_benchmark.csv")
    nominal_by_detector: Dict[str, List[float]] = {detector: [] for detector in DETECTOR_ORDER}
    for row in nominal_rows:
        nominal_by_detector[row["detector"]].append(float(row["macro_f1"]))
    summary_rows.append(
        {
            "coverage_level": "Nominal",
            "coverage_scale": 1.0,
            "mean_observed_fraction": round(
                mean(float(base_contexts[city]["observation_profile"]["observed_fraction"]) for city in CITY_ORDER),
                6,
            ),
            "baseline_macro_f1": round(mean(nominal_by_detector["baseline"]), 6),
            "flat_feature_logistic_macro_f1": round(mean(nominal_by_detector["flat_feature_logistic"]), 6),
            "weighted_logistic_macro_f1": round(mean(nominal_by_detector["weighted_logistic"]), 6),
            "gaussian_naive_bayes_macro_f1": round(mean(nominal_by_detector["gaussian_naive_bayes"]), 6),
            "knowledge_graph_macro_f1": round(mean(nominal_by_detector["knowledge_graph"]), 6),
            "kg_std_macro_f1": round(pstdev(nominal_by_detector["knowledge_graph"]), 6),
            "num_runs": len(nominal_by_detector["knowledge_graph"]),
        }
    )

    for coverage_level, scale in OBSERVATION_LEVELS:
        per_detector: Dict[str, List[float]] = {detector: [] for detector in DETECTOR_ORDER}
        observed_fraction_values: List[float] = []
        for city in CITY_ORDER:
            seed = CITY_BASE_SEEDS[city]
            print(
                f"[{timestamp()}] observation sweep: profile={coverage_level}, city={city}, seed={seed}",
                flush=True,
            )
            context = apply_observation_scale(base_contexts[city], scale)
            observed_fraction_values.append(float(context["observation_profile"]["observed_fraction"]))
            result = run_city_with_context(context, city, seed)
            for detector in DETECTOR_ORDER:
                macro_f1 = float(result[detector]["macro_f1"])
                per_detector[detector].append(macro_f1)
                raw_rows.append(
                    {
                        "coverage_level": coverage_level,
                        "coverage_scale": scale,
                        "city": city,
                        "seed": seed,
                        "detector": detector,
                        "macro_f1": macro_f1,
                        "observed_fraction": round(float(context["observation_profile"]["observed_fraction"]), 6),
                    }
                )
        summary_rows.append(
            {
                "coverage_level": coverage_level,
                "coverage_scale": scale,
                "mean_observed_fraction": round(mean(observed_fraction_values), 6),
                "baseline_macro_f1": round(mean(per_detector["baseline"]), 6),
                "flat_feature_logistic_macro_f1": round(mean(per_detector["flat_feature_logistic"]), 6),
                "weighted_logistic_macro_f1": round(mean(per_detector["weighted_logistic"]), 6),
                "gaussian_naive_bayes_macro_f1": round(mean(per_detector["gaussian_naive_bayes"]), 6),
                "knowledge_graph_macro_f1": round(mean(per_detector["knowledge_graph"]), 6),
                "kg_std_macro_f1": round(pstdev(per_detector["knowledge_graph"]), 6),
                "num_runs": len(per_detector["knowledge_graph"]),
            }
        )

    return raw_rows, summary_rows


def main() -> None:
    ensure_pipeline_directories()
    SENSITIVITY_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    threshold_raw, threshold_summary = run_threshold_sweep()
    observation_raw, observation_summary = run_observation_sweep()

    write_csv(SENSITIVITY_RESULTS_DIR / "kg_threshold_sweep_raw.csv", threshold_raw)
    write_csv(SENSITIVITY_RESULTS_DIR / "kg_threshold_sweep.csv", threshold_summary)
    write_csv(SENSITIVITY_RESULTS_DIR / "city_observation_sweep_raw.csv", observation_raw)
    write_csv(SENSITIVITY_RESULTS_DIR / "city_observation_sweep.csv", observation_summary)
    render_phase6_assets(PROJECT_ROOT)


if __name__ == "__main__":
    main()
