from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import csv
import random
import sys

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from experiments.common.detection_shared import (
    ATTACK_TYPES,
    CLASS_LABELS,
    GaussianNaiveBayesModel,
    MulticlassLogisticRegressionModel,
    MulticlassStats,
)
from experiments.common.project_paths import STATISTICS_RESULTS_DIR, ensure_pipeline_directories
from experiments.synthetic.urban_transport_kg_ids import (
    BaselineDetector as SyntheticBaselineDetector,
    FlatFeatureLogisticDetector as SyntheticFlatFeatureDetector,
    FlatFeatureNaiveBayesDetector as SyntheticNaiveBayesDetector,
    KnowledgeGraphDetector as SyntheticKGDetector,
    UrbanTransportExperiment,
    build_learning_dataset as build_synthetic_learning_dataset,
    message_label as synthetic_message_label,
    split_episodes as split_synthetic_episodes,
)
from experiments.city.real_city_benchmark import (
    BaselineDetector as CityBaselineDetector,
    FlatFeatureLogisticDetector as CityFlatFeatureDetector,
    FlatFeatureNaiveBayesDetector as CityNaiveBayesDetector,
    KGDetector as CityKGDetector,
    ROUTE_LIBRARY_CAP,
    EPISODE_VEHICLES,
    EPISODES_PER_ATTACK,
    TRAIN_EPISODES_PER_ATTACK,
    build_learning_dataset as build_city_learning_dataset,
    build_real_city_context,
    build_trip_library,
    message_label as city_message_label,
    run_episode,
    split_episodes as split_city_episodes,
)


BOOTSTRAP_REPLICATES = 1000
CONTROLLED_SEED = 42
CITY_SEEDS: Dict[str, int] = {
    "austin": 42,
    "houston": 43,
    "dallas": 44,
}
DETECTOR_ORDER: Sequence[str] = (
    "baseline",
    "flat_feature_logistic",
    "weighted_logistic",
    "gaussian_naive_bayes",
    "knowledge_graph",
)
DETECTOR_LABELS: Dict[str, str] = {
    "baseline": "Baseline",
    "flat_feature_logistic": "Flat-feature logistic",
    "weighted_logistic": "Weighted logistic",
    "gaussian_naive_bayes": "Gaussian naive Bayes",
    "knowledge_graph": "Knowledge-graph",
}


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def macro_f1_from_rows(rows: Sequence[Tuple[str, str]]) -> float:
    stats = MulticlassStats(CLASS_LABELS)
    for true_label, pred_label in rows:
        stats.add(true_label, pred_label)
    return stats.macro_f1()


def bootstrap_delta(
    rows_a: Sequence[Tuple[str, str]],
    rows_b: Sequence[Tuple[str, str]],
    seed: int,
) -> Tuple[float, float, float, float]:
    if len(rows_a) != len(rows_b):
        raise ValueError("Detector prediction rows must have the same length for paired bootstrap.")
    original_delta = macro_f1_from_rows(rows_a) - macro_f1_from_rows(rows_b)
    rng = random.Random(seed)
    deltas: List[float] = []
    total = len(rows_a)
    for _ in range(BOOTSTRAP_REPLICATES):
        indices = [rng.randrange(total) for _ in range(total)]
        sampled_a = [rows_a[idx] for idx in indices]
        sampled_b = [rows_b[idx] for idx in indices]
        deltas.append(macro_f1_from_rows(sampled_a) - macro_f1_from_rows(sampled_b))
    deltas.sort()
    lower = deltas[int(0.025 * len(deltas))]
    upper = deltas[int(0.975 * len(deltas))]
    p_value = 2.0 * min(
        sum(delta <= 0.0 for delta in deltas) / len(deltas),
        sum(delta >= 0.0 for delta in deltas) / len(deltas),
    )
    return original_delta, lower, upper, min(1.0, p_value)


def collect_prediction_rows(episodes, detector_factory, message_label_fn) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    for messages, _, _ in episodes:
        detector = detector_factory()
        for message in messages:
            prediction = detector.predict_label(message)
            true_label = message_label_fn(message)
            if true_label is None or prediction is None:
                continue
            rows.append((true_label, prediction))
    return rows


def run_controlled_predictions() -> Dict[str, List[Tuple[str, str]]]:
    print(f"[{timestamp()}] building controlled prediction rows", flush=True)
    simulator = UrbanTransportExperiment(seed=CONTROLLED_SEED)
    episodes = []
    for attack_type in ATTACK_TYPES:
        for _ in range(25):
            messages, attack_start = simulator.run_episode(attack_type)
            episodes.append((messages, attack_type, attack_start))
        for _ in range(max(5, 25 // 5)):
            messages, attack_start = simulator.run_episode(None)
            episodes.append((messages, None, attack_start))

    train_episodes, test_episodes = split_synthetic_episodes(episodes)
    x_train, y_train = build_synthetic_learning_dataset(train_episodes)
    learning_model = MulticlassLogisticRegressionModel()
    learning_model.fit(x_train, y_train)
    weighted_model = MulticlassLogisticRegressionModel()
    weighted_model.fit(x_train, y_train, class_weight_mode="balanced")
    gaussian_nb_model = GaussianNaiveBayesModel()
    gaussian_nb_model.fit(x_train, y_train)

    return {
        "baseline": collect_prediction_rows(
            test_episodes,
            SyntheticBaselineDetector,
            synthetic_message_label,
        ),
        "flat_feature_logistic": collect_prediction_rows(
            test_episodes,
            lambda: SyntheticFlatFeatureDetector(learning_model),
            synthetic_message_label,
        ),
        "weighted_logistic": collect_prediction_rows(
            test_episodes,
            lambda: SyntheticFlatFeatureDetector(weighted_model),
            synthetic_message_label,
        ),
        "gaussian_naive_bayes": collect_prediction_rows(
            test_episodes,
            lambda: SyntheticNaiveBayesDetector(gaussian_nb_model),
            synthetic_message_label,
        ),
        "knowledge_graph": collect_prediction_rows(
            test_episodes,
            SyntheticKGDetector,
            synthetic_message_label,
        ),
    }


def run_city_predictions(city_key: str, seed: int) -> Dict[str, List[Tuple[str, str]]]:
    print(f"[{timestamp()}] building city prediction rows: city={city_key}, seed={seed}", flush=True)
    context = build_real_city_context(city_key)
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

    train_episodes, test_episodes = split_city_episodes(
        episodes,
        train_count_per_group=TRAIN_EPISODES_PER_ATTACK,
    )
    x_train, y_train = build_city_learning_dataset(
        train_episodes,
        adjacency=context["adjacency"],
        segment_to_rsu=context["segment_to_rsu"],
    )
    learning_model = MulticlassLogisticRegressionModel()
    learning_model.fit(x_train, y_train)
    weighted_model = MulticlassLogisticRegressionModel()
    weighted_model.fit(x_train, y_train, class_weight_mode="balanced")
    gaussian_nb_model = GaussianNaiveBayesModel()
    gaussian_nb_model.fit(x_train, y_train)

    return {
        "baseline": collect_prediction_rows(
            test_episodes,
            lambda: CityBaselineDetector(context["adjacency"]),
            city_message_label,
        ),
        "flat_feature_logistic": collect_prediction_rows(
            test_episodes,
            lambda: CityFlatFeatureDetector(learning_model, context["adjacency"], context["segment_to_rsu"]),
            city_message_label,
        ),
        "weighted_logistic": collect_prediction_rows(
            test_episodes,
            lambda: CityFlatFeatureDetector(weighted_model, context["adjacency"], context["segment_to_rsu"]),
            city_message_label,
        ),
        "gaussian_naive_bayes": collect_prediction_rows(
            test_episodes,
            lambda: CityNaiveBayesDetector(gaussian_nb_model, context["adjacency"], context["segment_to_rsu"]),
            city_message_label,
        ),
        "knowledge_graph": collect_prediction_rows(
            test_episodes,
            lambda: CityKGDetector(
                context["adjacency"],
                context["segment_to_rsu"],
                context["segment_to_intersection"],
            ),
            city_message_label,
        ),
    }


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[{timestamp()}] wrote {path}", flush=True)


def comparison_row(
    setting: str,
    city: str,
    detector_a: str,
    detector_b: str,
    rows_a: Sequence[Tuple[str, str]],
    rows_b: Sequence[Tuple[str, str]],
    seed: int,
) -> Dict[str, object]:
    original_delta, lower, upper, p_value = bootstrap_delta(rows_a, rows_b, seed=seed)
    return {
        "setting": setting,
        "city": city,
        "detector_a": detector_a,
        "detector_a_label": DETECTOR_LABELS[detector_a],
        "detector_b": detector_b,
        "detector_b_label": DETECTOR_LABELS[detector_b],
        "metric": "macro_f1",
        "score_a": round(macro_f1_from_rows(rows_a), 6),
        "score_b": round(macro_f1_from_rows(rows_b), 6),
        "delta_a_minus_b": round(original_delta, 6),
        "ci_lower": round(lower, 6),
        "ci_upper": round(upper, 6),
        "bootstrap_p": round(p_value, 6),
        "significant_95": int(lower > 0.0 or upper < 0.0),
        "paired_messages": len(rows_a),
    }


def main() -> None:
    ensure_pipeline_directories()
    STATISTICS_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []

    controlled_predictions = run_controlled_predictions()
    for detector in DETECTOR_ORDER:
        rows.append(
            comparison_row(
                "controlled_single_run",
                "",
                detector,
                "knowledge_graph" if detector != "knowledge_graph" else "baseline",
                controlled_predictions[detector],
                controlled_predictions["knowledge_graph"] if detector != "knowledge_graph" else controlled_predictions["baseline"],
                seed=1000 + CONTROLLED_SEED,
            )
        )

    city_pairs = [
        ("knowledge_graph", "baseline"),
        ("flat_feature_logistic", "knowledge_graph"),
        ("weighted_logistic", "knowledge_graph"),
        ("gaussian_naive_bayes", "knowledge_graph"),
    ]
    for city, seed in CITY_SEEDS.items():
        city_predictions = run_city_predictions(city, seed)
        for detector_a, detector_b in city_pairs:
            rows.append(
                comparison_row(
                    "city_single_run",
                    city,
                    detector_a,
                    detector_b,
                    city_predictions[detector_a],
                    city_predictions[detector_b],
                    seed=2000 + seed,
                )
            )

    write_csv(STATISTICS_RESULTS_DIR / "significance_tests.csv", rows)


if __name__ == "__main__":
    main()
