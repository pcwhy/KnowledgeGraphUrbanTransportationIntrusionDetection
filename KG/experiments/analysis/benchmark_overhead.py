from __future__ import annotations

from pathlib import Path
from statistics import mean, stdev
from time import perf_counter
from typing import Callable, Dict, List, Sequence, Tuple
import csv
import json
import sys

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from experiments.common.project_paths import OVERHEAD_RESULTS_DIR, ensure_pipeline_directories
from experiments.synthetic.urban_transport_kg_ids import (
    ATTACK_TYPES,
    BaselineDetector,
    FlatFeatureContext,
    FlatFeatureLogisticDetector,
    KnowledgeGraphDetector,
    MulticlassLogisticRegressionModel,
    Message,
    SEGMENT_TO_RSU,
    UrbanTransportExperiment,
    build_learning_dataset,
    evaluate_detector,
    split_episodes,
)
from rendering.scripts.paper_asset_renderers import render_overhead_assets


FeatureRows = List[List[float]]
Labels = List[int]
Episodes = List[Tuple[List[Message], str | None, int]]


class MaskedLogisticDetector:
    def __init__(self, model: MulticlassLogisticRegressionModel, keep_indices: Sequence[int]) -> None:
        self.model = model
        self.keep_indices = list(keep_indices)
        self.context = FlatFeatureContext()

    def predict_label(self, message: Message):
        features = self.context.features_for(message)
        prediction = None
        if features is not None:
            masked = [features[index] for index in self.keep_indices]
            prediction = self.model.predict_label(masked)
        self.context.update(message)
        return prediction

    def observe(self, message: Message) -> bool:
        prediction = self.predict_label(message)
        return prediction not in (None, "benign")


def generate_episodes(seed: int = 42, episodes_per_case: int = 25) -> tuple[Episodes, Episodes]:
    simulator = UrbanTransportExperiment(seed=seed)
    episodes: Episodes = []
    for attack_type in ATTACK_TYPES:
        for _ in range(episodes_per_case):
            messages, attack_start = simulator.run_episode(attack_type)
            episodes.append((messages, attack_type, attack_start))
        for _ in range(max(5, episodes_per_case // 5)):
            messages, attack_start = simulator.run_episode(None)
            episodes.append((messages, None, attack_start))
    return split_episodes(episodes)


def masked_dataset(x_rows: FeatureRows, keep_indices: Sequence[int]) -> FeatureRows:
    return [[row[index] for index in keep_indices] for row in x_rows]


def time_logistic_training(
    x_rows: FeatureRows,
    y_rows: Labels,
    repeats: int = 7,
) -> Dict[str, float]:
    samples: List[float] = []
    for _ in range(repeats):
        model = MulticlassLogisticRegressionModel()
        start = perf_counter()
        model.fit(x_rows, y_rows)
        samples.append(perf_counter() - start)
    return summarize_samples(samples)


def time_detector_initialization(factory: Callable[[], object], repeats: int = 2000) -> Dict[str, float]:
    samples: List[float] = []
    for _ in range(repeats):
        start = perf_counter()
        factory()
        samples.append(perf_counter() - start)
    return summarize_samples(samples)


def time_episode_evaluation(factory: Callable[[], object], episodes: Episodes, repeats: int = 5) -> Dict[str, float]:
    samples: List[float] = []
    total_messages = sum(len(messages) for messages, _, _ in episodes)
    for _ in range(repeats):
        start = perf_counter()
        evaluate_detector("timed", factory, episodes)
        samples.append(perf_counter() - start)
    summary = summarize_samples(samples)
    summary["messages"] = total_messages
    summary["mean_us_per_message"] = summary["mean_seconds"] / total_messages * 1e6
    return summary


def summarize_samples(samples: List[float]) -> Dict[str, float]:
    return {
        "mean_seconds": mean(samples),
        "std_seconds": stdev(samples) if len(samples) > 1 else 0.0,
        "min_seconds": min(samples),
        "max_seconds": max(samples),
    }


def build_models(train_episodes: Episodes):
    x_train, y_train = build_learning_dataset(train_episodes)

    full_model = MulticlassLogisticRegressionModel()
    full_model.fit(x_train, y_train)

    # Remove RSU-related features.
    keep_no_rsu = [0, 1, 2, 3, 4, 7, 8, 10, 11, 12]
    x_no_rsu = masked_dataset(x_train, keep_no_rsu)
    no_rsu_model = MulticlassLogisticRegressionModel()
    no_rsu_model.fit(x_no_rsu, y_train)

    # Remove topology-related features.
    keep_no_topology = [0, 1, 2, 3, 4, 5, 6, 8, 10, 11]
    x_no_topology = masked_dataset(x_train, keep_no_topology)
    no_topology_model = MulticlassLogisticRegressionModel()
    no_topology_model.fit(x_no_topology, y_train)

    return {
        "full": (full_model, list(range(len(x_train[0]))), x_train, y_train),
        "no_rsu": (no_rsu_model, keep_no_rsu, x_no_rsu, y_train),
        "no_topology": (no_topology_model, keep_no_topology, x_no_topology, y_train),
    }


def write_outputs(root: Path, results: Dict[str, object]) -> None:
    ensure_pipeline_directories()
    generated = OVERHEAD_RESULTS_DIR
    generated.mkdir(parents=True, exist_ok=True)

    with (generated / "overhead_results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    comparison_rows = [
        {
            "component": "Logistic retraining (full feature set)",
            "mean_seconds": results["logistic_training"]["full"]["mean_seconds"],
            "std_seconds": results["logistic_training"]["full"]["std_seconds"],
        },
        {
            "component": "Logistic retraining (no RSU features)",
            "mean_seconds": results["logistic_training"]["no_rsu"]["mean_seconds"],
            "std_seconds": results["logistic_training"]["no_rsu"]["std_seconds"],
        },
        {
            "component": "Logistic retraining (no topology features)",
            "mean_seconds": results["logistic_training"]["no_topology"]["mean_seconds"],
            "std_seconds": results["logistic_training"]["no_topology"]["std_seconds"],
        },
        {
            "component": "KG initialization (full rules)",
            "mean_seconds": results["kg_initialization"]["full"]["mean_seconds"],
            "std_seconds": results["kg_initialization"]["full"]["std_seconds"],
        },
        {
            "component": "KG initialization (no RSU rule)",
            "mean_seconds": results["kg_initialization"]["no_rsu"]["mean_seconds"],
            "std_seconds": results["kg_initialization"]["no_rsu"]["std_seconds"],
        },
        {
            "component": "KG initialization (no topology rule)",
            "mean_seconds": results["kg_initialization"]["no_topology"]["mean_seconds"],
            "std_seconds": results["kg_initialization"]["no_topology"]["std_seconds"],
        },
    ]

    with (generated / "overhead_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0].keys()))
        writer.writeheader()
        writer.writerows(comparison_rows)

    inference_rows = [
        {
            "detector": "baseline",
            "messages": results["episode_evaluation"]["baseline"]["messages"],
            "mean_seconds": results["episode_evaluation"]["baseline"]["mean_seconds"],
            "mean_us_per_message": results["episode_evaluation"]["baseline"]["mean_us_per_message"],
        },
        {
            "detector": "flat_feature_logistic",
            "messages": results["episode_evaluation"]["logistic_full"]["messages"],
            "mean_seconds": results["episode_evaluation"]["logistic_full"]["mean_seconds"],
            "mean_us_per_message": results["episode_evaluation"]["logistic_full"]["mean_us_per_message"],
        },
        {
            "detector": "knowledge_graph",
            "messages": results["episode_evaluation"]["kg_full"]["messages"],
            "mean_seconds": results["episode_evaluation"]["kg_full"]["mean_seconds"],
            "mean_us_per_message": results["episode_evaluation"]["kg_full"]["mean_us_per_message"],
        },
    ]
    with (generated / "overhead_inference.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(inference_rows[0].keys()))
        writer.writeheader()
        writer.writerows(inference_rows)

    render_overhead_assets(root)


def main() -> None:
    root = PROJECT_ROOT
    train_episodes, test_episodes = generate_episodes()
    models = build_models(train_episodes)

    results = {
        "train_episodes": len(train_episodes),
        "test_episodes": len(test_episodes),
        "logistic_training": {
            "full": time_logistic_training(models["full"][2], models["full"][3]),
            "no_rsu": time_logistic_training(models["no_rsu"][2], models["no_rsu"][3]),
            "no_topology": time_logistic_training(models["no_topology"][2], models["no_topology"][3]),
        },
        "kg_initialization": {
            "full": time_detector_initialization(KnowledgeGraphDetector),
            "no_rsu": time_detector_initialization(lambda: KnowledgeGraphDetector(use_rsu_context=False)),
            "no_topology": time_detector_initialization(lambda: KnowledgeGraphDetector(use_topology=False)),
        },
        "episode_evaluation": {
            "logistic_full": time_episode_evaluation(lambda: FlatFeatureLogisticDetector(models["full"][0]), test_episodes),
            "kg_full": time_episode_evaluation(KnowledgeGraphDetector, test_episodes),
            "baseline": time_episode_evaluation(BaselineDetector, test_episodes),
        },
        "reconfigured_performance": {
            "logistic_no_rsu": evaluate_detector(
                "logistic_no_rsu",
                lambda: MaskedLogisticDetector(models["no_rsu"][0], models["no_rsu"][1]),
                test_episodes,
            ),
            "logistic_no_topology": evaluate_detector(
                "logistic_no_topology",
                lambda: MaskedLogisticDetector(models["no_topology"][0], models["no_topology"][1]),
                test_episodes,
            ),
            "kg_no_rsu": evaluate_detector(
                "kg_no_rsu",
                lambda: KnowledgeGraphDetector(use_rsu_context=False),
                test_episodes,
            ),
            "kg_no_topology": evaluate_detector(
                "kg_no_topology",
                lambda: KnowledgeGraphDetector(use_topology=False),
                test_episodes,
            ),
        },
    }
    write_outputs(root, results)
    print(f"Overhead benchmark completed. Results written to {OVERHEAD_RESULTS_DIR}.")


if __name__ == "__main__":
    main()
