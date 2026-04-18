from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence
import csv
import json
import math
import random
import sys

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from experiments.city.real_city_benchmark import (
    ATTACK_TYPES,
    EPISODE_VEHICLES,
    EPISODES_PER_ATTACK,
    ROUTE_LIBRARY_CAP,
    build_real_city_context,
    build_trip_library,
)
from experiments.common.project_paths import CITY_RESULTS_DIR, ensure_pipeline_directories
from rendering.scripts.paper_asset_renderers import render_city_assets


CITY_ORDER = ("austin", "houston", "dallas")
BASE_SEED = 42


def pearson_correlation(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or not xs:
        return 0.0
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    denom = den_x * den_y
    return num / denom if denom else 0.0


def average_ranks(values: Sequence[float]) -> List[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    idx = 0
    while idx < len(ordered):
        end = idx
        while end + 1 < len(ordered) and ordered[end + 1][1] == ordered[idx][1]:
            end += 1
        average_rank = (idx + end) / 2.0 + 1.0
        for position in range(idx, end + 1):
            ranks[ordered[position][0]] = average_rank
        idx = end + 1
    return ranks


def spearman_correlation(xs: Sequence[float], ys: Sequence[float]) -> float:
    return pearson_correlation(average_ranks(xs), average_ranks(ys))


def top_k_overlap(aadt_ranked: Iterable[str], demand_ranked: Iterable[str], k: int) -> float:
    aadt_top = list(aadt_ranked)[:k]
    demand_top = list(demand_ranked)[:k]
    if not aadt_top or not demand_top:
        return 0.0
    overlap = len(set(aadt_top) & set(demand_top))
    return overlap / min(len(aadt_top), len(demand_top))


def segment_traversal_counts(context: Dict[str, object], seed: int) -> Dict[str, int]:
    context["trip_library"] = build_trip_library(context, seed=seed, num_trips=ROUTE_LIBRARY_CAP)
    context["active_segments"] = sorted({segment for trip in context["trip_library"] for segment in trip})
    counts = {segment: 0 for segment in context["active_segments"]}
    counter = 0

    for _attack in ATTACK_TYPES:
        for _ in range(EPISODES_PER_ATTACK):
            episode_seed = seed + counter
            counter += 1
            rng_routes = random.Random(episode_seed)
            for _ in range(EPISODE_VEHICLES):
                route = rng_routes.choice(context["trip_library"])
                for segment in route:
                    counts[segment] = counts.get(segment, 0) + 1
    return counts


def analyze_city(city_key: str, seed: int) -> tuple[dict, list[dict]]:
    print(f"[validation] building context for {city_key}", flush=True)
    context = build_real_city_context(city_key)
    context["city"] = city_key
    counts = segment_traversal_counts(context, seed)

    detail_rows: List[dict] = []
    comparable_rows: List[dict] = []
    for segment in context["active_segments"]:
        aadt = float(context["segment_aadt"].get(segment, 0.0))
        demand = int(counts.get(segment, 0))
        row = {
            "city": city_key,
            "segment": segment,
            "aadt": round(aadt, 3),
            "sampled_traversals": demand,
            "observed_segment": int(segment in context["observed_segments"]),
        }
        detail_rows.append(row)
        if aadt > 0.0:
            comparable_rows.append(row)

    comparable_rows.sort(key=lambda row: row["segment"])
    aadt_values = [float(row["aadt"]) for row in comparable_rows]
    traversal_values = [float(row["sampled_traversals"]) for row in comparable_rows]

    aadt_ranked = [
        row["segment"]
        for row in sorted(
            comparable_rows,
            key=lambda row: (-float(row["aadt"]), row["segment"]),
        )
    ]
    demand_ranked = [
        row["segment"]
        for row in sorted(
            comparable_rows,
            key=lambda row: (-int(row["sampled_traversals"]), row["segment"]),
        )
    ]
    top_k = max(1, int(round(len(comparable_rows) * 0.10)))

    summary = {
        "city": city_key,
        "route_library_seed": seed,
        "episode_seed_start": seed,
        "episode_seed_end": seed + len(ATTACK_TYPES) * EPISODES_PER_ATTACK - 1,
        "active_segments": len(context["active_segments"]),
        "comparable_segments": len(comparable_rows),
        "log_pearson_r": round(
            pearson_correlation(
                [math.log1p(value) for value in aadt_values],
                [math.log1p(value) for value in traversal_values],
            ),
            4,
        ),
        "spearman_rho": round(spearman_correlation(aadt_values, traversal_values), 4),
        "top_decile_overlap_pct": round(100.0 * top_k_overlap(aadt_ranked, demand_ranked, top_k), 1),
    }

    aadt_positions = {segment: rank for rank, segment in enumerate(aadt_ranked, start=1)}
    demand_positions = {segment: rank for rank, segment in enumerate(demand_ranked, start=1)}
    for row in detail_rows:
        row["aadt_rank"] = aadt_positions.get(row["segment"], "")
        row["sampled_traversal_rank"] = demand_positions.get(row["segment"], "")

    print(
        "[validation] "
        f"{city_key}: active={summary['active_segments']}, comparable={summary['comparable_segments']}, "
        f"log-r={summary['log_pearson_r']:.3f}, rho={summary['spearman_rho']:.3f}, "
        f"top-decile overlap={summary['top_decile_overlap_pct']:.1f}%",
        flush=True,
    )
    return summary, detail_rows


def write_outputs(summaries: List[dict], detail_rows: List[dict]) -> None:
    ensure_pipeline_directories()
    CITY_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summary_path = CITY_RESULTS_DIR / "aadt_validation.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    detail_path = CITY_RESULTS_DIR / "aadt_validation_segments.csv"
    with detail_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detail_rows[0].keys()))
        writer.writeheader()
        writer.writerows(detail_rows)

    metadata = {
        "cities": list(CITY_ORDER),
        "base_seed": BASE_SEED,
        "route_library_cap": ROUTE_LIBRARY_CAP,
        "vehicles_per_episode": EPISODE_VEHICLES,
        "episodes_per_attack": EPISODES_PER_ATTACK,
        "attack_families": list(ATTACK_TYPES),
        "validation_scope": "AADT-grounded route-demand sampling only; cyber labels are not validated by this artifact.",
    }
    with (CITY_RESULTS_DIR / "aadt_validation.json").open("w", encoding="utf-8") as handle:
        json.dump({"metadata": metadata, "summary": summaries}, handle, indent=2)

    render_city_assets(PROJECT_ROOT)
    print(f"[output] wrote {summary_path.name}, {detail_path.name}, and aadt_validation.json", flush=True)


def main() -> None:
    summaries: List[dict] = []
    detail_rows: List[dict] = []
    for offset, city_key in enumerate(CITY_ORDER):
        summary, city_details = analyze_city(city_key, BASE_SEED + offset)
        summaries.append(summary)
        detail_rows.extend(city_details)
    write_outputs(summaries, detail_rows)


if __name__ == "__main__":
    main()
