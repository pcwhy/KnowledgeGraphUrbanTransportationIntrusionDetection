from __future__ import annotations

from pathlib import Path
import csv
import sys

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from experiments.common.project_paths import ANALYSIS_RESULTS_DIR, CITY_RESULTS_DIR, ensure_pipeline_directories


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_generalization_summary() -> list[dict[str, object]]:
    transfer_rows = _read_csv(CITY_RESULTS_DIR / "city_transfer.csv")
    config_rows = _read_csv(CITY_RESULTS_DIR / "real_city_configuration.csv")
    benchmark_rows = _read_csv(CITY_RESULTS_DIR / "real_city_benchmark.csv")
    kg_failure_rows = _read_csv(ANALYSIS_RESULTS_DIR / "kg_city_failure_summary.csv")

    config_index = {row["city"]: row for row in config_rows}
    failure_index = {row["city"]: row for row in kg_failure_rows}
    logistic_index = {
        row["city"]: row
        for row in benchmark_rows
        if row["detector"] == "flat_feature_logistic"
    }

    cities = ["austin", "houston", "dallas"]
    summary_rows: list[dict[str, object]] = []
    for city in cities:
        incoming = [
            float(row["delta_logistic_macro_f1"])
            for row in transfer_rows
            if row["target_city"] == city and row["source_city"] != city
        ]
        outgoing = [
            float(row["delta_logistic_macro_f1"])
            for row in transfer_rows
            if row["source_city"] == city and row["target_city"] != city
        ]
        summary_rows.append(
            {
                "city": city,
                "active_segments": int(config_index[city]["active_segments"]),
                "observed_segments": int(config_index[city]["observed_segments"]),
                "monitored_intersections": int(config_index[city]["monitored_intersections"]),
                "observed_fraction_pct": round(float(failure_index[city]["observed_fraction_pct"]), 3),
                "monitored_fraction_pct": round(float(failure_index[city]["monitored_fraction_pct"]), 3),
                "city_specific_logistic_macro_f1": round(float(logistic_index[city]["macro_f1"]), 4),
                "mean_incoming_logistic_delta": round(sum(incoming) / len(incoming), 4),
                "mean_outgoing_logistic_delta": round(sum(outgoing) / len(outgoing), 4),
            }
        )
    return summary_rows


def write_outputs(summary_rows: list[dict[str, object]]) -> None:
    ensure_pipeline_directories()
    ANALYSIS_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = ANALYSIS_RESULTS_DIR / "city_generalization_summary.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"[output] wrote {output_path.name}", flush=True)


def main() -> None:
    summary_rows = build_generalization_summary()
    write_outputs(summary_rows)


if __name__ == "__main__":
    main()
