from pathlib import Path
import sys

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from experiments.common.project_paths import SYNTHETIC_RESULTS_DIR, ensure_pipeline_directories
from experiments.synthetic.urban_transport_kg_ids import run_full_experiment, write_outputs


def main() -> None:
    ensure_pipeline_directories()
    print("[synthetic] starting synthetic experiment run", flush=True)
    results = run_full_experiment(seed=42, episodes_per_case=25)
    print("[synthetic] writing structured outputs", flush=True)
    write_outputs(results, SYNTHETIC_RESULTS_DIR)
    print(f"Experiment completed. Results written to {SYNTHETIC_RESULTS_DIR}.")


if __name__ == "__main__":
    main()
