# Experiment Execution

## Setup

Create a Python virtual environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Local Inputs

Place the required Austin cache files under:

```text
raw_data/real_city_cache/
```

The sequential experiment expects the same local graph and traffic cache naming used by the research codebase, including files such as:

- `austin_drive.graphml`
- `austin_txdot_aadt.geojson`
- `austin_txdot_permanent_count_stations.geojson`

If cache permissions are restrictive in your environment, you can run with local cache variables:

```bash
MPLCONFIGDIR="$(pwd)/.mplconfig" XDG_CACHE_HOME="$(pwd)/.cache" python experiments/kg_sequential_focus.py
```

## Main Experiment Command

```bash
python experiments/kg_sequential_focus.py
```

This runs the focused sequential localization study with:

- benign nominal operation
- regional GPS failure
- position spoofing
- intent hiding

## Practical Notes

- The experiment code is intentionally trimmed to the sequential IDS path only.
- `experiments/sequential_city_support.py` and `experiments/real_city_geo.py` are support modules, not separate public entry points.
- The real-city context uses OpenStreetMap, OSMnx, and TxDOT-based preprocessing through the local cache files above.
