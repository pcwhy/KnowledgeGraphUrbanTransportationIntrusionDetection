# Witness Evidence-based Vehicular Position Spoofing Detection

This folder is a GitHub-friendly export of the sequential localization IDS study only.

## Repository Layout

- `experiments/kg_sequential_focus.py`: main sequential IDS experiment
- `experiments/sequential_city_support.py`: minimal city-context and route-sampling support code used by the experiment
- `experiments/real_city_geo.py`: small geospatial helper module for cached AADT annotation
- `docs/`: lightweight execution and code-map notes
- `requirements.txt`: Python package list for the experiment environment

## Environment

Python 3.10 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

Run the sequential localization study with:

```bash
python experiments/kg_sequential_focus.py
```

For setup details and expected local inputs, see [docs/experiment_execution.md](/Users/yongxinliu/Career/CareerAtERAU/KG Transportation System Sequential/ToGithub/docs/experiment_execution.md).

## Notes

- This export is intentionally code-and-docs only.
- Manuscript files and generated paper assets are not included here.
- The experiment expects local city cache files under `raw_data/real_city_cache/`.
