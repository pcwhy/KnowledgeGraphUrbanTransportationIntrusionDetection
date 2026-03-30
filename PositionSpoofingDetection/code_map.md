# Code Map

## Experiment Entry Point

- `experiments/kg_sequential_focus.py`
  - focused sequential localization study with four operational states
  - builds train/test episodes, runs the robust sequential detector, and writes summary artifacts

## Supporting Modules

- `experiments/sequential_city_support.py`
  - builds the city-grounded context used by the sequential study
  - samples routes and provides the deterministic signal-state helper
- `experiments/real_city_geo.py`
  - provides the narrow TxDOT AADT download and edge-annotation utilities used by the support module

## Suggested Reading Order

1. `experiments/kg_sequential_focus.py`
2. `experiments/sequential_city_support.py`
3. `experiments/real_city_geo.py`
