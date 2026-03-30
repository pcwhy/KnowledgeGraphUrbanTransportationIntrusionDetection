# Repository Guide

## Purpose

This project implements a transportation-cybersecurity research pipeline with three detector families:

- a local-rule baseline,
- a flat-feature logistic detector,
- and a knowledge-graph detector backed by explicit transportation state.

The repository follows a three-layer workflow:

`raw_data/ -> experiments/ -> results/structured/`

## Top-Level Layout

### `raw_data/`

External inputs and rebuildable caches used by the experiments.

- `city_cache/`
  - cached OpenStreetMap graphs,
  - cached TxDOT Annual Average Daily Traffic layers,
  - cached TxDOT permanent-count-station layers.
  - see `raw_data/city_cache/README.md` for the local fetch and naming guide

#### How the city-cache inputs are obtained

- Drive-network graphs are derived from OpenStreetMap and stored as GraphML.
- The current Texas-city traffic-volume and station layers are derived from
  TxDOT data exports and stored as GeoJSON.
- The repository keeps the cached files needed to rerun the benchmark, but a
  developer porting the workflow to another city should fetch the equivalent
  source layers and normalize them into the same cache pattern:
  - `<city>_drive.graphml`
  - `<city>_txdot_aadt.geojson`
  - `<city>_txdot_permanent_count_stations.geojson`

### `experiments/`

Experiment logic only. These scripts should write machine-readable outputs rather than presentation-formatted deliverables.

- `common/project_paths.py`
  - shared path constants for the project pipeline.
- `synthetic/urban_transport_kg_ids.py`
  - synthetic simulator, detector definitions, and synthetic evaluation logic.
- `synthetic/run_experiments.py`
  - runner for the synthetic simulator study.
- `city/real_city_case_study.py`
  - real-city visual case-study generation and summary export.
- `city/real_city_benchmark.py`
  - Austin, Houston, and Dallas benchmark and cross-city transfer evaluation.
- `analysis/benchmark_overhead.py`
  - retraining, initialization, and inference overhead measurements.
- `analysis/export_kg_tikz.py`
  - KG snapshot export.

### `results/structured/`

Machine-readable result bridges produced locally by the experiments.

- `synthetic/`
  - synthetic CSV and JSON outputs generated after local runs.
- `city/`
  - real-city CSV and JSON outputs generated after local runs.
- `overhead/`
  - overhead benchmark CSV and JSON outputs generated after local runs.
- `assets/`
  - structured metadata generated after local runs.

## Canonical Pipeline

The intended dependency chain is:

1. `raw_data/` stores external inputs and caches.
2. `experiments/` produces machine-readable outputs only.
3. `results/structured/` stores CSV and JSON bridges generated after local
   experiment runs.

Experiment scripts should write structured outputs only.

## Main Commands

Run the synthetic study:

```bash
MPLCONFIGDIR="$PWD/.mplconfig" XDG_CACHE_HOME="$PWD/.cache" .venv/bin/python experiments/synthetic/run_experiments.py
```

Run the real-city case studies:

```bash
MPLCONFIGDIR="$PWD/.mplconfig" XDG_CACHE_HOME="$PWD/.cache" .venv/bin/python experiments/city/real_city_case_study.py
```

Run the real-city benchmark:

```bash
MPLCONFIGDIR="$PWD/.mplconfig" XDG_CACHE_HOME="$PWD/.cache" .venv/bin/python experiments/city/real_city_benchmark.py
```

Run the overhead benchmark:

```bash
MPLCONFIGDIR="$PWD/.mplconfig" XDG_CACHE_HOME="$PWD/.cache" .venv/bin/python experiments/analysis/benchmark_overhead.py
```

Regenerate the KG snapshot export:

```bash
MPLCONFIGDIR="$PWD/.mplconfig" XDG_CACHE_HOME="$PWD/.cache" .venv/bin/python experiments/analysis/export_kg_tikz.py
```

## Recommended Rebuild Order

For a full end-to-end refresh:

```bash
MPLCONFIGDIR="$PWD/.mplconfig" XDG_CACHE_HOME="$PWD/.cache" .venv/bin/python experiments/synthetic/run_experiments.py
MPLCONFIGDIR="$PWD/.mplconfig" XDG_CACHE_HOME="$PWD/.cache" .venv/bin/python experiments/city/real_city_case_study.py
MPLCONFIGDIR="$PWD/.mplconfig" XDG_CACHE_HOME="$PWD/.cache" .venv/bin/python experiments/city/real_city_benchmark.py
MPLCONFIGDIR="$PWD/.mplconfig" XDG_CACHE_HOME="$PWD/.cache" .venv/bin/python experiments/analysis/export_kg_tikz.py
MPLCONFIGDIR="$PWD/.mplconfig" XDG_CACHE_HOME="$PWD/.cache" .venv/bin/python experiments/analysis/benchmark_overhead.py
```

## Headless Rendering Note

The plotting scripts are configured for headless execution with Matplotlib's `Agg` backend because the macOS GUI backend can abort when launched from a non-GUI worker process. Use the `MPLCONFIGDIR` and `XDG_CACHE_HOME` environment variables shown above when running long plotting jobs from the terminal.

## Notes

- The current benchmark settings are defined in [real_city_benchmark.py](/Users/yongxinliu/Career/CareerAtERAU/KG Transportation System/experiments/city/real_city_benchmark.py).
- The current city-evaluation settings are `10 x 10` zoning, an `800`-route library, `500` vehicles per episode, `8` episodes per attack family, a `20/12` train-test split, and a `2.0%` attacker share.
