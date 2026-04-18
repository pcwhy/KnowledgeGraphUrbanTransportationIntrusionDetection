# Repository Guide

## Purpose

This project implements a transportation-cybersecurity research pipeline with three detector families:

- a local-rule baseline,
- a flat-feature logistic detector,
- and a knowledge-graph detector backed by explicit transportation state.

The repository follows the four-layer architecture required by the `scholar-engine` workflow:

`raw_data/ -> experiments/ -> results/structured/ -> rendering/`

## Top-Level Layout

### `raw_data/`

External inputs and rebuildable caches used by the experiments.

- `city_cache/`
  - cached OpenStreetMap graphs,
  - cached TxDOT Annual Average Daily Traffic layers,
  - cached TxDOT permanent-count-station layers.

### `experiments/`

Experiment logic only. These scripts should not write LaTeX or manuscript-facing prose.

- `common/project_paths.py`
  - shared path constants for the four-layer pipeline.
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
  - KG snapshot export for the manuscript figure.

### `results/structured/`

Machine-readable result bridges. This layer is the single source of truth for rendered paper assets.

- `synthetic/`
  - synthetic CSV and JSON outputs.
- `city/`
  - real-city CSV and JSON outputs.
- `overhead/`
  - overhead benchmark CSV and JSON outputs.
- `assets/`
  - structured metadata for figure-generation workflows.

### `rendering/`

Paper-facing and presentation-facing outputs.

- `scripts/paper_asset_renderers.py`
  - rebuilds paper figures and LaTeX tables from `results/structured/`.
- `paper/`
  - manuscript source, generated tables, generated figures, and compiled PDF.
- `slides/`
  - slide material and exported decks.

### `archive/`

Preserved legacy and upstream material.

- `upstream/`
  - motivating upstream autonomous-vehicle traffic-system code.
- `legacy/`
  - old archives and legacy working files.
- `snapshots/pre_restructure_20260324/`
  - full backup captured before the restructuring pass.

## Canonical Pipeline

The intended dependency chain is:

1. `raw_data/` stores external inputs and caches.
2. `experiments/` produces machine-readable outputs only.
3. `results/structured/` stores CSV and JSON bridges.
4. `rendering/` turns those bridges into figures, tables, and the paper PDF.

Experiment scripts should write structured outputs only. Paper-facing `.tex` snippets and figure assets should be generated downstream from the structured-results layer.

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

Regenerate the KG snapshot TikZ fragment:

```bash
MPLCONFIGDIR="$PWD/.mplconfig" XDG_CACHE_HOME="$PWD/.cache" .venv/bin/python experiments/analysis/export_kg_tikz.py
```

Refresh paper assets from structured outputs only:

```bash
MPLCONFIGDIR="$PWD/.mplconfig" XDG_CACHE_HOME="$PWD/.cache" .venv/bin/python rendering/scripts/paper_asset_renderers.py
```

Compile the paper:

```bash
cd rendering/paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

## Recommended Rebuild Order

For a full end-to-end refresh:

```bash
MPLCONFIGDIR="$PWD/.mplconfig" XDG_CACHE_HOME="$PWD/.cache" .venv/bin/python experiments/synthetic/run_experiments.py
MPLCONFIGDIR="$PWD/.mplconfig" XDG_CACHE_HOME="$PWD/.cache" .venv/bin/python experiments/city/real_city_case_study.py
MPLCONFIGDIR="$PWD/.mplconfig" XDG_CACHE_HOME="$PWD/.cache" .venv/bin/python experiments/city/real_city_benchmark.py
MPLCONFIGDIR="$PWD/.mplconfig" XDG_CACHE_HOME="$PWD/.cache" .venv/bin/python experiments/analysis/export_kg_tikz.py
MPLCONFIGDIR="$PWD/.mplconfig" XDG_CACHE_HOME="$PWD/.cache" .venv/bin/python experiments/analysis/benchmark_overhead.py
MPLCONFIGDIR="$PWD/.mplconfig" XDG_CACHE_HOME="$PWD/.cache" .venv/bin/python rendering/scripts/paper_asset_renderers.py
cd rendering/paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

If only styling or figure formatting changes, rerun the renderer and then rebuild the paper.

## Raw Data Acquisition

The real-city pipeline uses public data and caches the downloaded files in `raw_data/city_cache/`. The cache is rebuildable, but the cached files should be treated as the raw-data layer for ordinary experiment reruns.

The OpenStreetMap graphs are downloaded through OSMnx in [real_city_case_study.py](/Users/yongxinliu/Career/CareerAtERAU/KG Transportation System/experiments/city/real_city_case_study.py). The current city queries are:

- Austin: center `(30.2672, -97.7431)`, radius `5500` m.
- Houston: center `(29.7604, -95.3698)`, radius `6500` m.
- Dallas: center `(32.7767, -96.7970)`, radius `6000` m.

The TxDOT annual average daily traffic records are downloaded from:

```text
https://services.arcgis.com/KTcxiTD9dsQw4r7Z/ArcGIS/rest/services/TxDOT_AADT/FeatureServer/0/query
```

The query uses the graph envelope, `AADT_CUR IS NOT NULL`, and returns GeoJSON records with `AADT_CUR`, route-name fields, and geometry. Cached files are named `raw_data/city_cache/{city}_txdot_aadt.geojson`.

The TxDOT permanent count station records are downloaded from:

```text
https://services.arcgis.com/KTcxiTD9dsQw4r7Z/ArcGIS/rest/services/TxDOT_Permanent_Count_Stations/FeatureServer/0/query
```

The query uses the graph envelope and returns GeoJSON records for count-station locations. Cached files are named `raw_data/city_cache/{city}_txdot_permanent_count_stations.geojson`.

AADT records are assigned to directed graph edges through a nearest spatial join in EPSG:3857. Permanent count stations are snapped to nearest graph edges; those edges and their one hop neighbors define observed corridors and monitored intersections for the city benchmark.

## Headless Rendering Note

The plotting scripts are configured for headless execution with Matplotlib's `Agg` backend because the macOS GUI backend can abort when launched from a non-GUI worker process. Use the `MPLCONFIGDIR` and `XDG_CACHE_HOME` environment variables shown above when running long plotting jobs from the terminal.

## Asset Provenance

Per-asset provenance notes live in [README.md](/Users/yongxinliu/Career/CareerAtERAU/KG Transportation System/docs/assets/README.md). Each figure and table used in the paper has a companion Markdown note that records:

- source data,
- generating script,
- output files,
- and the manuscript role of the asset.

## Notes

- `rendering/paper/generated/` and `rendering/paper/figures/` contain derived outputs and are rebuildable.
- `figSimulator.png` and `figKGDetector.png` are static conceptual figures rather than Python-generated assets.
- The current benchmark settings are defined in [real_city_benchmark.py](/Users/yongxinliu/Career/CareerAtERAU/KG Transportation System/experiments/city/real_city_benchmark.py).
- The current city-evaluation settings are `10 x 10` zoning, an `800`-route library, `500` vehicles per episode, `8` episodes per attack family, a `20/12` train/test split, and a `2.0%` attacker share.
