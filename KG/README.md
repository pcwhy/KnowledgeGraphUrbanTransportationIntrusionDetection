# KG Transportation System

This repository is organized as a four-layer research pipeline:

1. `raw_data/`
   - External caches and preserved inputs used by the experiments.
2. `experiments/`
   - Synthetic, city-scale, and overhead experiment code.
3. `results/structured/`
   - Machine-readable outputs such as CSV and JSON files.
4. `rendering/`
   - Paper source, generated manuscript assets, rendering scripts, and slide material.

The current paper source is in `rendering/paper/`. The paper-facing figures and generated LaTeX tables are written under `rendering/paper/figures/` and `rendering/paper/generated/`, but those assets are rendered from the structured outputs in `results/structured/`.

## Directory Overview

- `raw_data/city_cache/`
  - Cached OpenStreetMap and TxDOT inputs for the Austin, Houston, and Dallas studies.
- `experiments/synthetic/`
  - Synthetic simulator, detectors, and synthetic runner.
- `experiments/city/`
  - Real-city case-study and city-scale benchmark code.
- `experiments/analysis/`
  - Overhead measurements and KG snapshot export.
- `experiments/common/`
  - Shared project-path helpers used across the pipeline.
- `results/structured/synthetic/`
  - Synthetic CSV and JSON outputs.
- `results/structured/city/`
  - Real-city CSV and JSON outputs.
- `results/structured/overhead/`
  - Overhead benchmark outputs.
- `results/structured/assets/`
  - Machine-readable asset metadata, such as KG snapshot metadata.
- `rendering/scripts/`
  - Renderers that turn structured outputs into paper-facing figures and LaTeX tables.
- `rendering/paper/`
  - IEEE paper source, compiled PDF, generated figures, and generated LaTeX snippets.
- `rendering/slides/`
  - Slide-generation material.
- `archive/`
  - Preserved upstream code, legacy archives, and the pre-restructure snapshot.

## Main Commands

Run the synthetic experiment:

```bash
.venv/bin/python experiments/synthetic/run_experiments.py
```

Run the real-city case-study visualizations:

```bash
.venv/bin/python experiments/city/real_city_case_study.py
```

Run the real-city benchmark:

```bash
.venv/bin/python experiments/city/real_city_benchmark.py
```

Run the overhead benchmark:

```bash
.venv/bin/python experiments/analysis/benchmark_overhead.py
```

Regenerate the KG snapshot TikZ asset:

```bash
.venv/bin/python experiments/analysis/export_kg_tikz.py
```

Rerender paper assets from structured results only:

```bash
MPLCONFIGDIR="$PWD/.mplconfig" XDG_CACHE_HOME="$PWD/.cache" .venv/bin/python rendering/scripts/paper_asset_renderers.py
```

Compile the paper:

```bash
cd rendering/paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

## Documentation

- [Repository Guide](docs/repository_guide.md)
- [Paper Asset Generation](docs/paper_asset_generation.md)
- [Asset Provenance Index](docs/assets/README.md)

## Snapshot

The pre-restructure snapshot is preserved in:

`archive/snapshots/pre_restructure_20260324/`
