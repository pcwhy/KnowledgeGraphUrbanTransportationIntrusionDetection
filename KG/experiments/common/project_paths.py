from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "raw_data"
CITY_CACHE_DIR = RAW_DATA_DIR / "city_cache"

RESULTS_DIR = PROJECT_ROOT / "results" / "structured"
SYNTHETIC_RESULTS_DIR = RESULTS_DIR / "synthetic"
CITY_RESULTS_DIR = RESULTS_DIR / "city"
OVERHEAD_RESULTS_DIR = RESULTS_DIR / "overhead"
ASSET_RESULTS_DIR = RESULTS_DIR / "assets"
ANALYSIS_RESULTS_DIR = RESULTS_DIR / "analysis"
STATISTICS_RESULTS_DIR = RESULTS_DIR / "statistics"
SENSITIVITY_RESULTS_DIR = RESULTS_DIR / "sensitivity"

RENDERING_DIR = PROJECT_ROOT / "rendering"
PAPER_DIR = RENDERING_DIR / "paper"
PAPER_GENERATED_DIR = PAPER_DIR / "generated"
PAPER_FIGURES_DIR = PAPER_DIR / "figures"
REAL_CITY_FIGURES_DIR = PAPER_FIGURES_DIR / "real_city"

ARCHIVE_DIR = PROJECT_ROOT / "archive"
DOCS_DIR = PROJECT_ROOT / "docs"


def ensure_pipeline_directories() -> None:
    for path in (
        CITY_CACHE_DIR,
        SYNTHETIC_RESULTS_DIR,
        CITY_RESULTS_DIR,
        OVERHEAD_RESULTS_DIR,
        ASSET_RESULTS_DIR,
        ANALYSIS_RESULTS_DIR,
        STATISTICS_RESULTS_DIR,
        SENSITIVITY_RESULTS_DIR,
        PAPER_GENERATED_DIR,
        PAPER_FIGURES_DIR,
        REAL_CITY_FIGURES_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
