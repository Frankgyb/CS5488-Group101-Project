"""Shared portable locations, configured by run_analysis.py or environment variables."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("CS5488_DATA_DIR", PROJECT_ROOT / "data/bi_aggregates_compact")).expanduser().resolve()
OUTPUT_ROOT = Path(os.environ.get("CS5488_OUTPUT_DIR", PROJECT_ROOT / "results")).expanduser().resolve()
LOOKUP_FILE = Path(os.environ.get("CS5488_LOOKUP_FILE", PROJECT_ROOT / "data/taxi_zone_lookup.csv")).expanduser().resolve()

# Keep Matplotlib's generated files with the outputs, not in a personal home folder.
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_ROOT / ".matplotlib"))
