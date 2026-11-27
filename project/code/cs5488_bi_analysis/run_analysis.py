"""Run the CS5488 BI analyses in dependency order using the current Python interpreter."""
import argparse
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from importlib import metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
STAGES = [
    "01_audit_aggregate_data.py",
    "02_trip_volume_and_platform_share.py",
    "03_weekly_stl_decomposition.py",
    "04_monthly_trends_and_smoothing.py",
    "05_weekday_significance_tests.py",
    "06_hourly_wait_analysis.py",
    "07_weekday_hour_wait_heatmap.py",
    "08_platform_borough_comparison.py",
    "09_platform_hourly_wait.py",
    "10_uber_borough_hour_heatmap.py",
    "11_annual_seasonality_exploration.py",
]
PACKAGES = ("numpy", "pandas", "pyarrow", "matplotlib", "seaborn", "scipy", "statsmodels")


def check_inputs(data, lookup):
    import pyarrow.parquet as pq

    expected = {(date(2023, 1, 1) + timedelta(days=i)).isoformat() for i in range(731)}
    shared = {"pickup_date", "pickup_hour", "PULocationID", "trip_count", "total_fare",
              "total_miles", "total_tips", "avg_wait_minutes"}
    schemas = {
        "l1_detail": shared | {"hvfhs_license_num", "valid_wait_count", "valid_fare_count",
                               "valid_trip_count", "total_valid_wait_minutes"},
        "l2_overview": shared,
    }
    summary = {}
    for table, required in schemas.items():
        files = sorted((data / table).rglob("*.parquet"))
        counts = Counter(p.parent.name.removeprefix("pickup_date=") for p in files)
        if set(counts) != expected or any(v != 1 for v in counts.values()):
            raise ValueError(f"{table}: expected one compact Parquet file for each of 731 dates; found {len(files)} files")
        for path in files:
            missing = required - set(pq.ParquetFile(path).schema_arrow.names)
            if missing:
                raise ValueError(f"{path.name}: missing fields {sorted(missing)}. Waiting-time analysis requires the corrected L1 table.")
        summary[table] = {"files": len(files), "required_columns": sorted(required)}
    if not lookup.is_file():
        raise ValueError(f"Taxi-zone lookup file not found: {lookup}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/bi_aggregates_compact",
                        help="Directory containing corrected l1_detail/ and l2_overview/")
    parser.add_argument("--out", type=Path, default=ROOT / "results", help="Output directory")
    parser.add_argument("--lookup", type=Path, default=ROOT / "data/taxi_zone_lookup.csv")
    parser.add_argument("--core-only", action="store_true", help="Skip exploratory annual decomposition (stage 11)")
    parser.add_argument("--check-only", action="store_true", help="Check packages, date coverage and input schemas without running analyses")
    args = parser.parse_args()
    data, output, lookup = (p.expanduser().resolve() for p in (args.data, args.out, args.lookup))
    if output == data or output in data.parents or data in output.parents:
        parser.error("Keep the input-data directory and output directory separate.")
    try:
        versions = {name: metadata.version(name) for name in PACKAGES}
        inputs = check_inputs(data, lookup)
    except (metadata.PackageNotFoundError, ImportError, OSError, ValueError) as exc:
        parser.exit(2, f"Input/dependency check failed: {exc}\nSee README.md and docs/INPUT_SCHEMA.md.\n")
    print("Input checks passed: 731 daily files per table; required columns present.", flush=True)
    if args.check_only:
        print(json.dumps({"software": versions, "inputs": inputs}, indent=2))
        return
    output.mkdir(parents=True, exist_ok=True)
    logs = output / "logs"
    logs.mkdir(exist_ok=True)
    env = os.environ.copy()
    env.update({"CS5488_DATA_DIR": str(data), "CS5488_OUTPUT_DIR": str(output),
                "CS5488_LOOKUP_FILE": str(lookup), "MPLCONFIGDIR": str(output / ".matplotlib"),
                "PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1"})
    stages = STAGES[:-1] if args.core_only else STAGES
    manifest = {"started_utc": datetime.now(timezone.utc).isoformat(),
                "python": sys.version, "software": versions, "input": str(data),
                "input_checks": inputs, "profile": "core" if args.core_only else "all", "stages": []}
    def save_manifest():
        (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for index, stage in enumerate(stages, 1):
        print(f"[{index}/{len(stages)}] {stage}", flush=True)
        start = time.perf_counter()
        log_path = logs / (Path(stage).stem + ".log")
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run([sys.executable, str(ROOT / "scripts" / stage)],
                                    cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        manifest["stages"].append({"script": stage, "exit_code": result.returncode,
                                   "elapsed_seconds": round(time.perf_counter() - start, 3),
                                   "log": str(log_path.relative_to(output))})
        save_manifest()
        if result.returncode:
            print(log_path.read_text(encoding="utf-8", errors="replace")[-5000:], file=sys.stderr)
            parser.exit(result.returncode, f"Stopped at {stage}. Details: {log_path}\n")
        print(f"  completed in {manifest['stages'][-1]['elapsed_seconds']:.1f}s", flush=True)
    result = subprocess.run([sys.executable, str(ROOT / "export_report_figures.py"), "--out", str(output)],
                            cwd=ROOT, env=env)
    if result.returncode:
        parser.exit(result.returncode, "Analysis completed, but report-figure export failed.\n")
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["report_figures_exported"] = 9
    save_manifest()
    print(f"All analyses completed. Outputs: {output}", flush=True)


if __name__ == "__main__":
    main()
