# CS5488 NYC Ride Hailing BI Analysis

This folder collects the BI analysis code developed for the CS5488 Group 101 project. It covers aggregate-data auditing, trip-volume patterns, weekly statistical tests, waiting-time analysis and Uber/Lyft comparisons for 2023–2024.

中文说明：本文件夹已整理之前的 11 个分析及核查脚本，统一命名并移除个人电脑路径。可将整个文件夹上传到小组仓库的 `project/code/cs5488_bi_analysis/`。原始大数据和生成结果不包含在代码包内。

The scripts consume the group's compact L1/L2 aggregate tables. The upstream Spark preprocessing, anomaly detection, clustering and parallel-scaling code remain in the main project repository; this folder does not duplicate those implementations.

## Quick start

Use Python 3.10 or later. From this folder:

```bash
python -m venv .venv
```

Activate the environment (`source .venv/bin/activate` on macOS/Linux, or `.venv\Scripts\activate` on Windows), then:

```bash
python -m pip install -r requirements.txt
python run_analysis.py --data "/path/to/bi_aggregates_compact"
```

`--data` must point to the parent directory containing **both** `l1_detail/` and `l2_overview/`, not to one monthly raw TLC file. Use the corrected L1 version containing `total_valid_wait_minutes`.

All 11 analyses run in order. Results, CSV tables, figures and logs are written to `results/`. The runner also creates `results/report_figures/` using the nine image filenames in the group's LaTeX report.

```bash
# Only check dependencies, date coverage and required input columns.
python run_analysis.py --data "/path/to/bi_aggregates_compact" --check-only

# Run the report workflow without exploratory annual decomposition.
python run_analysis.py --data "/path/to/bi_aggregates_compact" --core-only

# Choose a different output directory; all downstream scripts follow it.
python run_analysis.py --data "/path/to/bi_aggregates_compact" --out "/path/to/bi_results"
```

If input data is placed in `data/bi_aggregates_compact/`, `python run_analysis.py` needs no arguments. Input data is only read. Re-running into the same output directory replaces generated files of the same name.

## Folder contents

```text
cs5488_bi_analysis/
├── run_analysis.py                    # Runs all stages in dependency order
├── export_report_figures.py           # Exports the nine report charts
├── requirements.txt                  # Runtime dependencies
├── requirements-tested.txt           # Versions used for local verification
├── scripts/
│   ├── _config.py                    # Shared data/output/lookup paths
│   ├── 01_audit_aggregate_data.py
│   ├── 02_trip_volume_and_platform_share.py
│   ├── 03_weekly_stl_decomposition.py
│   ├── 04_monthly_trends_and_smoothing.py
│   ├── 05_weekday_significance_tests.py
│   ├── 06_hourly_wait_analysis.py
│   ├── 07_weekday_hour_wait_heatmap.py
│   ├── 08_platform_borough_comparison.py
│   ├── 09_platform_hourly_wait.py
│   ├── 10_uber_borough_hour_heatmap.py
│   └── 11_annual_seasonality_exploration.py
├── notebooks/00_load_l2_example.ipynb  # Optional original loading example
├── data/taxi_zone_lookup.csv          # Official TLC zone lookup
└── docs/
    ├── INPUT_SCHEMA.md
    ├── SCRIPT_MAP.md
    ├── VALIDATION.md
    ├── source_inventory.json
    └── tlc_2023_2024_manifest.csv
```

The optional notebook requires Jupyter, which is not required by the command-line workflow.

## Analysis scope and metric definitions

- Trip counts represent **471,960,468 recorded trips**, not the number of aggregate rows. L1 has 8,651,895 groups and L2 has 4,421,839 groups.
- Trip-volume charts use all recorded trips. Waiting-time charts use corrected L1 sums and counts for waits in **0–120 minutes**, inclusive: `SUM(total_valid_wait_minutes) / SUM(valid_wait_count)`. There are 467,359,108 valid waiting-time records.
- An empty valid-wait group can have a NULL sum. The scripts treat that sum as zero only when its valid count is zero. An unknown mean is not treated as a zero-minute wait.
- Date and hour groupings use **pickup time**, not request time. Weekdays/weekends are based on the calendar and do not separately classify holidays.
- Monthly platform shares use both platforms' combined recorded trips in that month. Platform borough composition uses each platform's own total as the denominator.
- Stage 05 uses regression on log daily trips, year-month fixed effects, Newey-West HAC standard errors and Holm-adjusted comparisons. It does not assign p-values to the fitted STL factors.
- Stage 11 is exploratory. Two years do not establish stable annual seasonality. Smooth curves are display aids, not forecasts; platform comparisons do not identify causal performance differences.

The date and total checks are intentionally fixed to this project's verified 2023–2024 release. Using another period requires updating and revalidating those checks, not merely changing the folder path.

## Individual scripts and dependencies

The numbered scripts can be run individually after their prerequisites have produced the required CSV files. See [SCRIPT_MAP.md](docs/SCRIPT_MAP.md). Shared defaults are configured by environment variables:

- `CS5488_DATA_DIR`: parent of `l1_detail/` and `l2_overview/`.
- `CS5488_OUTPUT_DIR`: common output directory for every stage.
- `CS5488_LOOKUP_FILE`: optional alternate taxi-zone lookup.

For example, after the daily CSV exists:

```bash
CS5488_OUTPUT_DIR="/path/to/bi_results" python scripts/05_weekday_significance_tests.py
```

The runner is the recommended entry point because it sets these variables and runs prerequisites automatically. Full diagnostic output is saved in `results/logs/`; `results/run_manifest.json` records software versions, stage status and timings.

## GitHub and the report

Upload this entire folder, including the small zone lookup and documentation. `.gitignore` excludes Parquet data, generated outputs, virtual environments and caches. Do not upload the large aggregate dataset as part of the code folder.

After a successful run, the nine PNGs in `results/report_figures/` can be copied into the existing repository directory `project/data/output/bi/`. They are generated locally; the scripts do not modify GitHub or automatically overwrite the report's existing image directory. The monthly figure is exported as a single panel using the same monthly means as stage 04; the original two-panel exploratory figure is also retained in the results.

## Sources and provenance

The included lookup and file manifest document the TLC source. The 24 monthly raw Parquet files are upstream source data, while these scripts require the already prepared L1/L2 tables.

- TLC trip records: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
- TLC field dictionary: https://www.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_hvfhs.pdf
- Taxi-zone lookup: https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv

[source_inventory.json](docs/source_inventory.json) maps the renamed scripts to their original filenames and records the original source hashes. The numerical analysis was preserved; packaging changes centralize paths, add an ordered runner and retain the existing checks. The aggregate audit now accepts any filename within the expected one-file-per-day layout. A small legacy notebook is retained with portable paths and cleared outputs.
