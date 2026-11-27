# Scripts and output mapping

The runner executes these in the displayed order. All paths below are relative to the configured output directory (default `results/`).

| Stage | Purpose | Inputs / prerequisites | Main outputs |
| --- | --- | --- | --- |
| 01 | Aggregate data-quality audit and L1/L2 reconciliation | Both Parquet tables | `data_quality/aggregate_audit.json` |
| 02 | Daily volume, weekday-hour volume, top zones and monthly platform shares | Both Parquet tables and zone lookup | `01_daily.csv`, `02_weekday_hour.csv`, `03_zones.csv`, `04_platform.csv` and matching PNGs |
| 03 | Weekly log-STL, residual candidates and sensitivity | Stage 02 daily CSV | `time_series/stl_daily_components.csv`, `weekday_factors.csv`, `monthly_daily_means.csv`, `anomaly_candidates.csv` |
| 04 | Monthly means, descriptive monthly ratios, rolling/block means | Stage 02 daily CSV and stage 03 monthly means | `time_series/monthly_pattern.png`, `monthly_seasonal_factors.csv`, rolling/block CSVs and figures |
| 05 | Weekday regression tests, HAC intervals and Holm correction | Stage 02 daily CSV | `time_series/weekly_significance.png`, `weekly_significance/*.csv`, summary and English analysis notes |
| 06 | Corrected L1 audit, pooled hourly waits, weekday/weekend and monthly comparisons | Corrected L1 | `hourly_wait/hourly_wait.csv`, `daily_hourly_sums.csv`, `audit_summary.json` and figures |
| 07 | Weekday-by-hour waits | Stage 02 volume matrix and stage 06 daily/hourly sums | `hourly_wait/weekday_hour_wait.png`, matrix, detailed table and summary |
| 08 | Platform totals, regional composition, borough-hour and day-type comparisons | Corrected L1 and zone lookup | `platform_comparison/platform_totals.csv`, borough/day-type tables, mix and wait-gap figures |
| 09 | Platform hourly wait comparison | Corrected L1, stage 06 hourly totals, stage 08 platform totals | `platform_comparison/03_hourly_platform_wait.png`, hourly tables and summary |
| 10 | Uber borough-hour wait heatmap | Stage 08 borough, day-type and platform totals | `platform_comparison/04_uber_borough_hour_wait.png`, matrix and summary |
| 11 | Exploratory annual MSTL and Fourier cross-checks | Stage 02 daily CSV | `time_series/annual_decomposition/`, annual figures and sensitivity results |

STL residual candidates in stage 03 are aggregate-day diagnostics. They are distinct from the team's trip-level Isolation Forest anomaly detector.

## Report figures

`export_report_figures.py` creates these filenames in `results/report_figures/`, matching the figures in the group's `report/report.pdf`:

| Report filename | Source |
| --- | --- |
| `fig01_weekday_effect.png` | Stage 05, `time_series/weekly_significance.png` |
| `fig02_monthly_trend.png` | Single-panel rendering of stage 04 monthly means with observed daily trips |
| `fig03_top15_zones.png` | Stage 02, `03_zones.png` |
| `fig04_weekday_hour_heat.png` | Stage 02, `02_weekday_hour.png` |
| `fig05_wait_weekday_hour.png` | Stage 07, `hourly_wait/weekday_hour_wait.png` |
| `fig06_platform_share.png` | Stage 02, `04_platform.png` |
| `fig07_borough_mix.png` | Stage 08, `platform_comparison/01_borough_platform_mix.png` |
| `fig08_wait_by_platform.png` | Stage 09, `platform_comparison/03_hourly_platform_wait.png` |
| `fig09_uber_pressure.png` | Stage 10, `platform_comparison/04_uber_borough_hour_wait.png` |

Stage 11 is retained for completeness but does not provide evidence that annual seasonality is stable. The report's monthly chart is based on observed monthly means, not the annual MSTL component.
