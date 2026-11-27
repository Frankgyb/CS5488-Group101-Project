# Packaging verification

The portable package was verified on 2026-11-27 using the existing corrected 2023–2024 L1/L2 aggregate tables.

## Checks completed

- All **11 analysis stages** completed successfully through `run_analysis.py` with a custom input path and a separate temporary output directory.
- **45 generated CSV files** were compared with the previously saved analysis outputs. All comparisons passed with relative tolerance `1e-9` and absolute tolerance `1e-8`; strings, row order and column structure were also compared.
- Main numerical fields in weekly STL, annual decomposition, pooled waiting-time, platform-hour and Uber borough-hour JSON summaries matched their existing baselines.
- L1 and L2 each reconciled to **471,960,468 recorded trips**. Their row counts were 8,651,895 and 4,421,839 respectively, with no missing L1/L2 group keys and no count mismatches.
- Waiting-time controls reconciled to **467,359,108 valid records**.
- All **nine report PNGs** were exported under the expected filenames.
- All 14 Python files passed syntax checks. Analysis scripts no longer contain personal absolute input paths.
- Input checking also passed when the runner was launched from outside the package directory.

Machine-readable counts and the list of compared CSVs are in `verification_summary.json`. Direct dependency versions are recorded in `requirements-tested.txt`; the verified interpreter was Python 3.14.6 on macOS.

Verification concerns numerical reproducibility and packaging. The annual analysis remains exploratory, and no causal interpretation is established by successful execution. Report rendering may vary slightly with plotting-library and font versions. The monthly report image is regenerated from the same monthly means as a single-panel chart, rather than copied from the Word file.

The original local scripts were retained. Changes in this package centralize input/output paths, add the run order and documentation, remove the audit's obsolete private dependency directory, and allow any filename within each compact daily partition. The aggregate audit and STL scripts now have explicit `main()` entry points. Analysis formulas and model settings are unchanged.
