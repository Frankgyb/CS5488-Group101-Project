# full_scores_rowkeyed/ — not tracked

Per-row Isolation Forest scores for all 4,672,734 scored trips (172 MB). The 13,577 anomaly
rows themselves are tracked one level up in `strict_anomalies_rowkeyed/`.

Rebuild:

    python project/code/ml/pipeline/02b_isolation_forest_shap_rowkeyed.py

See [`../../../README.md`](../../../README.md) for the complete tracked/untracked data map.
