# out_of_use/ — 不参与任何结论的产物与运行记录

这个目录放的是**被取代或被判定为不可复现**的东西。规则只有一条：**这里面的任何数字都不要引用**。
需要"这个数字从哪来"时在这里查；需要"现在的结论是什么"时去 `data/output/` 现役目录或
`handoff/ML/readme_Isolation_tree_Kmeans.md`。

- `_outputs/` —— 被取代的产物（旧版本流水线的输出、作废的簇表与图）
- `best_k.txt` —— 详见下表第 1 行

## 逐项溯源

| 文件 / 目录 | 由谁产生 | 为什么在这里 | 现在该用哪个 |
|---|---|---|---|
| `best_k.txt`（1 字节，内容 `4`） | `code/ml/out_of_use/kmeans_tuning.py:162-164`（该脚本的第二个输出） | 与本目录的 `tuning_results_untraced/tuning_results.csv` **同一分钟**（2026-11-18 03:11）成对生成；那次运行的输入规模无法从仓库重建，CSV 的列数比脚本 `:156` 的输出少一列 `elapsed_sec`、换行是 CRLF（被 Excel 另存过），按 WSSSE 反推约 9.2M 行而脚本采样率应对应 4,672 万行。**整套结果不可复现**，故与 CSV 一并移入本目录 | K 选择看 `data/output/clustering_robustness/`（10 种子 × K=2..8，K=2 全票，silhouette 0.2198） |
| `_outputs/anomaly_features_deprecated/` | `04_rename_clusters.py` 的 `ARCHIVE_DIR` 输出 + 早期迁移 | 旧簇名/旧编号的产物，含 `anomalies_enriched.parquet`（旧单位口径）、`consistency_profile.csv`（`spd_med` 量纲错）、按旧别名分组的 `fare_per_mile_by_cluster.csv` / `tip_ratio_top20.csv` | `data/output/anomaly_features/`（`cluster_money_summary.csv`、`cluster_sizes_named.csv`）与 `data/output/anomaly_features_v2/` |
| `_outputs/anomaly_features_stale/` | `04_rename_clusters.py` 的 `STALE_DIR` 输出（2026-11-22 起直接写这里） | `cluster_centers_named.csv` 列名是原始特征名、值是 **StandardScaler 空间**（`speed_mph` 中心 3.433 不是 mph）；`cluster_persona.csv` 的 `top_shap_driver` 是旧 SHAP 字符串。这两个曾两次被误读为物理量 | 物理量纲看 `data/output/unit_check/unit_check_report.txt`；簇驱动特征看 `data/output/shap_final/shap_top_drivers_final.csv` |
| `_outputs/if_results_10pct_of_10pct/` | 旧版 `02`（无 `row_uid` 的 IsolationForest 输出） | 旧 03 与 09 都读过它；其 silhouette 0.3662 是 Spark `squaredEuclidean` 尺度下的值（与欧氏尺度比值 1.66–1.70（K=2..8，非精确常数），复跑复现 0.3641） | `data/output/if_results_rowkeyed/`（13,577 / 0.291%） |
| `_outputs/if_results_2pct_precision/` | 早期 2% 精度试探 | 被 `if_results_rowkeyed` 取代 | 同上 |
| `_outputs/kmeans_anomaly_clusters_k2_legacy/` | 迁移遗留的重复副本（旧 ACL 禁止删除） | 内容与 `anomaly_clusters_rowkeyed/strict_anomalies_clustered/` 的 `part-00000-4bbbf451-*.parquet` **字节相同**（同一 Spark 作业 UUID） | `data/output/anomaly_clusters_rowkeyed/` |
| `_outputs/anomaly_clusters_positional_jsonvalid/`（5 个文件） | `03_kmeans_cluster_anomalies.py` 当年的 SHAP 位置连接输出（`shap_top3_per_cluster.csv`、`shap_cluster_summary.csv`、`shap_bar_cluster_0/1.png`），以及 2026-11-22 补移进来的 `k_selection_summary.csv`（K=2..8 的 WSSSE/silhouette 扫描表，K=2 记 0.3662） | 前四个文件里的簇标签是按**行位置**硬贴上去的推断结果，指标与簇号都不可信；`k_selection_summary.csv` 的 silhouette 列用的是 Spark 的 `squaredEuclidean` 度量（该 evaluator 只支持这一种，与全项目其余 sklearn 欧氏数字相差约 1.66-1.70×（K=2..8），见 README §8），两者都来自旧的无 `row_uid` 表，因此同置一处。`03.py` 已改为把这张表直接写到这里，重跑不会再污染现役目录 | 簇驱动特征看 `data/output/shap_final/shap_top_drivers_final.csv`；簇画像看 `data/output/anomaly_features_v2/consistency_profile_v2.csv`；K 选择看 `data/output/clustering_robustness/`（K=2，silhouette 0.2198）。现役簇目录只留 `03` 的簇划分本体（`cluster_centers.csv`、`cluster_sizes.csv`、`strict_anomalies_clustered/`）与 `07b` 的 K=4 自检 `selfcheck_k4/` |
| `_outputs/tuning_results_untraced/` | 2026-11-22 由旧 `data/output/tuning_results/` 移入（CSV + `best_k.txt`） | 见第 1 行；整份 CSV 与 `best_k.txt` 同源、同批判定为不可复现，放在一起便于追溯 | K 选择看 `clustering_robustness/` |
