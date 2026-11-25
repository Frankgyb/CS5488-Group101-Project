# selfcheck_k4 — 一次历史运行，不是当前代码的产物

> **一句话**：本目录里的 K 扫描表选出 K=4，但**用仓库当前的
> `project/code/ml/analysis/07b_shap_cluster_analysis_rowkeyed.py` 重跑会得到另一组数，
> 且 K=2 得分最高**。所以这里的数只作历史留存，**K 选择一律引 `data/output/clustering_robustness/`**。

## 这些文件是哪来的

全部写于 **2026-11-22 00:53:55 – 00:54:07**，紧接在 `02b` 写出 rowkeyed 表之后
（`if_results_rowkeyed/` 的 mtime 是 00:52:35，`02b` 日志结束于 00:52:48）。
所以它读的**确实是新的 rowkeyed 表**，不是旧的无 `row_uid` 表。

| 文件 | mtime | 内容 |
|---|---|---|
| `k_selection_summary.csv` | 00:53:55 | K=2..8 的 WSSSE / silhouette，记录值选 K=4 |
| `shap_cluster_summary.csv` | 00:53:55 | 逐簇 SHAP 汇总 |
| `shap_top3_per_cluster.csv` | 00:53:55 | 逐簇 top-3 驱动特征 |
| `shap_bar_cluster_0..3.png` / `shap_beeswarm_cluster_0..3.png` | 00:54 | 8 张图 |
| `alignment_audit.csv` / `.txt` | 00:54:07 | SHAP 行对齐审计（通过） |
| `strict_anomalies_clustered_rowkeyed/` | 00:54:07 | 13,577 行的 K=4 簇标签表 |

## 为什么说它不是当前代码的产物

三条独立证据：

**① 数值不可复现。** 在 `k_selection_diagnostics/` 里用同样的输入、同样的参数
（`seed=42 / maxIter=20 / tol=1e-4 / squaredEuclidean`）重复 6 次，**7 个 K 的 WSSSE 每次都完全相同**
（spread = 0.0，Spark 在这里是确定性的），但**与本表 7 行全部对不上**：

| K | 本表记录 | 同配置重跑（6 次相同） |
|---|---|---|
| 2 | 102,254.7 | 98,241.7 |
| 3 | 83,681.9 | 83,693.9 |
| 4 | 75,427.0 | 75,717.2 |
| 5 | 71,570.2 | 68,572.9 |
| 6 | 63,977.5 | 64,285.2 |
| 7 | 61,263.3 | 61,485.7 |
| 8 | 58,756.9 | 59,688.9 |

既然同一配置重跑 spread=0，就排除了"初始化运气"。差异只能来自**不同的代码或不同的参数**。

**② 表结构对不上。** `strict_anomalies_clustered_rowkeyed/` 有 `cluster_name` 列，
但只有 4 个簇中的 2 个有名字（`long_low_tip` 5,295 / `short_high_tip` 2,574，簇 2、簇 3 无名）。
而仓库里的 `07b` 只输出 `row_uid` + `cluster`（`07b_shap_cluster_analysis_rowkeyed.py:185`），
**根本不写 `cluster_name`**。说明当时运行的是另一版脚本。

**③ 用当前代码重跑，选不出 K=4。** 同配置重跑的 silhouette 是
K=2 **0.3641** > K=5 0.3108 > K=3 0.3082 > K=4 0.3028 …… **K=2 最高，K=4 排第三。**

结论：本次运行来自**一版早期 07 脚本**（作者已确认该版本已从仓库中删除），无法重建。

## 那这个目录还有什么用

- `alignment_audit.*`：记录 SHAP 表按 `row_uid` 与簇表对齐、7.105e-15 级最大偏差，
  审计本身有效，与 K 取几无关；
- `strict_anomalies_clustered_rowkeyed/`：**是现役输入**——
  `10_robust_clustering_rowkeyed.py` 与 `add_consistency_features_v2.py` 都读它
  （前者只用 9 个特征列 + `row_uid`，后者再用 `cluster` 列）。**不要删除或覆盖**；
- `shap_*.csv` / `*.png`：历史留存，**不要引用**。逐簇 SHAP 的权威版本在 `data/output/shap_final/`。

## 不要做的事

- ❌ 不要引用 `k_selection_summary.csv` 里的任何数值（包括"会选 K=4"这个结论）；
- ❌ 不要用当前 `07b` 重跑去"修正"这个目录——重跑会覆盖
  `strict_anomalies_clustered_rowkeyed/`，而新版本不写 `cluster_name` 列，会让下游两个脚本拿到残缺的表；
- ✅ 要引用 K 选择，用 `data/output/clustering_robustness/`（K=2，silhouette 0.2198，10/10 种子，
  bootstrap ARI 0.9593 ± 0.0283）；
- ✅ 要引用同配置的、可复现的 K 扫描，用 `data/output/k_selection_diagnostics/k_probe.csv`。

## 审计记录

完整过程（6 次重复实验、时间线、逐 K 对照）见：
- `data/output/k_selection_diagnostics/README.md` §8
- `data/output/k_selection_diagnostics/audit_07b_k_rows.py` 与其输出
  `k_spread_07b_audit.csv` / `k_spread_07b_audit.txt`
