# k_selection_diagnostics — 为什么同一个数据集会出现两个 silhouette

## 这个目录解决什么问题

仓库里关于"K 应该取几"曾同时存在两组数：

| 出处 | K=2 silhouette | 会选 K |
|---|---|---|
| 旧 `03` 的 K 扫描（无 `row_uid` 的旧表） | **0.3662** | K=2 |
| `07b` 自检的 K 扫描（rowkeyed 表，Spark） | 0.2731 | **K=4** |
| `10_robust_clustering_rowkeyed.py`（10 种子 × 50 次重启，sklearn） | **0.2198** | K=2 |

三个数看起来互相矛盾，2026-11-22 之前仓库里给出的解释是"0.3662 是一次幸运初始化"，
这个解释**是错的**。本目录是实测证据。

## 实测结论

用 `03` 的完全相同配置（同一张 `if_results_rowkeyed/strict_anomalies_rowkeyed`、
`seed=42 / maxIter=20 / tol=1e-4`、单次初始化）重跑，对每个 K 同时记录**两套度量**：

| K | WSSSE | Spark `squaredEuclidean` | sklearn `euclidean` | 比值 |
|---|---|---|---|---|
| 2 | 98,241.67 | **0.3641** | **0.2185** | 1.666 |
| 3 | 83,693.95 | 0.3082 | 0.1829 | 1.685 |
| 4 | 75,717.24 | 0.3028 | 0.1780 | 1.701 |
| 5 | 68,572.94 | 0.3108 | 0.1854 | 1.676 |
| 6 | 64,285.17 | 0.2842 | 0.1715 | 1.657 |
| 7 | 61,485.70 | 0.2818 | 0.1695 | 1.663 |
| 8 | 59,688.94 | 0.2729 | 0.1649 | 1.655 |

两点结论：

1. **同一个 K、同一批标签，两套度量给两个数**，比值在 1.655–1.701 之间。
   它不是精确常数——平方距离不是度量的线性缩放，比值随划分形状微动——但量级稳定。
   Spark 的 `ClusteringEvaluator` **只接受** `distanceMeasure="squaredEuclidean"`；
   传 `"euclidean"` 会直接抛 `IllegalArgumentException`。这就是两套数同时存在的原因。
2. **两套度量各自都选 K=2**（平方距离 max 0.3641 @K=2，欧氏 max 0.2185 @K=2），
   所以"旧表会不会选出不同的 K"这个担心不成立。

## 3. 被排除的假设（保留，避免重复踩）

最初对 0.3662 与 0.2198 的差距给过两个解释，都被实测否定：

- **"0.3662 是一次幸运/不幸的初始化"**——错误。它是可复现的：重跑得 0.3641，差 0.0021。
- **"表里有零向量/退化行把指标抬高了"**——错误。9 维标准化特征的行范数
  min / median / max = **1.714 / 2.759 / 7.959**；精确零向量 **0** 个，
  范数 < 1e-6 **0** 个，重复特征行 **0** 行。

真实的成因只有一条：**度量尺度不同**。

## 4. 0.3662 应该怎么引用

**不引用，或只作为"尺度差异"的例子引用。** 理由：

- 它与全项目其余 silhouette（`clustering_robustness/` 的 0.2198）不同尺度，并排会被误读；
- 它的 K≥4 行由单次初始化 + `maxIter=20` 得到，未收敛（例：K=4 得 0.2713，
  同工程不同初始化的 `k_sweep.csv` 给 0.1872，本目录重跑给 0.3028——三者都受初始化影响，
  只有 10 种子 × 50 重启的 `clustering_robustness/` 收敛到稳定值）。

正式引用一律用 `data/output/clustering_robustness/`：**K=2，silhouette 0.2198，10/10 种子，
bootstrap ARI 0.9593 ± 0.0283（袋外计分）**。

## 5. 文件清单

| 文件 | 内容 |
|---|---|
| `run_k_selection_probe.py` | 探测脚本。沿用 `03` 的配置，输出写在本目录内，约 40 秒跑完 |
| `k_probe.csv` | §2 表格的原始数据（列：`k,wssse,spark_squaredEuclidean,sklearn_euclidean,n_clusters_found,cluster_sizes`） |
| `probe_summary.json` | 输入路径、行数、两套度量的 argmax 判定、特征范数审计 |
| `audit_07b_k_rows.py` | §8 的审计脚本：同配置重复 6 次，检验 07b 那 7 行能否被复现 |
| `k_spread_07b_audit.csv` | 42 次拟合明细（`k,rep,wssse,silhouette,sizes`） |
| `k_spread_07b_audit_summary.csv` | 逐 K 的 min/max/spread，以及 07b 记录值是否落在区间内 |
| `k_spread_07b_audit.txt` | 上述审计的完整运行日志与 verdict |
| `_run_log.txt` | 审计脚本的 stdout 重定向（Spark 收尾信息在里面） |
| `README.md` | 本文件 |

## 6. 重跑

```
D:/Anaconda/python.exe project/data/output/k_selection_diagnostics/run_k_selection_probe.py
D:/Anaconda/python.exe project/data/output/k_selection_diagnostics/audit_07b_k_rows.py
```

需要 `pyspark` + Java 17。两个脚本都只读 `if_results_rowkeyed/`，只写本目录。
前者约 40 秒，后者约 4 分钟（7 个 K × 6 次重复）。

## 7. 顺带发现（与 K 选择无关，但影响引用）本目录重跑得到的 K=2 簇大小是 **9,735 / 3,842**，而现役产物是 **9,791 / 3,786**，
差 56 行。这说明 **`03` 的单次初始化结果本身不可精确重现**——K 相同，但边界点会漂，
因此：

- 固定 K=2 能管住"选几"，但管不住"边界点在哪一行"；
- 引用簇规模时用现役产物（`anomaly_features/cluster_sizes_named.csv`：`short_high_tip` 9,791 /
  `long_low_tip` 3,786），不要用本目录重跑的簇大小。

现役的 `anomaly_clusters_rowkeyed/strict_anomalies_clustered/` 与
`clustering_robustness/row_uid_cluster_robust.parquet` 的簇划分是**逐行相同**的
（9 列特征指纹连接，13,577/13,577 全匹配，ARI = 1.000），所以现役那两份可以放心混用。

---

## 8. `selfcheck_k4/k_selection_summary.csv` 为什么对不上（2026-11-22 审计）

07b 的自检表（`anomaly_clusters_rowkeyed/selfcheck_k4/k_selection_summary.csv`，写于
2026-11-22 00:53:55）记录 K=4 得分最高，这是仓库里"重跑会选 K=4"说法的唯一来源。
本目录的 `audit_07b_k_rows.py` 用同输入、同参数（`seed=42 / maxIter=20 / tol=1e-4 /
squaredEuclidean`）**重复 6 次**，结论：

**① Spark 在这里是确定性的。** K=2..8 每个 K 的 6 次重复 WSSSE 与 silhouette
**完全相同**（spread = 0.0）。所以"初始化运气"不能解释任何差异。

**② 但 07b 表里的 7 行与重跑的 7 行全部对不上。**

| K | 07b 记录 | 6 次重跑（完全相同） |
|---|---|---|
| 2 | 102,254.7 | 98,241.7 |
| 3 | 83,681.9 | 83,693.9 |
| 4 | 75,427.0 | 75,717.2 |
| 5 | 71,570.2 | 68,572.9 |
| 6 | 63,977.5 | 64,285.2 |
| 7 | 61,263.3 | 61,485.7 |
| 8 | 58,756.9 | 59,688.9 |

**③ 07b 读的不是旧表。** 它的产物 mtime 是 00:53:55–00:54:07，而 02b 写出
`if_results_rowkeyed/` 是 00:52:35、日志收尾 00:52:48 —— 相隔 1 分钟，读的就是新表。

**④ 运行时的代码不是仓库现在这一份。** `selfcheck_k4/strict_anomalies_clustered_rowkeyed/`
带 `cluster_name` 列，但 4 个簇里只有 2 个有名字；而仓库的 `07b` 在
`07b_shap_cluster_analysis_rowkeyed.py:185` 只输出 `row_uid` + `cluster`，**不写 `cluster_name`**。

结论：那次运行为**一版早期 07 脚本**的产物，作者确认该版本已从仓库删除，无法重建。
它的 K 扫描数值**不可引用**；详见 `anomaly_clusters_rowkeyed/selfcheck_k4/README.md`。

附带影响：当前代码重跑 07b 会得到 **K=2 最高（0.3641）、K=4 仅 0.3028（第三）**，
即"会选 K=4"在当前代码下**复现不出来**。

审计原始输出：`k_spread_07b_audit.csv`（42 次拟合明细）、`k_spread_07b_audit_summary.csv`
（逐 K 的 min/max/spread/是否覆盖记录值）、`k_spread_07b_audit.txt`（完整运行日志）。
