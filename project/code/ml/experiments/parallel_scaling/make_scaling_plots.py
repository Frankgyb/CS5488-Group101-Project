# Location: project/code/ml/experiments/parallel_scaling/make_scaling_plots.py | PROJECT_ROOT depth: 4
"""Aggregate the scaling-study CSVs into figures, SUMMARY.md and checks.json.

Reads/writes only ``data/output/scaling_study``. This script does not start
Spark and can be used to regenerate the figures from the persisted raw CSVs.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scaling_common import (
    OUT_ROOT, atomic_write_json, diff_snapshots, ensure_dir, resolve_path,
)

COLORS = {30: "#1f77b4", 120: "#d62728"}


def md_table(df: pd.DataFrame, cols: list[str] | None = None) -> str:
    if df is None or df.empty:
        return "_(empty)_"
    d = df[cols] if cols else df
    headers = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in d.iterrows():
        cells = []
        for v in row:
            if isinstance(v, float):
                cells.append("" if math.isnan(v) else f"{v:.3f}")
            elif isinstance(v, (np.floating,)):
                cells.append(f"{float(v):.3f}")
            elif isinstance(v, (np.integer,)):
                cells.append(str(int(v)))
            elif isinstance(v, bool):
                cells.append("yes" if v else "no")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def s2_panel(ax, summary: pd.DataFrame, job: str, metric: str, baseline: int) -> None:
    sub = summary[summary["job"] == job] if not summary.empty else pd.DataFrame()
    if sub.empty:
        ax.text(0.5, 0.5, f"{job}: no data", ha="center", va="center")
        ax.set_axis_off()
        return
    for shuffle in sorted(sub["shuffle_partitions"].unique()):
        g = sub[sub["shuffle_partitions"] == shuffle].sort_values("n_cores")
        if metric == "speedup":
            base_rows = g[g["n_cores"] == baseline]["fit_s_median"]
            if not len(base_rows):
                continue
            base = float(base_rows.iloc[0])
            med = base / g["fit_s_median"].to_numpy(dtype=float)
            low = base / g["fit_s_max"].to_numpy(dtype=float)
            high = base / g["fit_s_min"].to_numpy(dtype=float)
            yerr = np.vstack([med - low, high - med])
            ax.errorbar(g["n_cores"], med, yerr=yerr, marker="o", capsize=3,
                        color=COLORS.get(int(shuffle), None),
                        label=f"S={int(shuffle)}")
        else:
            ax.errorbar(g["n_cores"], g["parallel_eff_median"],
                        yerr=[
                            g["parallel_eff_median"] - g["parallel_eff_net_median"],
                            np.zeros(len(g)),
                        ],
                        marker="o", capsize=3,
                        color=COLORS.get(int(shuffle), None),
                        label=f"S={int(shuffle)}")
    if metric == "speedup":
        xs = np.array(sorted(sub["n_cores"].unique()), dtype=float)
        ax.plot(xs, xs / baseline, "k--", linewidth=1, label="ideal (linear)")
        ax.set_ylabel("speedup vs local[%d]" % baseline)
    else:
        ax.axhline(1.0, color="k", linestyle="--", linewidth=1)
        ax.set_ylabel("parallel_eff (CPU busy / requested threads)")
    ax.set_xlabel("local[N] threads")
    ax.set_xticks(sorted(sub["n_cores"].unique()))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    ax.set_title(job.upper())


def plot_s2(summary: pd.DataFrame, out_dir: Path) -> None:
    for metric, fname in [("speedup", "s2_speedup.png"),
                          ("parallel_eff", "s2_parallel_eff.png")]:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        s2_panel(axes[0], summary, "k1", metric, 2)
        s2_panel(axes[1], summary, "k2", metric, 2)
        fig.suptitle("S2: KMeans(k=2) vs BisectingKMeans(k=8), 2M rows")
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=160)
        plt.close(fig)


def plot_s3(summary: pd.DataFrame, out_dir: Path) -> None:
    if summary.empty:
        return
    s = summary.copy()
    s["x"] = s["n_jobs_eff"].astype(float)
    prod = s[s["n_jobs_req"] == -1]
    curve = s[s["n_jobs_req"] != -1].sort_values("x")
    for metric, fname in [("speedup", "s3_speedup.png"),
                          ("parallel_eff", "s3_parallel_eff.png")]:
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        if metric == "speedup":
            ax.plot(curve["x"], curve["speedup_vs_1"], marker="o",
                    color="#1f77b4", label="measured")
            xs = np.array([1, 2, 4, 8], dtype=float)
            ax.plot(xs, xs, "k--", linewidth=1, label="ideal (linear)")
            if len(prod):
                ax.scatter(prod["x"], prod["speedup_vs_1"], marker="*", s=140,
                           color="#d62728",
                           label="n_jobs=-1 (production, 20 threads)")
            ax.set_ylabel("score speedup vs n_jobs=1")
        else:
            ax.plot(curve["x"], curve["parallel_eff_median"], marker="o",
                    color="#1f77b4", label="measured")
            ax.axhline(1.0, color="k", linestyle="--", linewidth=1)
            if len(prod):
                ax.scatter(prod["x"], prod["parallel_eff_median"], marker="*", s=140,
                           color="#d62728", label="n_jobs=-1")
            ax.set_ylabel("parallel_eff")
            ax.set_ylim(0, max(1.2, float(s["parallel_eff_median"].max()) * 1.1))
        ax.set_xlabel("n_jobs (effective threads)")
        ax.set_xticks(sorted(s["x"].unique()))
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        ax.set_title("S3: IsolationForest.decision_function, 1M rows")
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=160)
        plt.close(fig)


def build_checks(k1: pd.DataFrame, k2: pd.DataFrame, f1: pd.DataFrame,
                 s1: pd.DataFrame, s2: pd.DataFrame, s3: pd.DataFrame,
                 protected: dict | None) -> dict:
    checks: dict = {}
    checks["k1_present"] = not k1.empty
    checks["k2_present"] = not k2.empty
    checks["s3_present"] = not f1.empty
    checks["k1_n_rows_unique"] = sorted(k1["n_rows"].unique().tolist()) if not k1.empty else []
    checks["k2_n_rows_unique"] = sorted(k2["n_rows"].unique().tolist()) if not k2.empty else []
    checks["s3_n_rows_unique"] = sorted(f1["n_rows"].unique().tolist()) if not f1.empty else []
    checks["k1_wssse_unique"] = sorted(k1["wssse"].round(3).unique().tolist()) if not k1.empty else []
    checks["k2_wssse_unique"] = sorted(k2["wssse"].round(3).unique().tolist()) if not k2.empty else []
    checks["k1_n_iter_unique"] = sorted(k1["n_iter"].astype(int).unique().tolist()) if not k1.empty else []
    checks["k2_n_iter_unique"] = sorted(k2["n_iter"].astype(int).unique().tolist()) if not k2.empty else []
    checks["k1_grids"] = int(s1[["n_cores", "shuffle_partitions"]].drop_duplicates().shape[0]) if not s1.empty else 0
    checks["k2_grids"] = int(s2[["n_cores", "shuffle_partitions"]].drop_duplicates().shape[0]) if not s2.empty else 0
    k1_main = s1[s1["n_cores"] != 1] if not s1.empty else s1
    k1_local1 = s1[s1["n_cores"] == 1] if not s1.empty else s1
    checks["k1_min_runs_per_grid"] = int(k1_main["n_runs"].min()) if not k1_main.empty else 0
    checks["k1_local1_runs"] = int(k1_local1["n_runs"].min()) if not k1_local1.empty else 0
    checks["k2_min_runs_per_grid"] = int(s2["n_runs"].min()) if not s2.empty else 0
    checks["s3_min_runs_per_job"] = int(s3["n_runs"].min()) if not s3.empty else 0
    checks["s3_all_score_equal"] = bool(s3["valid"].all()) if not s3.empty else False
    checks["protected"] = protected
    checks["protected_ok"] = (
        True if protected is None else
        (len(protected.get("added", [])) == 0
         and len(protected.get("removed", [])) == 0
         and len(protected.get("changed", [])) == 0)
    )
    checks["pass"] = bool(
        checks["k1_present"] and checks["k2_present"] and checks["s3_present"]
        and checks["k1_n_rows_unique"] == [2_000_000]
        and checks["k2_n_rows_unique"] == [2_000_000]
        and checks["s3_n_rows_unique"] == [1_000_000]
        and len(checks["k1_wssse_unique"]) == 1
        and len(checks["k2_wssse_unique"]) == 1
        and len(checks["k1_n_iter_unique"]) == 1
        and len(checks["k2_n_iter_unique"]) == 1
        and checks["k1_grids"] in (6, 7) and checks["k2_grids"] == 6
        and checks["k1_min_runs_per_grid"] >= 4
        and (checks["k1_local1_runs"] == 0 or checks["k1_local1_runs"] >= 2)
        and checks["k2_min_runs_per_grid"] >= 4
        and checks["s3_min_runs_per_job"] >= 4
        and checks["s3_all_score_equal"]
    )
    return checks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default=str(OUT_ROOT))
    ap.add_argument("--out-dir", default="")
    args = ap.parse_args()
    in_dir = resolve_path(args.in_dir)
    out_dir = resolve_path(args.out_dir) if args.out_dir else in_dir
    ensure_dir(out_dir)
    fig_dir = ensure_dir(out_dir / "figures")

    k1 = read_csv(in_dir / "s2_k1.csv")
    k2 = read_csv(in_dir / "s2_k2.csv")
    f1 = read_csv(in_dir / "s3_f1.csv")
    s1 = read_csv(in_dir / "s2_k1_summary.csv")
    s2 = read_csv(in_dir / "s2_k2_summary.csv")
    s3 = read_csv(in_dir / "s3_f1_summary.csv")

    if not s1.empty or not s2.empty:
        plot_s2(pd.concat([s1, s2], ignore_index=True), fig_dir)
    if not s3.empty:
        plot_s3(s3, fig_dir)

    before_path = in_dir / "protected_before.json"
    after_path = in_dir / "protected_after.json"
    protected = None
    if before_path.exists() and after_path.exists():
        try:
            before = json.loads(before_path.read_text(encoding="utf-8"))
            after = json.loads(after_path.read_text(encoding="utf-8"))
            protected = diff_snapshots(before, after)
        except Exception as exc:  # noqa: BLE001
            protected = {"error": str(exc)}

    checks = build_checks(k1, k2, f1, s1, s2, s3, protected)
    atomic_write_json(out_dir / "checks.json", checks)

    env = {}
    env_path = in_dir / "env.json"
    if env_path.exists():
        try:
            env = json.loads(env_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            env = {}

    lines = ["# Scaling study v3 — automatic summary", ""]
    lines.append(f"Generated: {pd.Timestamp.now().isoformat(timespec='seconds')}")
    lines.append("")
    if env:
        lines.append("## Environment")
        for key in ["python", "pyspark", "psutil", "sklearn", "cpu_model",
                    "physical_cores", "logical_cores", "ram_gb",
                    "power_scheme", "battery_plugged", "temp_free_gb",
                    "input_path", "input_parts", "baseline_cpu_pct"]:
            if key in env:
                lines.append(f"- {key}: `{env[key]}`")
        lines.append("")
    lines.append("## S2 K1: KMeans(k=2), 2M rows")
    lines.append(md_table(s1, [
        "n_cores", "shuffle_partitions", "n_runs", "fit_s_median",
        "fit_s_mad", "speedup_vs_baseline", "efficiency",
        "parallel_eff_median", "amdahl_alpha", "amdahl_r2",
    ]))
    lines.append("")
    lines.append("## S2 K2: BisectingKMeans(k=8), 2M rows")
    lines.append(md_table(s2, [
        "n_cores", "shuffle_partitions", "n_runs", "fit_s_median",
        "fit_s_mad", "speedup_vs_baseline", "efficiency",
        "parallel_eff_median", "amdahl_alpha", "amdahl_r2",
    ]))
    lines.append("")
    lines.append("## S3 F1: IsolationForest.decision_function, 1M rows")
    lines.append(md_table(s3, [
        "n_jobs_req", "n_jobs_eff", "n_runs", "score_s_median",
        "score_s_mad", "speedup_vs_1", "efficiency",
        "parallel_eff_median", "max_abs_diff_max", "threads_observed_median",
    ]))
    lines.append("")
    lines.append("## Automatic checks")
    lines.append(f"- pass: **{checks['pass']}**")
    for key, value in checks.items():
        if key != "pass":
            lines.append(f"- {key}: `{value}`")
    lines.append("")
    lines.append("_Generated by make_scaling_plots.py; no project files were modified._")
    (out_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[done] wrote {out_dir / 'SUMMARY.md'} and {out_dir / 'checks.json'}", flush=True)
    print(json.dumps(checks, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
