#!/usr/bin/env python3
"""Analyze the post-hoc replication of the K2 S=30 local[8] cell.

Why this script exists
----------------------
The original K2 n=4 run had one high-leverage timing point (143.793 s) in the
S=30 local[8] cell.  We re-ran *only that cell* with the same protocol
(1 warm-up + 4 measured runs, same 2M subset, same KMeans configuration) to
see whether the slow run was a one-off environment event.  The primary n=4
result is deliberately kept unchanged; this script reports:

* primary n=4
* replication n=4
* pooled n=8 as an explicitly unbalanced sensitivity analysis

It does not overwrite ``s2_k2.csv`` or the primary summaries.  No Spark is
started here; everything is read from the persisted JSONL/CSV files.

Usage
-----
    python project/code/ml/experiments/parallel_scaling/analyze_k2_replication.py
    python .../analyze_k2_replication.py --study-dir ... --replication-dir ... \
        --out-dir ...

Outputs (default under data/output/scaling_study/replication/analysis/):
    k2_s30_c8_runs.csv
    k2_s30_c8_summary.csv
    k2_s30_c8_summary.json
    k2_s30_c8_replication.md
    figures/k2_s30_c8_replication.png
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[3]
DEFAULT_STUDY = PROJECT_ROOT / "data" / "output" / "scaling_study"
DEFAULT_REPLICATION = DEFAULT_STUDY / "replication" / "k2_s30_c8"
DEFAULT_OUT = DEFAULT_STUDY / "replication" / "analysis"

SEED = 42
N_BOOT = 20_000
JOB = "k2"
CORES = 8
SHUFFLE = 30
BASELINE_CORE = 2


def resolve_path(value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def cell_rows(rows: list[dict], batch: str) -> list[dict]:
    out: list[dict] = []
    for r in rows:
        if r.get("job") != JOB:
            continue
        if int(r.get("n_cores", -1)) != CORES:
            continue
        if int(r.get("shuffle_partitions", -1)) != SHUFFLE:
            continue
        if bool(r.get("is_warmup")):
            continue
        rec = dict(r)
        rec["batch"] = batch
        out.append(rec)
    out.sort(key=lambda x: str(x.get("started_at", "")))
    return out


def primary_baseline_values(study_dir: Path) -> list[float]:
    df = pd.read_csv(study_dir / "s2_k2.csv")
    mask = (
        (df["job"] == JOB)
        & (df["n_cores"] == BASELINE_CORE)
        & (df["shuffle_partitions"] == SHUFFLE)
        & (~df["is_warmup"].astype(bool))
    )
    vals = df.loc[mask, "fit_s"].astype(float)
    if vals.empty:
        raise SystemExit(f"no primary baseline rows in {study_dir / 's2_k2.csv'}")
    return [float(v) for v in vals]


def primary_baseline(study_dir: Path) -> float:
    return float(np.median(primary_baseline_values(study_dir)))


def summarize_values(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    med = float(np.median(arr))
    return {
        "n": int(arr.size),
        "median": round(med, 3),
        "mean": round(float(np.mean(arr)), 3),
        "min": round(float(np.min(arr)), 3),
        "max": round(float(np.max(arr)), 3),
        "mad": round(float(np.median(np.abs(arr - med))), 3),
        "max_over_median": round(float(np.max(arr) / med), 3) if med > 0 else float("nan"),
        "values": [round(float(v), 3) for v in arr],
    }


def median_ci(values: list[float], n_boot: int = N_BOOT, seed: int = SEED) -> list[float]:
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boots = np.median(arr[idx], axis=1)
    return [round(float(np.percentile(boots, 2.5)), 3), round(float(np.percentile(boots, 97.5)), 3)]


def ratio_ci(num: list[float], den: list[float], n_boot: int = N_BOOT, seed: int = SEED) -> list[float]:
    n = np.asarray(num, dtype=float)
    d = np.asarray(den, dtype=float)
    rng = np.random.default_rng(seed)
    idx_n = rng.integers(0, n.size, size=(n_boot, n.size))
    idx_d = rng.integers(0, d.size, size=(n_boot, d.size))
    boots = np.median(n[idx_n], axis=1) / np.median(d[idx_d], axis=1)
    return [round(float(np.percentile(boots, 2.5)), 3), round(float(np.percentile(boots, 97.5)), 3)]


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if v is None or (isinstance(v, float) and math.isnan(v)):
                vals.append("—")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def make_figure(primary: list[float], replication: list[float], pooled: list[float],
                baseline: float, summary: dict, out_dir: Path) -> Path:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / "k2_s30_c8_replication.png"

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ax = axes[0]
    groups = [("primary", primary), ("replication", replication), ("pooled", pooled)]
    rng = np.random.default_rng(SEED)
    for i, (label, vals) in enumerate(groups):
        x = np.full(len(vals), i, dtype=float) + rng.uniform(-0.08, 0.08, len(vals))
        ax.scatter(x, vals, s=38, alpha=0.85, label=f"{label} (n={len(vals)})")
        med = float(np.median(vals))
        ax.plot([i - 0.22, i + 0.22], [med, med], color="black", lw=2)
        ax.text(i + 0.24, med, f"{med:.1f}s", va="center", fontsize=9)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylabel("fit_s (s)")
    ax.set_title("K2 S=30 local[8]: primary vs replication")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    rows = [
        ("primary", summary["primary"]["speedup_vs_primary_local2"], summary["primary"]["speedup_ci95"]),
        ("pooled\n(sensitivity)", summary["pooled_sensitivity"]["speedup_vs_primary_local2"], summary["pooled_sensitivity"]["speedup_ci95"]),
    ]
    for i, (label, val, ci) in enumerate(rows):
        ax.errorbar(i, val, yerr=[[val - ci[0]], [ci[1] - val]], fmt="o", capsize=6, ms=8)
        ax.text(i + 0.08, val, f"{val:.2f}×\n[{ci[0]:.2f},{ci[1]:.2f}]", va="center", fontsize=9)
    ax.axhline(1.0, color="grey", lw=1, ls="--")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([r[0] for r in rows])
    ax.set_ylabel(f"speed-up vs primary local[{BASELINE_CORE}] (×)")
    ax.set_title("Replication does not replace the primary n=4 result")
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--study-dir", default=str(DEFAULT_STUDY))
    ap.add_argument("--replication-dir", default=str(DEFAULT_REPLICATION))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    study_dir = resolve_path(args.study_dir)
    repl_dir = resolve_path(args.replication_dir)
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    primary_jsonl = study_dir / "s2_k2_runs.jsonl"
    repl_jsonl = repl_dir / "s2_k2_runs.jsonl"
    for path in (primary_jsonl, repl_jsonl):
        if not path.exists():
            raise SystemExit(f"missing JSONL: {path}")

    primary = cell_rows(read_jsonl(primary_jsonl), "primary")
    replication = cell_rows(read_jsonl(repl_jsonl), "replication")
    if not primary:
        raise SystemExit("no primary measured rows for K2 S=30 local[8]")
    if not replication:
        raise SystemExit("no replication measured rows for K2 S=30 local[8]")

    p_vals = [float(r["fit_s"]) for r in primary]
    r_vals = [float(r["fit_s"]) for r in replication]
    pooled = p_vals + r_vals
    baseline_values = primary_baseline_values(study_dir)
    baseline = float(np.median(baseline_values))

    p_stats = summarize_values(p_vals)
    r_stats = summarize_values(r_vals)
    pooled_stats = summarize_values(pooled)

    p_ci = median_ci(p_vals, args.n_boot, args.seed)
    r_ci = median_ci(r_vals, args.n_boot, args.seed + 1)
    pooled_ci = median_ci(pooled, args.n_boot, args.seed + 2)

    p_speed = baseline / p_stats["median"]
    pooled_speed = baseline / pooled_stats["median"]
    # Resample the primary local[2] baseline values as well, matching
    # bootstrap_ci.py; fixing the baseline would understate the interval.
    p_speed_ci = ratio_ci(baseline_values, p_vals, args.n_boot, args.seed + 3)
    pooled_speed_ci = ratio_ci(baseline_values, pooled, args.n_boot, args.seed + 4)
    repl_vs_primary = r_stats["median"] / p_stats["median"]

    summary = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "job": JOB,
        "cell": {"n_cores": CORES, "shuffle_partitions": SHUFFLE, "n_rows": 2_000_000},
        "primary_local2_median_s": round(float(baseline), 3),
        "speedup_ci_note": (
            "Speed-up CIs resample the primary local[2] baseline values as well as "
            "the 8-core values, matching bootstrap_ci.py; they are not fixed-baseline "
            "conditional intervals."
        ),
        "note": (
            "Post-hoc replication of one cell after observing a 143.793 s high-leverage "
            "point. The primary n=4 result is not overwritten. The replication has no "
            "contemporaneous local[2] baseline, so no replication-only speed-up is "
            "claimed. Pooled n=8 is an unbalanced sensitivity analysis: only this cell "
            "has 8 measured runs; the other K2 cells remain n=4."
        ),
        "primary": {
            **p_stats,
            "median_ci95": p_ci,
            "speedup_vs_primary_local2": round(float(p_speed), 3),
            "speedup_ci95": p_speed_ci,
        },
        "replication": {
            **r_stats,
            "median_ci95": r_ci,
            "median_ratio_vs_primary": round(float(repl_vs_primary), 3),
            "median_change_pct_vs_primary": round(float((repl_vs_primary - 1.0) * 100.0), 1),
        },
        "pooled_sensitivity": {
            **pooled_stats,
            "median_ci95": pooled_ci,
            "speedup_vs_primary_local2": round(float(pooled_speed), 3),
            "speedup_ci95": pooled_speed_ci,
            "unbalanced": True,
        },
        "interpretation": {
            "replication_reproduces_primary_slow_tail": bool(r_stats["max_over_median"] > 1.3),
            "replication_max_over_median": r_stats["max_over_median"],
            "primary_max_over_median": p_stats["max_over_median"],
        },
    }

    runs = pd.DataFrame(primary + replication)
    runs["batch"] = runs["batch"].astype(str)
    runs = runs[["batch", "run_idx", "fit_s", "started_at", "finished_at",
                 "cpu_pct_mean", "cpu_pct_baseline", "parallel_eff",
                 "wssse", "n_iter"]]
    runs.to_csv(out_dir / "k2_s30_c8_runs.csv", index=False, encoding="utf-8")

    summary_df = pd.DataFrame([
        {
            "dataset": "primary",
            "n": p_stats["n"],
            "fit_s_median": p_stats["median"],
            "ci95_low": p_ci[0],
            "ci95_high": p_ci[1],
            "fit_s_min": p_stats["min"],
            "fit_s_max": p_stats["max"],
            "fit_s_mad": p_stats["mad"],
            "speedup_vs_primary_local2": round(float(p_speed), 3),
            "speedup_ci95_low": p_speed_ci[0],
            "speedup_ci95_high": p_speed_ci[1],
            "max_over_median": p_stats["max_over_median"],
            "note": "original n=4; main result",
        },
        {
            "dataset": "replication",
            "n": r_stats["n"],
            "fit_s_median": r_stats["median"],
            "ci95_low": r_ci[0],
            "ci95_high": r_ci[1],
            "fit_s_min": r_stats["min"],
            "fit_s_max": r_stats["max"],
            "fit_s_mad": r_stats["mad"],
            "speedup_vs_primary_local2": None,
            "speedup_ci95_low": None,
            "speedup_ci95_high": None,
            "max_over_median": r_stats["max_over_median"],
            "note": "post-hoc; no contemporaneous local[2] baseline",
        },
        {
            "dataset": "pooled_sensitivity",
            "n": pooled_stats["n"],
            "fit_s_median": pooled_stats["median"],
            "ci95_low": pooled_ci[0],
            "ci95_high": pooled_ci[1],
            "fit_s_min": pooled_stats["min"],
            "fit_s_max": pooled_stats["max"],
            "fit_s_mad": pooled_stats["mad"],
            "speedup_vs_primary_local2": round(float(pooled_speed), 3),
            "speedup_ci95_low": pooled_speed_ci[0],
            "speedup_ci95_high": pooled_speed_ci[1],
            "max_over_median": pooled_stats["max_over_median"],
            "note": "unbalanced: only this cell has n=8",
        },
    ])
    summary_df.to_csv(out_dir / "k2_s30_c8_summary.csv", index=False, encoding="utf-8")
    with (out_dir / "k2_s30_c8_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    fig = make_figure(p_vals, r_vals, pooled, baseline, summary, out_dir)

    md = [
        "# K2 S=30 local[8] replication",
        "",
        f"- Generated: {summary['generated_at']}",
        "- Trigger: the primary n=4 run had one high-leverage 143.793 s point.",
        "- Primary is unchanged; this is a post-hoc replication, not a replacement.",
        "",
        "## Summary",
        "",
        md_table(summary_df),
        "",
        "## Raw measured runs",
        "",
        md_table(runs[["batch", "run_idx", "fit_s", "cpu_pct_mean", "wssse", "n_iter"]]),
        "",
        "## Reading",
        "",
        f"- primary: {p_stats['n']} runs, median {p_stats['median']:.3f} s, "
        f"bootstrap 95% CI [{p_ci[0]:.3f}, {p_ci[1]:.3f}].",
        f"- replication: {r_stats['n']} runs, median {r_stats['median']:.3f} s, "
        f"bootstrap 95% CI [{r_ci[0]:.3f}, {r_ci[1]:.3f}]; median ratio vs primary "
        f"{repl_vs_primary:.3f} ({summary['replication']['median_change_pct_vs_primary']:+.1f}%).",
        f"- pooled n=8 sensitivity: median {pooled_stats['median']:.3f} s, "
        f"CI [{pooled_ci[0]:.3f}, {pooled_ci[1]:.3f}]; speed-up vs primary local[2] "
        f"{pooled_speed:.3f}× [{pooled_speed_ci[0]:.3f}, {pooled_speed_ci[1]:.3f}].",
        "- Speed-up CIs resample the primary local[2] values as well as the 8-core "
        "values; they therefore match the method in `bootstrap_ci.json`, not a "
        "fixed-baseline conditional interval.",
        f"- replication max/median = {r_stats['max_over_median']:.3f}; "
        f"a new >1.3× slow tail is {'present' if r_stats['max_over_median'] > 1.3 else 'not present'}.",
        "",
        f"Figure: `{fig.relative_to(PROJECT_ROOT).as_posix()}`",
        "",
        "The pooled row is unbalanced and is not a new main result. The primary n=4 "
        "table remains the primary result; the replication answers only whether the "
        "143.8 s point recurs in the same cell.",
        "",
    ]
    (out_dir / "k2_s30_c8_replication.md").write_text("\n".join(md), encoding="utf-8")

    print(f"[done] wrote {out_dir}")
    print(summary_df.to_string(index=False))
    print(f"[note] primary median={p_stats['median']:.3f}s; replication median={r_stats['median']:.3f}s; "
          f"pooled={pooled_stats['median']:.3f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
