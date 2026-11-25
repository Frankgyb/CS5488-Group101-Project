# Location: project/code/ml/experiments/parallel_scaling/bootstrap_ci.py | PROJECT_ROOT depth: 4
"""Bootstrap 95% CIs for the v3 scaling medians.

Reads the persisted run-level CSVs under ``data/output/scaling_study`` and
writes ``bootstrap_ci.json``. No Spark session is started.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scaling_common import OUT_ROOT, resolve_path

RNG = np.random.default_rng(20261125)
B = 20000


def ci_median(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    med = float(np.median(values))
    draws = np.array([np.median(RNG.choice(values, size=len(values), replace=True)) for _ in range(B)])
    return med, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def ci_ratio(base: np.ndarray, cell: np.ndarray) -> tuple[float, float, float]:
    base = np.asarray(base, dtype=float)
    cell = np.asarray(cell, dtype=float)
    point = float(np.median(base) / np.median(cell))
    draws = []
    for _ in range(B):
        b = np.median(RNG.choice(base, size=len(base), replace=True))
        c = np.median(RNG.choice(cell, size=len(cell), replace=True))
        draws.append(b / c)
    return point, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def s2(job: str, in_dir: Path) -> dict:
    df = pd.read_csv(in_dir / f"s2_{job}.csv")
    df = df[~df["is_warmup"].astype(bool)]
    out = {"job": job, "cells": [], "speedups": []}
    for shuffle in sorted(df["shuffle_partitions"].unique()):
        sub = df[df["shuffle_partitions"] == shuffle]
        base = sub[sub["n_cores"] == 2]["fit_s"].to_numpy()
        for cores in sorted(sub["n_cores"].unique()):
            vals = sub[sub["n_cores"] == cores]["fit_s"].to_numpy()
            med, lo, hi = ci_median(vals)
            out["cells"].append({
                "shuffle": int(shuffle), "cores": int(cores), "n": int(len(vals)),
                "median": round(med, 3), "ci95": [round(lo, 3), round(hi, 3)],
            })
            if cores != 2:
                point, lo2, hi2 = ci_ratio(base, vals)
                out["speedups"].append({
                    "shuffle": int(shuffle), "cores": int(cores),
                    "speedup_vs_2": round(point, 3),
                    "ci95": [round(lo2, 3), round(hi2, 3)],
                })
    return out


def s3(in_dir: Path) -> dict:
    df = pd.read_csv(in_dir / "s3_f1.csv")
    df = df[~df["is_warmup"].astype(bool)]
    base = df[df["n_jobs_req"] == 1]["score_s"].to_numpy()
    out = {"cells": [], "speedups": []}
    for req in sorted(df["n_jobs_req"].unique()):
        vals = df[df["n_jobs_req"] == req]["score_s"].to_numpy()
        med, lo, hi = ci_median(vals)
        out["cells"].append({
            "n_jobs_req": int(req), "n": int(len(vals)),
            "median": round(med, 4), "ci95": [round(lo, 4), round(hi, 4)],
        })
        if req != 1:
            point, lo2, hi2 = ci_ratio(base, vals)
            out["speedups"].append({
                "n_jobs_req": int(req), "speedup_vs_1": round(point, 3),
                "ci95": [round(lo2, 3), round(hi2, 3)],
            })
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default=str(OUT_ROOT))
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    in_dir = resolve_path(args.in_dir)
    out_path = resolve_path(args.out) if args.out else in_dir / "bootstrap_ci.json"
    result = {"k1": s2("k1", in_dir), "k2": s2("k2", in_dir),
              "s3": s3(in_dir)}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    out_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
