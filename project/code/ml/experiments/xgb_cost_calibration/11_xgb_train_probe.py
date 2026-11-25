# Location: project/code/ml/experiments/xgb_cost_calibration/11_xgb_train_probe.py | PROJECT_ROOT depth: 4
"""Probe B: single-node XGBoost throughput on the L1 sample (driver side, no Spark worker).

Reads ``data/output/xgb_cost_calibration/sample5pct`` (produced by 10_scan_and_sample.py)
directly through PyArrow, builds a ``QuantileDMatrix`` and measures:

  1. load time and resident matrix size,
  2. ``s/round`` at ``max_depth=6`` and ``max_depth=8`` on the 90/10 time-agnostic split,
  3. ``nthread`` scaling (4 / 8 / 20) on a row slice.

Writes ``metrics/xgb_train_probe.json``. This is the route recommended in
the report's future-work section (P2) — it needs no Python worker, so it is not
affected by the PySpark/Windows daemon limitation documented there.

Usage:
    python project/code/ml/experiments/xgb_cost_calibration/11_xgb_train_probe.py
    ... --rows 5000000 --thread-axis 4,8,20 --rounds 40
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from calibration_common import (  # noqa: E402
    FEATURES, LABEL, SAMPLE_DIR, env_meta, open_tee, write_metrics,
)

BASE_PARAMS = dict(
    objective="reg:squarederror",
    tree_method="hist",
    eta=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=20,
    reg_lambda=2.0,
    max_bin=128,
)


def rmse(pred, truth) -> float:
    import numpy as np

    return float(np.sqrt(np.mean((pred - truth) ** 2)))


def load_sample(rows: int | None):
    """Arrow -> float32 matrix; ``rows`` limits how many rows are read."""
    import numpy as np
    import pyarrow.dataset as ds

    if not SAMPLE_DIR.is_dir():
        raise FileNotFoundError(
            f"{SAMPLE_DIR} missing — run 10_scan_and_sample.py first")
    table = ds.dataset(str(SAMPLE_DIR)).to_table(columns=FEATURES + [LABEL])
    if rows:
        table = table.slice(0, rows)
    n = table.num_rows
    matrix = np.empty((n, len(FEATURES)), dtype=np.float32)
    for i, column in enumerate(FEATURES):
        matrix[:, i] = table.column(column).to_numpy(zero_copy_only=False).astype(
            np.float32, copy=False)
    target = table.column(LABEL).to_numpy(zero_copy_only=False).astype(
        np.float32, copy=False)
    return matrix, target


def main(argv: list[str] | None = None) -> int:
    import time

    import xgboost as xgb

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=0,
                        help="rows to read; 0 = whole sample (default)")
    parser.add_argument("--threads", type=int, default=20, help="nthread for the main fits")
    parser.add_argument("--rounds", type=int, default=40, help="boosting rounds at depth 6")
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--second-depth", type=int, default=8)
    parser.add_argument("--second-rounds", type=int, default=15)
    parser.add_argument("--thread-axis", default="4,8,20",
                        help="comma separated nthread values for the scaling probe")
    parser.add_argument("--thread-axis-rows", type=int, default=5_000_000)
    parser.add_argument("--thread-axis-rounds", type=int, default=10)
    args = parser.parse_args(argv)

    log = open_tee("11_xgb_train_probe")
    result: dict = {"stage": "xgb_train_probe"}

    start = time.perf_counter()
    X, y = load_sample(args.rows or None)
    result["load_s"] = round(time.perf_counter() - start, 1)
    n, p = X.shape
    result.update(n_rows=n, n_features=p,
                  matrix_gb=round(X.nbytes / 1e9, 2))
    log(f"loaded {n:,} rows x {p} features in {result['load_s']} s "
        f"({result['matrix_gb']} GB float32)")

    cut = int(n * 0.9)
    Xtr, ytr, Xva, yva = X[:cut], y[:cut], X[cut:], y[cut:]
    result.update(n_train=int(cut), n_valid=int(n - cut))

    start = time.perf_counter()
    dtr = xgb.QuantileDMatrix(Xtr, label=ytr, max_bin=BASE_PARAMS["max_bin"])
    dva = xgb.QuantileDMatrix(Xva, label=yva, ref=dtr, max_bin=BASE_PARAMS["max_bin"])
    result["dmatrix_build_s"] = round(time.perf_counter() - start, 1)
    log(f"QuantileDMatrix built in {result['dmatrix_build_s']} s")

    params = dict(BASE_PARAMS, max_depth=args.depth, nthread=args.threads)
    start = time.perf_counter()
    booster = xgb.train(params, dtr, num_boost_round=args.rounds,
                        evals=[(dva, "valid")], verbose_eval=False)
    train_s = time.perf_counter() - start
    result[f"depth{args.depth}"] = {
        "rounds": args.rounds,
        "train_s": round(train_s, 1),
        "s_per_round": round(train_s / args.rounds, 3),
        "valid_rmse": rmse(booster.predict(dva), yva),
        "nthread": args.threads,
    }
    log(f"depth{args.depth}: {args.rounds} rounds in {train_s:.1f} s "
        f"({train_s / args.rounds:.3f} s/round), valid RMSE "
        f"{result[f'depth{args.depth}']['valid_rmse']:.4f}")

    if args.second_depth:
        deep = dict(BASE_PARAMS, max_depth=args.second_depth,
                    min_child_weight=50, nthread=args.threads)
        start = time.perf_counter()
        xgb.train(deep, dtr, num_boost_round=args.second_rounds,
                  evals=[(dva, "valid")], verbose_eval=False)
        deep_s = time.perf_counter() - start
        result[f"depth{args.second_depth}"] = {
            "rounds": args.second_rounds,
            "train_s": round(deep_s, 1),
            "s_per_round": round(deep_s / args.second_rounds, 3),
            "nthread": args.threads,
        }
        log(f"depth{args.second_depth}: {args.second_rounds} rounds in {deep_s:.1f} s "
            f"({deep_s / args.second_rounds:.3f} s/round)")

    axis = [int(v) for v in args.thread_axis.split(",") if v.strip()]
    slice_rows = min(args.thread_axis_rows, n)
    dslice = xgb.QuantileDMatrix(X[:slice_rows], label=y[:slice_rows],
                                 max_bin=BASE_PARAMS["max_bin"])
    scaling: dict[str, dict] = {}
    for threads in axis:
        tuned = dict(BASE_PARAMS, max_depth=args.depth, nthread=threads)
        start = time.perf_counter()
        xgb.train(tuned, dslice, num_boost_round=args.thread_axis_rounds, verbose_eval=False)
        seconds = time.perf_counter() - start
        scaling[str(threads)] = {
            "rounds": args.thread_axis_rounds,
            "train_s": round(seconds, 2),
            "s_per_round": round(seconds / args.thread_axis_rounds, 3),
        }
        log(f"nthread={threads}: {args.thread_axis_rounds} rounds on "
            f"{slice_rows:,} rows in {seconds:.2f} s ({seconds / args.thread_axis_rounds:.3f} s/round)")
    result["thread_scaling"] = {
        "rows": slice_rows,
        "rounds": args.thread_axis_rounds,
        "by_threads": scaling,
    }
    baseline = min(scaling.values(), key=lambda v: v["s_per_round"]) if scaling else None
    first = scaling.get(str(min(axis)), None) if axis else None
    if first and baseline:
        result["thread_scaling"]["speedup_vs_min_threads"] = round(
            first["s_per_round"] / baseline["s_per_round"], 2)

    result["env"] = env_meta()
    log(f"metrics -> {write_metrics('xgb_train_probe', result)}")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
