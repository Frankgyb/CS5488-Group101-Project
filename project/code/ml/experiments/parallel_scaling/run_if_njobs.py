# Location: project/code/ml/experiments/parallel_scaling/run_if_njobs.py | PROJECT_ROOT depth: 4
"""S3/F1: IsolationForest scoring throughput vs n_jobs on a fixed 1M subset.

The forest is fitted once (150 trees, max_samples=512, max_features=0.8,
seed=42, n_jobs=4 as in 02b), then only decision_function is rotated.
No Spark session is started here; the 1M subset is read with pandas.
"""
from __future__ import annotations

import argparse
import json
import math
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from scaling_common import (
    CpuSampler, FEATURES, LOGICAL_CORES, OUT_ROOT, SEED, SUBSET_1M,
    ThreadSampler, append_jsonl, baseline_cpu, ensure_dir, iso_now,
    parallel_eff, read_jsonl, resolve_path,
)


def load_matrix(subset: Path) -> np.ndarray:
    from sklearn.preprocessing import StandardScaler

    df = pd.read_parquet(subset, columns=FEATURES)
    x_raw = df[FEATURES].to_numpy(dtype=np.float64)
    return StandardScaler().fit_transform(x_raw)


def fit_forest(x: np.ndarray):
    from sklearn.ensemble import IsolationForest

    clf = IsolationForest(
        n_estimators=150,
        max_samples=512,
        max_features=0.8,
        random_state=SEED,
        n_jobs=4,
    )
    t0 = time.perf_counter()
    try:
        clf.fit(x)
    except (PermissionError, OSError) as exc:
        # A restricted Windows sandbox may deny joblib's pipe directory. The
        # fit is not a timed quantity, so retry single-threaded.
        print(f"[warn] fit with n_jobs=4 failed ({type(exc).__name__}: {exc}); "
              f"retrying n_jobs=1", flush=True)
        clf.set_params(n_jobs=1)
        clf.fit(x)
    return clf, time.perf_counter() - t0


def score_once(clf, x: np.ndarray, n_jobs: int) -> tuple[np.ndarray, float]:
    from joblib import parallel_backend

    clf.set_params(n_jobs=n_jobs)
    with parallel_backend("threading", n_jobs=n_jobs):
        t0 = time.perf_counter()
        scores = clf.decision_function(x)
        elapsed = time.perf_counter() - t0
    return scores, elapsed


def summarize(raw: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(raw)
    if df.empty:
        return pd.DataFrame()
    df["is_warmup"] = df["is_warmup"].astype(bool)
    meas = df[~df["is_warmup"]].copy()
    if meas.empty:
        return pd.DataFrame()
    base_rows = meas[meas["n_jobs_req"] == 1]["score_s"]
    base = float(base_rows.median()) if len(base_rows) else float("nan")
    rows: list[dict] = []
    for req in sorted(meas["n_jobs_req"].unique()):
        g = meas[meas["n_jobs_req"] == req]
        med = float(g["score_s"].median())
        mad = float((g["score_s"] - med).abs().median())
        speedup = base / med if med > 0 and math.isfinite(base) else float("nan")
        n_eff = float(g["n_jobs_eff"].iloc[0])
        efficiency = speedup / n_eff if math.isfinite(speedup) and n_eff > 0 else float("nan")
        rows.append({
            "n_jobs_req": int(req),
            "n_jobs_eff": int(n_eff),
            "n_rows": int(g["n_rows"].iloc[0]),
            "n_runs": int(len(g)),
            "score_s_median": round(med, 3),
            "score_s_min": round(float(g["score_s"].min()), 3),
            "score_s_max": round(float(g["score_s"].max()), 3),
            "score_s_mad": round(mad, 3),
            "speedup_vs_1": round(speedup, 3) if math.isfinite(speedup) else float("nan"),
            "efficiency": round(efficiency, 3) if math.isfinite(efficiency) else float("nan"),
            "parallel_eff_median": round(float(g["parallel_eff"].median()), 3),
            "parallel_eff_net_median": round(float(g["parallel_eff_net"].median()), 3),
            "max_abs_diff_max": float(g["max_abs_diff"].max()),
            "threads_observed_median": int(g["threads_observed"].median()),
            "valid": bool(g["max_abs_diff"].max() < 1e-10),
        })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default=str(SUBSET_1M))
    ap.add_argument("--njobs", default="1,2,4,8,-1")
    ap.add_argument("--repeats", type=int, default=5,
                    help="total calls per n_jobs, including warm-up")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--out-dir", default=str(OUT_ROOT))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    args.subset = resolve_path(args.subset)
    out_dir = resolve_path(args.out_dir)
    ensure_dir(out_dir)
    njobs_list = [int(x.strip()) for x in args.njobs.split(",") if x.strip()]
    if not args.subset.exists():
        raise SystemExit(f"subset not found: {args.subset}")

    total_calls = len(njobs_list) * max(1, args.repeats)
    if args.dry_run:
        print(json.dumps({
            "subset": str(args.subset),
            "njobs": njobs_list,
            "calls": total_calls,
            "repeats": args.repeats,
            "warmup": args.warmup,
            "estimated_minutes": round(total_calls * 5 / 60 + 1, 1),
        }, ensure_ascii=False, indent=2))
        return 0

    jsonl_path = out_dir / "s3_f1_runs.jsonl"
    csv_path = out_dir / "s3_f1.csv"
    summary_path = out_dir / "s3_f1_summary.csv"
    errors_path = out_dir / "errors.jsonl"

    print(f"[load] {args.subset}", flush=True)
    x = load_matrix(args.subset)
    n_rows = int(x.shape[0])
    print(f"[load] rows={n_rows} features={x.shape[1]}", flush=True)

    clf, fit_s = fit_forest(x)
    print(f"[fit] IsolationForest(150,512,0.8,seed={SEED},n_jobs=4) "
          f"fit_s={fit_s:.2f}", flush=True)

    # Reference scores at n_jobs=1. Every other point must match these.
    ref_scores, ref_s = score_once(clf, x, 1)
    print(f"[ref] n_jobs=1 score_s={ref_s:.2f} sum={ref_scores.sum():.6f}", flush=True)

    raw = read_jsonl(jsonl_path)
    done = {
        (int(r["n_jobs_req"]), int(r["run_idx"])) for r in raw
        if r.get("n_jobs_req") is not None and r.get("run_idx") is not None
    }
    total_calls = max(1, args.repeats)
    warmup = min(max(0, args.warmup), total_calls - 1)

    for req in njobs_list:
        n_eff = LOGICAL_CORES if req == -1 else req
        if args.resume and all((req, i) in done for i in range(total_calls)):
            print(f"[skip] n_jobs={req}: complete", flush=True)
            continue
        print(f"[grid] n_jobs={req} (eff={n_eff}) calls={total_calls} "
              f"warmup={warmup}", flush=True)
        base = baseline_cpu(3.0)
        for run_idx in range(total_calls):
            if (req, run_idx) in done:
                print(f"[skip] n_jobs={req} run {run_idx}", flush=True)
                continue
            is_warmup = run_idx < warmup
            started_at = iso_now()
            try:
                with CpuSampler() as cs, ThreadSampler() as ts:
                    scores, score_s = score_once(clf, x, req)
                finished_at = iso_now()
                max_abs_diff = float(np.max(np.abs(scores - ref_scores)))
                eff, eff_net = parallel_eff(cs.mean, n_eff, base["mean"])
                rec = {
                    "n_rows": n_rows,
                    "n_jobs_req": req,
                    "n_jobs_eff": n_eff,
                    "run_idx": run_idx,
                    "is_warmup": bool(is_warmup),
                    "fit_s_reference": round(fit_s, 3),
                    "score_s": round(score_s, 3),
                    "cpu_pct_mean": round(cs.mean, 2),
                    "cpu_pct_max": round(cs.pct_max, 2),
                    "cpu_pct_baseline": round(base["mean"], 2),
                    "parallel_eff": round(eff, 4),
                    "parallel_eff_net": round(eff_net, 4),
                    "score_sum": float(scores.sum()),
                    "score_mean": float(scores.mean()),
                    "score_std": float(scores.std()),
                    "max_abs_diff": max_abs_diff,
                    "threads_observed": int(ts.max_threads),
                    "started_at": started_at,
                    "finished_at": finished_at,
                }
                append_jsonl(jsonl_path, rec)
                print(
                    f"[run] n_jobs={req} idx={run_idx} warmup={is_warmup} "
                    f"score_s={score_s:.3f} cpu={cs.mean:.1f}% "
                    f"eff={eff:.2f} thr={ts.max_threads} "
                    f"maxdiff={max_abs_diff:.2e}", flush=True,
                )
                if max_abs_diff >= 1e-10:
                    err = {
                        "ts": iso_now(), "stage": "s3", "n_jobs_req": req,
                        "run_idx": run_idx,
                        "error": f"score changed with n_jobs: max_abs_diff={max_abs_diff:.3e}",
                    }
                    append_jsonl(errors_path, err)
                    print(f"[error] {err['error']}", flush=True)
                    break
            except Exception as exc:  # noqa: BLE001
                err = {
                    "ts": iso_now(), "stage": "s3", "n_jobs_req": req,
                    "run_idx": run_idx, "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc()[-4000:],
                }
                append_jsonl(errors_path, err)
                print(f"[error] {err['error']}", flush=True)
                break

    raw = read_jsonl(jsonl_path)
    if raw:
        pd.DataFrame(raw).to_csv(csv_path, index=False, encoding="utf-8")
        summary = summarize(raw)
        summary.to_csv(summary_path, index=False, encoding="utf-8")
        print(f"[done] wrote {csv_path} and {summary_path}", flush=True)
        print(summary.to_string(index=False), flush=True)
        if len(summary) and not bool(summary["valid"].all()):
            print("[WARN] some n_jobs points changed scores; see max_abs_diff", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
