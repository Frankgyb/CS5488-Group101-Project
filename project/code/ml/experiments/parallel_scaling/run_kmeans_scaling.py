# Location: project/code/ml/experiments/parallel_scaling/run_kmeans_scaling.py | PROJECT_ROOT depth: 4
"""S2: KMeans / BisectingKMeans core-count and parallelism scaling (v3).

One Spark JVM per (n_cores, shuffle) grid; the 2M-row subset is fixed and
shared by every grid. Warm-up calls are recorded but excluded from summaries.
Reads ``data/output/scaling_study/subsets`` and writes to the study directory.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

from scaling_common import (
    CpuSampler, FEATURES, LOGICAL_CORES, OUT_ROOT, SEED, SUBSET_2M,
    append_jsonl, baseline_cpu, count_jvm, ensure_dir, iso_now, new_spark,
    parallel_eff, read_jsonl, resolve_path,
)


def parse_int_list(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def make_estimator(job: str):
    from pyspark.ml.clustering import BisectingKMeans, KMeans

    if job == "k1":
        return KMeans(
            featuresCol="features", predictionCol="prediction",
            k=2, seed=SEED, maxIter=20, tol=1e-4,
        )
    if job == "k2":
        return BisectingKMeans(
            featuresCol="features", predictionCol="prediction",
            k=8, seed=SEED, maxIter=20,
        )
    raise ValueError(f"unknown job {job!r}")


def prepare_data(spark, subset: Path):
    from pyspark.ml.feature import StandardScaler, VectorAssembler

    # RawLocalFileSystem.listStatus on a Python-created directory is denied
    # by this Windows sandbox; explicit file paths bypass the directory list.
    if subset.is_dir():
        files = sorted(subset.glob("part-*.parquet"))
        if not files:
            files = sorted(subset.glob("*.parquet"))
        if not files:
            raise RuntimeError(f"no parquet files under {subset}")
        df = spark.read.parquet(*[str(f) for f in files]).select(*FEATURES)
    else:
        df = spark.read.parquet(str(subset)).select(*FEATURES)
    assembler = VectorAssembler(inputCols=FEATURES, outputCol="features_raw")
    vec = assembler.transform(df)
    scaler = StandardScaler(
        inputCol="features_raw", outputCol="features", withMean=True, withStd=True,
    )
    scaled = scaler.fit(vec).transform(vec)
    scaled.cache()
    n_rows = scaled.count()
    return scaled, int(n_rows)


def fit_once(job: str, sdf):
    model = make_estimator(job).fit(sdf)
    summary = getattr(model, "summary", None)
    wssse = float("nan")
    n_iter = None
    if summary is not None:
        try:
            wssse = float(summary.trainingCost)
        except Exception:  # noqa: BLE001
            pass
        try:
            n_iter = int(summary.numIter)
        except Exception:  # noqa: BLE001
            n_iter = None
    return model, wssse, n_iter


def run_grid(args, job: str, n_cores: int, shuffle: int, jsonl_path: Path,
             errors_path: Path, state: dict) -> None:
    total_calls = max(1, args.repeats)
    warmup = min(max(0, args.warmup), total_calls - 1)
    expected = list(range(total_calls))

    raw = read_jsonl(jsonl_path)
    done = {
        int(r["run_idx"]) for r in raw
        if r.get("job") == job
        and int(r.get("n_cores", -1)) == n_cores
        and int(r.get("shuffle_partitions", -1)) == shuffle
        and r.get("run_idx") is not None
    }
    if args.resume and all(i in done for i in expected):
        print(f"[skip] {job} cores={n_cores} shuffle={shuffle}: already complete", flush=True)
        return

    master = f"local[{n_cores}]"
    print(f"[grid] {job} cores={n_cores} shuffle={shuffle} calls={total_calls} "
          f"(warmup={warmup}, measured={total_calls - warmup})", flush=True)

    t_boot = time.perf_counter()
    spark = new_spark(master, shuffle, f"{job}-c{n_cores}-s{shuffle}")
    spark_boot_s = time.perf_counter() - t_boot
    try:
        t0 = time.perf_counter()
        sdf, n_rows = prepare_data(spark, args.subset)
        prep_s = time.perf_counter() - t0
        if n_rows != args.rows:
            raise RuntimeError(f"subset has {n_rows} rows, expected {args.rows}")
        input_partitions = int(sdf.rdd.getNumPartitions())
        base = baseline_cpu(3.0)
        print(f"[data] rows={n_rows} input_partitions={input_partitions} "
              f"prep_s={prep_s:.1f} baseline_cpu={base['mean']:.1f}%", flush=True)

        consecutive_failures = 0
        for run_idx in expected:
            if run_idx in done:
                print(f"[skip] run {run_idx}", flush=True)
                continue
            is_warmup = run_idx < warmup
            javaw_before = count_jvm()
            free_ram_gb = psutil.virtual_memory().available / 1e9
            started_at = iso_now()
            try:
                with CpuSampler() as sampler:
                    t_fit = time.perf_counter()
                    _model, wssse, n_iter = fit_once(job, sdf)
                    fit_s = time.perf_counter() - t_fit
                finished_at = iso_now()
                eff, eff_net = parallel_eff(sampler.mean, n_cores, base["mean"])
                rec = {
                    "job": job,
                    "run_idx": run_idx,
                    "is_warmup": bool(is_warmup),
                    "n_cores": n_cores,
                    "shuffle_partitions": shuffle,
                    "n_rows": n_rows,
                    "input_partitions": input_partitions,
                    "spark_boot_s": round(spark_boot_s, 3),
                    "prep_s": round(prep_s, 3),
                    "fit_s": round(fit_s, 3),
                    "wssse": wssse,
                    "n_iter": n_iter,
                    "cpu_pct_mean": round(sampler.mean, 2),
                    "cpu_pct_max": round(sampler.pct_max, 2),
                    "cpu_pct_baseline": round(base["mean"], 2),
                    "parallel_eff": round(eff, 4),
                    "parallel_eff_net": round(eff_net, 4),
                    "javaw_before": javaw_before,
                    "javaw_after": count_jvm(),
                    "free_ram_gb": round(free_ram_gb, 2),
                    "started_at": started_at,
                    "finished_at": finished_at,
                }
                append_jsonl(jsonl_path, rec)
                print(
                    f"[run] {job} c={n_cores} s={shuffle} idx={run_idx} "
                    f"warmup={is_warmup} fit_s={fit_s:.2f} "
                    f"cpu={sampler.mean:.1f}% eff={eff:.2f} "
                    f"wssse={wssse:.1f}", flush=True,
                )
                consecutive_failures = 0
            except Exception as exc:  # noqa: BLE001
                consecutive_failures += 1
                err = {
                    "ts": iso_now(), "job": job, "n_cores": n_cores,
                    "shuffle_partitions": shuffle, "run_idx": run_idx,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc()[-4000:],
                }
                append_jsonl(errors_path, err)
                print(f"[error] {err['error']}", flush=True)
                if consecutive_failures >= 2:
                    print(f"[skip] {job} c={n_cores} s={shuffle}: "
                          f"2 consecutive failures, skipping grid", flush=True)
                    state["skipped_grids"] += 1
                    break
        if state["skipped_grids"] >= 3:
            raise RuntimeError("3 grids skipped; stopping this job")
    finally:
        try:
            spark.stop()
        except Exception:  # noqa: BLE001
            pass
        time.sleep(1.0)


def summarize(raw: list[dict], job: str, baseline_core: int) -> pd.DataFrame:
    df = pd.DataFrame(raw)
    if df.empty:
        return pd.DataFrame()
    df["is_warmup"] = df["is_warmup"].astype(bool)
    meas = df[~df["is_warmup"]].copy()
    if meas.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for shuffle in sorted(meas["shuffle_partitions"].unique()):
        sub = meas[meas["shuffle_partitions"] == shuffle]
        base_rows = sub[sub["n_cores"] == baseline_core]["fit_s"]
        base = float(base_rows.median()) if len(base_rows) else float("nan")

        # T(p) = a + b/p, fitted on median fit_s per core count.
        def _fit_amdahl(series: pd.Series) -> tuple[float, float, float, float]:
            aa = bb = al = rr = float("nan")
            if len(series) >= 3:
                x = 1.0 / series.index.to_numpy(dtype=float)
                y = series.to_numpy(dtype=float)
                try:
                    slope, intercept = np.polyfit(x, y, 1)
                    bb, aa = float(slope), float(intercept)
                    if aa + bb > 0:
                        al = aa / (aa + bb)
                    y_hat = aa + bb * x
                    ss_res = float(np.sum((y - y_hat) ** 2))
                    ss_tot = float(np.sum((y - y.mean()) ** 2))
                    rr = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
                except Exception:  # noqa: BLE001
                    pass
            return aa, bb, al, rr

        # Main fit excludes local[1]: it is a P/E-mixed reference point, not the
        # denominator.  Keep the all-points fit only as a sensitivity check.
        pts_main = sub[sub["n_cores"] >= baseline_core].groupby("n_cores")["fit_s"].median().sort_index()
        a, b, alpha, r2 = _fit_amdahl(pts_main)
        pts_all = sub.groupby("n_cores")["fit_s"].median().sort_index()
        _, _, alpha_all, r2_all = _fit_amdahl(pts_all)

        for n_cores in sorted(sub["n_cores"].unique()):
            g = sub[sub["n_cores"] == n_cores]
            med = float(g["fit_s"].median())
            mad = float((g["fit_s"] - med).abs().median())
            speedup = base / med if med > 0 and math.isfinite(base) else float("nan")
            efficiency = float("nan")
            if math.isfinite(speedup) and n_cores >= baseline_core:
                efficiency = speedup / (n_cores / baseline_core)
            rows.append({
                "job": job,
                "baseline_core": baseline_core,
                "n_cores": int(n_cores),
                "shuffle_partitions": int(shuffle),
                "n_rows": int(g["n_rows"].iloc[0]),
                "n_runs": int(len(g)),
                "fit_s_median": round(med, 3),
                "fit_s_min": round(float(g["fit_s"].min()), 3),
                "fit_s_max": round(float(g["fit_s"].max()), 3),
                "fit_s_mad": round(mad, 3),
                "speedup_vs_baseline": round(speedup, 3) if math.isfinite(speedup) else float("nan"),
                "efficiency": round(efficiency, 3) if math.isfinite(efficiency) else float("nan"),
                "parallel_eff_median": round(float(g["parallel_eff"].median()), 3),
                "parallel_eff_net_median": round(float(g["parallel_eff_net"].median()), 3),
                "input_partitions": int(g["input_partitions"].iloc[0]),
                "amdahl_a": round(a, 3),
                "amdahl_b": round(b, 3),
                "amdahl_alpha": round(alpha, 3),
                "amdahl_r2": round(r2, 3),
                "amdahl_alpha_with_local1": round(alpha_all, 3),
                "amdahl_r2_with_local1": round(r2_all, 3),
            })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True, choices=["k1", "k2"])
    ap.add_argument("--rows", type=int, default=2_000_000)
    ap.add_argument("--subset", default=str(SUBSET_2M))
    ap.add_argument("--cores", default="2,4,8")
    ap.add_argument("--shuffle", default="30,120")
    ap.add_argument("--order", choices=["sequential", "random"],
                    default="sequential",
                    help="grid execution order; random reduces heat/order drift")
    ap.add_argument("--order-seed", type=int, default=42)
    ap.add_argument("--repeats", type=int, default=5,
                    help="total calls per grid, including warm-up")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--baseline", type=int, default=2)
    ap.add_argument("--out-dir", default=str(OUT_ROOT))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    args.subset = resolve_path(args.subset)
    out_dir = resolve_path(args.out_dir)
    ensure_dir(out_dir)
    args.cores_list = parse_int_list(args.cores)
    args.shuffle_list = parse_int_list(args.shuffle)

    if not args.subset.exists():
        raise SystemExit(f"subset not found: {args.subset}")

    grids = [(n, s) for n in args.cores_list for s in args.shuffle_list]
    if args.order == "random":
        random.Random(args.order_seed).shuffle(grids)
    total_calls = len(grids) * max(1, args.repeats)
    if args.dry_run:
        est_per_call = 85 if args.job == "k1" else 70
        print(json.dumps({
            "job": args.job, "subset": str(args.subset), "rows": args.rows,
            "grids": [{"n_cores": n, "shuffle": s} for n, s in grids],
            "order": args.order, "order_seed": args.order_seed,
            "calls": total_calls, "repeats": args.repeats, "warmup": args.warmup,
            "baseline": args.baseline,
            "estimated_minutes": round(total_calls * est_per_call / 60 + 3, 1),
        }, ensure_ascii=False, indent=2))
        return 0

    jsonl_path = out_dir / f"s2_{args.job}_runs.jsonl"
    csv_path = out_dir / f"s2_{args.job}.csv"
    summary_path = out_dir / f"s2_{args.job}_summary.csv"
    errors_path = out_dir / "errors.jsonl"

    state = {"skipped_grids": 0}
    for n_cores, shuffle in grids:
        run_grid(args, args.job, n_cores, shuffle, jsonl_path, errors_path, state)

    raw = [r for r in read_jsonl(jsonl_path) if r.get("job") == args.job]
    if raw:
        pd.DataFrame(raw).to_csv(csv_path, index=False, encoding="utf-8")
        summary = summarize(raw, args.job, args.baseline)
        summary.to_csv(summary_path, index=False, encoding="utf-8")
        print(f"[done] wrote {csv_path} and {summary_path}", flush=True)
        print(summary.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
