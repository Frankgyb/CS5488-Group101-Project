# Location: project/code/ml/experiments/parallel_scaling/analyze_k2_increase.py | PROJECT_ROOT depth: 4
"""Compare the archived K2 n=2 run with the new K2 n=4 run.

Reads only persisted CSVs/JSON; starts no Spark session.  Outputs:

* ``k2_n2_vs_n4.csv``: per-grid median/MAD, speed-up and 95% CI comparison.
* ``k2_drift.json``: per-grid time/order drift diagnostics for both runs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scaling_common import OUT_ROOT, PROJECT_ROOT, resolve_path

DEFAULT_OLD = (
    PROJECT_ROOT / "data" / "output" / "scaling_study_archive" / "k2_n2_20261125"
)


def measured(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["is_warmup"] = df["is_warmup"].astype(bool)
    return df[~df["is_warmup"]].copy()


def load_bootstrap(path: Path) -> dict:
    if not path.exists():
        return {"cells": {}, "speedups": {}}
    data = json.loads(path.read_text(encoding="utf-8"))
    k2 = data.get("k2", {})
    cells = {
        (int(c["shuffle"]), int(c["cores"])): c
        for c in k2.get("cells", [])
    }
    speedups = {
        (int(s["shuffle"]), int(s["cores"])): s
        for s in k2.get("speedups", [])
    }
    return {"cells": cells, "speedups": speedups}


def med_mad(values: pd.Series) -> tuple[float, float]:
    values = pd.Series(values, dtype=float)
    med = float(values.median())
    mad = float((values - med).abs().median())
    return med, mad


def drift_table(df: pd.DataFrame) -> dict:
    out: dict[str, dict] = {}
    for (shuffle, cores), group in df.groupby(
        ["shuffle_partitions", "n_cores"], sort=True
    ):
        group = group.sort_values("started_at")
        y = group["fit_s"].to_numpy(dtype=float)
        t = np.arange(len(y), dtype=float)
        slope = float(np.polyfit(t, y, 1)[0]) if len(y) >= 2 else float("nan")
        k = min(2, len(y))
        first = float(np.mean(y[:k]))
        last = float(np.mean(y[-k:]))
        delta_pct = (last - first) / first * 100.0 if first else float("nan")
        out[f"{int(shuffle)}|{int(cores)}"] = {
            "n": int(len(y)),
            "slope_s_per_run": round(slope, 4),
            "first2_mean_s": round(first, 4),
            "last2_mean_s": round(last, 4),
            "delta_pct": round(delta_pct, 2),
            "flag_gt_20pct": bool(abs(delta_pct) > 20.0),
            "run_order": [int(x) for x in group["run_idx"].tolist()],
            "fit_s": [round(float(x), 4) for x in y],
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-dir", default=str(DEFAULT_OLD))
    ap.add_argument("--new-dir", default=str(OUT_ROOT))
    ap.add_argument("--out-dir", default="")
    args = ap.parse_args()

    old_dir = resolve_path(args.old_dir)
    new_dir = resolve_path(args.new_dir)
    out_dir = resolve_path(args.out_dir) if args.out_dir else new_dir
    old_csv = old_dir / "s2_k2.csv"
    new_csv = new_dir / "s2_k2.csv"
    if not old_csv.exists():
        raise SystemExit(f"archived old K2 CSV not found: {old_csv}")
    if not new_csv.exists():
        raise SystemExit(f"new K2 CSV not found: {new_csv}")

    old = measured(old_csv)
    new = measured(new_csv)
    old_bs = load_bootstrap(old_dir / "bootstrap_ci.json")
    new_bs = load_bootstrap(new_dir / "bootstrap_ci.json")

    old_base = {
        int(s): float(old[
            (old["shuffle_partitions"] == s) & (old["n_cores"] == 2)
        ]["fit_s"].median())
        for s in sorted(old["shuffle_partitions"].unique())
    }
    new_base = {
        int(s): float(new[
            (new["shuffle_partitions"] == s) & (new["n_cores"] == 2)
        ]["fit_s"].median())
        for s in sorted(new["shuffle_partitions"].unique())
    }

    keys = sorted(
        set(map(tuple, old[["shuffle_partitions", "n_cores"]].to_numpy()))
        | set(map(tuple, new[["shuffle_partitions", "n_cores"]].to_numpy()))
    )
    rows: list[dict] = []
    for shuffle, cores in keys:
        shuffle, cores = int(shuffle), int(cores)
        key = (shuffle, cores)
        o = old[
            (old["shuffle_partitions"] == shuffle)
            & (old["n_cores"] == cores)
        ]["fit_s"]
        n = new[
            (new["shuffle_partitions"] == shuffle)
            & (new["n_cores"] == cores)
        ]["fit_s"]
        omed, omad = med_mad(o) if len(o) else (float("nan"), float("nan"))
        nmed, nmad = med_mad(n) if len(n) else (float("nan"), float("nan"))
        ospeed = old_base[shuffle] / omed if omed else float("nan")
        nspeed = new_base[shuffle] / nmed if nmed else float("nan")
        delta_pct = (nmed - omed) / omed * 100.0 if omed else float("nan")

        ocell = old_bs["cells"].get(key, {})
        ncell = new_bs["cells"].get(key, {})
        oci = ocell.get("ci95") or [float("nan"), float("nan")]
        nci = ncell.get("ci95") or [float("nan"), float("nan")]
        owidth = float(oci[1]) - float(oci[0])
        nwidth = float(nci[1]) - float(nci[0])
        osp = old_bs["speedups"].get(key, {})
        nsp = new_bs["speedups"].get(key, {})
        ospci = osp.get("ci95") or [float("nan"), float("nan")]
        nspci = nsp.get("ci95") or [float("nan"), float("nan")]

        rows.append({
            "shuffle": shuffle,
            "n_cores": cores,
            "n_old": int(len(o)),
            "n_new": int(len(n)),
            "median_old": round(omed, 3),
            "median_new": round(nmed, 3),
            "delta_pct": round(delta_pct, 2),
            "mad_old": round(omad, 3),
            "mad_new": round(nmad, 3),
            "min_old": round(float(o.min()), 3) if len(o) else float("nan"),
            "max_old": round(float(o.max()), 3) if len(o) else float("nan"),
            "min_new": round(float(n.min()), 3) if len(n) else float("nan"),
            "max_new": round(float(n.max()), 3) if len(n) else float("nan"),
            "speedup_old": round(ospeed, 3),
            "speedup_new": round(nspeed, 3),
            "ci_old_lo": round(float(oci[0]), 3),
            "ci_old_hi": round(float(oci[1]), 3),
            "ci_new_lo": round(float(nci[0]), 3),
            "ci_new_hi": round(float(nci[1]), 3),
            "ci_width_old": round(owidth, 3),
            "ci_width_new": round(nwidth, 3),
            "ci_width_ratio": round(nwidth / owidth, 3) if owidth else float("nan"),
            "speedup_ci_old_lo": round(float(ospci[0]), 3),
            "speedup_ci_old_hi": round(float(ospci[1]), 3),
            "speedup_ci_new_lo": round(float(nspci[0]), 3),
            "speedup_ci_new_hi": round(float(nspci[1]), 3),
        })

    out_dir.mkdir(parents=True, exist_ok=True)
    cmp_path = out_dir / "k2_n2_vs_n4.csv"
    drift_path = out_dir / "k2_drift.json"
    pd.DataFrame(rows).to_csv(cmp_path, index=False, encoding="utf-8")
    drift = {"old": drift_table(old), "new": drift_table(new)}

    def finite(obj):
        if isinstance(obj, dict):
            return {k: finite(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [finite(v) for v in obj]
        if isinstance(obj, float) and not np.isfinite(obj):
            return None
        return obj

    drift_path.write_text(
        json.dumps(finite(drift), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"[done] wrote {cmp_path} and {drift_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
