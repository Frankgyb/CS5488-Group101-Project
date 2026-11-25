# Location: project/code/ml/experiments/parallel_scaling/prepare_subset.py | PROJECT_ROOT depth: 4
"""Create the fixed 2M / 1M subsets used by the v3 scaling study.

Reads ``data/output/sampled_km`` read-only and writes only under
``data/output/scaling_study/subsets``.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from scaling_common import (
    FEATURES, INPUT_PATH, OUT_ROOT, SEED, SUBSET_2M, ensure_dir, iso_now,
    list_part_files, resolve_path,
)


def meta_path(out: Path) -> Path:
    return out.parent / (out.name + "_meta.json")


def read_parquet_any(path: Path, columns: list[str]) -> pd.DataFrame:
    """Read a parquet file or a directory of part-*.parquet files."""
    if path.is_dir():
        files = sorted(path.glob("part-*.parquet"))
        if not files:
            files = sorted(path.glob("*.parquet"))
        if not files:
            raise RuntimeError(f"no parquet files under {path}")
        return pd.concat(
            [pd.read_parquet(f, columns=columns) for f in files],
            ignore_index=True,
        )
    return pd.read_parquet(path, columns=columns)


def existing_ok(out: Path, rows: int) -> bool:
    if not out.exists():
        return False
    if out.is_dir():
        if not (out / "_SUCCESS").exists():
            return False
        mp = meta_path(out)
        if not mp.exists():
            return False
        try:
            meta = json.loads(mp.read_text(encoding="utf-8"))
            return int(meta.get("rows", -1)) == rows
        except (OSError, ValueError, TypeError):
            return False
    mp = meta_path(out)
    if not mp.exists():
        return False
    try:
        meta = json.loads(mp.read_text(encoding="utf-8"))
        return int(meta.get("rows", -1)) == rows
    except (OSError, ValueError, TypeError):
        return False


def make_full_subset(rows: int, out: Path, partitions: int, seed: int) -> dict:
    """Pandas/pyarrow version: write N part files plus _SUCCESS.

    Spark/Hadoop on this Windows sandbox fails at RawLocalFileSystem
    setPermission (ChangeFileModeByMask error 5), so the subset is
    materialised without Spark.  Spark can still read the resulting
    directory normally.
    """
    input_parts = len(list_part_files(INPUT_PATH))
    if input_parts == 0:
        raise RuntimeError(f"no part-*.parquet files under {INPUT_PATH}")
    parts = partitions if partitions > 0 else input_parts

    src = read_parquet_any(INPUT_PATH, FEATURES)
    if rows > len(src):
        raise RuntimeError(f"requested {rows} rows but input has only {len(src)}")
    sub = src.sample(n=rows, random_state=seed).reset_index(drop=True)
    del src

    ensure_dir(out)
    for i in range(parts):
        chunk = sub.iloc[i::parts]
        chunk.to_parquet(out / f"part-{i:05d}.parquet", index=False)
    (out / "_SUCCESS").write_text("", encoding="utf-8")

    meta = {
        "kind": "pandas_full",
        "source": str(INPUT_PATH),
        "out": str(out),
        "rows": int(len(sub)),
        "partitions": parts,
        "input_parts": input_parts,
        "seed": seed,
        "created_at": iso_now(),
    }
    meta_path(out).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return meta


def make_pandas_subset(rows: int, source: Path, out_file: Path, seed: int) -> dict:
    df = read_parquet_any(source, FEATURES)
    if rows > len(df):
        raise RuntimeError(f"requested {rows} rows but source has only {len(df)}")
    sub = df.sample(n=rows, random_state=seed)
    ensure_dir(out_file.parent)
    sub.to_parquet(out_file, index=False)
    meta = {
        "kind": "pandas",
        "source": str(source),
        "out": str(out_file),
        "rows": int(len(sub)),
        "seed": seed,
        "created_at": iso_now(),
    }
    meta_path(out_file).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=2_000_000)
    ap.add_argument("--out", default=str(SUBSET_2M))
    ap.add_argument("--partitions", type=int, default=0,
                    help="0 = mirror the number of part files in the source")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--from-subset", default="",
                    help="pandas mode: read this parquet instead of sampled_km")
    ap.add_argument("--if-exists", choices=["skip", "error", "overwrite"], default="skip")
    args = ap.parse_args()

    out = resolve_path(args.out)

    if args.from_subset:
        source = resolve_path(args.from_subset)
        if out.exists() and args.if_exists == "skip" and existing_ok(out, args.rows):
            print(json.dumps({"status": "skip", "out": str(out)}, ensure_ascii=False))
            return 0
        if out.exists() and args.if_exists == "error":
            raise SystemExit(f"{out} already exists")
        if out.exists():
            # Never touch anything outside the scaling_study output tree.
            if OUT_ROOT.resolve() not in out.resolve().parents:
                raise SystemExit(f"refusing to remove {out}: outside {OUT_ROOT}")
            if out.is_dir():
                shutil.rmtree(out)
            else:
                out.unlink()
            mp = meta_path(out)
            if mp.exists():
                mp.unlink()
        meta = make_pandas_subset(args.rows, source, out, args.seed)
        print(json.dumps(meta, ensure_ascii=False, indent=2))
        return 0

    if out.exists() and args.if_exists == "skip" and existing_ok(out, args.rows):
        print(json.dumps({"status": "skip", "out": str(out)}, ensure_ascii=False))
        return 0
    if out.exists():
        if args.if_exists == "error":
            raise SystemExit(f"{out} already exists")
        # Never touch anything outside the scaling_study output tree.
        if OUT_ROOT.resolve() not in out.resolve().parents:
            raise SystemExit(f"refusing to remove {out}: outside {OUT_ROOT}")
        shutil.rmtree(out)
        mp = meta_path(out)
        if mp.exists():
            mp.unlink()

    ensure_dir(out.parent)
    meta = make_full_subset(args.rows, out, args.partitions, args.seed)
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
