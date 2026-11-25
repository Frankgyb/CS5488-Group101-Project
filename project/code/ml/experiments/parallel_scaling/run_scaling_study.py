# Location: project/code/ml/experiments/parallel_scaling/run_scaling_study.py | PROJECT_ROOT depth: 4
"""Unattended driver for the v3 single-machine scaling study.

Pipeline: preflight -> 2M/1M subsets -> smoke -> K1/K2 -> IF scoring -> plots.
All paths are project-relative; the only writable trees are
``data/output/scaling_study`` and ``data/log/scaling_study``.

Typical use (from any cwd):
    python project/code/ml/experiments/parallel_scaling/run_scaling_study.py
    python .../run_scaling_study.py --analysis-only
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import psutil

from scaling_common import (
    INPUT_PATH, LOG_DIR, LOGICAL_CORES, OUT_ROOT, PHYSICAL_CORES, PROJECT_ROOT,
    SPARK_LOCAL_DIR, SUBSET_1M, SUBSET_2M, atomic_write_json, baseline_cpu,
    ensure_dir, iso_now, jvm_processes, keep_awake, list_part_files,
    powercfg_summary, resolve_path,
)

SCRIPT_DIR = Path(__file__).resolve().parent


class PreflightError(RuntimeError):
    pass


def collect_env(python: str) -> dict:
    import pyspark
    import sklearn

    try:
        battery = psutil.sensors_battery()
    except (NotImplementedError, AttributeError):
        battery = None
    power = powercfg_summary()
    scheme = ""
    for line in (power.get("active_scheme") or "").splitlines():
        if line.strip():
            scheme = line.strip()
            break
    parts = list_part_files(INPUT_PATH)
    return {
        "python": sys.version.split()[0],
        "python_executable": python,
        "pyspark": pyspark.__version__,
        "psutil": psutil.__version__,
        "sklearn": sklearn.__version__,
        "pandas": pd.__version__,
        "cpu_model": platform.processor() or platform.machine(),
        "physical_cores": PHYSICAL_CORES,
        "logical_cores": LOGICAL_CORES,
        "ram_gb": round(psutil.virtual_memory().total / 1e9, 2),
        "battery_plugged": None if battery is None else bool(battery.power_plugged),
        "power_scheme": scheme,
        "powercfg_raw": power,
        "temp_free_gb": round(psutil.disk_usage(str(OUT_ROOT)).free / 1e9, 2),
        "out_free_gb": round(psutil.disk_usage(str(PROJECT_ROOT)).free / 1e9, 2),
        "input_path": str(INPUT_PATH),
        "input_parts": len(parts),
        "input_success": (INPUT_PATH / "_SUCCESS").exists(),
        "out_dir": str(OUT_ROOT),
        "spark_local_dir": str(SPARK_LOCAL_DIR),
        "started_at": iso_now(),
    }


def preflight(python: str, out_dir: Path) -> dict:
    ensure_dir(out_dir)
    ensure_dir(SPARK_LOCAL_DIR)
    missing = [
        name for name in ["pyspark", "psutil", "sklearn", "pandas", "numpy",
                          "matplotlib", "pyarrow"]
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        raise PreflightError(f"missing python modules: {missing}")

    env = collect_env(python)
    atomic_write_json(out_dir / "env.json", env)

    if env["battery_plugged"] is False:
        raise PreflightError("not on AC power; plug in before the run")
    if not env["input_success"]:
        raise PreflightError(f"missing _SUCCESS under {INPUT_PATH}")
    if env["input_parts"] == 0:
        raise PreflightError(f"no part-*.parquet under {INPUT_PATH}")
    if env["temp_free_gb"] < 10:
        raise PreflightError(f"study/output free {env['temp_free_gb']:.1f} GB < 10 GB")
    if env["out_free_gb"] < 5:
        raise PreflightError(f"project free {env['out_free_gb']:.1f} GB < 5 GB")

    jvms = jvm_processes()
    if jvms:
        raise PreflightError(f"existing Java processes: {jvms}")

    base = baseline_cpu(5.0)
    env["baseline_cpu_pct"] = round(base["mean"], 2)
    env["baseline_cpu_max_pct"] = round(base["max"], 2)
    atomic_write_json(out_dir / "env.json", env)
    if base["mean"] > 20.0:
        raise PreflightError(
            f"idle CPU {base['mean']:.1f}% > 20%; close other apps and rerun"
        )
    if base["mean"] > 10.0:
        print(f"[warn] idle CPU {base['mean']:.1f}% (10-20%); "
              f"parallel_eff_net will be reported too", flush=True)
    return env


def update_status(status_path: Path, state: dict, **changes) -> None:
    state.update(changes)
    state["updated_at"] = iso_now()
    atomic_write_json(status_path, state)


def write_failure(out_dir: Path, status: dict, step: str, cmd: list[str],
                  returncode: int, log_path: Path | None) -> None:
    tail = ""
    if log_path is not None and log_path.exists():
        try:
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            tail = "\n".join(lines[-60:])
        except OSError as exc:
            tail = f"(could not read log: {exc})"
    text = [
        "# Scaling study — FAILURE",
        "",
        f"- time: {iso_now()}",
        f"- step: `{step}`",
        f"- returncode: `{returncode}`",
        f"- command: `{' '.join(cmd)}`",
        f"- log: `{log_path}`",
        "",
        "## Last 60 log lines",
        "```",
        tail,
        "```",
    ]
    (out_dir / "FAILURE.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def run_step(out_dir: Path, status_path: Path, status: dict, index: int,
             total: int, name: str, cmd: list[str], python: str,
             log_dir: Path, sandbox_temp: bool) -> None:
    log_path = log_dir / f"{index:02d}_{name}.log"
    update_status(status_path, status, status="running", step=name,
                  step_index=index, total_steps=total)
    print(f"[step {index}/{total}] {name}", flush=True)
    print(f"  cmd: {' '.join(cmd)}", flush=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYSPARK_PYTHON"] = python
    env["PYSPARK_DRIVER_PYTHON"] = python
    if sandbox_temp:
        env["SCALING_SANDBOX_TEMP"] = "1"
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), stdout=log,
                              stderr=subprocess.STDOUT, env=env)
    status.setdefault("commands", []).append({
        "step": name, "cmd": cmd, "returncode": proc.returncode,
        "log": str(log_path), "finished_at": iso_now(),
    })
    if proc.returncode != 0:
        write_failure(out_dir, status, name, cmd, proc.returncode, log_path)
        update_status(status_path, status, status="failed", step=name,
                      finished_at=iso_now(), error=f"{name} exit {proc.returncode}")
        raise SystemExit(proc.returncode)
    print(f"[step {index}/{total}] {name} OK", flush=True)


def check_smoke(out_dir: Path) -> dict:
    csv_path = out_dir / "smoke" / "s2_k1.csv"
    if not csv_path.exists():
        raise PreflightError(f"smoke csv missing: {csv_path}")
    df = pd.read_csv(csv_path)
    if df.empty:
        raise PreflightError("smoke csv is empty")
    measured = df[~df["is_warmup"].astype(bool)]
    if measured.empty:
        raise PreflightError("smoke has no measured run")
    n_rows = sorted(df["n_rows"].unique().tolist())
    fit_min = float(measured["fit_s"].min())
    fit_max = float(measured["fit_s"].max())
    eff = measured["parallel_eff"].tolist()
    checks = {
        "rows": n_rows,
        "measured_runs": int(len(measured)),
        "fit_s_min": fit_min,
        "fit_s_max": fit_max,
        "parallel_eff": eff,
        "pass": bool(
            n_rows == [2_000_000]
            and len(measured) >= 1
            and 5.0 <= fit_min <= 300.0
            and 5.0 <= fit_max <= 300.0
            and all(0.05 <= x <= 2.5 for x in eff)
        ),
    }
    atomic_write_json(out_dir / "smoke_check.json", checks)
    if not checks["pass"]:
        raise PreflightError(f"smoke check failed: {checks}")
    return checks


def write_manifest(out_dir: Path) -> dict:
    files: dict[str, dict] = {}
    for path in sorted(out_dir.rglob("*")):
        if not path.is_file() or path.name in {"MANIFEST.json", "run_status.json"}:
            continue
        rel = str(path.relative_to(out_dir)).replace("\\", "/")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        files[rel] = {"bytes": path.stat().st_size, "sha256": digest}
    manifest = {"generated_at": iso_now(), "root": str(out_dir), "files": files}
    atomic_write_json(out_dir / "MANIFEST.json", manifest)
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--out-dir", default=str(OUT_ROOT))
    ap.add_argument("--include-local1", action="store_true")
    ap.add_argument("--skip-smoke", action="store_true")
    ap.add_argument("--skip-s3", action="store_true")
    ap.add_argument("--only", choices=["all", "k2"], default="all",
                    help="k2 = formal K2-only run (prepare_2m, s2_k2, plots, bootstrap)")
    ap.add_argument("--repeats-k1", type=int, default=5)
    ap.add_argument("--repeats-k2", type=int, default=3)
    ap.add_argument("--repeats-f1", type=int, default=5)
    ap.add_argument("--k2-order", choices=["sequential", "random"],
                    default="sequential")
    ap.add_argument("--analysis-only", action="store_true",
                    help="regenerate plots/checks/bootstrap from persisted CSVs")
    ap.add_argument("--sandbox-temp", action="store_true",
                    help="opt-in DSH sandbox temp-dir workaround")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out_dir = resolve_path(args.out_dir)
    ensure_dir(out_dir)
    log_dir = ensure_dir(LOG_DIR / f"run_{time.strftime('%Y%m%d_%H%M%S')}")
    status_path = out_dir / "run_status.json"
    python = args.python

    if status_path.exists() and not args.force:
        try:
            old = json.loads(status_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            old = {}
        if old.get("status") == "running" and old.get("pid") and psutil.pid_exists(old["pid"]):
            raise SystemExit(f"another run is active (pid {old['pid']}); use --force")

    steps: list[tuple[str, list[str]]] = []
    if args.analysis_only:
        steps = [
            ("plots", [python, "-u", str(SCRIPT_DIR / "make_scaling_plots.py"),
                       "--in-dir", str(out_dir), "--out-dir", str(out_dir)]),
            ("bootstrap", [python, "-u", str(SCRIPT_DIR / "bootstrap_ci.py"),
                           "--in-dir", str(out_dir), "--out",
                           str(out_dir / "bootstrap_ci.json")]),
        ]
    elif args.only == "k2":
        steps = [
            ("prepare_2m", [python, "-u", str(SCRIPT_DIR / "prepare_subset.py"),
                            "--rows", "2000000", "--out", str(SUBSET_2M),
                            "--if-exists", "skip"]),
            ("s2_k2", [
                python, "-u", str(SCRIPT_DIR / "run_kmeans_scaling.py"),
                "--job", "k2", "--rows", "2000000", "--subset", str(SUBSET_2M),
                "--cores", "2,4,8", "--shuffle", "30,120",
                "--repeats", str(args.repeats_k2), "--warmup", "1",
                "--order", args.k2_order, "--order-seed", "42",
                "--baseline", "2", "--out-dir", str(out_dir), "--resume",
            ]),
            ("plots", [python, "-u", str(SCRIPT_DIR / "make_scaling_plots.py"),
                       "--in-dir", str(out_dir), "--out-dir", str(out_dir)]),
            ("bootstrap", [python, "-u", str(SCRIPT_DIR / "bootstrap_ci.py"),
                           "--in-dir", str(out_dir), "--out",
                           str(out_dir / "bootstrap_ci.json")]),
        ]
    else:
        steps = [
            ("prepare_2m", [python, "-u", str(SCRIPT_DIR / "prepare_subset.py"),
                            "--rows", "2000000", "--out", str(SUBSET_2M),
                            "--if-exists", "skip"]),
            ("prepare_1m", [python, "-u", str(SCRIPT_DIR / "prepare_subset.py"),
                            "--rows", "1000000", "--from-subset", str(SUBSET_2M),
                            "--out", str(SUBSET_1M), "--if-exists", "skip"]),
        ]
        if not args.skip_smoke:
            steps.append(("smoke", [
                python, "-u", str(SCRIPT_DIR / "run_kmeans_scaling.py"),
                "--job", "k1", "--rows", "2000000", "--subset", str(SUBSET_2M),
                "--cores", "4", "--shuffle", "120", "--repeats", "2",
                "--warmup", "1", "--baseline", "2",
                "--out-dir", str(out_dir / "smoke"),
            ]))
        steps.append(("s2_k1", [
            python, "-u", str(SCRIPT_DIR / "run_kmeans_scaling.py"),
            "--job", "k1", "--rows", "2000000", "--subset", str(SUBSET_2M),
            "--cores", "2,4,8", "--shuffle", "30,120",
            "--repeats", str(args.repeats_k1), "--warmup", "1",
            "--baseline", "2", "--out-dir", str(out_dir), "--resume",
        ]))
        if args.include_local1:
            steps.append(("s2_k1_local1", [
                python, "-u", str(SCRIPT_DIR / "run_kmeans_scaling.py"),
                "--job", "k1", "--rows", "2000000", "--subset", str(SUBSET_2M),
                "--cores", "1", "--shuffle", "120", "--repeats", "3",
                "--warmup", "1", "--baseline", "2", "--out-dir", str(out_dir),
                "--resume",
            ]))
        steps.append(("s2_k2", [
            python, "-u", str(SCRIPT_DIR / "run_kmeans_scaling.py"),
            "--job", "k2", "--rows", "2000000", "--subset", str(SUBSET_2M),
            "--cores", "2,4,8", "--shuffle", "30,120",
            "--repeats", str(args.repeats_k2), "--warmup", "1",
            "--order", args.k2_order, "--order-seed", "42",
            "--baseline", "2", "--out-dir", str(out_dir), "--resume",
        ]))
        if not args.skip_s3:
            steps.append(("s3_f1", [
                python, "-u", str(SCRIPT_DIR / "run_if_njobs.py"),
                "--subset", str(SUBSET_1M), "--njobs", "1,2,4,8,-1",
                "--repeats", str(args.repeats_f1), "--warmup", "1",
                "--out-dir", str(out_dir), "--resume",
            ]))
        steps.extend([
            ("plots", [python, "-u", str(SCRIPT_DIR / "make_scaling_plots.py"),
                       "--in-dir", str(out_dir), "--out-dir", str(out_dir)]),
            ("bootstrap", [python, "-u", str(SCRIPT_DIR / "bootstrap_ci.py"),
                           "--in-dir", str(out_dir), "--out",
                           str(out_dir / "bootstrap_ci.json")]),
        ])

    total = len(steps)
    if args.dry_run:
        print(json.dumps({
            "python": python, "out_dir": str(out_dir),
            "analysis_only": bool(args.analysis_only),
            "only": args.only,
            "steps": [{"index": i + 1, "name": n, "cmd": c}
                      for i, (n, c) in enumerate(steps)],
            "total_steps": total,
        }, ensure_ascii=False, indent=2))
        return 0

    status = {
        "status": "running", "step": "preflight" if not args.analysis_only else "analysis",
        "step_index": 0, "total_steps": total, "started_at": iso_now(),
        "updated_at": iso_now(), "finished_at": None, "pid": os.getpid(),
        "out_dir": str(out_dir), "commands": [], "error": None,
    }
    update_status(status_path, status)

    with keep_awake():
        try:
            if not args.analysis_only:
                env = preflight(python, out_dir)
                status["env"] = {k: env[k] for k in
                                 ["python", "pyspark", "physical_cores",
                                  "logical_cores", "ram_gb", "battery_plugged",
                                  "temp_free_gb", "input_parts",
                                  "baseline_cpu_pct"]}
                update_status(status_path, status, step="prepare_2m", step_index=1)

            for index, (name, cmd) in enumerate(steps, start=1):
                run_step(out_dir, status_path, status, index, total, name, cmd,
                         python, log_dir, args.sandbox_temp)
                if name == "smoke":
                    check_smoke(out_dir)

            write_manifest(out_dir)
            checks_path = out_dir / "checks.json"
            checks = json.loads(checks_path.read_text(encoding="utf-8")) if checks_path.exists() else {}
            final_status = "done" if checks.get("pass") else "done_with_warnings"
            update_status(status_path, status, status=final_status, step="finished",
                          finished_at=iso_now(), checks_pass=bool(checks.get("pass")))
            print(f"[run] {final_status}; checks_pass={checks.get('pass')}", flush=True)
            return 0
        except PreflightError as exc:
            write_failure(out_dir, status, status.get("step", "preflight"),
                          ["preflight"], 2, None)
            update_status(status_path, status, status="failed", step="preflight",
                          finished_at=iso_now(), error=str(exc))
            print(f"[preflight] FAILED: {exc}", flush=True)
            return 2
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 1
            update_status(status_path, status, status="failed",
                          finished_at=iso_now(), error=f"exit {code}")
            print(f"[run] FAILED with exit {code}", flush=True)
            return code
        except Exception as exc:  # noqa: BLE001
            write_failure(out_dir, status, status.get("step", "unknown"),
                          ["run_scaling_study"], 1, None)
            update_status(status_path, status, status="failed",
                          finished_at=iso_now(), error=f"{type(exc).__name__}: {exc}")
            print(f"[run] FAILED: {exc}", flush=True)
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
