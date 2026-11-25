# Location: project/code/ml/experiments/parallel_scaling/scaling_common.py | PROJECT_ROOT depth: 4
"""Shared helpers for the v3 single-machine scaling study.

This module is imported by the experiment scripts only. It never writes outside
``data/output/scaling_study`` and ``data/log/scaling_study``.
"""
from __future__ import annotations

import contextlib
import ctypes
import json
import os
import secrets
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

import psutil

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[3]  # .../project/
INPUT_PATH = PROJECT_ROOT / "data" / "output" / "sampled_km"
OUT_ROOT = PROJECT_ROOT / "data" / "output" / "scaling_study"
SUBSET_DIR = OUT_ROOT / "subsets"
SUBSET_2M = SUBSET_DIR / "subset_2m"
SUBSET_1M = SUBSET_DIR / "subset_1m.parquet"
FIG_DIR = OUT_ROOT / "figures"
LOG_DIR = PROJECT_ROOT / "data" / "log" / "scaling_study"
SPARK_LOCAL_DIR = OUT_ROOT / "_spark_tmp"
SPARK_GATEWAY_BASE = OUT_ROOT / "_spark_gateway"

FEATURES = [
    "log_miles", "log_fare", "log_tips", "speed_mph", "log_duration",
    "hour_sin", "hour_cos", "day_sin", "day_cos",
]
SEED = 42
LOGICAL_CORES = psutil.cpu_count(logical=True) or 1
PHYSICAL_CORES = psutil.cpu_count(logical=False) or LOGICAL_CORES

_SANDBOX_PATCHED = False


def configure_sandbox_temp(base: Path | None = None) -> None:
    """Opt-in workaround for the restricted DSH Windows sandbox.

    Normal runs should leave this alone. Set ``SCALING_SANDBOX_TEMP=1`` when
    PySpark's java_gateway cannot create its own temp directory. The base
    directory must be writable (pre-create it if the sandbox requires that).
    """
    global _SANDBOX_PATCHED
    if _SANDBOX_PATCHED:
        return
    base = Path(base or SPARK_GATEWAY_BASE)
    base.mkdir(parents=True, exist_ok=True)

    def _sandbox_mkdtemp(prefix=None, dir=None, **kwargs):
        parent = Path(dir) if dir is not None else base
        for _ in range(10):
            path = parent / ((prefix or "tmp") + secrets.token_hex(4))
            try:
                os.mkdir(path)
                return str(path)
            except FileExistsError:
                continue
        raise RuntimeError(f"could not allocate a unique temp dir under {parent}")

    tempfile.mkdtemp = _sandbox_mkdtemp
    os.environ["TEMP"] = str(base)
    os.environ["TMP"] = str(base)
    tempfile.tempdir = str(base)
    _SANDBOX_PATCHED = True


def iso_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_path(value: str | Path) -> Path:
    """Resolve a CLI path relative to the project root (never the caller cwd)."""
    p = Path(value)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


# Backwards-compatible alias used by the scratch scripts.
resolve_here = resolve_path


def atomic_write_json(path: Path, obj) -> None:
    ensure_dir(path.parent)
    tmp = path.parent / (path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: Path, obj) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def count_jvm() -> int:
    names = {"java.exe", "javaw.exe"}
    n = 0
    for proc in psutil.process_iter(["name"]):
        try:
            if (proc.info.get("name") or "").lower() in names:
                n += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return n


def jvm_processes() -> list[dict]:
    names = {"java.exe", "javaw.exe"}
    out: list[dict] = []
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if (proc.info.get("name") or "").lower() in names:
                out.append({
                    "pid": proc.info["pid"],
                    "name": proc.info["name"],
                    "cmdline": " ".join(proc.info.get("cmdline") or [])[:300],
                })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return out


def baseline_cpu(seconds: float = 3.0) -> dict:
    psutil.cpu_percent(interval=None)
    time.sleep(0.1)
    vals: list[float] = []
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        vals.append(psutil.cpu_percent(interval=None))
        time.sleep(0.1)
    vals = vals[2:] or vals or [0.0]
    return {
        "mean": float(statistics.fmean(vals)),
        "max": float(max(vals)),
        "samples": len(vals),
        "seconds": seconds,
    }


class CpuSampler:
    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.samples: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.mean = 0.0
        self.pct_max = 0.0

    def __enter__(self):
        psutil.cpu_percent(interval=None)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            self.samples.append(psutil.cpu_percent(interval=None))
            time.sleep(self.interval)

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        vals = self.samples[2:] or self.samples or [0.0]
        self.mean = float(statistics.fmean(vals))
        self.pct_max = float(max(vals))
        return False


class ThreadSampler:
    def __init__(self, interval: float = 0.05):
        self.interval = interval
        self.samples: list[int] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.max_threads = 0

    def __enter__(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self):
        proc = psutil.Process()
        while not self._stop.is_set():
            try:
                self.samples.append(proc.num_threads())
            except psutil.Error:
                pass
            time.sleep(self.interval)

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.max_threads = max(self.samples) if self.samples else 0
        return False


def parallel_eff(cpu_pct_mean: float, n_units: float, baseline_pct: float = 0.0) -> tuple[float, float]:
    if n_units <= 0:
        return 0.0, 0.0
    eff = (cpu_pct_mean / 100.0) * LOGICAL_CORES / n_units
    net = (max(0.0, cpu_pct_mean - baseline_pct) / 100.0) * LOGICAL_CORES / n_units
    return float(eff), float(net)


@contextlib.contextmanager
def keep_awake():
    """Prevent Windows idle sleep while a long experiment is running."""
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    if os.name != "nt":
        yield
        return
    kernel32 = ctypes.windll.kernel32
    kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    try:
        yield
    finally:
        kernel32.SetThreadExecutionState(ES_CONTINUOUS)


def build_spark(master: str, shuffle: int, app_tag: str = "S2"):
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    if os.environ.get("SCALING_SANDBOX_TEMP") == "1":
        configure_sandbox_temp()
    ensure_dir(SPARK_LOCAL_DIR)
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder
        .appName(f"Scaling-{app_tag}-{master}-S{shuffle}")
        .master(master)
        .config("spark.local.dir", str(SPARK_LOCAL_DIR))
        .config("spark.sql.shuffle.partitions", str(shuffle))
        .config("spark.default.parallelism", str(shuffle))
        .config("spark.driver.memory", "8g")
        .config("spark.driver.memoryOverhead", "2g")
        .config("spark.memory.fraction", "0.6")
        .config("spark.memory.storageFraction", "0.3")
        .config("spark.memory.offHeap.enabled", "false")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.executor.extraJavaOptions", "-XX:+UseG1GC")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def new_spark(master: str, shuffle: int, app_tag: str = "S2"):
    """Build a session and make sure getOrCreate did not reuse a stale one."""
    spark = build_spark(master, shuffle, app_tag)
    if spark.sparkContext.master != master:
        spark.stop()
        time.sleep(1.0)
        spark = build_spark(master, shuffle, app_tag)
    if spark.sparkContext.master != master:
        raise RuntimeError(
            f"Spark master is {spark.sparkContext.master!r}, expected {master!r}"
        )
    return spark


def powercfg_summary() -> dict:
    out: dict[str, str] = {}
    commands = {
        "active_scheme": ["powercfg", "/getactivescheme"],
        "standby_idle": ["powercfg", "/query", "SCHEME_CURRENT", "SUB_SLEEP", "STANDBYIDLE"],
        "monitor_idle": ["powercfg", "/query", "SCHEME_CURRENT", "SUB_VIDEO", "VIDEOIDLE"],
    }
    for key, args in commands.items():
        try:
            r = subprocess.run(
                args, capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=15,
            )
            out[key] = (r.stdout or "") + (r.stderr or "")
        except Exception as exc:  # noqa: BLE001
            out[key] = f"ERROR: {exc}"
    return out


def list_part_files(path: Path) -> list[Path]:
    return sorted(path.glob("part-*.parquet"))


def snapshot_tree(paths: list[Path]) -> dict:
    snap: dict[str, dict] = {}
    for root in paths:
        if not root.exists():
            snap[str(root)] = {"exists": False}
            continue
        for p in root.rglob("*"):
            if p.is_file():
                try:
                    st = p.stat()
                    snap[str(p)] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns}
                except OSError:
                    continue
    return snap


def diff_snapshots(before: dict, after: dict) -> dict:
    added = sorted(set(after) - set(before))
    removed = sorted(set(before) - set(after))
    changed = sorted(k for k in set(before) & set(after) if before[k] != after[k])
    return {"added": added, "removed": removed, "changed": changed}
