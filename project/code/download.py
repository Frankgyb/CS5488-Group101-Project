"""Download the 24 monthly FHVHV parquet files into project/data/parquet/.

Location: project/code/download.py | PROJECT_ROOT depth: 1
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import requests


# =========================
# Configuration
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]      # .../project
OUTPUT_DIR = PROJECT_ROOT / "data" / "parquet"

BASE_URL = (
    "https://d37ci6vzurychx.cloudfront.net/trip-data/"
    "fhvhv_tripdata_{year}-{month:02d}.parquet"
)

YEARS = [2023, 2024]
MONTHS = list(range(1, 13))

# Set to True to test the network with only the first file
TEST_ONLY_FIRST = False

CHUNK_SIZE = 8 * 1024 * 1024       # 8 MB
MAX_RETRIES = 4
TIMEOUT = (15, 120)                # connect timeout, read timeout


def human_size(num_bytes: Optional[int]) -> str:
    if num_bytes is None:
        return "unknown size"
    size = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024.0 or unit == "TB":
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{num_bytes} B"


def get_remote_size(session: requests.Session, url: str) -> Optional[int]:
    """Return the remote file size via HTTP HEAD, or None if unavailable."""
    try:
        resp = session.head(url, allow_redirects=True, timeout=(10, 30))
        if resp.status_code == 200:
            content_length = resp.headers.get("Content-Length")
            if content_length and content_length.isdigit():
                return int(content_length)
        print(f"  [WARN] HEAD did not return size, HTTP {resp.status_code}")
    except Exception as exc:
        print(f"  [WARN] HEAD failed: {exc}")
    return None


def download_one(session: requests.Session, url: str, dest: Path) -> bool:
    """Download one file with .part resume support and retries."""
    expected_size = get_remote_size(session, url)

    # Skip if the file already exists with a matching size
    if dest.exists():
        local_size = dest.stat().st_size
        if expected_size is not None and local_size == expected_size:
            print(f"  [SKIP] Exists and size matches: {dest.name} ({human_size(local_size)})")
            return True
        if expected_size is None and local_size > 0:
            print(f"  [SKIP] Exists but size cannot be verified: {dest.name} ({human_size(local_size)})")
            return True
        print(
            f"  [INFO] Size mismatch, re-downloading: "
            f"{dest.name} ({human_size(local_size)} != {human_size(expected_size)})"
        )

    part_path = dest.parent / (dest.name + ".part")

    # Promote a complete .part file directly to the final name
    if part_path.exists() and expected_size is not None:
        part_size = part_path.stat().st_size
        if part_size == expected_size:
            part_path.replace(dest)
            print(f"  [DONE] Promoted existing .part file: {dest.name}")
            return True
        if part_size > expected_size:
            print(f"  [INFO] .part is larger than expected, deleting: {part_path.name}")
            part_path.unlink()

    for attempt in range(1, MAX_RETRIES + 1):
        existing_size = part_path.stat().st_size if part_path.exists() else 0

        headers = {}
        mode = "wb"

        if existing_size > 0:
            headers["Range"] = f"bytes={existing_size}-"
            mode = "ab"
            print(f"  [RESUME] From {human_size(existing_size)}: {dest.name}")
        else:
            print(f"  [DOWNLOAD] {dest.name}")

        try:
            with session.get(
                url,
                stream=True,
                headers=headers,
                timeout=TIMEOUT,
            ) as resp:
                # HTTP 416 means the requested range is invalid; usually .part is complete or corrupt
                if existing_size > 0 and resp.status_code == 416:
                    if expected_size is not None and existing_size == expected_size:
                        part_path.replace(dest)
                        print(f"  [DONE] {dest.name}")
                        return True
                    print("  [INFO] HTTP 416 returned, deleting corrupt .part and retrying")
                    part_path.unlink(missing_ok=True)
                    raise IOError("HTTP 416, .part deleted")

                # Server ignored the Range header and returned the whole file
                if existing_size > 0 and resp.status_code == 200:
                    print("  [INFO] Server does not support resume, restarting from zero")
                    existing_size = 0
                    mode = "wb"

                resp.raise_for_status()

                total_size = expected_size
                if total_size is None:
                    content_length = resp.headers.get("Content-Length")
                    if content_length and content_length.isdigit():
                        cl = int(content_length)
                        total_size = existing_size + cl if resp.status_code == 206 else cl

                downloaded = existing_size
                start_time = time.time()
                last_report_time = start_time

                with open(part_path, mode) as f:
                    for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                        if not chunk:
                            continue

                        f.write(chunk)
                        downloaded += len(chunk)

                        now = time.time()
                        if now - last_report_time >= 5:
                            elapsed = max(now - start_time, 0.001)
                            speed = downloaded / elapsed
                            if total_size:
                                percent = downloaded / total_size * 100
                                print(
                                    f"    {percent:6.2f}%  "
                                    f"{human_size(downloaded)} / {human_size(total_size)}  "
                                    f"{human_size(int(speed))}/s"
                                )
                            else:
                                print(
                                    f"    {human_size(downloaded)}  "
                                    f"{human_size(int(speed))}/s"
                                )
                            last_report_time = now

                # Final size check before renaming
                actual_size = part_path.stat().st_size
                if total_size is not None and actual_size != total_size:
                    raise IOError(
                        f"Size check failed: local {actual_size} bytes, "
                        f"remote {total_size} bytes"
                    )

                part_path.replace(dest)
                print(f"  [DONE] {dest.name} ({human_size(actual_size)})")
                return True

        except Exception as exc:
            print(f"  [RETRY {attempt}/{MAX_RETRIES}] {dest.name} failed: {exc}")
            if attempt == MAX_RETRIES:
                print(f"  [FAILED] {dest.name}, keeping .part for next run")
                return False
            sleep_seconds = 2 ** attempt
            print(f"  [WAIT] Retrying in {sleep_seconds}s...")
            time.sleep(sleep_seconds)

    return False


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tasks = []
    for year in YEARS:
        for month in MONTHS:
            url = BASE_URL.format(year=year, month=month)
            filename = f"fhvhv_tripdata_{year}-{month:02d}.parquet"
            tasks.append((url, OUTPUT_DIR / filename))

    if TEST_ONLY_FIRST:
        tasks = tasks[:1]
        print("[TEST MODE] Only the first file will be downloaded\n")

    print("=" * 80)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Number of files: {len(tasks)}")
    print("=" * 80)

    session = requests.Session()
    session.headers.update({
        "User-Agent": "CS5488-Group101-Downloader/1.0"
    })

    success = 0
    failed_files = []

    try:
        for index, (url, dest) in enumerate(tasks, start=1):
            print(f"\n[{index}/{len(tasks)}] {dest.name}")
            print(f"  URL: {url}")

            if download_one(session, url, dest):
                success += 1
            else:
                failed_files.append(dest.name)

    finally:
        session.close()

    print("\n" + "=" * 80)
    print(f"Finished: {success}/{len(tasks)} succeeded")
    if failed_files:
        print("Failed files:")
        for name in failed_files:
            print(f"  - {name}")
        print("Re-run the script to resume from .part files.")
        sys.exit(1)
    else:
        print("All files downloaded successfully.")


if __name__ == "__main__":
    main()