"""
compact_bi_partitions.py
Purpose: Reduce number of small files per partition while preserving daily partitioning.
Input:  data/output/bi_aggregates/l1_detail/ and l2_overview/
Output: data/output/bi_aggregates_compact/l1_detail/ and l2_overview/
Each daily partition will contain exactly one Parquet file (instead of ~80).
Compression: zstd for smaller size.
"""

import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

# ---------- Configuration ----------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

BASE_INPUT = str(PROJECT_ROOT / "data" / "output" / "bi_aggregates")
BASE_OUTPUT = str(PROJECT_ROOT / "data" / "output" / "bi_aggregates_compact")

TABLES = ["l1_detail", "l2_overview"]


# ---------- Helper function ----------
def compact_table(table_name):
    input_dir = os.path.join(BASE_INPUT, table_name)
    output_dir = os.path.join(BASE_OUTPUT, table_name)

    print(f"\nProcessing {table_name}...")
    print(f"Reading from: {input_dir}")

    # Read entire table (PyArrow automatically handles many small files)
    dataset = pq.ParquetDataset(input_dir)
    table = dataset.read()

    # Get partition column name (should be 'pickup_date')
    # We'll group by pickup_date manually
    pdf = table.to_pandas()
    print(f"  Total rows: {len(pdf):,}")

    # Ensure pickup_date is present
    if "pickup_date" not in pdf.columns:
        print("  ERROR: 'pickup_date' column not found!")
        return

    # Group by pickup_date and write each group as a single file
    unique_dates = pdf["pickup_date"].unique()
    print(f"  Unique dates: {len(unique_dates)}")

    for i, date_val in enumerate(sorted(unique_dates)):
        subset = pdf[pdf["pickup_date"] == date_val]

        # Build output path: output_dir/pickup_date=YYYY-MM-DD/part-00001.parquet
        partition_dir = os.path.join(output_dir, f"pickup_date={date_val}")
        os.makedirs(partition_dir, exist_ok=True)
        out_file = os.path.join(partition_dir, "part-00001.parquet")

        # Convert back to PyArrow table and write with zstd compression
        sub_table = pa.Table.from_pandas(subset, preserve_index=False)
        pq.write_table(sub_table, out_file, compression="zstd")

        if (i + 1) % 100 == 0:
            print(f"  Progress: {i + 1}/{len(unique_dates)} dates written")

    # Count final files
    total_files = sum(len(files) for _, _, files in os.walk(output_dir) if any(f.endswith('.parquet') for f in files))
    print(f"  Done! Output directory: {output_dir}")
    print(f"  Total files now: {total_files} (was ~{len(unique_dates) * 80})")


# ---------- Main ----------
if __name__ == "__main__":
    for tbl in TABLES:
        compact_table(tbl)
    print("\nAll tables compacted successfully.")
    print(f"Compact output root: {BASE_OUTPUT}")