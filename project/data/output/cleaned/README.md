# cleaned/ — not tracked

The partitioned cleaning output: 75 Parquet files under `pickup_year=*/pickup_month=*/`,
16 GB total. It is the input of every downstream stage and is excluded from version control;
this README keeps the folder visible.

Rebuild:

    python project/code/preprocessing/preprocessing.py

See [`../../README.md`](../../README.md) for the column groups and partition layout.
