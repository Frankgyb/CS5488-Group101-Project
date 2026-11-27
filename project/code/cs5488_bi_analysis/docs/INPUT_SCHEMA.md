# Required aggregate inputs

The workflow expects the group's **compact daily tables**, not raw monthly TLC trip records:

```text
bi_aggregates_compact/
├── l1_detail/
│   ├── pickup_date=2023-01-01/<one-file>.parquet
│   ├── ...
│   └── pickup_date=2024-12-31/<one-file>.parquet
└── l2_overview/
    ├── pickup_date=2023-01-01/<one-file>.parquet
    ├── ...
    └── pickup_date=2024-12-31/<one-file>.parquet
```

There must be 731 daily files in each table. Each Parquet file must also contain `pickup_date` as a column. The scripts read file contents without re-inferencing a duplicate partition field.

## Shared columns

| Column | Meaning |
| --- | --- |
| `pickup_date` | Calendar date, consistent with the enclosing partition |
| `pickup_hour` | Integer 0–23 |
| `PULocationID` | TLC pickup-zone identifier |
| `trip_count` | Positive number of recorded trips represented by the group |
| `total_fare` | Sum of base passenger fares |
| `total_miles` | Sum of recorded trip miles |
| `total_tips` | Sum of recorded tips |
| `avg_wait_minutes` | Stored group mean; audited, not used as an unweighted mean |

L1 additionally requires:

| Column | Meaning |
| --- | --- |
| `hvfhs_license_num` | `HV0003` or `HV0005` |
| `valid_wait_count` | Count of records with wait in 0–120 minutes, inclusive |
| `valid_fare_count` | Count passing the nonnegative base-fare rule |
| `valid_trip_count` | Count passing the source trip-validity rule |
| `total_valid_wait_minutes` | Sum of wait minutes over valid-wait records only |

L1 grouping keys are date, hour, zone and platform. L2 omits platform. L1 and L2 counts must reconcile. Monetary and distance sums are audited with floating-point differences recorded separately.

The initial handoff did not contain `total_valid_wait_minutes`; its average could include invalid waits. **Use the corrected L1 delivery.** A count and an unfiltered mean are insufficient to reconstruct the sum over valid records. The workflow does not fabricate that missing numerator.

Stage 01 can report invalid stored L2 mean waits as historical quality diagnostics. Formal waiting-time charts use corrected L1. Stage 06 additionally checks that stored L1 means agree with valid sums/counts; see its generated `hourly_wait/audit_summary.json`.

## Fixed reconciliation totals

| Quantity | Expected value |
| --- | ---: |
| Period | 2023-01-01 to 2024-12-31 |
| Calendar days | 731 |
| Total recorded trips | 471,960,468 |
| L1 aggregate groups | 8,651,895 |
| L2 aggregate groups | 4,421,839 |
| Valid waiting-time records | 467,359,108 |

The package contains only the small public taxi-zone lookup. Obtain the aggregate tables from the team's existing preprocessing outputs.
