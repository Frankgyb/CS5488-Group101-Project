"""Audit the supplied compact L1/L2 tables; no raw trip data is modified."""

from _config import PROJECT_ROOT, DATA_ROOT, OUTPUT_ROOT, LOOKUP_FILE
from pathlib import Path
from collections import Counter
from datetime import date, timedelta
import sys
import json
import time

import pyarrow.parquet as pq
import pandas as pd
import numpy as np

ROOT = DATA_ROOT
OUT = OUTPUT_ROOT / "data_quality/aggregate_audit.json"
EXPECTED = {(date(2023, 1, 1) + timedelta(days=i)).isoformat() for i in range(731)}
ADDITIVE = ['trip_count', 'total_fare', 'total_miles', 'total_tips']
KEY = ['pickup_hour', 'PULocationID']
def main():
    report = {'input_path': str(ROOT), 'audit_scope': 'All supplied Parquet files; no original trip records or producer scripts supplied.', 'tables': {}, 'reconciliation': {}, 'daily': []}
    stats = {}
    for name in ['l1_detail', 'l2_overview']:
        files = sorted((ROOT/name).rglob('*.parquet'))
        stats[name] = {
            'files': len(files), 'bytes': sum(p.stat().st_size for p in files),
            'rows': 0, 'trip_count': 0, 'nulls': Counter(), 'valid_counts': Counter(),
            'license_trip_counts': Counter(), 'location_ids': set(), 'schema_variants': set(),
            'duplicate_keys': 0, 'nonpositive_trip_count_groups': 0,
            'invalid_valid_count_groups': 0, 'invalid_hour_groups': 0,
            'negative_mean_wait_groups': 0, 'over_120_mean_wait_groups': 0,
            'nonfinite_nonnull_wait_groups': 0, 'all_wait_invalid_groups': 0,
            'mean_wait_min': float('inf'), 'mean_wait_max': -float('inf'),
            'mean_wait_extreme_examples': [], 'folder_date_mismatch_rows': 0,
            'missing_dates': sorted(EXPECTED-{p.parent.name.split('=',1)[1] for p in files}),
            'unexpected_dates': sorted({p.parent.name.split('=',1)[1] for p in files}-EXPECTED),
        }

    count_mismatch_groups = 0
    missing_group_keys = 0
    sum_max_diff = Counter()
    started = time.perf_counter()
    for i, day in enumerate(sorted(EXPECTED)):
        frames = {}
        day_report = {'date': day}
        for name in stats:
            day_files = sorted((ROOT/name/f'pickup_date={day}').glob('*.parquet'))
            if len(day_files) != 1:
                raise ValueError(f'Expected one compact Parquet file for {name}/{day}; found {len(day_files)}')
            path = day_files[0]
            pf = pq.ParquetFile(path)
            table = pf.read()
            df = table.to_pandas()
            s = stats[name]
            s['rows'] += len(df)
            s['schema_variants'].add(str(pf.schema_arrow.remove_metadata()))
            s['columns'] = list(df.columns)
            s['nulls'].update({c:int(df[c].isna().sum()) for c in df})
            s['trip_count'] += int(df.trip_count.sum())
            s['location_ids'].update(int(x) for x in df.PULocationID.dropna().unique())
            s['folder_date_mismatch_rows'] += int((df.pickup_date.astype(str)!=day).sum())
            s['invalid_hour_groups'] += int((~df.pickup_hour.between(0,23)).sum())
            s['nonpositive_trip_count_groups'] += int((df.trip_count<=0).sum())
            keys = KEY + (['hvfhs_license_num'] if name=='l1_detail' else [])
            s['duplicate_keys'] += int(df.duplicated(keys).sum())
            wait = df.avg_wait_minutes
            s['negative_mean_wait_groups'] += int((wait<0).sum())
            s['over_120_mean_wait_groups'] += int((wait>120).sum())
            s['nonfinite_nonnull_wait_groups'] += int((wait.notna() & ~np.isfinite(wait)).sum())
            s['mean_wait_min'] = min(s['mean_wait_min'],float(wait.min()))
            s['mean_wait_max'] = max(s['mean_wait_max'],float(wait.max()))
            if len(s['mean_wait_extreme_examples']) < 3:
                extreme = df[(wait<0)|(wait>120)]
                for _,row in extreme.head(3-len(s['mean_wait_extreme_examples'])).iterrows():
                    s['mean_wait_extreme_examples'].append({'date':day,'hour':int(row.pickup_hour),'zone':int(row.PULocationID),'trip_count':int(row.trip_count),'avg_wait_minutes':float(row.avg_wait_minutes)})
            day_report[name+'_trip_count'] = int(df.trip_count.sum())
            if name=='l1_detail':
                for c in ['valid_wait_count','valid_fare_count','valid_trip_count']:
                    s['valid_counts'][c] += int(df[c].sum())
                    s['invalid_valid_count_groups'] += int(((df[c]<0)|(df[c]>df.trip_count)).sum())
                    day_report[c] = int(df[c].sum())
                s['all_wait_invalid_groups'] += int((df.valid_wait_count==0).sum())
                s['license_trip_counts'].update({str(k):int(v) for k,v in df.groupby('hvfhs_license_num',observed=True).trip_count.sum().items()})
            frames[name] = df
        a=frames['l1_detail'].groupby(KEY,dropna=False)[ADDITIVE].sum()
        b=frames['l2_overview'].set_index(KEY)[ADDITIVE]
        missing_group_keys += len(a.index.symmetric_difference(b.index))
        common=a.index.intersection(b.index)
        delta=(a.loc[common]-b.loc[common]).abs()
        count_mismatch_groups += int((delta.trip_count!=0).sum())
        for c in ADDITIVE:sum_max_diff[c]=max(sum_max_diff[c],float(delta[c].max()))
        report['daily'].append(day_report)
        if (i+1)%200==0:print(f'Audited {i+1}/731 dates',flush=True)

    for name,s in stats.items():
        s['location_ids']=sorted(s['location_ids'])
        s['schema_variants']=sorted(s['schema_variants'])
        s['valid_rates']={k:v/s['trip_count'] for k,v in s['valid_counts'].items()}
        report['tables'][name]=s
    report['reconciliation']={'missing_group_keys':missing_group_keys,'trip_count_mismatch_groups':count_mismatch_groups,'max_absolute_group_sum_differences':dict(sum_max_diff)}
    report['elapsed_seconds']=time.perf_counter()-started
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    summary={k:v for k,v in report.items() if k!='daily'}
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
