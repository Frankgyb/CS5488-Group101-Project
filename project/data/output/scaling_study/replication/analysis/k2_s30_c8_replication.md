# K2 S=30 local[8] replication

- Generated: 2026-11-25T18:27:47+08:00
- Trigger: the primary n=4 run had one high-leverage 143.793 s point.
- Primary is unchanged; this is a post-hoc replication, not a replacement.

## Summary

| dataset | n | fit_s_median | ci95_low | ci95_high | fit_s_min | fit_s_max | fit_s_mad | speedup_vs_primary_local2 | speedup_ci95_low | speedup_ci95_high | max_over_median | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| primary | 4 | 98.448 | 93.818 | 143.793 | 93.818 | 143.793 | 2.424 | 1.792 | 1.227 | 2.221 | 1.461 | original n=4; main result |
| replication | 4 | 108.435 | 90.849 | 124.752 | 90.849 | 124.752 | 12.097 | — | — | — | 1.15 | post-hoc; no contemporaneous local[2] baseline |
| pooled_sensitivity | 8 | 99.612 | 93.818 | 124.752 | 90.849 | 143.793 | 7.278 | 1.771 | 1.414 | 2.195 | 1.444 | unbalanced: only this cell has n=8 |

## Raw measured runs

| batch | run_idx | fit_s | cpu_pct_mean | wssse | n_iter |
| --- | --- | --- | --- | --- | --- |
| primary | 1 | 93.818 | 7.35 | 9404845.652462509 | 20 |
| primary | 2 | 98.666 | 6.77 | 9404845.652462509 | 20 |
| primary | 3 | 98.231 | 6.8 | 9404845.652462509 | 20 |
| primary | 4 | 143.793 | 10.21 | 9404845.652462509 | 20 |
| replication | 1 | 90.849 | 15.45 | 9404845.652462509 | 20 |
| replication | 2 | 124.752 | 16.95 | 9404845.652462509 | 20 |
| replication | 3 | 116.312 | 16.65 | 9404845.652462509 | 20 |
| replication | 4 | 100.558 | 17.93 | 9404845.652462509 | 20 |

## Reading

- primary: 4 runs, median 98.448 s, bootstrap 95% CI [93.818, 143.793].
- replication: 4 runs, median 108.435 s, bootstrap 95% CI [90.849, 124.752]; median ratio vs primary 1.101 (+10.1%).
- pooled n=8 sensitivity: median 99.612 s, CI [93.818, 124.752]; speed-up vs primary local[2] 1.771× [1.414, 2.195].
- Speed-up CIs resample the primary local[2] values as well as the 8-core values; they therefore match the method in `bootstrap_ci.json`, not a fixed-baseline conditional interval.
- replication max/median = 1.150; a new >1.3× slow tail is not present.

Figure: `data/output/scaling_study/replication/analysis/figures/k2_s30_c8_replication.png`

The pooled row is unbalanced and is not a new main result. The primary n=4 table remains the primary result; the replication answers only whether the 143.8 s point recurs in the same cell.
