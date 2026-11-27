"""读取修正后的 L1，合并平台，以有效等待总和/有效记录数分析上车小时。"""

from _config import PROJECT_ROOT, DATA_ROOT, OUTPUT_ROOT, LOOKUP_FILE
from pathlib import Path
import argparse
from collections import Counter
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = PROJECT_ROOT
DEFAULT_DATA = DATA_ROOT / "l1_detail"
SUM_COLS = ["trip_count", "valid_wait_count", "total_valid_wait_minutes"]
READ_COLS = ["pickup_date", "pickup_hour", "PULocationID", "hvfhs_license_num",
             "avg_wait_minutes", *SUM_COLS]


def summarize(df, keys):
    out = df.groupby(keys, observed=True, dropna=False)[SUM_COLS].sum().reset_index()
    denominator = out.valid_wait_count.where(out.valid_wait_count > 0)
    out["avg_valid_wait_minutes"] = out.total_valid_wait_minutes / denominator
    out["valid_wait_rate"] = out.valid_wait_count / out.trip_count.where(out.trip_count > 0)
    return out


def run(data, output):
    paths = sorted(data.rglob("*.parquet"))
    if not paths:
        raise ValueError(f"没有 Parquet 文件：{data}")
    stats = Counter()
    nulls = Counter()
    max_mean_diff = 0.0
    pieces = []
    dates = []
    for p in paths:
        pf = pq.ParquetFile(p)
        if not set(READ_COLS).issubset(pf.schema_arrow.names):
            raise ValueError(f"未发现修正后的完整字段：{p}")
        df = pf.read(columns=READ_COLS).to_pandas()
        stats["files"] += 1
        stats["aggregate_rows"] += len(df)
        nulls.update({c: int(df[c].isna().sum()) for c in READ_COLS})
        required = [c for c in READ_COLS if c not in ["avg_wait_minutes", "total_valid_wait_minutes"]]
        if df[required].isna().any().any():
            raise ValueError(f"必需列含 NULL：{p}")
        missing_sum = df.total_valid_wait_minutes.isna()
        if (missing_sum & (df.valid_wait_count > 0)).any():
            raise ValueError(f"有效计数大于零但等待总和缺失：{p}")
        # 空有效集合的 SUM 可以为 NULL；仅在计数为零时按加法单位元 0 处理。
        # 不把缺失的平均等待时间填成 0。
        stats["zero_count_null_sum_groups"] += int(missing_sum.sum())
        df.loc[missing_sum, "total_valid_wait_minutes"] = 0.0
        if not np.isfinite(df[SUM_COLS].to_numpy()).all():
            raise ValueError(f"非有限汇总值：{p}")
        if not ((df.trip_count > 0) & (df.valid_wait_count >= 0) &
                (df.valid_wait_count <= df.trip_count)).all():
            raise ValueError(f"计数范围异常：{p}")
        if not df.pickup_hour.isin(range(24)).all():
            raise ValueError(f"小时范围异常：{p}")
        if not ((df.total_valid_wait_minutes >= -1e-8) &
                (df.total_valid_wait_minutes <= 120 * df.valid_wait_count + 1e-6)).all():
            raise ValueError(f"有效等待总和不符合现有 0–120 分钟口径：{p}")
        date = pd.Timestamp(p.parent.name.split("=", 1)[1])
        if not df.pickup_date.astype(str).eq(date.strftime("%Y-%m-%d")).all():
            raise ValueError(f"文件内日期与目录不符：{p}")
        if df.duplicated(["pickup_date", "pickup_hour", "PULocationID", "hvfhs_license_num"]).any():
            raise ValueError(f"分组键重复：{p}")
        dates.append(date)
        valid = df.valid_wait_count > 0
        stats["zero_valid_groups"] += int((~valid).sum())
        stats["negative_stored_mean_groups"] += int((df.avg_wait_minutes < 0).sum())
        stats["stored_mean_above_120_groups"] += int((df.avg_wait_minutes > 120).sum())
        stats["positive_count_missing_mean_groups"] += int((valid & df.avg_wait_minutes.isna()).sum())
        expected = df.loc[valid, "total_valid_wait_minutes"] / df.loc[valid, "valid_wait_count"]
        actual = df.loc[valid, "avg_wait_minutes"]
        if len(expected):
            delta = (expected - actual).abs()
            max_mean_diff = max(max_mean_diff, float(delta.max()))
            stats["stored_mean_mismatch_groups"] += int((~np.isclose(expected, actual, atol=1e-8, rtol=1e-9)).sum())
        one = df.groupby("pickup_hour", observed=True)[SUM_COLS].sum().reset_index()
        one["pickup_date"] = date
        pieces.append(one)

    calendar = pd.DatetimeIndex(sorted(dates))
    expected_calendar = pd.date_range("2023-01-01", "2024-12-31")
    if not calendar.equals(expected_calendar):
        raise ValueError("日期覆盖不符合当前 731 天数据，需检查缺失/重复日期。")
    daily = pd.concat(pieces, ignore_index=True)
    daily["day_type"] = np.where(daily.pickup_date.dt.dayofweek < 5, "Mon–Fri", "Sat–Sun")
    daily["month"] = daily.pickup_date.dt.strftime("%Y-%m")
    hourly = summarize(daily, ["pickup_hour"])
    if hourly.pickup_hour.tolist() != list(range(24)):
        raise ValueError("全期 24 小时不完整")
    hourly["mean_daily_valid_trips"] = hourly.valid_wait_count / len(calendar)
    split = summarize(daily, ["day_type", "pickup_hour"])
    monthly = summarize(daily, ["month", "pickup_hour"])
    totals = daily[SUM_COLS].sum()
    peak = hourly.loc[hourly.avg_valid_wait_minutes.idxmax()]
    low = hourly.loc[hourly.avg_valid_wait_minutes.idxmin()]
    overall = float(totals.total_valid_wait_minutes / totals.valid_wait_count)
    output.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(output / "hourly_wait.csv", index=False)
    split.to_csv(output / "weekday_weekend_wait.csv", index=False)
    monthly.to_csv(output / "monthly_hourly_wait.csv", index=False)
    daily.to_csv(output / "daily_hourly_sums.csv", index=False)
    summary = {
        "input": str(data.resolve()), "period": [str(calendar.min().date()), str(calendar.max().date())],
        "time_basis": "pickup_hour; request_hour is unavailable",
        "pooling": "sum(total_valid_wait_minutes)/sum(valid_wait_count), both platforms, all zones",
        "validity_rule_from_handoff": "0 <= wait_minutes <= 120; raw records not independently available",
        "checks": dict(stats), "nulls": dict(nulls),
        "max_stored_mean_absolute_difference": max_mean_diff,
        "trips": int(totals.trip_count), "valid_wait_trips": int(totals.valid_wait_count),
        "total_valid_wait_minutes": float(totals.total_valid_wait_minutes),
        "overall_valid_wait_minutes": overall,
        "valid_wait_rate": float(totals.valid_wait_count / totals.trip_count),
        "peak_hour": int(peak.pickup_hour), "peak_wait_minutes": float(peak.avg_valid_wait_minutes),
        "lowest_hour": int(low.pickup_hour), "lowest_wait_minutes": float(low.avg_valid_wait_minutes),
        "range_minutes": float(peak.avg_valid_wait_minutes - low.avg_valid_wait_minutes),
    }
    (output / "audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    blue, orange = "#17637b", "#d88535"
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.8), sharex=True,
                             gridspec_kw={"height_ratios": [2.4, 1]}, layout="constrained")
    fig.suptitle("Average waiting time by pickup hour", fontsize=17, fontweight="bold")
    ax = axes[0]
    ax.set_title("2023–2024 · Both platforms pooled · Request-to-pickup wait, valid records only", fontsize=10, pad=12)
    ax.plot(hourly.pickup_hour, hourly.avg_valid_wait_minutes, marker="o", lw=2.3, ms=4.5, color=blue)
    ax.axhline(overall, ls="--", color="#858585", lw=1, label=f"All-hours mean: {overall:.2f} min")
    ax.set(ylabel="Average wait (minutes)", ylim=(0, float(peak.avg_valid_wait_minutes) * 1.25))
    for row, label in [(peak, "Highest"), (low, "Lowest")]:
        h, val = int(row.pickup_hour), float(row.avg_valid_wait_minutes)
        ax.scatter([h], [val], s=48, color=orange, zorder=4)
        ax.annotate(f"{label}: {h:02d}:00–{h:02d}:59\n{val:.2f} min", xy=(h, val),
                    xytext=(8 if h < 18 else -8, 18 if label == "Highest" else -36),
                    textcoords="offset points", ha="left" if h < 18 else "right", fontsize=10)
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="y", alpha=.2)
    ax = axes[1]
    ax.bar(hourly.pickup_hour, hourly.mean_daily_valid_trips / 1000, color="#abc6ce", width=.75)
    ax.set(ylabel="Valid trips/day\n(thousands)", xlabel="Pickup hour (local clock time)",
           xticks=range(24), xlim=(-.5, 23.5))
    ax.grid(axis="y", alpha=.2)
    fig.savefig(output / "hourly_wait.png", dpi=240, bbox_inches="tight")
    fig.savefig(output / "hourly_wait.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.5), layout="constrained")
    for label, color in [("Mon–Fri", blue), ("Sat–Sun", orange)]:
        sub = split.loc[split.day_type == label]
        ax.plot(sub.pickup_hour, sub.avg_valid_wait_minutes, "o-", label=label, color=color, ms=4)
    ax.set(title="Waiting time by pickup hour: weekdays vs weekends | 2023–2024",
           xlabel="Pickup hour (local clock time)", ylabel="Average valid wait (minutes)",
           xticks=range(24), xlim=(-.5, 23.5))
    ax.set_ylim(0, split.avg_valid_wait_minutes.max() * 1.18)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=.2)
    fig.savefig(output / "weekday_weekend_wait.png", dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(hourly[["pickup_hour", "avg_valid_wait_minutes", "valid_wait_rate"]].round(4).to_string(index=False))
    print("DAY_TYPE_PEAKS", split.loc[split.groupby("day_type").avg_valid_wait_minutes.idxmax(),
                                      ["day_type", "pickup_hour", "avg_valid_wait_minutes"]].to_dict("records"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=OUTPUT_ROOT / "hourly_wait")
    args = parser.parse_args()
    run(args.data.expanduser(), args.out)
