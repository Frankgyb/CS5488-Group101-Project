"""两张平台比较图：平台自身区域构成；行政区×上车小时的等待差值。"""

from _config import PROJECT_ROOT, DATA_ROOT, OUTPUT_ROOT, LOOKUP_FILE
from pathlib import Path
import argparse
import json
from collections import Counter

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import PercentFormatter

ROOT = PROJECT_ROOT
DATA = DATA_ROOT / "l1_detail"
LOOKUP = LOOKUP_FILE
BOROUGHS = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]
PLATFORMS = ["HV0003", "HV0005"]
METRICS = ["trip_count", "valid_wait_count", "total_valid_wait_minutes"]
DAY_TYPES = ["Mon–Fri", "Sat–Sun"]
COLORS = ["#17637b", "#d88535"]


def add_rates(table):
    out = table.copy()
    out["avg_valid_wait_minutes"] = out.total_valid_wait_minutes / out.valid_wait_count.where(out.valid_wait_count > 0)
    out["valid_wait_rate"] = out.valid_wait_count / out.trip_count.where(out.trip_count > 0)
    return out


def differences(table, keys):
    # 缺平台或无有效行程时保留 NaN，不填成 0 分钟。
    wide = table.pivot(index=keys, columns="hvfhs_license_num",
                       values=["avg_valid_wait_minutes", "valid_wait_count", "valid_wait_rate"])
    wide.columns = [f"{metric}_{platform}" for metric, platform in wide.columns]
    out = wide.reset_index()
    out["wait_difference_minutes"] = out.avg_valid_wait_minutes_HV0005 - out.avg_valid_wait_minutes_HV0003
    out["both_platforms_observed"] = (out.valid_wait_count_HV0003 > 0) & (out.valid_wait_count_HV0005 > 0)
    return out


def compute(data, output):
    lookup = pd.read_csv(LOOKUP, keep_default_na=False)
    if lookup.LocationID.duplicated().any():
        raise ValueError("区域维表存在重复 LocationID。")
    zone_map = lookup.set_index("LocationID").Borough
    files = sorted(data.rglob("*.parquet"))
    if not files:
        raise ValueError(f"找不到数据：{data}")
    cols = ["pickup_date", "pickup_hour", "PULocationID", "hvfhs_license_num", *METRICS]
    blocks, days = [], []
    counts = Counter()
    for path in files:
        d = pq.ParquetFile(path).read(columns=cols).to_pandas()
        counts["files"] += 1
        counts["source_groups"] += len(d)
        nonnullable = [c for c in cols if c != "total_valid_wait_minutes"]
        if d[nonnullable].isna().any().any():
            raise ValueError(f"必需字段缺失：{path}")
        if not set(d.hvfhs_license_num.unique()).issubset(PLATFORMS):
            raise ValueError("发现额外平台，请检查比较范围。")
        if not ((d.trip_count > 0) & (d.valid_wait_count >= 0) & (d.valid_wait_count <= d.trip_count)).all():
            raise ValueError("计数无效。")
        absent_sum = d.total_valid_wait_minutes.isna()
        if (absent_sum & (d.valid_wait_count > 0)).any():
            raise ValueError("有有效记录，但有效等待总和缺失。")
        counts["null_sum_with_zero_valid_count_groups"] += int(absent_sum.sum())
        d.loc[absent_sum, "total_valid_wait_minutes"] = 0
        if not np.isfinite(d[METRICS]).all().all():
            raise ValueError("聚合数值包含无穷或缺失。")
        if not ((d.total_valid_wait_minutes >= -1e-8) &
                (d.total_valid_wait_minutes <= 120*d.valid_wait_count+1e-6)).all():
            raise ValueError("等待总和超出有效口径范围。")
        if not d.pickup_hour.isin(range(24)).all():
            raise ValueError("上车小时无效。")
        if d.duplicated(["pickup_date", "pickup_hour", "PULocationID", "hvfhs_license_num"]).any():
            raise ValueError("源聚合键重复。")
        date = pd.Timestamp(path.parent.name.split("=", 1)[1])
        if not d.pickup_date.astype(str).eq(date.strftime("%Y-%m-%d")).all():
            raise ValueError("目录日期与数据不符。")
        days.append(date)
        d["Borough"] = d.PULocationID.map(zone_map)
        counts["unmapped_trips"] += int(d.loc[d.Borough.isna(), "trip_count"].sum())
        d["Borough"] = d.Borough.fillna("Unmapped")
        original_count = int(d.trip_count.sum())
        a = d.groupby(["Borough", "pickup_hour", "hvfhs_license_num"], observed=True)[METRICS].sum().reset_index()
        if int(a.trip_count.sum()) != original_count:
            raise ValueError("维度关联/汇总改变行程数。")
        a["pickup_date"] = date
        a["day_type"] = DAY_TYPES[0] if date.dayofweek < 5 else DAY_TYPES[1]
        a["month"] = date.strftime("%Y-%m")
        blocks.append(a)
    if not pd.DatetimeIndex(sorted(days)).equals(pd.date_range("2023-01-01", "2024-12-31")):
        raise ValueError("日期不完整或重复。")
    daily = pd.concat(blocks, ignore_index=True)
    total = int(daily.trip_count.sum())
    if total != 471_960_468:
        raise ValueError("总量与当前已核查版本不一致。")
    platform = add_rates(daily.groupby("hvfhs_license_num", observed=True)[METRICS].sum())
    mix = daily.groupby(["Borough", "hvfhs_license_num"], observed=True).trip_count.sum().reset_index()
    mix["share_of_platform_trips"] = mix.trip_count / mix.hvfhs_license_num.map(platform.trip_count)
    mix["platform_share_in_borough"] = mix.trip_count / mix.groupby("Borough").trip_count.transform("sum")
    if not np.allclose(mix.groupby("hvfhs_license_num").share_of_platform_trips.sum(), 1):
        raise ValueError("所有区域的平台内部占比未合计为 1。")
    within = daily.loc[daily.Borough.isin(BOROUGHS)]
    hourly = add_rates(within.groupby(["Borough", "pickup_hour", "hvfhs_license_num"], observed=True)[METRICS].sum().reset_index())
    daytype = add_rates(within.groupby(["day_type", "Borough", "pickup_hour", "hvfhs_license_num"], observed=True)[METRICS].sum().reset_index())
    monthly = add_rates(within.groupby(["month", "day_type", "Borough", "pickup_hour", "hvfhs_license_num"], observed=True)[METRICS].sum().reset_index())
    diff = differences(hourly, ["Borough", "pickup_hour"])
    daydiff = differences(daytype, ["day_type", "Borough", "pickup_hour"])
    monthdiff = differences(monthly, ["month", "day_type", "Borough", "pickup_hour"])
    # 月份正差比例只作描述，未控制区域内部差异，也不是显著性检验。
    stability = monthdiff.groupby(["day_type", "Borough", "pickup_hour"]).agg(
        observed_months=("wait_difference_minutes", "count"),
        positive_months=("wait_difference_minutes", lambda s: int((s > 0).sum())),
        median_monthly_difference=("wait_difference_minutes", "median"),
    ).reset_index()
    daydiff = daydiff.merge(stability, on=["day_type", "Borough", "pickup_hour"], validate="one_to_one")
    output.mkdir(parents=True, exist_ok=True)
    platform.to_csv(output / "platform_totals.csv")
    mix.to_csv(output / "borough_platform_mix.csv", index=False)
    hourly.to_csv(output / "borough_hour_platform_wait.csv", index=False)
    diff.to_csv(output / "borough_hour_wait_difference.csv", index=False)
    daytype.to_csv(output / "daytype_borough_hour_platform_wait.csv", index=False)
    daydiff.to_csv(output / "daytype_borough_hour_wait_difference.csv", index=False)
    monthdiff.to_csv(output / "monthly_wait_difference.csv", index=False)
    audit = {
        "input": str(data.resolve()), "period": ["2023-01-01", "2024-12-31"],
        "source_checks": dict(counts), "total_trips": total,
        "valid_wait_trips": int(platform.valid_wait_count.sum()),
        "excluded_from_five_borough_figures_trips": int(daily.loc[~daily.Borough.isin(BOROUGHS), "trip_count"].sum()),
        "mix_denominator": "All recorded trips within each platform, including non-five-borough records",
        "wait_formula": "sum(total_valid_wait_minutes)/sum(valid_wait_count)",
        "difference_sign": "HV0005 minus HV0003, minutes",
        "time_basis": "pickup_hour, day type from pickup_date; Mon-Fri vs Sat-Sun without holiday reclassification",
        "all_hourly_cells_have_both_platforms": bool(diff.both_platforms_observed.all()),
        "all_daytype_cells_have_both_platforms": bool(daydiff.both_platforms_observed.all()),
        "minimum_valid_count_per_daytype_borough_hour_platform": int(daytype.valid_wait_count.min()),
        "positive_difference_daytype_cells": int((daydiff.wait_difference_minutes > 0).sum()),
        "total_daytype_cells": len(daydiff),
        "daytype_diff_min_minutes": float(daydiff.wait_difference_minutes.min()),
        "daytype_diff_max_minutes": float(daydiff.wait_difference_minutes.max()),
        "valid_wait_rate_range_by_platform": {
            key: [float(g.valid_wait_rate.min()), float(g.valid_wait_rate.max())]
            for key, g in daytype.groupby("hvfhs_license_num")},
        "caution": "Descriptive pooled comparisons; not adjusted for within-borough zone/date/trip mix or causal platform strategy effects.",
    }
    (output / "audit_summary.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    return platform, mix, daydiff, audit


def save(fig, output, stem):
    fig.savefig(output / f"{stem}.png", dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot(platform, mix, daydiff, audit, output):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    # 图 1：平台内区域构成，分母使用各平台全量行程。
    shares = mix.pivot(index="Borough", columns="hvfhs_license_num", values="share_of_platform_trips").reindex(BOROUGHS)
    fig, ax = plt.subplots(figsize=(10.8, 6.4))
    fig.subplots_adjust(left=.17, right=.96, top=.77, bottom=.18)
    fig.text(.035, .95, "Where each platform's trips originate", fontsize=20, weight="bold")
    fig.text(.035, .905, "Share of each platform's recorded trips by pickup borough  |  2023–2024", color="#475569", fontsize=11)
    y = np.arange(len(BOROUGHS))
    for offset, key, color in zip([-.19, .19], PLATFORMS, COLORS):
        values = shares[key].to_numpy()
        bars = ax.barh(y + offset, values, height=.34, color=color,
                       label=f"{key}  ·  {platform.loc[key, 'trip_count']/1e6:.2f}M trips")
        ax.bar_label(bars, labels=[f"{v:.1%}" for v in values], padding=6, fontsize=11)
    ax.set_yticks(y, BOROUGHS)
    ax.invert_yaxis()
    ax.set_xlim(0, .46)
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_xlabel("Share of all trips within each platform", labelpad=10)
    ax.grid(axis="x", alpha=.16)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", bbox_to_anchor=(-.005, 1.005), ncol=2, frameon=False, fontsize=10)
    excluded = audit["excluded_from_five_borough_figures_trips"]
    fig.text(.035, .055, f"Denominator includes all source records. Five boroughs shown; {excluded:,} other-area trips are not drawn.",
             fontsize=9, color="#475569")
    fig.text(.035, .028, "Source: corrected L1 aggregates. Percentages describe composition, not platform share within a borough.",
             fontsize=9, color="#475569")
    save(fig, output, "01_borough_platform_mix")

    # 图 2：星期类型两个面板，共用以 0 为中心的对称色标。
    bound = np.ceil(daydiff.wait_difference_minutes.abs().max()*2)/2
    bound = max(.5, float(bound))
    norm = TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound)
    fig, axes = plt.subplots(2, 1, figsize=(15, 8.8))
    fig.subplots_adjust(left=.12, right=.91, top=.81, bottom=.21, hspace=.65)
    fig.text(.025, .957, "Where and when does the waiting-time gap widen?", fontsize=21, weight="bold")
    fig.text(.025, .915, "HV0005 minus HV0003  ·  Mean valid request-to-pickup wait, in minutes  ·  2023–2024", fontsize=11, color="#475569")
    fig.text(.025, .875, "Red / positive: HV0005 waits longer     Blue / negative: HV0003 waits longer", fontsize=11, color="#475569")
    for ax, daytype in zip(axes, DAY_TYPES):
        a = daydiff.loc[daydiff.day_type == daytype].pivot(index="Borough", columns="pickup_hour", values="wait_difference_minutes").reindex(index=BOROUGHS, columns=range(24))
        values = a.to_numpy()
        image = ax.imshow(np.ma.masked_invalid(values), aspect="auto", cmap="RdBu_r", norm=norm)
        ax.set_xticks(range(24), [f"{h:02d}" for h in range(24)], fontsize=9)
        ax.set_yticks(range(5), BOROUGHS, fontsize=10)
        ax.set_title(daytype, loc="left", fontsize=13, weight="bold", pad=8)
        ax.set_xlabel("Pickup hour (local clock time)", labelpad=6, fontsize=10)
        for r in range(5):
            for c in range(24):
                value = values[r, c]
                if not np.isfinite(value):
                    ax.text(c, r, "NA", ha="center", va="center", fontsize=8, color="#555555")
                    continue
                rgba = plt.get_cmap("RdBu_r")(norm(value))
                luminance = .2126*rgba[0]+.7152*rgba[1]+.0722*rgba[2]
                ax.text(c, r, f"{value:.1f}" if round(value, 1) else "0.0", ha="center", va="center",
                         fontsize=8.5, color="white" if luminance < .52 else "#182631")
        for spine in ax.spines.values():
            spine.set_visible(False)
    cax = fig.add_axes([.935, .24, .013, .54])
    cb = fig.colorbar(image, cax=cax)
    cb.set_label("Wait difference (minutes)", fontsize=10)
    fig.text(.025, .125, "Each mean = sum of valid waiting minutes / valid trip count. Valid range follows the handoff rule: 0–120 min.", fontsize=9, color="#475569")
    fig.text(.025, .095, "Hours refer to pickup, not request. Mon–Fri / Sat–Sun are calendar groupings; holidays are not reclassified.", fontsize=9, color="#475569")
    fig.text(.025, .065, "Descriptive comparison: zone, date and trip composition can still differ within cells. It does not establish a strategy or causal effect.", fontsize=9, color="#475569")
    fig.text(.025, .035, f"Source: corrected L1 aggregates. Every cell has both platforms; minimum valid sample per platform/cell: {audit['minimum_valid_count_per_daytype_borough_hour_platform']:,}.", fontsize=9, color="#475569")
    save(fig, output, "02_borough_hour_wait_gap")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--out", type=Path, default=OUTPUT_ROOT / "platform_comparison")
    args = parser.parse_args()
    platform, mix, diff, audit = compute(args.data.expanduser(), args.out)
    plot(platform, mix, diff, audit, args.out)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print("Largest gaps:")
    print(diff.nlargest(6, "wait_difference_minutes")[["day_type", "Borough", "pickup_hour", "wait_difference_minutes", "positive_months"]].round(3).to_string(index=False))
    print("Smallest gaps:")
    print(diff.nsmallest(6, "wait_difference_minutes")[["day_type", "Borough", "pickup_hour", "wait_difference_minutes", "positive_months"]].round(3).to_string(index=False))
