"""Plot each platform's trip-weighted request-to-pickup wait by pickup hour."""

from _config import PROJECT_ROOT, DATA_ROOT, OUTPUT_ROOT, LOOKUP_FILE
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

ROOT = PROJECT_ROOT
SOURCE = DATA_ROOT / "l1_detail"
OUT = OUTPUT_ROOT / "platform_comparison"
PLATFORMS = ["HV0003", "HV0005"]
COLORS = {"HV0003": "#17637b", "HV0005": "#d88535"}
METRICS = ["trip_count", "valid_wait_count", "total_valid_wait_minutes"]


def run(data, output):
    files = sorted(data.rglob("*.parquet"))
    if not files:
        raise ValueError(f"No L1 parquet files found: {data}")
    pieces, dates = [], []
    source_groups = 0
    cols = ["pickup_date", "pickup_hour", "hvfhs_license_num", *METRICS]
    for path in files:
        df = pq.ParquetFile(path).read(columns=cols).to_pandas()
        source_groups += len(df)
        required = [c for c in cols if c != "total_valid_wait_minutes"]
        if df[required].isna().any().any():
            raise ValueError(f"Missing required data: {path}")
        if not set(df.hvfhs_license_num.unique()).issubset(PLATFORMS):
            raise ValueError("Unexpected platform")
        if not df.pickup_hour.isin(range(24)).all():
            raise ValueError("Invalid pickup hour")
        if not ((df.trip_count > 0) & (df.valid_wait_count >= 0) &
                (df.valid_wait_count <= df.trip_count)).all():
            raise ValueError("Invalid counts")
        missing = df.total_valid_wait_minutes.isna()
        if (missing & (df.valid_wait_count > 0)).any():
            raise ValueError("Missing wait total with positive valid count")
        df.loc[missing, "total_valid_wait_minutes"] = 0.0
        if not np.isfinite(df[METRICS].to_numpy()).all():
            raise ValueError("Nonfinite aggregate")
        if not ((df.total_valid_wait_minutes >= -1e-8) &
                (df.total_valid_wait_minutes <= 120 * df.valid_wait_count + 1e-6)).all():
            raise ValueError("Wait sum outside source validity range")
        date = pd.Timestamp(path.parent.name.split("=", 1)[1])
        if not df.pickup_date.astype(str).eq(date.strftime("%Y-%m-%d")).all():
            raise ValueError("File date differs from partition date")
        dates.append(date)
        part = df.groupby(["pickup_hour", "hvfhs_license_num"], observed=True)[METRICS].sum().reset_index()
        part["pickup_date"] = date
        pieces.append(part)
    expected_days = pd.date_range("2023-01-01", "2024-12-31")
    if not pd.DatetimeIndex(sorted(dates)).equals(expected_days):
        raise ValueError("Date coverage differs from 731 complete source days")
    daily = pd.concat(pieces, ignore_index=True)
    if daily.duplicated(["pickup_date", "pickup_hour", "hvfhs_license_num"]).any():
        raise ValueError("Repeated daily aggregate key")
    hourly = daily.groupby(["hvfhs_license_num", "pickup_hour"], observed=True)[METRICS].sum()
    expected_index = pd.MultiIndex.from_product([PLATFORMS, range(24)], names=hourly.index.names)
    hourly = hourly.reindex(expected_index)
    if hourly.isna().any().any() or (hourly.valid_wait_count <= 0).any():
        raise ValueError("Missing platform-hour values")
    hourly["avg_valid_wait_minutes"] = hourly.total_valid_wait_minutes / hourly.valid_wait_count
    hourly["valid_wait_rate"] = hourly.valid_wait_count / hourly.trip_count
    hourly["mean_daily_trips"] = hourly.trip_count / len(expected_days)
    hourly = hourly.reset_index()
    totals = hourly.groupby("hvfhs_license_num")[METRICS].sum()
    if int(totals.trip_count.sum()) != 471_960_468 or int(totals.valid_wait_count.sum()) != 467_359_108:
        raise ValueError("Source totals differ from corrected historical controls")
    # Independent existing exports cover all source areas; borough-only exports do not.
    old_totals = pd.read_csv(output / "platform_totals.csv").set_index("hvfhs_license_num").loc[PLATFORMS]
    for c in METRICS:
        np.testing.assert_allclose(totals.loc[PLATFORMS, c], old_totals[c], rtol=1e-10, atol=1e-5)
    pooled = hourly.groupby("pickup_hour")[METRICS].sum()
    old_hourly = pd.read_csv(OUTPUT_ROOT / "hourly_wait/hourly_wait.csv").set_index("pickup_hour")
    for c in METRICS:
        np.testing.assert_allclose(pooled[c], old_hourly.loc[range(24), c], rtol=1e-10, atol=1e-5)
    wide = hourly.pivot(index="pickup_hour", columns="hvfhs_license_num", values="avg_valid_wait_minutes")
    gap = wide["HV0005"] - wide["HV0003"]
    summary = {
        "source": str(data.resolve()), "period": ["2023-01-01", "2024-12-31"],
        "source_files": len(files), "source_groups": source_groups,
        "total_trips": int(totals.trip_count.sum()), "valid_wait_trips": int(totals.valid_wait_count.sum()),
        "formula": "sum(total_valid_wait_minutes) / sum(valid_wait_count), separately by platform and pickup_hour",
        "scope": "All source pickup areas and all days pooled",
        "time_basis": "Pickup hour as recorded in the source, not request hour; hour h means h:00 through h:59",
        "validity": "0 <= request-to-pickup wait <= 120 minutes, from the corrected handoff",
        "interpolation": "PCHIP through 24 hourly averages for display only; no smoothing of the statistics or extrapolation",
        "checks": {"matches_existing_platform_totals": True, "matches_existing_all_platform_24h_totals": True},
        "platforms": {},
        "gap": {"definition": "HV0005 minus HV0003, minutes", "positive_hours": int((gap > 0).sum()),
                "min_hour": int(gap.idxmin()), "min_minutes": float(gap.min()),
                "max_hour": int(gap.idxmax()), "max_minutes": float(gap.max())},
        "interpretation": "Descriptive pooled comparison. Location, date and trip mix are not standardized; not a causal test of platform performance or incentives."
    }
    for key in PLATFORMS:
        a = hourly.loc[hourly.hvfhs_license_num.eq(key)].set_index("pickup_hour")
        v = a.avg_valid_wait_minutes
        t = totals.loc[key]
        summary["platforms"][key] = {
            "trip_count": int(t.trip_count), "valid_wait_count": int(t.valid_wait_count),
            "overall_wait_minutes": float(t.total_valid_wait_minutes / t.valid_wait_count),
            "valid_wait_rate": float(t.valid_wait_count / t.trip_count),
            "peak_hour": int(v.idxmax()), "peak_minutes": float(v.max()),
            "lowest_hour": int(v.idxmin()), "lowest_minutes": float(v.min()),
            "hourly_valid_rate_min": float(a.valid_wait_rate.min()),
            "hourly_valid_rate_max": float(a.valid_wait_rate.max())
        }
    output.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(output / "hourly_platform_wait.csv", index=False)
    wide.assign(HV0005_minus_HV0003_minutes=gap).to_csv(output / "hourly_platform_wait_comparison.csv")
    (output / "hourly_platform_wait_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    draw(hourly, summary, output)
    write_notes(summary, output)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def draw(hourly, summary, output):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#94a3b8", "text.color": "#172b3a",
                         "axes.labelcolor": "#334155", "xtick.color": "#475569",
                         "ytick.color": "#475569", "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(12.8, 7.1))
    fig.subplots_adjust(left=.085, right=.965, top=.75, bottom=.23)
    fig.text(.06, .94, "Average waiting time across the day", fontsize=23, weight="bold")
    fig.text(.06, .895, "Two-platform comparison by pickup hour, 2023–2024", fontsize=12, color="#475569")
    x = np.arange(24)
    xx = np.linspace(0, 23, 600)
    for key in PLATFORMS:
        a = hourly.loc[hourly.hvfhs_license_num.eq(key)].set_index("pickup_hour").loc[x]
        y = a.avg_valid_wait_minutes.to_numpy()
        interp = PchipInterpolator(x, y)
        np.testing.assert_allclose(interp(x), y, atol=1e-12)
        s = summary["platforms"][key]
        ax.plot(xx, interp(xx), lw=3, color=COLORS[key],
                label=f"{key}   All-hours mean: {s['overall_wait_minutes']:.2f} min")
        h, peak = s["peak_hour"], s["peak_minutes"]
        offset = (55, 30) if key == "HV0005" else (65, -30)
        ax.annotate(f"{key} peak: {peak:.2f} min\n{h:02d}:00–{h:02d}:59",
                    xy=(h, peak), xytext=offset, textcoords="offset points",
                    fontsize=10, color=COLORS[key], ha="left", va="center",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": .9, "pad": 3},
                    arrowprops={"arrowstyle": "-", "color": COLORS[key], "lw": 1})
    ymax = np.ceil(hourly.avg_valid_wait_minutes.max() + .7)
    ax.set(xlim=(0, 23), ylim=(0, ymax), ylabel="Average request-to-pickup wait (minutes)",
           xlabel="Pickup hour (local clock time)")
    ax.set_xticks(x, [f"{h:02d}" for h in x], fontsize=9)
    ax.set_yticks(np.arange(0, ymax + .01, 1))
    ax.grid(axis="y", color="#dce3ea", alpha=.8, lw=.7)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.035), ncol=2, frameon=False, fontsize=11)
    fig.text(.06, .12, "Each hourly mean = total valid waiting minutes ÷ valid trip count; all source areas and dates pooled.",
             fontsize=9.5, color="#475569")
    fig.text(.06, .08, "Valid waits: 0–120 minutes. Smooth curves interpolate the 24 hourly means for display only.",
             fontsize=9.5, color="#475569")
    fig.text(.06, .04, "Source: corrected L1 aggregates. Pickup hour is not request hour; trip composition differs between platforms.",
             fontsize=9, color="#64748b")
    fig.savefig(output / "03_hourly_platform_wait.png", dpi=220, facecolor="white")
    fig.savefig(output / "03_hourly_platform_wait.pdf", facecolor="white")
    plt.close(fig)


def write_notes(s, output):
    a, b = [s["platforms"][k] for k in PLATFORMS]
    g = s["gap"]
    rows = []
    for key, p in s["platforms"].items():
        rows.append(f"| {key} | {p['overall_wait_minutes']:.2f} | {p['peak_hour']:02d}:00–{p['peak_hour']:02d}:59 ({p['peak_minutes']:.2f}) | {p['lowest_hour']:02d}:00–{p['lowest_hour']:02d}:59 ({p['lowest_minutes']:.2f}) | {p['valid_wait_rate']:.2%} |")
    report = "# Average waiting time by platform and pickup hour\n\n"
    report += "Figure: `03_hourly_platform_wait.png` / `.pdf`. Period: 2023–2024.\n\n"
    report += "## Method\n\n"
    report += "For each platform and pickup hour, we divided total valid request-to-pickup waiting minutes by the number of valid waiting-time records. All dates and pickup areas were pooled. Valid waits follow the corrected source definition of 0–120 minutes, inclusive. This is a trip-weighted mean, not an unweighted average of L1 group means or daily averages. Curves use shape-preserving interpolation through the 24 hourly averages for presentation only. The time axis records pickup hour, not request hour.\n\n"
    report += "## Results\n\n| Platform | Overall mean (min) | Highest hour (min) | Lowest hour (min) | Valid wait coverage |\n| --- | ---: | --- | --- | ---: |\n" + "\n".join(rows) + "\n\n"
    report += f"HV0005's mean wait exceeded HV0003's in {g['positive_hours']} of 24 hourly bins. The largest observed difference was {g['max_minutes']:.2f} minutes at {g['max_hour']:02d}:00–{g['max_hour']:02d}:59. The smallest difference was {g['min_minutes']:.2f} minutes at {g['min_hour']:02d}:00–{g['min_hour']:02d}:59.\n\n"
    report += "## Report-ready paragraph\n\n"
    report += f"Average request-to-pickup waiting time varied by pickup hour for both platforms. Across all hours, the trip-weighted mean was {a['overall_wait_minutes']:.2f} minutes for HV0003 and {b['overall_wait_minutes']:.2f} minutes for HV0005. HV0005 recorded a higher mean wait in {g['positive_hours']} of the 24 hourly bins, with the largest difference of {g['max_minutes']:.2f} minutes at {g['max_hour']:02d}:00–{g['max_hour']:02d}:59. The longest mean waits occurred at {a['peak_hour']:02d}:00–{a['peak_hour']:02d}:59 for HV0003 ({a['peak_minutes']:.2f} minutes) and {b['peak_hour']:02d}:00–{b['peak_hour']:02d}:59 for HV0005 ({b['peak_minutes']:.2f} minutes). These descriptive patterns identify periods for further operational investigation. Differences in location, date and trip composition were not standardized, so the chart does not establish a causal difference in dispatch efficiency or the effectiveness of driver incentives.\n\n"
    report += "## 数据口径与复现\n\n- 使用修正后的 L1；全部区域纳入，未只筛选纽约五个行政区。\n- 以有效行程数加权；不能对已有平均等待时长直接再求平均。\n- 小时 h 指 h:00–h:59 的上车记录，不是该小时发出请求的记录。\n- 同时保留有效记录数、有效比例，便于检查平台覆盖差异。\n- 已核对两平台总量及合并后的全部24小时，与既有全量输出一致。\n- 此图为描述性比较，没有进行平台差异显著性检验。\n\n运行根目录脚本 `scripts/09_platform_hourly_wait.py`。明细为 `hourly_platform_wait.csv`，宽表为 `hourly_platform_wait_comparison.csv`，核对结果为 `hourly_platform_wait_summary.json`。\n"
    (output / "hourly_platform_wait_report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    run(args.data, args.output)
