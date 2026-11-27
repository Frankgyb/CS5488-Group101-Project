"""Uber mean request-to-pickup wait by borough and pickup hour, 2023–2024."""

from _config import PROJECT_ROOT, DATA_ROOT, OUTPUT_ROOT, LOOKUP_FILE
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

ROOT = PROJECT_ROOT
OUT = OUTPUT_ROOT / "platform_comparison"
SOURCE = OUT / "borough_hour_platform_wait.csv"
BOROUGHS = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]
METRICS = ["trip_count", "valid_wait_count", "total_valid_wait_minutes"]


def run():
    source = pd.read_csv(SOURCE)
    data = source.loc[source.hvfhs_license_num.eq("HV0003")].copy()
    keys = ["Borough", "pickup_hour"]
    if data.duplicated(keys).any():
        raise ValueError("Duplicate borough-hour keys")
    expected = pd.MultiIndex.from_product([BOROUGHS, range(24)], names=keys)
    if set(data.set_index(keys).index) != set(expected):
        raise ValueError("Expected all five boroughs and all 24 hours")
    data = data.set_index(keys).reindex(expected).reset_index()
    if not np.isfinite(data[METRICS].to_numpy()).all():
        raise ValueError("Missing or nonfinite totals")
    if not ((data.trip_count > 0) & (data.valid_wait_count > 0) &
            (data.valid_wait_count <= data.trip_count)).all():
        raise ValueError("Invalid trip counts")
    avg = data.total_valid_wait_minutes / data.valid_wait_count
    if not avg.between(0, 120).all():
        raise ValueError("Invalid mean waiting time")
    np.testing.assert_allclose(avg, data.avg_valid_wait_minutes, rtol=1e-10, atol=1e-10)
    data["avg_valid_wait_minutes"] = avg
    data["valid_wait_rate"] = data.valid_wait_count / data.trip_count
    # Independently reconcile the existing weekday/weekend export into all-day cells.
    daytypes = pd.read_csv(OUT / "daytype_borough_hour_platform_wait.csv")
    check = daytypes.loc[daytypes.hvfhs_license_num.eq("HV0003")].groupby(keys)[METRICS].sum().reindex(expected)
    for metric in METRICS:
        np.testing.assert_allclose(data[metric], check[metric], rtol=1e-10, atol=1e-6)
    mix = pd.read_csv(OUT / "borough_platform_mix.csv")
    mix = mix.loc[mix.hvfhs_license_num.eq("HV0003")].set_index("Borough")
    boroughs = data.groupby("Borough")[METRICS].sum().reindex(BOROUGHS)
    np.testing.assert_array_equal(boroughs.trip_count, mix.loc[BOROUGHS, "trip_count"])
    boroughs["avg_valid_wait_minutes"] = boroughs.total_valid_wait_minutes / boroughs.valid_wait_count
    boroughs["valid_wait_rate"] = boroughs.valid_wait_count / boroughs.trip_count
    platform = pd.read_csv(OUT / "platform_totals.csv").set_index("hvfhs_license_num").loc["HV0003"]
    excluded_trips = int(platform.trip_count - data.trip_count.sum())
    excluded_valid = int(platform.valid_wait_count - data.valid_wait_count.sum())
    if excluded_trips < 0 or excluded_valid < 0:
        raise ValueError("Five-borough totals exceed all-region platform totals")
    matrix = data.pivot(index="Borough", columns="pickup_hour", values="avg_valid_wait_minutes").reindex(index=BOROUGHS, columns=range(24))
    data.to_csv(OUT / "uber_borough_hour_wait.csv", index=False)
    matrix.to_csv(OUT / "uber_borough_hour_wait_matrix.csv")
    boroughs.to_csv(OUT / "uber_borough_wait_totals.csv")
    peak = data.loc[data.avg_valid_wait_minutes.idxmax()]
    low = data.loc[data.avg_valid_wait_minutes.idxmin()]
    coverage_low = data.loc[data.valid_wait_rate.idxmin()]
    def cell(row):
        return {"borough": row.Borough, "pickup_hour": int(row.pickup_hour),
                "mean_wait_minutes": float(row.avg_valid_wait_minutes),
                "valid_wait_count": int(row.valid_wait_count),
                "valid_wait_rate": float(row.valid_wait_rate)}
    summary = {
        "platform": "Uber (HV0003)", "source": str(SOURCE),
        "period": ["2023-01-01", "2024-12-31"], "scope": "Five NYC pickup boroughs; all dates pooled",
        "cells": len(data), "total_trips": int(data.trip_count.sum()),
        "valid_wait_trips": int(data.valid_wait_count.sum()),
        "excluded_other_area_trips": excluded_trips, "excluded_other_area_valid_wait_trips": excluded_valid,
        "formula": "sum(total_valid_wait_minutes)/sum(valid_wait_count) for each Uber borough-hour cell",
        "time_basis": "Pickup hour, not request hour; h denotes h:00–h:59",
        "valid_wait_definition": "0–120 minutes inclusive, as defined in the corrected L1 handoff",
        "highest_cell": cell(peak), "lowest_cell": cell(low), "lowest_coverage_cell": cell(coverage_low),
        "minimum_valid_wait_count_per_cell": int(data.valid_wait_count.min()),
        "borough_overall_mean_minutes": boroughs.avg_valid_wait_minutes.to_dict(),
        "checks": {"all_120_cells_complete": True, "means_match_source": True,
                    "totals_match_daytype_export": True, "borough_counts_match_composition_export": True},
        "interpretation": "Descriptive geographic and hourly associations, without holiday, date, within-borough zone or trip-mix adjustment. No significance or causal incentive-effect test."
    }
    (OUT / "uber_borough_hour_wait_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    plot(matrix, summary)
    report(data, boroughs, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def plot(matrix, summary):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "text.color": "#172b3a", "axes.labelcolor": "#334155",
                         "xtick.color": "#475569", "ytick.color": "#475569",
                         "pdf.fonttype": 42})
    values = matrix.to_numpy()
    vmin = np.floor(values.min() * 2) / 2
    vmax = np.ceil(values.max() * 2) / 2
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("YlOrRd")
    fig, ax = plt.subplots(figsize=(15.5, 6.4))
    fig.subplots_adjust(left=.115, right=.895, top=.76, bottom=.27)
    fig.text(.035, .935, "Uber Operational Pressure Hotspots", fontsize=23, weight="bold")
    fig.text(.035, .875, "Mean request-to-pickup wait in minutes, 2023–2024. All days pooled.", fontsize=12, color="#475569")
    im = ax.imshow(values, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_xticks(range(24), [f"{h:02d}" for h in range(24)], fontsize=10)
    ax.set_yticks(range(5), BOROUGHS, fontsize=11)
    ax.set_xlabel("Pickup hour (local clock time)", labelpad=13, fontsize=11)
    ax.set_xticks(np.arange(-.5, 24, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 5, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(which="major", length=0, pad=8)
    for r in range(5):
        for c in range(24):
            value = values[r, c]
            rgba = cmap(norm(value))
            luminance = .2126 * rgba[0] + .7152 * rgba[1] + .0722 * rgba[2]
            ax.text(c, r, f"{value:.1f}", ha="center", va="center", fontsize=9,
                    color="white" if luminance < .48 else "#172b3a")
    for spine in ax.spines.values():
        spine.set_visible(False)
    cax = fig.add_axes([.92, .27, .014, .49])
    cb = fig.colorbar(im, cax=cax)
    cb.outline.set_visible(False)
    cb.set_label("Average wait (minutes)", fontsize=10, labelpad=10)
    fig.text(.035, .145, "Each cell = total valid waiting minutes ÷ valid trip count. Valid waits: 0–120 minutes. Darker red means longer waits.", fontsize=9.5, color="#475569")
    fig.text(.035, .095, "Hours refer to pickup, not request. Boroughs combine multiple pickup zones; within-borough variation is not shown.", fontsize=9.5, color="#475569")
    fig.text(.035, .045, f"Source: corrected L1 aggregates, Uber (HV0003). Five boroughs shown; {summary['excluded_other_area_trips']:,} other-area trips excluded.", fontsize=9.5, color="#64748b")
    fig.savefig(OUT / "04_uber_borough_hour_wait.png", dpi=220, facecolor="white")
    fig.savefig(OUT / "04_uber_borough_hour_wait.pdf", facecolor="white")
    plt.close(fig)


def report(data, boroughs, summary):
    high, low = summary["highest_cell"], summary["lowest_cell"]
    rows = []
    for name in BOROUGHS:
        a = data.loc[data.Borough.eq(name)]
        peak = a.loc[a.avg_valid_wait_minutes.idxmax()]
        minimum = a.loc[a.avg_valid_wait_minutes.idxmin()]
        rows.append(f"| {name} | {boroughs.loc[name, 'avg_valid_wait_minutes']:.2f} | {int(peak.pickup_hour):02d}:00–{int(peak.pickup_hour):02d}:59 ({peak.avg_valid_wait_minutes:.2f}) | {int(minimum.pickup_hour):02d}:00–{int(minimum.pickup_hour):02d}:59 ({minimum.avg_valid_wait_minutes:.2f}) |")
    text = "# Uber average waiting time by pickup borough and hour\n\n"
    text += "Figure: `04_uber_borough_hour_wait.png` / `.pdf`. Five boroughs × 24 pickup hours, 2023–2024, all days pooled.\n\n"
    text += "## Method\n\nWithin each Uber (HV0003) borough-hour cell, the mean equals total valid request-to-pickup waiting minutes divided by valid waiting-time records. The valid range is 0–120 minutes inclusive, following the corrected handoff. Means are weighted by valid trip count, not averaged across aggregate rows. Time is indexed by pickup hour rather than request hour.\n\n"
    text += "| Borough | Overall mean (min) | Highest hour (min) | Lowest hour (min) |\n| --- | ---: | --- | --- |\n" + "\n".join(rows) + "\n\n"
    text += "## Report-ready paragraph\n\n"
    text += f"Uber's mean request-to-pickup waiting time varied across pickup boroughs and hours. The highest borough-hour mean was observed in {high['borough']} at {high['pickup_hour']:02d}:00–{high['pickup_hour']:02d}:59 ({high['mean_wait_minutes']:.2f} minutes), while the lowest occurred in {low['borough']} at {low['pickup_hour']:02d}:00–{low['pickup_hour']:02d}:59 ({low['mean_wait_minutes']:.2f} minutes). These patterns identify locations and time periods for further operational investigation. The pooled averages do not control for differences in date, pickup zone or trip composition, and do not establish that driver shortages or incentive policies cause the observed differences.\n\n"
    text += "## 数据核对与解释\n\n"
    text += f"- 120个行政区×小时组合均有有效数据；每格至少 {summary['minimum_valid_wait_count_per_cell']:,} 笔有效等待记录。\n"
    text += f"- 五行政区纳入 {summary['total_trips']:,} 笔行程，其中 {summary['valid_wait_trips']:,} 笔具有有效等待时间；排除其他区域行程 {summary['excluded_other_area_trips']:,} 笔。\n"
    text += "- 各格分子、分母已与既有工作日/周末导出合并结果核对；各行政区订单总量与地域构成表一致。\n"
    coverage = summary["lowest_coverage_cell"]
    text += f"- 有效等待覆盖最低的格为 {coverage['borough']} {coverage['pickup_hour']:02d}:00–{coverage['pickup_hour']:02d}:59，为 {coverage['valid_wait_rate']:.2%}；最高等待格覆盖率为 {high['valid_wait_rate']:.2%}。无效或缺失等待记录不计入均值。\n"
    text += "- 此处区域指五个行政区，而非全部 TLC taxi zones；不区分工作日/周末，不做显著性检验。\n\n复现：运行项目根目录 `scripts/10_uber_borough_hour_heatmap.py`。图表数据：`uber_borough_hour_wait.csv`；矩阵：`uber_borough_hour_wait_matrix.csv`；核对摘要：`uber_borough_hour_wait_summary.json`。\n"
    (OUT / "uber_borough_hour_wait_report.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    run()
