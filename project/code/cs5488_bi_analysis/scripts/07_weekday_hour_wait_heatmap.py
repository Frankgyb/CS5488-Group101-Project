"""从已核查的每日×小时汇总表生成星期×小时平均等待热力图。"""

from _config import PROJECT_ROOT, DATA_ROOT, OUTPUT_ROOT, LOOKUP_FILE
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = PROJECT_ROOT
OUT = OUTPUT_ROOT / "hourly_wait"
WEEK = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
SUM_COLS = ["trip_count", "valid_wait_count", "total_valid_wait_minutes"]


def run():
    source = OUT / "daily_hourly_sums.csv"
    daily = pd.read_csv(source, parse_dates=["pickup_date"])
    audit = json.loads((OUT / "audit_summary.json").read_text())
    calendar = pd.date_range(*audit["period"])
    expected = pd.MultiIndex.from_product(
        [calendar, range(24)], names=["pickup_date", "pickup_hour"]
    )
    actual = pd.MultiIndex.from_frame(daily[["pickup_date", "pickup_hour"]])
    if actual.has_duplicates or not actual.sort_values().equals(expected):
        raise ValueError("每日×小时数据存在重复或缺失，停止绘图。")
    if not np.isfinite(daily[SUM_COLS].to_numpy()).all():
        raise ValueError("可加总指标含缺失值或非有限数。")
    if not ((daily.trip_count > 0) & (daily.valid_wait_count >= 0)
            & (daily.valid_wait_count <= daily.trip_count)
            & (daily.total_valid_wait_minutes >= 0)
            & (daily.total_valid_wait_minutes <= 120 * daily.valid_wait_count + 1e-6)).all():
        raise ValueError("计数或有效等待时长超出已有数据口径。")
    totals = daily[SUM_COLS].sum()
    if (int(totals.trip_count) != audit["trips"]
            or int(totals.valid_wait_count) != audit["valid_wait_trips"]
            or not np.isclose(totals.total_valid_wait_minutes,
                              audit["total_valid_wait_minutes"], rtol=1e-12)):
        raise ValueError("汇总表与修正后 L1 的核查总数不一致。")

    daily["weekday_index"] = daily.pickup_date.dt.dayofweek
    grouped = daily.groupby(["weekday_index", "pickup_hour"], sort=True)
    table = grouped[SUM_COLS].sum()
    table["calendar_days"] = grouped.pickup_date.nunique()
    table["avg_valid_wait_minutes"] = (
        table.total_valid_wait_minutes / table.valid_wait_count.where(table.valid_wait_count > 0)
    )
    table["valid_wait_rate"] = table.valid_wait_count / table.trip_count
    table["mean_daily_trips"] = table.trip_count / table.calendar_days
    table = table.reset_index()
    table.insert(1, "weekday", table.weekday_index.map(dict(enumerate(WEEK))))
    heat = table.pivot(index="weekday", columns="pickup_hour", values="avg_valid_wait_minutes")
    heat = heat.reindex(index=WEEK, columns=range(24))
    if heat.shape != (7, 24) or heat.isna().any().any():
        raise ValueError("星期×小时的有效等待均值不完整。")

    # 与现有行程量热力图逐格对账，保证两张图使用相同时间和行程范围。
    trips = table.pivot(index="weekday", columns="pickup_hour", values="mean_daily_trips")
    trips = trips.reindex(index=WEEK, columns=range(24))
    reference = pd.read_csv(OUTPUT_ROOT / "02_weekday_hour.csv", index_col="weekday")
    reference.columns = reference.columns.astype(int)
    if not np.allclose(trips, reference.loc[WEEK, range(24)], rtol=1e-12, atol=1e-8):
        raise ValueError("新图使用的行程量与原行程量热力图不一致。")
    hourly = daily.groupby("pickup_hour")[SUM_COLS].sum()
    hourly_mean = hourly.total_valid_wait_minutes / hourly.valid_wait_count
    old_hourly = pd.read_csv(OUT / "hourly_wait.csv").set_index("pickup_hour")
    if not np.allclose(hourly_mean, old_hourly.loc[range(24), "avg_valid_wait_minutes"], rtol=1e-12):
        raise ValueError("等待均值与现有24小时分析不一致。")

    table.to_csv(OUT / "weekday_hour_wait.csv", index=False)
    heat.to_csv(OUT / "weekday_hour_wait_matrix.csv")
    sns.set_theme(style="whitegrid", context="notebook", font="DejaVu Sans")
    fig, ax = plt.subplots(figsize=(11, 4.3))
    sns.heatmap(
        heat, cmap="YlGnBu", ax=ax,
        vmin=np.floor(heat.to_numpy().min() * 2) / 2,
        vmax=np.ceil(heat.to_numpy().max() * 2) / 2,
        cbar_kws={"label": "Average wait (minutes)"},
    )
    ax.set(title="Average waiting time by weekday and hour, 2023–2024",
           xlabel="Pickup hour (0–23, local time)", ylabel="")
    ax.tick_params(axis="both", labelrotation=0)
    fig.subplots_adjust(left=.07, right=.98, top=.88, bottom=.22)
    fig.text(.07, .065,
             "Both platforms pooled. Request-to-pickup wait, valid records (0–120 min).",
             fontsize=8.5, color="#626262")
    fig.text(.07, .02,
             "Each cell = total valid waiting minutes / valid trip count for that weekday and pickup hour.",
             fontsize=8.5, color="#626262")
    for extension in ["png", "pdf"]:
        fig.savefig(OUT / f"weekday_hour_wait.{extension}", dpi=240,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)

    peak = table.loc[table.avg_valid_wait_minutes.idxmax()]
    low = table.loc[table.avg_valid_wait_minutes.idxmin()]
    metric_cols = ["weekday", "pickup_hour", "avg_valid_wait_minutes",
                   "valid_wait_count", "valid_wait_rate", "mean_daily_trips"]
    summary = {
        "source": str(source), "period": audit["period"],
        "date_hour_rows": len(daily), "heatmap_cells": len(table),
        "formula": "SUM(total_valid_wait_minutes) / SUM(valid_wait_count)",
        "time_basis": "Pickup weekday and pickup hour, local clock time",
        "population": "Both platforms and all pickup zones pooled",
        "validity_rule": audit["validity_rule_from_handoff"],
        "trips": int(totals.trip_count), "valid_wait_trips": int(totals.valid_wait_count),
        "overall_mean_minutes": float(totals.total_valid_wait_minutes / totals.valid_wait_count),
        "highest_cell": peak[metric_cols].to_dict(),
        "lowest_cell": low[metric_cols].to_dict(),
        "checks": ["Complete date-hour grid without duplicates",
                   "Totals match corrected L1 audit",
                   "All 168 trip-volume cells match original trip heatmap",
                   "All 24 hourly waiting means match existing hourly analysis"],
    }
    (OUT / "weekday_hour_wait_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("TOP_5_WAIT_CELLS")
    print(table.nlargest(5, "avg_valid_wait_minutes")[metric_cols].round(4).to_string(index=False))


if __name__ == "__main__":
    run()
