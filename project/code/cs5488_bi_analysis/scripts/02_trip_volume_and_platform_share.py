"""纽约网约车 BI：python scripts/02_trip_volume_and_platform_share.py；只做 L2：python scripts/02_trip_volume_and_platform_share.py --l2-only。"""

from _config import PROJECT_ROOT, DATA_ROOT, OUTPUT_ROOT, LOOKUP_FILE
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # 直接保存图片，运行时不用弹出绘图窗口
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter
import pandas as pd
import seaborn as sns

PROJECT = PROJECT_ROOT
DATA = DATA_ROOT
LOOKUP = LOOKUP_FILE
OUT = OUTPUT_ROOT
DAYS = pd.date_range("2023-01-01", "2024-12-31")
EXPECTED_TRIPS = 471_960_468   # 当前交付版本的对账数；更换数据时同步修改
WEEK = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
BLUE, ORANGE = "#17637b", "#d88535"


def read_table(root, name, columns):
    folder = root / name
    if not folder.is_dir():
        raise SystemExit(f"未找到 {folder}；请用 --data 指向含 l2_overview 的父目录。")
    # 文件内部已有 pickup_date；不要让 Arrow 再从目录推断同名分区列
    frame = pd.read_parquet(folder, columns=columns, engine="pyarrow", partitioning=None)
    print(f"{name}: {len(frame):,} 行，选列后的表占 {frame.memory_usage(deep=True).sum()/1e6:.1f} MB")
    return frame


def save_fig(fig, out, name):
    fig.savefig(out / (name + ".png"), dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def format_time_axis(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", rotation=30)


def l2_figures(root, out):
    # 第 1 步：读取 L2；下面所有汇总加的是 trip_count
    l2 = read_table(root, "l2_overview", ["pickup_date", "pickup_hour", "PULocationID", "trip_count"])
    hourly = l2.groupby(["pickup_date", "pickup_hour"], observed=True)["trip_count"].sum()
    zones = l2.groupby("PULocationID", as_index=False)["trip_count"].sum()
    del l2
    hourly = hourly.unstack(fill_value=0)
    hourly.index = pd.to_datetime(hourly.index.astype(str))
    hourly = hourly.sort_index()
    if not hourly.index.equals(DAYS) or not hourly.columns.isin(range(24)).all():
        raise ValueError("日期覆盖或小时不符合当前两年数据，停止生成正式图。")
    hourly = hourly.reindex(columns=range(24), fill_value=0)

    # 第 2 步：每日趋势，一天一行。滚动窗口含当天及之前 6 天
    daily = hourly.sum(axis=1).rename("trips").to_frame()
    daily.index.name = "pickup_date"
    daily["mean_7d"] = daily["trips"].rolling(7).mean()
    total = int(daily.trips.sum())
    if total != EXPECTED_TRIPS:
        raise ValueError(f"L2 总量 {total:,} 不符合已核查版本")
    daily.to_csv(out / "01_daily.csv")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(daily.index, daily.trips / 1000, color="#acc6d0", lw=.9, label="Daily")
    ax.plot(daily.index, daily.mean_7d / 1000, color=BLUE, lw=2, label="7-day mean")
    ax.set(title="Recorded trips by pickup date", xlabel="", ylabel="Trips (thousands)")
    ax.set_xlim(daily.index.min(), daily.index.max())
    format_time_axis(ax)
    ax.legend(frameon=False)
    save_fig(fig, out, "01_daily")

    # 第 3 步：每天已经只有 24 个数；按星期取均值，分母就是该星期的日历天数
    heat = hourly.groupby(hourly.index.dayofweek).mean()
    heat.index = pd.Index(WEEK, name="weekday")
    heat.to_csv(out / "02_weekday_hour.csv")
    fig, ax = plt.subplots(figsize=(11, 3.8))
    sns.heatmap(heat / 1000, cmap="YlGnBu", ax=ax, cbar_kws={"label": "Mean trips (thousands)"})
    ax.set(title="When do pickups concentrate?  |  2023–2024", xlabel="Pickup hour (0–23)", ylabel="")
    ax.tick_params(axis="both", labelrotation=0)
    save_fig(fig, out, "02_weekday_hour")

    # 第 4 步：每个上车区一个总数，再关联官方区名；排名分母仍是全部行程
    lookup = pd.read_csv(LOOKUP, keep_default_na=False, usecols=["LocationID", "Zone", "Borough"])
    zones = zones.merge(lookup, how="left", left_on="PULocationID", right_on="LocationID",
                        validate="many_to_one")
    if int(zones.trip_count.sum()) != total:
        raise ValueError("区域表关联改变了行程总数")
    zones["Zone"] = zones["Zone"].fillna("Unmatched")
    zones["share_of_all"] = zones.trip_count / total
    zones = zones.sort_values("trip_count", ascending=False)
    zones.to_csv(out / "03_zones.csv", index=False)
    top = zones.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 5.8))
    labels = top.Zone + "  ·  " + top.PULocationID.astype(str)
    ax.barh(labels, top.trip_count / 1e6, color=BLUE)
    ax.set(title=f"Top 15 pickup zones  |  {top.share_of_all.sum():.1%} of all recorded trips",
           xlabel="Trips (millions), 2023–2024", ylabel="")
    save_fig(fig, out, "03_zones")
    return daily.trips


def platform_figure(root, out, daily):
    # 第 5 步（只在分析平台时需要）：读 L1，先按日期/平台汇总，再转月份
    l1 = read_table(root, "l1_detail", ["pickup_date", "hvfhs_license_num", "trip_count"])
    plate = l1.groupby(["pickup_date", "hvfhs_license_num"], observed=True).trip_count.sum().unstack(fill_value=0)
    del l1
    plate.index = pd.to_datetime(plate.index.astype(str))
    if not plate.sort_index().sum(axis=1).equals(daily):
        raise ValueError("L1 与 L2 的每日总量对不上")
    monthly = plate.groupby(plate.index.to_period("M")).sum()
    monthly.index.name = "month"
    share = monthly.div(monthly.sum(axis=1), axis=0)
    table = pd.concat([monthly.stack().rename("trips"), share.stack().rename("share")], axis=1)
    table.to_csv(out / "04_platform.csv")
    fig, ax = plt.subplots(figsize=(10, 4))
    positions = list(range(len(share)))
    bottom = pd.Series(0., index=share.index)
    for platform, color in zip(share.columns, [BLUE, ORANGE], strict=True):
        ax.bar(positions, share[platform].to_numpy(), bottom=bottom.to_numpy(),
               label=platform, width=.84, color=color)
        bottom = bottom + share[platform]
    ax.set_xticks(range(0, len(share), 3), labels=share.index.astype(str)[::3], rotation=30)
    ax.set(title="Platform share of recorded trips by month", xlabel="Pickup month",
           ylabel="Share of trips", ylim=(0, 1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.24), ncol=2, frameon=False)
    save_fig(fig, out, "04_platform")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA, help="含 l2_overview 和 l1_detail 的文件夹")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--l2-only", action="store_true", help="只做前三张图")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    daily = l2_figures(args.data.expanduser(), args.out)
    if not args.l2_only:
        platform_figure(args.data.expanduser(), args.out, daily)
    print(f"已核对 {len(daily)} 天、{int(daily.sum()):,} 条行程。图和 CSV：{args.out.resolve()}")
