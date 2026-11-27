"""原始 daily 行程的 7/31 天平均、公历月平均与描述性月度因子。"""

from _config import PROJECT_ROOT, DATA_ROOT, OUTPUT_ROOT, LOOKUP_FILE
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter
from scipy.interpolate import PchipInterpolator

ROOT = PROJECT_ROOT
SOURCE = OUTPUT_ROOT / "01_daily.csv"
OUT = OUTPUT_ROOT / "time_series"
BLUE, ORANGE = "#17637b", "#d88535"


def block_means(y, window):
    # 从第一天起，每 window 天为一组；分组之间不重叠，不按公历月份切分。
    source = y.rename("trips").to_frame().reset_index()
    source["block"] = np.arange(len(source)) // window
    table = source.groupby("block").agg(
        start_date=("pickup_date", "min"),
        end_date=("pickup_date", "max"),
        days=("trips", "size"),
        total_trips=("trips", "sum"),
        mean_trips_per_day=("trips", "mean"),
    )
    table["midpoint"] = table.start_date + (table.end_date - table.start_date) / 2
    table["complete_block"] = table.days == window
    if int(table.total_trips.sum()) != int(y.sum()):
        raise ValueError("分组后行程总数未对齐。")
    return table


def base_figure(y, title):
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    fig.subplots_adjust(left=.09, right=.98, top=.88, bottom=.22)
    ax.plot(y.index, y / 1000, color="#d5dfe2", lw=.6, label="Observed daily trips", zorder=1)
    ax.set(title=title, ylabel="Trips per day (thousands)",
           xlim=(y.index.min(), y.index.max()), ylim=(380, 930))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.grid(axis="y", alpha=.18)
    ax.set_axisbelow(True)
    return fig, ax


def save_figure(fig, name):
    for extension in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{extension}", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_monthly_pattern(y):
    # 每个公历月一个日均值；不用固定 31 天分组，也不使用 STL 成分。
    daily = y.to_frame().reset_index()
    daily["year"] = daily.pickup_date.dt.year
    daily["month"] = daily.pickup_date.dt.month
    monthly = daily.groupby(["year", "month"], as_index=False).agg(
        start_date=("pickup_date", "min"),
        end_date=("pickup_date", "max"),
        days=("trips", "size"),
        total_trips=("trips", "sum"),
        mean_trips_per_day=("trips", "mean"),
    )
    yearly = daily.groupby("year").trips.agg(["sum", "size", "mean"])
    monthly["year_mean_trips_per_day"] = monthly.year.map(yearly["mean"])
    monthly["seasonal_factor"] = monthly.mean_trips_per_day / monthly.year_mean_trips_per_day
    monthly["factor_minus_one"] = monthly.seasonal_factor - 1
    monthly["midpoint"] = monthly.start_date + (monthly.end_date - monthly.start_date) / 2
    if len(monthly) != 24 or not monthly.groupby("year").size().eq(12).all():
        raise ValueError("应覆盖两年各 12 个完整月份。")
    if not monthly.days.eq(monthly.start_date.dt.days_in_month).all():
        raise ValueError("月份不完整，不生成正式月度对比图。")
    if int(monthly.total_trips.sum()) != int(y.sum()):
        raise ValueError("月度汇总总行程数不一致。")
    for year, part in monthly.groupby("year"):
        np.testing.assert_allclose(
            np.average(part.seasonal_factor, weights=part.days), 1.0, rtol=1e-12,
        )
        np.testing.assert_allclose(
            part.mean_trips_per_day * part.days, part.total_trips, rtol=1e-12,
        )
    # 对照此前已经生成的实际月内日均表，确保没有换成模型拟合值。
    prior = pd.read_csv(OUT / "monthly_daily_means.csv", index_col="month")
    current = monthly.pivot(index="month", columns="year", values="mean_trips_per_day")
    np.testing.assert_allclose(current[[2023, 2024]], prior[["2023", "2024"]], rtol=1e-12)
    monthly.to_csv(OUT / "monthly_seasonal_factors.csv", index=False)

    # 仅改变连线方式：在 24 个实际月均值之间做保形平滑插值。
    # 不外推到首末月中点之外，不改变用于下图的月均值或季节因子。
    month_x = mdates.date2num(monthly.midpoint.to_numpy())
    interpolator = PchipInterpolator(month_x, monthly.mean_trips_per_day.to_numpy(), extrapolate=False)
    curve_x = np.linspace(month_x[0], month_x[-1], 1200)
    curve_y = interpolator(curve_x)
    np.testing.assert_allclose(interpolator(month_x), monthly.mean_trips_per_day, rtol=1e-12)

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.8))
    fig.subplots_adjust(left=.105, right=.975, top=.93, bottom=.15, hspace=.34)
    ax = axes[0]
    ax.plot(y.index, y / 1000, color="#aec9d1", lw=.7, label="Observed daily trips")
    ax.plot(mdates.num2date(curve_x), curve_y / 1000, color=BLUE,
            lw=2.2, label="Monthly means (smooth interpolation)")
    ax.set(title="Daily trips and monthly averages, 2023–2024",
           ylabel="Trips per day (thousands)",
           xlim=(y.index.min(), y.index.max()), ylim=(380, 930))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.legend(frameon=False, ncol=2, loc="lower left", fontsize=9)
    ax.grid(axis="y", alpha=.2)

    ax = axes[1]
    positions = np.arange(12)
    for offset, year, color in ((-.19, 2023, BLUE), (.19, 2024, ORANGE)):
        values = monthly.loc[monthly.year == year].set_index("month").factor_minus_one.reindex(range(1, 13))
        ax.bar(positions + offset, values.to_numpy(), width=.35, color=color, label=str(year))
    ax.set_xticks(positions, ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.axhline(0, color="#555555", lw=.9)
    ax.set(title="Monthly seasonal factors relative to each year's daily mean",
           ylabel="Seasonal factor minus 1", xlabel="Pickup month", ylim=(-.10, .09))
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=1))
    ax.legend(frameon=False, ncol=2, loc="upper left")
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
    fig.text(.105, .035,
             "Top: smooth interpolation through monthly means. Factor = monthly daily mean / that year's daily mean; bars show factor − 1.\n"
             "Descriptive ratios, not STL factors: within-year trend and weekday mix are not removed. Only two years are available.",
             fontsize=8.1, color="#555555", linespacing=1.5)
    save_figure(fig, "monthly_pattern")

    summary = {
        "source": str(SOURCE), "observations": len(y), "total_trips": int(y.sum()),
        "calendar_months": len(monthly),
        "top_panel": "Observed daily trips plus PCHIP smooth interpolation through calendar-month daily means at month midpoints; no extrapolation",
        "factor_definition": "Calendar-month mean daily trips divided by mean daily trips in the same calendar year",
        "bar_definition": "seasonal_factor - 1, shown as percentage",
        "normalization": "Within each year the day-weighted arithmetic mean of monthly factors is 1",
        "year_daily_means": {str(k): float(v) for k, v in yearly["mean"].items()},
        "caveats": ["Descriptive seasonal indices, not STL or MSTL seasonal components.",
                    "Not adjusted for within-year trend, weekday mix, holidays or other effects.",
                    "Two annual cycles cannot establish stable recurring annual seasonality."],
        "checks": ["All 24 calendar months complete, including February 2024 with 29 days",
                   "Monthly sums reconcile to source", "Day-weighted factors average to 1 within each year",
                   "Monthly daily means match the earlier monthly_daily_means.csv"],
    }
    (OUT / "monthly_pattern_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(monthly[["year", "month", "days", "mean_trips_per_day", "seasonal_factor"]].round(4).to_string(index=False))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(SOURCE, parse_dates=["pickup_date"]).set_index("pickup_date").sort_index()
    y = source.trips
    expected = pd.date_range("2023-01-01", "2024-12-31", name="pickup_date")
    if not y.index.equals(expected) or y.isna().any() or not np.isfinite(y).all() or (y <= 0).any():
        raise ValueError("daily 数据日期或计数无效。")
    if int(y.sum()) != 471_960_468:
        raise ValueError("总量与当前已核对版本不同。")

    rolling = y.to_frame()
    blocks = {}
    for window in (7, 31):
        # 每天计算当天及此前 window-1 天的算术平均；不足完整窗口时留空。
        rolling[f"mean_{window}d"] = y.rolling(window, min_periods=window).mean()
        np.testing.assert_allclose(
            rolling[f"mean_{window}d"].iloc[window-1:].to_numpy(),
            np.convolve(y.to_numpy(dtype=float), np.ones(window) / window, mode="valid"),
            rtol=1e-12,
        )
        blocks[window] = block_means(y, window)
        blocks[window].to_csv(OUT / f"daily_mean_{window}d_blocks.csv")
    # 保持与原 daily 文件中的 7 天移动平均定义一致。
    np.testing.assert_allclose(rolling.mean_7d, source.mean_7d, equal_nan=True, rtol=1e-12)
    rolling.to_csv(OUT / "daily_mean_7d_31d_rolling.csv")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = base_figure(y, "Daily trips averaged in 7-day and 31-day blocks, 2023–2024")
    for window, color, width in ((7, BLUE, 1.4), (31, ORANGE, 2.4)):
        table = blocks[window].loc[blocks[window].complete_block]
        ax.plot(table.midpoint, table.mean_trips_per_day / 1000,
                color=color, lw=width, marker="o", ms=2.5 if window == 7 else 4,
                label=f"One mean per {window}-day block", zorder=3)
    ax.legend(frameon=False, ncol=3, fontsize=9, loc="upper left")
    remainder7 = int(blocks[7].loc[~blocks[7].complete_block, "days"].sum())
    remainder31 = int(blocks[31].loc[~blocks[31].complete_block, "days"].sum())
    fig.text(.09, .07,
             "Non-overlapping blocks begin on 2023-01-01; each mean is plotted at the block midpoint.\n"
             f"Only full blocks are plotted. The final {remainder7} days (7-day groups) / {remainder31} days (31-day groups) are retained in CSV only.",
             fontsize=8.5, color="#555555", linespacing=1.5)
    save_figure(fig, "daily_mean_7d_31d_blocks")

    fig, ax = base_figure(y, "Daily trips with 7-day and 31-day rolling means, 2023–2024")
    for window, color, width in ((7, BLUE, 1.5), (31, ORANGE, 2.3)):
        ax.plot(rolling.index, rolling[f"mean_{window}d"] / 1000,
                color=color, lw=width, label=f"{window}-day rolling mean", zorder=3)
    ax.legend(frameon=False, ncol=3, fontsize=9, loc="upper left")
    fig.text(.09, .07,
             "Each date averages that day and the preceding 6 or 30 days. Both lines use the same observed daily counts.\n"
             "Only full windows are shown. These are arithmetic averages, not seasonal adjustments or STL trend estimates.",
             fontsize=8.5, color="#555555", linespacing=1.5)
    save_figure(fig, "daily_mean_7d_31d_rolling")
    draw_monthly_pattern(y)

    common = rolling[["mean_7d", "mean_31d"]].dropna()
    summary = {
        "source": str(SOURCE), "observations": len(y), "total_trips": int(y.sum()),
        "block_origin": "2023-01-01",
        "block_method": "Non-overlapping 7/31-day arithmetic means, plotted at block midpoint; complete blocks only",
        "full_blocks": {str(w): int(blocks[w].complete_block.sum()) for w in (7, 31)},
        "partial_block_days_excluded_from_plots": {"7": remainder7, "31": remainder31},
        "rolling_method": "Trailing arithmetic mean with min_periods equal to window",
        "mean_absolute_day_to_day_change_on_common_rolling_dates": {
            name: float(common[name].diff().abs().dropna().mean()) for name in common.columns
        },
        "checks": "Counts reconcile; rolling means match direct convolution; 7-day mean matches original daily file",
    }
    (OUT / "daily_means_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
