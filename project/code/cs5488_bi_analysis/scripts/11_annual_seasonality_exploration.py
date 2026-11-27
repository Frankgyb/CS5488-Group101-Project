"""在已有 daily 数据上分解周周期、约 365 天周期和慢趋势。仅作回顾性探索。"""

from _config import PROJECT_ROOT, DATA_ROOT, OUTPUT_ROOT, LOOKUP_FILE
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels
from statsmodels.api import RLM, robust
from statsmodels.tsa.seasonal import MSTL, STL


ROOT = PROJECT_ROOT
SOURCE = OUTPUT_ROOT / "01_daily.csv"
TIME_SERIES = OUTPUT_ROOT / "time_series"
OUT = TIME_SERIES / "annual_decomposition"
BLUE, ORANGE, GREEN = "#17637b", "#d88535", "#397b69"
MAIN_WINDOW = 731
TREND_WINDOWS = (551, 731, 1097)
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def read_daily():
    source = pd.read_csv(SOURCE, parse_dates=["pickup_date"])
    if source.pickup_date.duplicated().any():
        raise ValueError("日数据含重复日期。")
    y = source.set_index("pickup_date").sort_index().trips.asfreq("D")
    expected = pd.date_range("2023-01-01", "2024-12-31", freq="D")
    if not y.index.equals(expected):
        raise ValueError("日期范围与当前两年版本不符。")
    if y.isna().any() or not np.isfinite(y).all() or (y <= 0).any():
        raise ValueError("日期缺失或计数无效；不自动填零、不插值。")
    if len(y) != 731 or int(y.sum()) != 471_960_468:
        raise ValueError("每日计数与已核对的全量版本不一致。")
    return y


def fit_mstl(y, trend_window):
    # 7、365 是周期；13、13 是各季节子序列的平滑窗口。
    # 只有两轮年度观测。seasonal_deg=0 限制季节子序列的局部变化，
    # 避免默认一次局部回归用两点逐日插值、几乎吸收全部残差。
    # 保留 2024-02-29：365 天是近似周期，不是精确公历年度模型。
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit = MSTL(
            np.log(y), periods=(7, 365), windows=(13, 13), iterate=2,
            stl_kwargs={"robust": True, "trend": trend_window, "seasonal_deg": 0},
        ).fit()
    if caught:
        raise ValueError("MSTL 发出警告，请检查周期是否保留：" +
                         "; ".join(str(w.message) for w in caught))
    if set(fit.seasonal.columns) != {"seasonal_7", "seasonal_365"}:
        raise ValueError("周周期或年度周期未成功估计。")
    # 将各季节因子的全样本几何平均规范化为 1，常数移回趋势。
    centers = fit.seasonal.mean()
    result = pd.DataFrame({
        "trips": y,
        "log_trend": fit.trend + centers.sum(),
        "log_weekly": fit.seasonal.seasonal_7 - centers.seasonal_7,
        "log_annual": fit.seasonal.seasonal_365 - centers.seasonal_365,
        "log_remainder": fit.resid,
    })
    for component in ("trend", "weekly", "annual"):
        name = "trend_level" if component == "trend" else f"{component}_factor"
        result[name] = np.exp(result[f"log_{component}"])
    result["seasonally_adjusted_trips"] = (
        y / result.weekly_factor / result.annual_factor
    )
    result["remainder_fraction"] = np.expm1(result.log_remainder)
    result["annual_factor_display_31d"] = np.exp(
        result.log_annual.rolling(31, center=True, min_periods=31).mean()
    )
    # 在周因子、残差因子均取 1 的同一基线上，将年度因素换算为每日行程增减。
    # 这是相对趋势的模型偏移量；乘法分解各因素不能直接作为可相加的订单份额。
    result["annual_trip_deviation"] = result.trend_level * (result.annual_factor - 1)
    result["annual_trip_deviation_display_31d"] = (
        result.annual_trip_deviation.rolling(31, center=True, min_periods=31).mean()
    )
    # 数值校验是分解恒等式和归一化，不能证明年度因素可识别。
    log_sum = result[["log_trend", "log_weekly", "log_annual", "log_remainder"]].sum(axis=1)
    np.testing.assert_allclose(log_sum, np.log(y), rtol=0, atol=1e-10)
    np.testing.assert_allclose(
        np.exp(result[["log_weekly", "log_annual"]].mean()), [1, 1], atol=1e-10,
    )
    np.testing.assert_allclose(
        result.trend_level + result.annual_trip_deviation,
        result.trend_level * result.annual_factor, rtol=1e-12,
    )
    display_columns = ["annual_factor_display_31d", "annual_trip_deviation_display_31d"]
    if not np.isfinite(result.drop(columns=display_columns)).all().all():
        raise ValueError("分解结果含非有限数值。")
    return result


def fourier_crosscheck(y):
    """少量公历年度谐波 + 星期控制 + 线性时间项，核对增长方向。"""
    t_years = np.arange(len(y)) / 365.25
    phase = (y.index.dayofyear.to_numpy() - 1) / np.where(y.index.is_leap_year, 366, 365)
    rows = []
    for harmonics in (2, 3, 4):
        columns = {"intercept": np.ones(len(y)), "time_years": t_years}
        for day in range(1, 7):
            columns[f"weekday_{day}"] = (y.index.dayofweek == day).astype(float)
        for k in range(1, harmonics + 1):
            columns[f"sin_{k}"] = np.sin(2 * np.pi * k * phase)
            columns[f"cos_{k}"] = np.cos(2 * np.pi * k * phase)
        design = pd.DataFrame(columns, index=y.index)
        fit = RLM(np.log(y), design, M=robust.norms.HuberT()).fit()
        rows.append({
            "annual_harmonics": harmonics,
            "annualized_loglinear_growth": float(np.expm1(fit.params.time_years)),
        })
    # 这里检查的是不同形状约束下的增长方向，不做独立样本显著性推断。
    return pd.DataFrame(rows)


def save_figure(fig, name):
    for extension in ("png", "pdf"):
        fig.savefig(TIME_SERIES / f"{name}.{extension}", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_decomposition(y, main, trends, weekly_only, summary):
    fig, axes = plt.subplots(3, 1, figsize=(11.8, 9.8), sharex=True)
    fig.subplots_adjust(left=.105, right=.975, top=.90, bottom=.12, hspace=.38)
    fig.suptitle("Daily trips: annual variation and slower change", x=.105, y=.982,
                 ha="left", fontsize=17, fontweight="bold")
    fig.text(.105, .944, "NYC high-volume FHV, 2023–2024 · exploratory decomposition with only two annual cycles",
             fontsize=10, color="#555555")
    ax = axes[0]
    ax.plot(y.index, y / 1000, color="#bdcdd1", lw=.75, label="Observed daily trips")
    ax.plot(y.index, weekly_only / 1000, color=BLUE, lw=1.9,
            label="Previous weekly-only STL trend")
    ax.set(title="1. Removing the weekly cycle still leaves variation within each year",
           ylabel="Trips per day\n(thousands)")
    ax.legend(frameon=False, ncol=2, loc="upper left", fontsize=9)

    ax = axes[1]
    ax.plot(y.index, main.annual_trip_deviation / 1000, color="#edcfb1", lw=.7,
            label="Annual deviation from trend")
    ax.plot(y.index, main.annual_trip_deviation_display_31d / 1000, color=ORANGE, lw=2,
            label="31-day mean")
    ax.axhline(0, color="#737373", lw=.8)
    ax.set(title="2. Annual variation expressed as trips above or below trend",
           ylabel="Annual deviation\n(thousand trips/day)")
    ax.legend(frameon=False, ncol=2, loc="lower left", fontsize=9)

    ax = axes[2]
    ax.fill_between(y.index, trends.min(axis=1) / 1000, trends.max(axis=1) / 1000,
                    color="#a7c6bd", alpha=.6, label="Range across three smoothing settings")
    ax.plot(y.index, main.trend_level / 1000, color=GREEN, lw=2.5,
            label="Trend after weekly + annual decomposition")
    ax.set(title="3. The remaining slow trend rises gradually", ylabel="Typical trend level\n(thousand trips/day)")
    ax.set_ylim(590, 680)
    ax.legend(frameon=False, loc="upper left", fontsize=8.8)
    change = summary["trend_year_mean_change"]
    ax.text(.98, .15, f"2024 vs 2023 mean trend level: {change:+.1%}",
            transform=ax.transAxes, ha="right", fontsize=11, fontweight="bold", color=GREEN)

    for ax in axes:
        ax.set_xlim(y.index.min(), y.index.max())
        ax.axvline(pd.Timestamp("2024-01-01"), color="#858585", ls="--", lw=.8, alpha=.65)
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
        ax.title.set_fontsize(11)
        ax.title.set_ha("left")
        ax.title.set_position((0, 1))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.text(.105, .035,
             "Annual deviation = trend × (annual factor − 1), with weekly and residual factors set to 1. Zero means no annual deviation.\n"
             "Log-MSTL: periods 7 and 365 days; locally constant seasonal smoothing. Shading shows parameter sensitivity, not confidence.\n"
             "Only two annual cycles: annual pattern and trend curvature remain uncertain. All 731 days, including Feb 29, are retained.",
             fontsize=8.3, color="#555555", linespacing=1.6)
    save_figure(fig, "01_annual_decomposition")


def draw_year_comparison(y, main):
    table = pd.DataFrame({"observed_daily_mean": y, "trend_daily_mean": main.trend_level})
    monthly = table.groupby([table.index.year, table.index.month]).mean()
    monthly.index.names = ["year", "month"]
    monthly.to_csv(OUT / "monthly_comparison.csv")
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.2), sharex=True)
    fig.subplots_adjust(left=.11, right=.97, top=.86, bottom=.12, hspace=.32)
    fig.suptitle("Compare 2023 and 2024 on the same calendar", x=.11, y=.97,
                 ha="left", fontsize=16, fontweight="bold")
    fig.text(.11, .92, "Monthly averages of daily values; February 2024 includes all 29 days", fontsize=10, color="#555555")
    for year, color in ((2023, BLUE), (2024, ORANGE)):
        block = monthly.loc[year]
        axes[0].plot(block.index, block.observed_daily_mean / 1000, "o-", color=color,
                     label=str(year), lw=2, ms=4)
        axes[1].plot(block.index, block.trend_daily_mean / 1000, "o-", color=color,
                     label=str(year), lw=2, ms=4)
    axes[0].set(title="Recorded daily trips averaged within each month", ylabel="Mean trips/day\n(thousands)")
    axes[1].set(title="Estimated slow trend after weekly + annual decomposition",
                ylabel="Typical trend level\n(thousand trips/day)", xticks=range(1, 13), xticklabels=MONTHS)
    for ax in axes:
        ax.grid(axis="y", alpha=.18)
        ax.legend(frameon=False, ncol=2, loc="upper left")
        ax.set_xlim(.8, 12.2)
        ax.set_ylim(570, 720)
        ax.title.set_fontsize(11)
    fig.text(.11, .045, "Lower panel is a model component, not observed monthly counts. Two annual cycles only; no forecast is shown.",
             fontsize=8.7, color="#555555")
    save_figure(fig, "02_year_comparison")


def draw_annual_pattern(y, main):
    """沿用 weekly_pattern 的上下结构，以月份展示年度行程偏移量。"""
    monthly = main.groupby([main.index.year, main.index.month]).agg(
        days=("trips", "size"),
        observed_mean_trips_per_day=("trips", "mean"),
        trend_mean_trips_per_day=("trend_level", "mean"),
        annual_deviation_mean_trips_per_day=("annual_trip_deviation", "mean"),
    )
    monthly.index.names = ["year", "month"]
    # 按日求偏移量再算月内日均，避免以月度总量混入月份天数差异。
    np.testing.assert_allclose(
        (monthly.annual_deviation_mean_trips_per_day * monthly.days).sum(),
        main.annual_trip_deviation.sum(), rtol=1e-10,
    )
    monthly.to_csv(TIME_SERIES / "annual_pattern_by_month.csv")

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.8))
    fig.subplots_adjust(left=.105, right=.975, top=.93, bottom=.16, hspace=.34)
    ax = axes[0]
    ax.plot(y.index, y / 1000, color="#aec9d1", lw=.7, label="Observed daily trips")
    ax.plot(y.index, main.trend_level / 1000, color=BLUE, lw=2.2,
            label="Slow trend after weekly + annual adjustment")
    ax.set(title="Daily trips and slow trend, 2023–2024", ylabel="Trips per day\n(thousands)")
    ax.set_xlim(y.index.min(), y.index.max())
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.legend(frameon=False, ncol=2, loc="lower left", fontsize=8.5)
    ax.grid(axis="y", alpha=.2)

    ax = axes[1]
    x = np.arange(12)
    for offset, year, color in ((-.19, 2023, BLUE), (.19, 2024, ORANGE)):
        values = monthly.loc[year].annual_deviation_mean_trips_per_day.reindex(range(1, 13))
        ax.bar(x + offset, values.to_numpy() / 1000, width=.35, color=color, label=str(year))
    ax.set_xticks(x, MONTHS)
    ax.axhline(0, color="#555555", lw=.9)
    ax.set(title="Estimated annual component by month",
           ylabel="Annual deviation\n(thousand trips/day)", xlabel="Pickup month")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + .22 * (hi - lo))
    ax.legend(frameon=False, ncol=2, loc="upper right")
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
    fig.text(.105, .02,
             "Bars: monthly mean of trend × (annual factor − 1); positive / negative means above / below the trend baseline.\n"
             "Both years come from one joint fit. Similar bars are not independent evidence of a recurring annual pattern.\n"
             "Only two annual cycles are available; the annual component and slow trend remain exploratory.",
             fontsize=8.1, color="#555555", linespacing=1.5)
    save_figure(fig, "annual_pattern")


def write_readme(summary, annual, sensitivity, fourier):
    raw23, raw24 = annual.loc[2023], annual.loc[2024]
    content = f"""# 行程量：周周期、年度因素与慢趋势

## 本次新增

图片统一放在输出目录下的 `time_series/`，与原来的 `weekly_pattern.png/pdf` 并列；本子文件夹保留模型分解明细。

`annual_pattern.png/pdf`：参照周周期图的上下布局，上方为每日行程与剥离周、年度因素后的慢趋势，下方为 2023、2024 各月份的年度行程偏移量，单位为千笔/天。配套 `annual_pattern_by_month.csv` 也在上一层。

`01_annual_decomposition.png/pdf`：原始每日行程与原周周期趋势、年度因素对应的行程增减、剥离周与年度因素后的慢趋势。三个面板统一使用千笔行程/天，中图以零为参照。

`02_year_comparison.png/pdf`：将 2023、2024 放在同一月份轴上，对比实际月内日均行程量和模型慢趋势。

## 数据与结果

来源为输出目录内已核对的 `01_daily.csv`。2023-01-01 至 2024-12-31，共 731 天、471,960,468 笔行程。没有重新加载大型 L1，没有抽样，没有将缺失值补零。2024-02-29 保留。

| 年度 | 天数 | 实际总行程数 | 实际日均行程数 |
| --- | ---: | ---: | ---: |
| 2023 | 365 | {raw23.total_trips:,.0f} | {raw23.mean_daily_trips:,.0f} |
| 2024 | 366 | {raw24.total_trips:,.0f} | {raw24.mean_daily_trips:,.0f} |

实际日均行程量同比增长 {summary['observed_daily_mean_change']:.2%}，采用日均口径避免将闰年多一天误当成需求增长。

主模型的 2024 年平均趋势水平比 2023 年高 {summary['trend_year_mean_change']:.2%}。在 551、731、1097 三种趋势窗口下，该指标为 {sensitivity.trend_year_mean_change.min():.2%}–{sensitivity.trend_year_mean_change.max():.2%}。不同平滑设置均显示温和上升。

另外用星期控制、少量公历年度 Fourier 项和线性时间项作交叉检查，2–4 阶年度谐波得到的年化对数线性增长率为 {fourier.annualized_loglinear_growth.min():.2%}–{fourier.annualized_loglinear_growth.max():.2%}。该指标与两年平均趋势水平之比定义不同，只用于检查增长方向，不视为精度证明。

## 方法与解读

```text
log(每日行程量) = 慢趋势 + 周周期 + 年度因素 + 残差
每日行程量 = 趋势水平 × 周因子 × 年度因子 × 残差因子
```

使用 statsmodels {statsmodels.__version__} 的 MSTL：`periods=(7, 365)`、`windows=(13, 13)`、`iterate=2`、`robust=True`、`trend=731`、`seasonal_deg=0`。13 是季节子序列平滑窗口，731 是趋势平滑窗口；它们不代表额外季节周期。周和年度对数因子分别按全样本均值中心化，使因子几何平均为 1，常数移回趋势，保持分解恒等式。

只有两轮年度数据时，默认季节局部一次拟合几乎可以在每个年度位置插值。本数据试算的对数残差标准差仅约 0.00013。因此采用局部常数季节平滑，减少将偶发波动全部吸收入年度因素的问题。这个约束仍不能解决只有两轮年度观测的识别限制。

中图原来显示 `年度因子 − 1` 的百分比，现已换算为行程量单位：

```text
年度行程偏移量（笔/天） = 趋势水平 ×（年度因子 − 1）
```

这表示将周因子和残差因子固定为 1 时，加入年度因素后相对慢趋势基线的模型行程增减。正数表示高于基线，负数表示低于基线，零表示没有年度偏移。例如趋势水平为每天 640,000 笔、年度因子为 0.95，偏移量就是 −32,000 笔/天，图上为 −32（千笔/天）。它不是实际每日总行程量，也不是能与其他乘法成分直接相加的独立订单份额。

年度因素面板的浅线显示上述偏移量，粗线显示它的 31 天居中算术平均；首末 15 天不画平滑值。该平滑仅帮助看图，没有替换模型内用于计算趋势的季节成分。底层年度因子仍保留在 CSV 中。慢趋势面板阴影是三种趋势窗口的结果范围，不是置信区间。

`annual_pattern` 下方柱状图将未额外平滑的每日年度偏移量按各年各月求日均值，不用月总量，因此 28、29、30、31 天的月份可以按每日水平比较。两个年份的柱子来自同一个联合拟合，且模型对跨年的季节变化施加了平滑约束；柱形相似不能当作年度规律稳定的独立证据。

365 天周期近似年度，闰年日期可能错位一天，不能解释为精确的公历日期或节假日效应。Fourier 检查用各公历年的实际天数定位年内相位。

**适合报告的表述：**“2023–2024 年记录行程量呈明显的周期波动。剥离周周期与探索性的年度因素后，长期水平温和上升；2024 年实际日均行程量较 2023 年增长约 2.7%。因此总体规模较稳定，但并非完全无增长。”

**限制：**两年只提供两轮年度变化。趋势平滑窗口较长，会限制趋势曲率；不要据此断言没有局部拐点，也不能把提取的年度模式认定为已证实的稳定季节规律。年度因素可能混入天气、节假日及一次性事件。行程记录量也不等于所有潜在出行需求。该分析回顾性使用全部数据，不是预测；指数化的对数趋势是典型基线，不是经过偏差校正的算术期望订单量。

## 复现与数据文件

在项目根目录运行：

```bash
python scripts/11_annual_seasonality_exploration.py
```

- `daily_components.csv`：每日完整分解，包含季节调整后的行程量、年度因子、年度行程偏移量 `annual_trip_deviation` 及其 31 天显示用均值 `annual_trip_deviation_display_31d`。
- `annual_summary.csv`：真实年度计数、日均及趋势平均水平。
- `monthly_comparison.csv`：分年度的月内日均行程与趋势平均水平。
- `annual_factors_by_month.csv`：按年份和月份汇总的年度因子几何均值。
- `trend_sensitivity.csv`、`trend_sensitivity_summary.csv`：三种趋势窗口的每日结果和年际变化。
- `fourier_crosscheck.csv`：另一组年度曲线约束下的增长方向检查。
- `summary.json`：参数、版本与核查结果。

已核对完整日期、总行程数、两个季节周期均保留、分解重构恒等式及因子几何平均归一化。原周周期分析和原始 daily 文件保持不变。
"""
    (OUT / "README.md").write_text(content, encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    y = read_daily()
    fits = {w: fit_mstl(y, w) for w in TREND_WINDOWS}
    result = fits[MAIN_WINDOW]
    result.to_csv(OUT / "daily_components.csv")
    trends = pd.DataFrame({f"trend_{w}": fit.trend_level for w, fit in fits.items()})
    trends.to_csv(OUT / "trend_sensitivity.csv")
    sensitivity = pd.DataFrame([
        {"trend_window": w,
         "trend_year_mean_change": float(fit.trend_level.loc["2024"].mean() /
                                         fit.trend_level.loc["2023"].mean() - 1)}
        for w, fit in fits.items()
    ])
    sensitivity.to_csv(OUT / "trend_sensitivity_summary.csv", index=False)
    fourier = fourier_crosscheck(y)
    fourier.to_csv(OUT / "fourier_crosscheck.csv", index=False)
    annual = y.groupby(y.index.year).agg(days="size", total_trips="sum", mean_daily_trips="mean")
    annual.index.name = "year"
    annual["mean_trend_level"] = result.trend_level.groupby(y.index.year).mean()
    annual.to_csv(OUT / "annual_summary.csv")
    monthly_log = result.log_annual.groupby([y.index.year, y.index.month]).mean()
    monthly_log.index.names = ["year", "month"]
    np.exp(monthly_log).rename("geometric_mean_annual_factor").to_csv(OUT / "annual_factors_by_month.csv")
    weekly_only = np.exp(STL(np.log(y), period=7, seasonal=13, trend=31, robust=True).fit().trend)
    summary = {
        "input": str(SOURCE), "period": [str(y.index[0].date()), str(y.index[-1].date())],
        "observations": len(y), "total_trips": int(y.sum()), "statsmodels_version": statsmodels.__version__,
        "method": "MSTL(log daily trips), periods=(7,365), windows=(13,13), iterate=2, robust=True, seasonal_deg=0",
        "main_trend_window": MAIN_WINDOW, "sensitivity_trend_windows": list(TREND_WINDOWS),
        "observed_daily_mean_change": float(annual.loc[2024, "mean_daily_trips"] / annual.loc[2023, "mean_daily_trips"] - 1),
        "trend_year_mean_change": float(annual.loc[2024, "mean_trend_level"] / annual.loc[2023, "mean_trend_level"] - 1),
        "trend_sensitivity_change_range": [float(sensitivity.trend_year_mean_change.min()), float(sensitivity.trend_year_mean_change.max())],
        "fourier_annualized_growth_range": [float(fourier.annualized_loglinear_growth.min()), float(fourier.annualized_loglinear_growth.max())],
        "log_residual_std": float(result.log_remainder.std()),
        "annual_display": {
            "metric": "annual_trip_deviation",
            "formula": "trend_level * (annual_factor - 1)",
            "assumption": "Weekly and residual multiplicative factors held at 1",
            "chart_units": "thousand trips/day",
            "display_smoothing": "31-day centered arithmetic mean of annual_trip_deviation; 15 endpoint days omitted",
        },
        "checks": {"dates_complete": True, "total_reconciles": True, "both_seasonal_components_retained": True,
                   "reconstruction_matches_log_observed": True, "seasonal_geometric_means_equal_one": True},
        "caveats": ["Only two annual cycles; exploratory, not confirmed stable seasonality.",
                    "365-day annual period approximates the calendar; leap day retained.",
                    "Wide trend smoother constrains curvature; sensitivity envelope is not a confidence interval.",
                    "No causal or forecasting claims; exponential log trend is a typical baseline, not bias-corrected expected count."],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    draw_decomposition(y, result, trends, weekly_only, summary)
    draw_year_comparison(y, result)
    draw_annual_pattern(y, result)
    write_readme(summary, annual, sensitivity, fourier)
    print(annual.to_string())
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
