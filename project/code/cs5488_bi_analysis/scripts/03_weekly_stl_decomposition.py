"""对完整 daily 行程序列做回顾性周季节分解；不是预测模型。"""

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
from statsmodels.tsa.seasonal import STL

ROOT = PROJECT_ROOT
SOURCE = OUTPUT_ROOT / "01_daily.csv"
OUT = OUTPUT_ROOT / "time_series"
OUT.mkdir(parents=True, exist_ok=True)
DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# 用原始每日总量 trips，不能用已经平滑的 mean_7d 做周期识别。
def main():
    df = pd.read_csv(SOURCE, parse_dates=["pickup_date"]).set_index("pickup_date").sort_index()
    if df.index.has_duplicates:
        raise ValueError("日期重复，先检查来源。")
    y = df["trips"].asfreq("D")
    if y.isna().any() or not np.isfinite(y).all() or (y <= 0).any():
        raise ValueError("日期缺失或值无效。不要把未知缺口自动补零；log 分解要求正值。")
    assert len(y) == 731 and int(y.sum()) == 471_960_468, "与当前已核查版本不一致"

    # log(y) = trend + weekly_seasonal + remainder
    # period=7 是每周 7 天；seasonal=13 是平滑窗口，不是 13 天周期。
    fit = STL(np.log(y), period=7, seasonal=13, trend=31, robust=True).fit()
    result = pd.DataFrame({
        "trips": y,
        "log_trend": fit.trend,
        "log_weekly": fit.seasonal,
        "log_remainder": fit.resid,
        "trend_level": np.exp(fit.trend),
        "weekly_factor": np.exp(fit.seasonal),
        "fitted_typical_level": np.exp(fit.trend + fit.seasonal),
        "deviation_fraction": np.expm1(fit.resid),
    }, index=y.index)
    result.to_csv(OUT / "stl_daily_components.csv")

    def weekday_factors(series, trend_window=31):
        model = STL(np.log(series), period=7, seasonal=13, trend=trend_window, robust=True).fit()
        log_factors = model.seasonal.groupby(series.index.dayofweek).mean().reindex(range(7))
        # 几何平均归一化到 1；不是所有日期的算术均值或因果效应。
        return np.exp(log_factors - log_factors.mean())

    factors = pd.DataFrame({"2023": weekday_factors(y.loc["2023"]),
                            "2024": weekday_factors(y.loc["2024"]),
                            "pooled": weekday_factors(y)})
    factors.index = pd.Index(DAY_NAMES, name="weekday")
    factors.to_csv(OUT / "weekday_factors.csv")

    # 月度比较用日均值，避免月份天数不同；这里只是探索年度模式。
    monthly = y.groupby([y.index.year, y.index.month]).mean().unstack(0)
    monthly.index.name = "month"
    monthly.to_csv(OUT / "monthly_daily_means.csv")

    # 样本内残差异常筛查；候选不等于脏数据，更不能据此给出节日等因果解释。
    median = float(fit.resid.median())
    mad = float((fit.resid - median).abs().median())
    if mad == 0:
        raise ValueError("残差 MAD 为零，不能计算稳健分数。")
    result["robust_residual_score"] = (fit.resid - median) / (1.4826 * mad)
    interior = result.iloc[15:-15].copy()  # 端点平滑更不稳定，筛查中暂不纳入
    ranked = interior.reindex(interior.robust_residual_score.abs().sort_values(ascending=False).index)
    ranked.head(10).to_csv(OUT / "anomaly_candidates.csv")

    # 简单诊断与参数敏感性，不把自相关或季节强度当显著性检验。
    sensitivity = pd.DataFrame({f"trend_{w}": weekday_factors(y, w) for w in [21, 31, 61]})
    sensitivity.index = pd.Index(DAY_NAMES, name="weekday")
    sensitivity.to_csv(OUT / "window_sensitivity.csv")
    summary = {
        "source": str(SOURCE), "observations": len(y), "total_trips": int(y.sum()),
        "method": "STL of log daily trips, period=7, seasonal=13, trend=31, robust=True",
        "factor_normalization": "Geometric mean of seven pooled weekday factors is one",
        "raw_autocorrelation": {str(k): float(y.autocorr(k)) for k in [1, 7, 14, 21, 28]},
        "weekly_factors": factors.pooled.to_dict(),
        "residual_lag7_autocorrelation": float(fit.resid.autocorr(7)),
        "caveats": [
            "Retrospective descriptive decomposition, not a forecast or causal estimate.",
            "Only two years; annual seasonality cannot be established reliably.",
            "Slow annual/calendar variation remains in trend and/or remainder of weekly STL.",
            "Exponential of fitted log level is a typical multiplicative baseline, not a bias-corrected arithmetic expected count.",
            "Anomaly scores rank candidates only; no automatic deletion or formal p-values.",
        ],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                        "axes.spines.top": False, "axes.spines.right": False})
    blue, orange = "#17637b", "#d88535"
    fig, axes = plt.subplots(2, 1, figsize=(10, 7.2), layout="constrained")
    ax = axes[0]
    ax.plot(y.index, y / 1000, color="#aec9d1", lw=.7, label="Observed daily trips")
    ax.plot(y.index, result.trend_level / 1000, color=blue, lw=2, label="STL trend level")
    ax.set(title="Daily trips and slower variation | 2023–2024", ylabel="Trips (thousands)")
    ax.set_xlim(y.index.min(), y.index.max())
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.legend(frameon=False, ncol=2)
    ax.grid(axis="y", alpha=.2)
    ax = axes[1]
    x = np.arange(7)
    for offset, col, color in [(-.19, "2023", blue), (.19, "2024", orange)]:
        ax.bar(x + offset, factors[col] - 1, width=.35, color=color, label=col)
    ax.set_xticks(x, DAY_NAMES)
    ax.axhline(0, color="#555555", lw=.8)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set(title="Weekly pattern is similar in both years", ylabel="Seasonal factor minus 1",
           xlabel="Within-year log-STL factors; geometric-mean factor = 1")
    ax.legend(frameon=False, ncol=2)
    ax.grid(axis="y", alpha=.2)
    fig.savefig(OUT / "weekly_pattern.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "weekly_pattern.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6.4), layout="constrained")
    for col, color in [(2023, blue), (2024, orange)]:
        axes[0].plot(monthly.index, monthly[col] / 1000, "o-", label=str(col), color=color)
    axes[0].set(title="Monthly daily averages | exploratory, only two years",
                xlabel="Month", ylabel="Mean trips/day (thousands)", xticks=range(1, 13))
    axes[0].legend(frameon=False, ncol=2)
    axes[0].grid(alpha=.2)
    axes[1].plot(result.index, result.deviation_fraction, lw=.8, color=blue)
    axes[1].axhline(0, color="#555555", lw=.8)
    axes[1].set(title="Deviation from fitted trend + weekly pattern", ylabel="Observed / fitted − 1")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[1].grid(alpha=.2)
    fig.savefig(OUT / "monthly_and_remainder.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    assert np.allclose(result.log_trend + result.log_weekly + result.log_remainder, np.log(y))
    assert np.allclose(np.exp(np.log(factors).mean(axis=0)), 1)
    print(factors.round(4).to_string())
    print(f"Saved exploratory results to {OUT}")


if __name__ == "__main__":
    main()
