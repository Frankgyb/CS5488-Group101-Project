"""Export nine charts under the filenames already used by report/main.tex."""
import argparse
import json
from pathlib import Path
import shutil

FIGURES = {
    "time_series/weekly_significance.png": "fig01_weekday_effect.png",
    "03_zones.png": "fig03_top15_zones.png",
    "02_weekday_hour.png": "fig04_weekday_hour_heat.png",
    "hourly_wait/weekday_hour_wait.png": "fig05_wait_weekday_hour.png",
    "04_platform.png": "fig06_platform_share.png",
    "platform_comparison/01_borough_platform_mix.png": "fig07_borough_mix.png",
    "platform_comparison/03_hourly_platform_wait.png": "fig08_wait_by_platform.png",
    "platform_comparison/04_uber_borough_hour_wait.png": "fig09_uber_pressure.png",
}


def monthly_panel(output, destination):
    """The report uses only the daily/monthly panel of the original two-panel chart."""
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from scipy.interpolate import PchipInterpolator

    daily = pd.read_csv(output / "01_daily.csv", parse_dates=["pickup_date"])
    monthly = pd.read_csv(output / "time_series/monthly_seasonal_factors.csv", parse_dates=["midpoint"])
    x = mdates.date2num(monthly.midpoint.to_numpy())
    f = PchipInterpolator(x, monthly.mean_trips_per_day.to_numpy(), extrapolate=False)
    np.testing.assert_allclose(f(x), monthly.mean_trips_per_day, rtol=1e-12)
    curve_x = np.linspace(x[0], x[-1], 1200)
    fig, ax = plt.subplots(figsize=(10.5, 3.4), layout="constrained")
    ax.plot(daily.pickup_date, daily.trips / 1000, color="#aec9d1", lw=.7, label="Observed daily trips")
    ax.plot(mdates.num2date(curve_x), f(curve_x) / 1000, color="#17637b", lw=2.2,
            label="Monthly means (smooth interpolation)")
    ax.set(ylabel="Trips per day (thousands)", xlim=(daily.pickup_date.min(), daily.pickup_date.max()), ylim=(380, 930))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.legend(frameon=False, ncol=2, loc="lower left", fontsize=9)
    ax.grid(axis="y", alpha=.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(destination, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()
    output = args.out.expanduser().resolve()
    target = output / "report_figures"
    target.mkdir(parents=True, exist_ok=True)
    for source, name in FIGURES.items():
        shutil.copyfile(output / source, target / name)
    monthly_panel(output, target / "fig02_monthly_trend.png")
    manifest = dict(FIGURES)
    manifest["01_daily.csv + time_series/monthly_seasonal_factors.csv (single-panel rendering)"] = "fig02_monthly_trend.png"
    (target / "figure_sources.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Nine report figures exported: {target}")


if __name__ == "__main__":
    main()
