"""Supplement STL with daily log-regression, HAC inference and Holm adjustment.

Run: python scripts/05_weekday_significance_tests.py
The inferential effects below are regression estimates, not tests of fitted STL values.
"""

from _config import PROJECT_ROOT, DATA_ROOT, OUTPUT_ROOT, LOOKUP_FILE
from pathlib import Path
import json
import hashlib
import platform

import numpy as np
import pandas as pd
from scipy import stats, special
import statsmodels
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


ROOT = PROJECT_ROOT
OUT = OUTPUT_ROOT / "time_series"
DETAIL = OUT / "weekly_significance"
WEEK = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
PRIMARY_LAG = 14
ALPHA = 0.05


def design(df, interaction=False, month_slopes=False):
    month = df.pickup_date.dt.strftime("%Y-%m")
    x = pd.get_dummies(month, prefix="month", dtype=float)
    for i, name in enumerate(WEEK[1:], 1):
        x[f"dow_{name}"] = (df.pickup_date.dt.dayofweek == i).astype(float)
    if interaction:
        for name in WEEK[1:]:
            x[f"year2024_dow_{name}"] = x[f"dow_{name}"] * (df.pickup_date.dt.year == 2024)
    if month_slopes:
        day = df.pickup_date.dt.day.astype(float)
        within = (day - day.groupby(month).transform("mean")) / 31
        for m in sorted(month.unique()):
            x[f"slope_{m}"] = (month == m) * within
    if np.linalg.matrix_rank(x) != x.shape[1]:
        raise ValueError("Regression design is not full rank")
    return x


def fit_model(df, x, lag=PRIMARY_LAG):
    return sm.OLS(np.log(df.trips.to_numpy()), x).fit(
        cov_type="HAC", cov_kwds={"maxlags": lag, "kernel": "bartlett", "use_correction": True},
        use_t=False,
    )


def centered_contrasts(x, year=None):
    b = np.zeros((7, x.shape[1]))
    for i, name in enumerate(WEEK[1:], 1):
        b[i, x.columns.get_loc(f"dow_{name}")] = 1
        if year == 2024:
            b[i, x.columns.get_loc(f"year2024_dow_{name}")] = 1
    return b - b.mean(axis=0)


def contrast_table(fit, contrasts, labels):
    estimate = contrasts @ fit.params.to_numpy()
    covariance = contrasts @ fit.cov_params().to_numpy() @ contrasts.T
    se = np.sqrt(np.diag(covariance))
    if not np.isfinite(se).all() or (se <= 0).any():
        raise ValueError("Invalid contrast standard errors")
    z = estimate / se
    logp = np.log(2) + stats.norm.logsf(np.abs(z))
    # A positive floor avoids reporting numerical underflow as a true p = 0.
    p = np.exp(np.maximum(logp, np.log(np.finfo(float).tiny)))
    reject, adjusted, _, _ = multipletests(p, alpha=ALPHA, method="holm")
    critical = stats.norm.ppf(1 - ALPHA / 2)
    lower, upper = estimate - critical * se, estimate + critical * se
    out = pd.DataFrame({
        "weekday": labels, "log_effect": estimate, "hac_standard_error": se, "z_statistic": z,
        "p_value": p, "log10_p_value": logp / np.log(10),
        "holm_p_value": adjusted, "significant_holm_05": reject,
        "factor": np.exp(estimate), "effect_pct": np.expm1(estimate) * 100,
        "factor_ci95_low": np.exp(lower), "factor_ci95_high": np.exp(upper),
        "effect_ci95_low_pct": np.expm1(lower) * 100, "effect_ci95_high_pct": np.expm1(upper) * 100,
    })
    check = fit.t_test(contrasts)
    assert np.allclose(check.effect, estimate) and np.allclose(check.sd, se)
    return out


def joint_test(fit, x, prefix):
    cols = [i for i, name in enumerate(x.columns) if name.startswith(prefix)]
    r = np.eye(x.shape[1])[cols]
    test = fit.wald_test(r, use_f=False, scalar=True)
    stat, df = float(test.statistic), len(cols)
    # For six restrictions the chi-square survival has this exact stable form.
    assert df == 6
    a = stat / 2
    logp = -a + special.logsumexp([0.0, np.log(a), 2 * np.log(a) - np.log(2)])
    probability = float(np.exp(logp))
    return {"chi_square": stat, "df": df,
            "p_value": probability if probability > 0 else None,
            "log10_p_value": float(logp / np.log(10)), "p_display": "< 0.001" if logp < np.log(.001) else f"= {probability:.4f}"}


def ptext(value):
    return "<0.001" if value < .001 else f"{value:.3f}"


def run():
    DETAIL.mkdir(parents=True, exist_ok=True)
    source = OUTPUT_ROOT / "01_daily.csv"
    df = pd.read_csv(source, parse_dates=["pickup_date"]).sort_values("pickup_date").reset_index(drop=True)
    expected = pd.date_range("2023-01-01", "2024-12-31")
    if not pd.DatetimeIndex(df.pickup_date).equals(expected):
        raise ValueError("Expected 731 consecutive unique dates")
    assert len(df) == 731 and int(df.trips.sum()) == 471_960_468
    assert np.isfinite(df.trips).all() and (df.trips > 0).all()
    x = design(df)
    fit = fit_model(df, x)
    contrasts = centered_contrasts(x)
    table = contrast_table(fit, contrasts, WEEK)
    assert np.isclose(table.log_effect.sum(), 0) and np.isclose(np.exp(table.log_effect.mean()), 1)
    overall = joint_test(fit, x, "dow_")
    table.to_csv(DETAIL / "weekday_effects.csv", index=False)
    pd.DataFrame({"term": x.columns, "coefficient": fit.params, "hac_standard_error": fit.bse}).to_csv(DETAIL / "model_coefficients.csv", index=False)

    # Secondary question: do the six independent weekday contrasts differ by year?
    xi = design(df, interaction=True)
    fi = fit_model(df, xi)
    interaction_test = joint_test(fi, xi, "year2024_dow_")
    l23, l24 = centered_contrasts(xi, 2023), centered_contrasts(xi, 2024)
    year_tables = []
    for year, contrasts_year in [(2023, l23), (2024, l24)]:
        t = contrast_table(fi, contrasts_year, WEEK)
        t.insert(0, "year", year)
        year_tables.append(t)
    years = pd.concat(year_tables, ignore_index=True)
    # One family of 14 year-specific comparisons, not two unadjusted families.
    reject, adjusted, _, _ = multipletests(years.p_value, method="holm", alpha=ALPHA)
    years["holm_p_value"] = adjusted
    years["significant_holm_05"] = reject
    years.to_csv(DETAIL / "weekday_effects_by_year.csv", index=False)
    changes = contrast_table(fi, l24 - l23, WEEK)
    changes.to_csv(DETAIL / "weekday_factor_changes_2024_vs_2023.csv", index=False)

    sensitivity = []
    for slopes in [False, True]:
        xs = design(df, month_slopes=slopes)
        for lag in [7, 14, 28]:
            fs = fit_model(df, xs, lag)
            ts = contrast_table(fs, centered_contrasts(xs), WEEK)
            jt = joint_test(fs, xs, "dow_")
            for row in ts.to_dict("records"):
                row.update({"specification": "month_FE_plus_month_slopes" if slopes else "month_FE",
                            "hac_lag": lag, "joint_chi_square": jt["chi_square"], "joint_log10_p_value": jt["log10_p_value"]})
                sensitivity.append(row)
    sensitivity = pd.DataFrame(sensitivity)
    sensitivity.to_csv(DETAIL / "sensitivity.csv", index=False)
    # Two further lags for the year-interaction question, all reported.
    year_sensitivity = []
    for lag in [7, 14, 28]:
        f = fit_model(df, xi, lag)
        test = joint_test(f, xi, "year2024_dow_")
        year_sensitivity.append({"hac_lag": lag, **test})
    pd.DataFrame(year_sensitivity).to_csv(DETAIL / "year_interaction_sensitivity.csv", index=False)

    residual = pd.Series(fit.resid.to_numpy())
    summary = {
        "source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "observations": len(df), "recorded_trips": int(df.trips.sum()),
        "period": [str(df.pickup_date.min().date()), str(df.pickup_date.max().date())],
        "analysis_unit": "Calendar day; all days receive equal weight, not trip-level replication",
        "formula": "log(daily_trips) = year_month_fixed_effect + six_weekday_indicators + error",
        "year_month_categories": 24, "parameters": len(fit.params), "residual_df": int(fit.df_resid),
        "reference_in_fit": "Monday", "reported_reference": "Equal-weight geometric mean across seven fitted weekday levels within the same year-month",
        "normalization": "delta_d = beta_d - mean(beta_Mon,...,beta_Sun), with beta_Mon=0; factor=exp(delta_d)",
        "covariance": {"type": "Newey-West HAC", "kernel": "Bartlett", "maxlags": PRIMARY_LAG, "finite_sample_correction": True, "inference": "asymptotic normal and chi-square"},
        "multiplicity": "Holm for seven pooled weekday contrasts. Fourteen year-specific contrasts and seven year-change contrasts are separate secondary families.",
        "confidence_intervals": "Pointwise 95% HAC intervals, not simultaneous intervals; significance is determined by Holm-adjusted p values",
        "global_weekday_test": overall, "year_interaction_test": interaction_test,
        "residual_autocorrelations": {str(l): float(residual.autocorr(l)) for l in [1,7,14,28]},
        "r_squared": float(fit.rsquared),
        "sensitivity_effect_range_pct": sensitivity.groupby("weekday").effect_pct.agg(["min","max"]).to_dict("index"),
        "sensitivity_rejections": sensitivity.groupby("weekday").significant_holm_05.agg(["sum","count"]).astype(int).to_dict("index"),
        "software": {"python": platform.python_version(), "statsmodels": statsmodels.__version__, "pandas": pd.__version__, "numpy": np.__version__},
        "limits": ["Regression-based supplement, not p values or error bars for the original STL factor estimates.",
                   "Exploratory inference after visual inspection, not a preregistered confirmatory analysis.",
                   "HAC relies on regression and weak-dependence assumptions; does not remove confounding or all possible long-memory dependence.",
                   "Month fixed effects do not separately identify holidays, weather, events, platform strategy or causal weekday effects.",
                   "Logged response implies geometric/typical levels, not arithmetic mean trip counts.",
                   "Statistical evidence for weekly variation does not establish annual seasonality, permanent stability or causality.",
                   "Null underflow p values are stored as null with finite log10_p_value, and never reported as zero."],
    }
    (DETAIL / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    make_plot(table, overall)
    make_report(table, years, changes, overall, interaction_test, sensitivity, summary)
    print(json.dumps({"overall": overall, "year_interaction": interaction_test}, indent=2))
    print(table[["weekday","effect_pct","effect_ci95_low_pct","effect_ci95_high_pct","holm_p_value","significant_holm_05"]].to_string(index=False))
    print("YEAR_CHANGES\n", changes[["weekday","effect_pct","holm_p_value","significant_holm_05"]].to_string(index=False))
    print("SENSITIVITY\n", sensitivity.groupby("weekday").agg(effect_min=("effect_pct","min"),effect_max=("effect_pct","max"),significant_models=("significant_holm_05","sum")).to_string())


def make_plot(table, overall):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(10.8, 6.1))
    fig.subplots_adjust(left=.1,right=.71,top=.77,bottom=.24)
    pos = np.arange(7)
    effects = table.effect_pct.to_numpy() / 100
    low = table.effect_ci95_low_pct.to_numpy() / 100
    high = table.effect_ci95_high_pct.to_numpy() / 100
    for i, row in table.iterrows():
        color = "#17637b" if row.significant_holm_05 else "#9a9a9a"
        ax.errorbar(effects[i],i,xerr=[[effects[i]-low[i]],[high[i]-effects[i]]],
                    fmt="o",color=color,capsize=4,elinewidth=2,ms=7)
    ax.set(yticks=pos,yticklabels=WEEK,xlabel="Estimated deviation from weekday-neutral baseline",xlim=(-.20,.24))
    ax.invert_yaxis()
    ax.axvline(0,color="#777777",ls="--",lw=1)
    ax.xaxis.set_major_formatter(PercentFormatter(1,decimals=0))
    ax.grid(axis="x",alpha=.18)
    fig.suptitle("Weekly trip pattern: statistical evidence",x=.1,ha="left",y=.96,fontsize=18,fontweight="bold")
    fig.text(.1,.88,"2023–2024 · 731 daily observations · Log regression with year-month fixed effects",fontsize=10.5)
    fig.text(.1,.825,f"Overall weekday test: Wald χ²(6) = {overall['chi_square']:,.1f}, p {overall['p_display']}",fontsize=12,color="#17637b")
    ax.text(1.02,1.04,"Effect [95% CI]",transform=ax.transAxes,fontsize=10,fontweight="bold")
    ax.text(1.40,1.04,"Holm p",transform=ax.transAxes,fontsize=10,fontweight="bold")
    for i,row in table.iterrows():
        ax.text(1.02,i,f"{row.effect_pct:+.1f}% [{row.effect_ci95_low_pct:+.1f}, {row.effect_ci95_high_pct:+.1f}]",
                transform=ax.get_yaxis_transform(),va="center",fontsize=10)
        ax.text(1.40,i,ptext(row.holm_p_value),transform=ax.get_yaxis_transform(),va="center",fontsize=10)
    fig.text(.1,.13,"Intervals: pointwise 95% Newey–West HAC (14 lags). Holm correction covers seven weekday tests.",fontsize=9,color="#555555")
    fig.text(.1,.085,"Baseline: geometric mean of seven fitted weekday levels. Effects are conditional associations.",fontsize=9,color="#555555")
    fig.text(.1,.04,"This regression supplements the STL figure. These intervals do not apply to the original STL factors.",fontsize=9,color="#555555")
    for ext in ["png","pdf"]:
        fig.savefig(OUT / f"weekly_significance.{ext}",dpi=240,bbox_inches="tight")
    plt.close(fig)


def make_report(table, years, changes, overall, interaction, sensitivity, summary):
    f = table.set_index("weekday")
    rows = "\n".join(f"| {r.weekday} | {r.factor:.4f} | {r.effect_pct:+.2f}% | [{r.effect_ci95_low_pct:+.2f}%, {r.effect_ci95_high_pct:+.2f}%] | {ptext(r.holm_p_value)} | {'Yes' if r.significant_holm_05 else 'No'} |" for r in table.itertuples())
    year_sentence = ("The secondary joint test detected a difference in the weekday profile between 2023 and 2024. The similar visual patterns should therefore not be described as statistically identical."
                     if interaction["log10_p_value"] < np.log10(.05) else
                     "The secondary joint test did not detect a difference in the weekday profile between 2023 and 2024 at the 5% level. This does not establish equivalence or prove that the profiles are identical.")
    robust_days = sensitivity.groupby("weekday").significant_holm_05.sum()
    stability = "; ".join(f"{d}: {int(robust_days[d])}/6 specifications reject" for d in WEEK)
    sensitivity_sentence = ("All seven weekday contrasts retained their direction and remained significant after Holm correction across the six sensitivity specifications using 7-, 14- and 28-day HAC lags, with or without linear trends within each year-month."
                            if (robust_days == 6).all() else
                            "Sensitivity results varied across specifications and are reported in full below.")
    report = f"""# Statistical Assessment of the Weekly Trip Pattern

## Report-ready text

We assessed weekly variation using 731 daily trip counts from 1 January 2023 to 31 December 2024. The unit of analysis was the calendar day, rather than the approximately 472 million individual trips represented by the aggregates. We fitted an ordinary least squares regression of log daily trip counts on weekday indicators and 24 year-month fixed effects. The fixed effects account for differences in average log trip levels across calendar months. Inference used Newey–West heteroskedasticity- and autocorrelation-consistent (HAC) standard errors with a Bartlett kernel, a maximum lag of 14 days and a finite-sample covariance correction. A joint Wald test evaluated the null hypothesis of no weekday differences. Seven weekday-specific tests used the Holm procedure to control the family-wise error rate at 5%. The reported 95% confidence intervals are pointwise rather than simultaneous. [1–3]

The joint test provided strong statistical evidence of weekday variation (Wald χ²(6) = {overall['chi_square']:.2f}, p {overall['p_display']}). Relative to the geometric mean of the seven fitted weekday levels within a given year-month, the estimated Friday factor was {f.loc['Fri','factor']:.4f}, corresponding to a {f.loc['Fri','effect_pct']:.2f}% higher typical trip level (95% CI: {f.loc['Fri','effect_ci95_low_pct']:.2f}% to {f.loc['Fri','effect_ci95_high_pct']:.2f}%). Saturday had the highest estimated factor, {f.loc['Sat','factor']:.4f}, corresponding to an {f.loc['Sat','effect_pct']:.2f}% higher typical level (95% CI: {f.loc['Sat','effect_ci95_low_pct']:.2f}% to {f.loc['Sat','effect_ci95_high_pct']:.2f}%). Monday had the lowest estimated factor, {f.loc['Mon','factor']:.4f}, or {abs(f.loc['Mon','effect_pct']):.2f}% below the baseline. All seven contrasts were significant after Holm correction, although Thursday's estimated difference was comparatively small (+{f.loc['Thu','effect_pct']:.2f}%, adjusted p = {f.loc['Thu','holm_p_value']:.3f}). Statistical significance should therefore be interpreted alongside effect size. The weekday-specific results are reported in Table 1.

{sensitivity_sentence}

{year_sentence} The interaction test yielded Wald χ²(6) = {interaction['chi_square']:.2f}, p {interaction['p_display']}. This is a separate question from whether a weekly pattern exists in the pooled series.

These findings provide a regression-based inferential supplement to the descriptive STL decomposition. The regression coefficients, confidence intervals and p values are not inferential quantities for the original STL estimates. Because the analysis followed inspection of the same data, the results should be interpreted as exploratory. They establish conditional associations in the recorded daily series, rather than causal effects of weekdays. Holidays, weather and special events were not separately modelled, and evidence of weekly variation does not establish stable annual seasonality or guarantee that the pattern will persist in future years.

## Table 1. Weekday effects relative to the weekday-neutral baseline

| Weekday | Factor | Estimated difference | Pointwise 95% HAC CI | Holm-adjusted p | Significant at 5% |
|---|---:|---:|---|---:|---|
{rows}

The neutral baseline is the equal-weight geometric mean of the seven model-implied weekday levels within the same year-month. It is not the arithmetic mean of observed daily trips, and Monday is not the reference for the percentages in this table. Monday is only the treatment-coding reference used to fit the regression. A factor of 1 denotes no deviation from the neutral baseline. P values are two-sided. Values below 0.001 are displayed as `<0.001`, never as zero. Rounding may affect displayed interval endpoints.

## Model and hypotheses

```text
log(Trips_t) = alpha_year-month(t) + beta_weekday(t) + error_t
beta_Mon = 0 for model identification
delta_d = beta_d - (beta_Mon + ... + beta_Sun) / 7
Factor_d = exp(delta_d)
Percentage difference_d = 100 * (Factor_d - 1)

Primary joint null: beta_Tue = beta_Wed = ... = beta_Sun = 0
Individual null for weekday d: delta_d = 0, equivalently Factor_d = 1
Secondary year-interaction null: all six 2024 × weekday coefficients = 0
```

The primary model has {summary['parameters']} parameters and {summary['residual_df']} residual degrees of freedom. All 731 observations receive equal weight. No observations were dropped to improve significance. HAC uses the chronological, complete daily sequence. The maximum lag of 14 covers two weeks of residual dependence; it does not imply that observations beyond 14 days are necessarily independent. Normal and chi-square reference distributions provide asymptotic inference under the regression and dependence assumptions. [1,2]

## Sensitivity analysis

All six combinations of HAC maximum lags (7, 14 and 28 days) and time controls (year-month fixed effects, with or without separate linear trends within each year-month) are retained in `weekly_significance/sensitivity.csv`. They were specified before fitting the inferential models and are not a search for favourable p values. They are not preregistered analyses.

Holm-adjusted rejection counts across the six specifications: {stability}.

Year-specific effects are in `weekly_significance/weekday_effects_by_year.csv`, with Holm correction across 14 tests. Estimated changes in normalized factors between 2023 and 2024 are in `weekly_significance/weekday_factor_changes_2024_vs_2023.csv`, with Holm correction across seven secondary tests. The change percentages refer to ratios of normalized weekday factors, not year-on-year growth in actual trip counts. Interaction-test lag sensitivity is in `weekly_significance/year_interaction_sensitivity.csv`. The pooled, year-specific and change contrasts form distinct stated families; they do not constitute one jointly adjusted family of every possible claim.

## Figure caption

**Figure. Regression-based weekday effects on daily recorded trips, 2023–2024.** Points show percentage deviations from the equal-weight geometric mean of seven fitted weekday levels after adjustment for year-month fixed effects. Error bars show pointwise 95% Newey–West HAC confidence intervals using 14 lags. The right column reports Holm-adjusted p values for seven weekday contrasts. The joint Wald test evaluates the absence of weekday differences. This figure supplements the STL decomposition and does not attach confidence intervals to the original STL seasonal factors.

## Reproduction and sources

Run `python scripts/05_weekday_significance_tests.py` from the project root.

Data: `01_daily.csv` under the configured output directory, generated from the group's corrected aggregate data. Complete date coverage and 471,960,468 recorded trips were reconciled before modelling. Original STL outputs remain unchanged.

1. statsmodels, OLSResults.get_robustcov_results: https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLSResults.get_robustcov_results.html
2. statsmodels, RegressionResults.wald_test: https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.RegressionResults.wald_test.html
3. statsmodels, multipletests (Holm correction): https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html

Software: statsmodels {statsmodels.__version__}, Python {platform.python_version()}. Model settings, data hash, diagnostics and limitations are saved in `weekly_significance/summary.json`.
"""
    (OUT / "weekly_significance_report_EN.md").write_text(report,encoding="utf-8")


if __name__ == "__main__":
    run()
