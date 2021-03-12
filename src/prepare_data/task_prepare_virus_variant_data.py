import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytask
import seaborn as sns
import statsmodels.api as sm
from sid import get_colors

from src.config import BLD
from src.config import SRC
from src.simulation.plotting import style_plot
from src.testing.shared import get_date_from_year_and_week

OUT_PATH = BLD / "data" / "virus_strains"


@pytask.mark.depends_on(
    {
        "rki": SRC / "original_data" / "virus_strains_rki.csv",
        "cologne": SRC / "original_data" / "virus_strains_cologne.csv",
    }
)
@pytask.mark.produces(
    {
        "rki_strains": OUT_PATH / "rki_strains.csv",
        "co_daily": OUT_PATH / "cologne_strains_daily.csv",
        "co_weekly": OUT_PATH / "cologne_strains_weekly.csv",
        "b117": OUT_PATH / "b117.pkl",
        "b1351": OUT_PATH / "b1351.pkl",
        "fig": OUT_PATH / "figures" / "averaged_strain_shares.png",
    }
)
def task_prepare_virus_variant_data(depends_on, produces):
    fig_path = produces["fig"].parent
    rki = pd.read_csv(depends_on["rki"])
    rki = _prepare_rki_data(rki)
    rki.to_csv(produces["rki_strains"])

    co_daily = pd.read_csv(depends_on["cologne"])
    co_daily = _prepare_co_data(co_daily)
    co_daily.to_csv(produces["co_daily"])

    co_weekly = _make_cologne_data_weekly(co_daily)
    co_weekly.to_csv(produces["co_weekly"])

    for col in co_daily:
        fig, ax = _plot_cologne_data(co_daily=co_daily, co_weekly=co_weekly, col=col)
        path = fig_path / f"{col}_cologne.png"
        fig.savefig(path, dpi=200, transparent=False, facecolor="w")

    for col in co_weekly:
        fig, ax = _rki_vs_cologne_data(rki=rki, co_weekly=co_weekly, col=col)
        path = fig_path / f"{col}_rki_vs_cologne.png"
        fig.savefig(path, dpi=200, transparent=False, facecolor="w")

    strain_data = _merge_rki_and_cologne_data(rki, co_weekly)

    # average over rki and cologne shares
    b117 = strain_data.groupby("date")["share_b117"].mean()
    b1351 = strain_data.groupby("date")["share_b1351"].mean()

    # change frequency to daily
    daily_index = pd.date_range(pd.Timestamp("2020-03-01"), b117.index.max())
    extended_b117 = b117.reindex(daily_index).interpolate()
    # extrapolate into the past
    extrapolated_fitted = _extrapolate_into_the_past(extended_b117)
    extended_b117 = extended_b117.fillna(extrapolated_fitted)
    extended_b117.to_pickle(produces["b117"])
    b117.name = "raw_share_b117"

    # we don't fit a model for b1351 because the values are too small.
    b1351 = b1351.reindex(daily_index).interpolate().fillna(0)
    b1351.to_pickle(produces["b1351"])

    fig, ax = _plot_final_shares([b117, b1351, extended_b117])
    fig.savefig(produces["fig"])


def _prepare_rki_data(rki):
    rki = rki[rki["week"].notnull()].copy(deep=True)
    rki["year"] = 2021
    rki["date"] = rki.apply(get_date_from_year_and_week, axis=1)
    rki = rki.set_index("date")
    as_float_cols = ["pct_b117", "pct_b1351", "n_tested_for_variants"]
    rki[as_float_cols] = rki[as_float_cols].astype(float)
    rki["share_b117"] = rki["pct_b117"] / 100
    rki["share_b1351"] = rki["pct_b1351"] / 100
    rki = rki[["share_b117", "share_b1351", "n_tested_for_variants"]]
    return rki


def _prepare_co_data(co):
    keep_cols = ["n_b117_cum", "n_b1351_cum", "n_tests_positive_cum", "date"]
    co = co[keep_cols].dropna().copy(deep=True)
    co["date"] = pd.to_datetime(co["date"], dayfirst=True)
    co = co.set_index("date").astype(int).sort_index()
    for col in co:
        assert (co[col].diff().dropna() >= 0).all(), col
        co[col.replace("_cum", "")] = co[col].diff()
    co = co.loc["2021-02-04":]
    co["share_b117"] = co["n_b117"] / co["n_tests_positive"]
    co["share_b1351"] = co["n_b1351"] / co["n_tests_positive"]
    co = co.rename(columns={"n_tests_positive": "n_tested_for_variants"})
    return co


def _make_cologne_data_weekly(co):
    co_small = co[["n_b117", "n_b1351", "n_tested_for_variants"]].reset_index()
    co_weekly = co_small.groupby(pd.Grouper(key="date", freq="W")).sum()
    # remove latest, incomplete week
    co_weekly = co_weekly[:-1]
    co_weekly["share_b117"] = co_weekly["n_b117"] / co_weekly["n_tested_for_variants"]
    co_weekly["share_b1351"] = co_weekly["n_b1351"] / co_weekly["n_tested_for_variants"]
    co_weekly = co_weekly[["n_tested_for_variants", "share_b117", "share_b1351"]]
    return co_weekly


def _rki_vs_cologne_data(rki, co_weekly, col):
    colors = get_colors("categorical", 2)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(x=rki.index, y=rki[col], color=colors[0], linewidth=2, label="RKI")
    sns.lineplot(
        x=co_weekly.index,
        y=co_weekly[col],
        color=colors[1],
        linewidth=2,
        label="Cologne",
    )
    nice_name = col.replace("_", " ").title()
    ax.set_title(f"{nice_name} Acc. to RKI and in Cologne")
    fig, ax = style_plot(fig, ax)
    return fig, ax


def _plot_cologne_data(co_daily, co_weekly, col):
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = get_colors("categorical", 3)
    if col.endswith("_cum"):
        color = colors[0]
    elif col.startswith("share"):
        color = colors[1]
    else:
        color = colors[2]
    if col in co_weekly:
        sns.lineplot(
            x=co_weekly.index,
            y=co_weekly[col],
            color=color,
            linewidth=2,
            linestyle="--",
            label="weekly",
        )
        label = "daily"
    else:
        label = None
    sns.lineplot(
        x=co_daily.index, y=co_daily[col], color=color, linewidth=2, label=label
    )
    nice_name = col.replace("_", " ").title()
    ax.set_title(f"{nice_name} in Cologne")
    fig, ax = style_plot(fig, ax)
    return fig, ax


def _merge_rki_and_cologne_data(rki, co_weekly):
    rki = rki.copy(deep=True)
    co_weekly = co_weekly.copy(deep=True)

    co_weekly["source"] = "cologne"
    rki["source"] = "rki"
    co_weekly = co_weekly.set_index("source", append=True)
    rki = rki.set_index("source", append=True)

    strain_data = pd.concat([rki, co_weekly]).sort_index()
    return strain_data


def _plot_final_shares(shares):
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = get_colors("categorical", len(shares))
    line_styles = ["-", "--", "dotted"]
    for color, sr, style in zip(colors, shares, line_styles):
        sns.lineplot(
            x=sr.index,
            y=sr,
            color=color,
            linewidth=2,
            label=sr.name,
            linestyle=style,
        )
    ax.set_title("Share of Virus Variants Over Time")
    fig, ax = style_plot(fig, ax)
    return fig, ax


def _extrapolate_into_the_past(sr):
    """Use data on the virus strains to extrapolate their incidence into the past."""
    endog = np.log(sr.dropna())
    exog = pd.DataFrame(index=endog.index)
    exog["constant"] = 1
    exog["days_since_start"] = (exog.index - exog.index.min()).days

    model = sm.OLS(endog=endog, exog=exog)
    results = model.fit()
    results.summary()

    assert results.rsquared > 0.9, (
        "Your fit of the virus strain trend has worsened considerably. "
        "Check the fitness plot in the "
    )

    full_x = pd.DataFrame(index=sr.index)
    full_x["days_since_start"] = (full_x.index - exog.index.min()).days
    full_x["constant"] = 1
    extrapolated = np.exp(full_x.dot(results.params))
    extrapolated.name = sr.name
    return extrapolated


def _fitness_plot(actual, fitted):
    """Compare the actual and fitted share becoming immune."""
    colors = get_colors("categorical", 4)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(
        x=actual.index,
        y=actual,
        label="actual data",
        linewidth=2,
        color=colors[0],
    )
    sns.lineplot(x=fitted.index, y=fitted, label="fitted", linewidth=2, color=colors[3])
    ax.set_title("Fitness Plot")
    fig, ax = style_plot(fig, ax)
    fig.tight_layout()
    return fig, ax
