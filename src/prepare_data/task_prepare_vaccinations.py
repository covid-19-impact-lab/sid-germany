import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytask
import seaborn as sns
import statsmodels.api as sm
from sid.colors import get_colors

from src.config import BLD
from src.config import POPULATION_GERMANY
from src.simulation.plotting import style_plot


plt.rcParams.update(
    {
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
    }
)

OUT_PATH = BLD / "data" / "vaccinations"


@pytask.mark.depends_on(BLD / "data" / "raw_time_series" / "vaccinations.xlsx")
@pytask.mark.produces(
    {
        "vaccination_shares": OUT_PATH / "vaccination_shares.pkl",
        "share_becoming_immune": OUT_PATH / "share_becoming_immune.pkl",
        "fig_first_dose": OUT_PATH / "first_dose.png",
        "fig_vaccination_shares": OUT_PATH / "vaccination_shares.png",
        "fig_share_immune": OUT_PATH / "share_becoming_immune.png",
        "fig_fit": OUT_PATH / "fitness_prediction.png",
    }
)
def task_prepare_vaccination_data(depends_on, produces):
    df = pd.read_excel(depends_on, sheet_name="Impfungen_proTag")
    df = _clean_vaccination_data(df)
    # this is for comparing with newspaper sites
    fig, ax = _plot_series(df["share_with_first_dose"], "Share with 1st Dose")
    fig.savefig(produces["fig_first_dose"], dpi=200, transparent=False, facecolor="w")
    plt.close()

    vaccination_shares = df["share_with_first_dose"].diff().dropna()
    vaccination_shares.to_pickle(produces["vaccination_shares"])
    fig, ax = _plot_series(vaccination_shares, "Share Receiving 1st Dose Per Day")
    fig.savefig(
        produces["fig_vaccination_shares"], dpi=200, transparent=False, facecolor="w"
    )
    plt.close()

    share_becoming_immune = _calculate_share_becoming_immune_from_vaccination(df)

    # because of strong weekend effects we smooth and extrapolate into the future
    smoothed = share_becoming_immune.rolling(7, min_periods=1).mean().dropna()
    fitted, prediction = _get_vaccination_prediction(smoothed)

    fig, ax = fitness_plot(share_becoming_immune, smoothed, fitted)
    fig.savefig(produces["fig_fit"], dpi=200, transparent=False, facecolor="w")
    plt.close()

    start_date = smoothed.index.min() - pd.Timedelta(days=1)
    past = pd.Series(data=0, index=pd.date_range("2020-01-01", start_date))
    expanded = pd.concat([past, smoothed, prediction])
    assert (
        expanded.index.is_monotonic
    ), "share_becoming_immune's index is not monotonic."
    assert (
        not expanded.index.duplicated().any()
    ), "Duplicate dates in the expanded share_becoming_immune Series."
    expanded.to_pickle(produces["share_becoming_immune"])

    title = "Actual and Extrapolated Share Becoming Immune Through Vaccination"
    fig, ax = _plot_series(expanded["2021-01-01":], title, label="extrapolated")
    sns.lineplot(
        x=share_becoming_immune.index, y=share_becoming_immune, label="actual data"
    )
    fig.savefig(produces["fig_share_immune"], dpi=200, transparent=False, facecolor="w")
    plt.close()


def _clean_vaccination_data(df):
    # drop last two rows (empty and total vaccinations)
    df = df[df["Datum"].isnull().cumsum() == 0].copy(deep=True)
    df["date"] = pd.to_datetime(df["Datum"], format="%m/%d/%yyyy")
    # check date conversion was correct
    assert df["date"].min() == pd.Timestamp(year=2020, month=12, day=27)
    df = df.set_index("date")
    df["received_first_dose"] = df["Erstimpfung"].cumsum()
    df["share_with_first_dose"] = df["received_first_dose"] / POPULATION_GERMANY
    return df


def _calculate_share_becoming_immune_from_vaccination(df):
    """Calculate the share of individuals becoming immune from vaccination.

    We ignore booster shots and simply assume immunity will start
    21 days after the first shot (Hunter2021) with 75% probability
    (90% for Pfizer but AstraZeneca is lower and protection against
    mutations is lower)

    """
    share_immune = 0.75 * df["share_with_first_dose"]
    share_immune.index = share_immune.index + pd.Timedelta(weeks=3)
    share_becoming_immune = share_immune.diff().dropna()
    share_becoming_immune.name = "share_becoming_immune"
    return share_becoming_immune


def _get_vaccination_prediction(smoothed):
    """Predict the vaccination data into the future using log smoothed data."""
    # exponential trend since March
    to_use = smoothed["2021-03-01":]
    y = np.log(to_use)
    exog = pd.DataFrame(index=y.index)
    exog["constant"] = 1
    exog["days_since_march"] = _get_days_since_march_first(exog)

    model = sm.OLS(endog=y, exog=exog)
    results = model.fit()
    assert results.rsquared > 0.8, (
        "Your fit of the vaccination trend has worsened considerably. "
        "Check the fitness plot: bld/data/vaccinations/fitness_prediction.png."
    )
    fitted = np.exp(exog.dot(results.params))

    start = exog.index.max() + pd.Timedelta(days=1)
    end = start + pd.Timedelta(weeks=8)
    future_x = pd.DataFrame(index=pd.date_range(start, end))
    future_x["days_since_march"] = _get_days_since_march_first(future_x)
    future_x["constant"] = 1
    prediction = np.exp(future_x.dot(results.params))
    return fitted, prediction


def _get_days_since_march_first(df):
    """Get the number of days since March 1st from a date index."""
    return (df.index - pd.Timestamp("2021-03-01")).days


def fitness_plot(actual, smoothed, fitted):
    """Compare the actual, smoothed and fitted share becoming immune."""
    colors = get_colors("categorical", 4)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(
        x=actual.index,
        y=actual,
        label="actual data",
        linewidth=2,
        color=colors[0],
    )
    sns.lineplot(
        x=smoothed.index, y=smoothed, label="smoothed", linewidth=2, color=colors[1]
    )
    sns.lineplot(x=fitted.index, y=fitted, label="fitted", linewidth=2, color=colors[3])
    ax.set_title("Fitness Plot")
    fig, ax = style_plot(fig, ax)
    fig.tight_layout()
    return fig, ax


def _plot_series(sr, title, label=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(x=sr.index, y=sr, label=label)
    ax.set_title(title)
    fig, ax = style_plot(fig, ax)
    fig.tight_layout()
    return fig, ax
