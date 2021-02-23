"""Prepare the ARS data."""
import matplotlib.pyplot as plt
import pandas as pd
import pytask
import seaborn as sns
from sid.colors import get_colors

from src.config import BLD
from src.config import SRC
from src.simulation.plotting import style_plot
from src.testing.task_calculate_test_capacity import (
    get_date_from_year_and_week,
)

plt.rcParams.update(
    {
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
    }
)

sns.set_palette(get_colors("categorical", 12))


OUT_PATH = BLD / "data" / "testing"

PRODUCTS = {
    "test_shares_by_age_group": OUT_PATH / "test_shares_by_age_group.pkl",
    "positivity_rate_by_age_group": OUT_PATH / "positivity_rate_by_age_group.pkl",
    "positivity_rate_overall": OUT_PATH / "positivity_rate_overall.pkl",
    "test_shares_by_age_group_png": OUT_PATH / "test_shares_by_age_group.png",
    "positivity_rate_by_age_group_png": OUT_PATH / "positivity_rate_by_age_group.png",
    "positivity_rate_overall_png": OUT_PATH / "positivity_rate_overall.png",
}


@pytask.mark.depends_on(SRC / "original_data" / "testing" / "ars_data_raw.xlsx")
@pytask.mark.produces(PRODUCTS)
def task_prepare_ars_data(depends_on, produces):
    """Calculate and save quantaties of interest of the ARS data.

    We do not export the ARS data at the moment because:
    1. it is weekly and not daily frequency data (not relevant for shares and rates)
    2. it would be necessary to upscale the number of tests to account for
       the fact that only a fraction of laboratories report ARS data.

    """
    ars = pd.read_excel(depends_on)
    ars = _clean_ars_data(ars)

    test_shares_by_age_group = _calculate_test_shares_by_age_group(ars)
    test_shares_by_age_group.to_pickle(produces["test_shares_by_age_group"])

    fig, ax = _plot_frame(test_shares_by_age_group)
    fig.savefig(produces["test_shares_by_age_group_png"])

    positivity_rates = ars["positivity_rate"].unstack()
    positivity_rates = _convert_from_weekly_to_daily(positivity_rates)
    positivity_rates.index.name = "date"
    positivity_rates.to_pickle(produces["positivity_rate_by_age_group"])

    fig, ax = _plot_frame(positivity_rates)
    fig.savefig(produces["positivity_rate_by_age_group_png"])

    positivity_rate_overall = (
        ars.groupby("date")["n_positive_tests"].sum()
        / ars.groupby("date")["n_tests"].sum()
    )
    positivity_rate_overall = _convert_from_weekly_to_daily(positivity_rate_overall)
    positivity_rate_overall.name = "positivity_rate_overall"
    positivity_rate_overall.to_pickle(produces["positivity_rate_overall"])

    fig, ax = _plot_frame(positivity_rate_overall.to_frame())
    fig.savefig(produces["positivity_rate_overall_png"])


def _clean_ars_data(ars):
    cleaned = ars[ars["year"].notnull()].copy()
    keep_cols = [
        "age_group",
        "n_tests",
        "pct_of_tests_positive",
        "n_tests_per_100_000",
        "n_positive_tests_per_100_000",
        "week",
        "year",
    ]
    cleaned = cleaned[keep_cols]
    assert cleaned.notnull().all().all()

    cleaned["date"] = cleaned.apply(get_date_from_year_and_week, axis=1)
    cleaned["positivity_rate"] = cleaned["pct_of_tests_positive"] / 100
    cleaned["n_positive_tests"] = cleaned["n_tests"] * cleaned["positivity_rate"]
    cleaned["age_group"] = cleaned["age_group"].replace({">=80": "80-100"})
    cleaned["age_group_rki"] = pd.Categorical(
        cleaned["age_group"],
        ordered=True,
        categories=["0-4", "5-14", "15-34", "35-59", "60-79", "80-100"],
    )
    cleaned = cleaned.drop(columns=["age_group"])
    cleaned = cleaned.set_index(["date", "age_group_rki"])
    return cleaned


def _calculate_test_shares_by_age_group(ars):
    """Calculate which share of tests each age group gets.


    Args:
        ars (pandas.DataFrame): DataFrame with dates and age_groups
            as MultiIndex. The columns contain "n_tests", the number
            of tests done in the group on each date.

    Returns:
        age_group_shares (pandas.DataFrame):

    """
    total_tests = ars.groupby("date")["n_tests"].sum()
    tests_by_age_group = ars["n_tests"].unstack()
    age_group_shares = tests_by_age_group.div(total_tests, axis=0)
    assert (age_group_shares.sum(axis=1).between(0.9999, 1.0001)).all()
    age_group_shares = _convert_from_weekly_to_daily(age_group_shares)
    age_group_shares.index.name = "date"
    return age_group_shares


def _convert_from_weekly_to_daily(short):
    """Version of convert_weekly_to_daily for DatetimeIndex pandas objects.

    Each week is filled with the observation of the end of the week.
    Together with `get_date_from_year_and_week` taking the Sunday of
    each week, this yields the week's values for Mon through Sun to be
    the values of reported for that week.


    Args:
        short (pandas.Series or pandas.DataFrame): index must be a DateTime index

    Returns:
        expanded (pandas.Series): the original Series expanded to have
            one entry for every day. Values are filled with the next
            available value from the future.

    """
    new_dates = pd.date_range(
        short.index.min() - pd.Timedelta(days=6), short.index.max()
    )
    expanded = short.reindex(new_dates)
    expanded = expanded.fillna(method="backfill")
    return expanded


def _plot_frame(df):
    fig, ax = plt.subplots(figsize=(15, 5))
    for col in df:
        sns.lineplot(ax=ax, x=df.index, y=df[col], label=col)
    fig, ax = style_plot(fig, ax)
    return fig, ax
