"""Prepare the ARS data."""
import pandas as pd
import pytask

from src.config import BLD
from src.config import SRC
from src.testing.task_calculate_test_capacity import (
    get_date_from_year_and_week,
)

OUT_PATH = BLD / "data" / "testing"

PRODUCTS = {
    "full_weekly": OUT_PATH / "ars_cleaned_full_weekly.csv",
    "test_shares_by_age_group": OUT_PATH / "test_shares_by_age_group.csv",
    "positivity_rate_by_age_group": OUT_PATH / "positivity_rate_by_age_group.csv",
}


@pytask.mark.depends_on(SRC / "original_data" / "testing" / "ars_data_raw.xlsx")
@pytask.mark.produces(PRODUCTS)
def task_prepare_ars_data(depends_on, produces):
    ars = pd.read_excel(depends_on)
    ars = _clean_ars_data(ars)

    test_shares_by_age_group = _calculate_test_shares_by_age_group(ars)
    test_shares_by_age_group.to_csv(produces["test_shares_by_age_group"])

    positivity_rates = ars["positivity_rate_by_age_group"].unstack()
    positivity_rates = _convert_from_weekly_to_daily(positivity_rates)
    positivity_rates.to_csv(produces["positivity_rate_by_age_group"])

    ars.to_csv(produces["full_weekly"])


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
    cleaned["positivity_rate_by_age_group"] = cleaned["pct_of_tests_positive"] / 100
    cleaned["n_positive_tests"] = (
        cleaned["n_tests"] * cleaned["positivity_rate_by_age_group"]
    )
    cleaned["age_group"] = pd.Categorical(
        cleaned["age_group"],
        ordered=True,
        categories=["0-4", "5-14", "15-34", "35-59", "60-79", ">=80"],
    )
    cleaned = cleaned.set_index(["date", "age_group"])
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
