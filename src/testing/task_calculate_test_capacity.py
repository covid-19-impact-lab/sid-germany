import datetime

import matplotlib.pyplot as plt
import pandas as pd
import pytask
import seaborn as sns

from src.config import BLD
from src.config import POPULATION_GERMANY
from src.simulation.plotting import style_plot


OUT_PATH = BLD / "data" / "processed_time_series"
PRODUCTS = {
    "n_laboratories_overall": OUT_PATH / "n_laboratories_overall.png",
    "n_laboratories_after_july": OUT_PATH / "n_laboratories_after_july.png",
    "data": OUT_PATH / "testing_capacity.csv",
    "daily_capacity_de": OUT_PATH / "daily_capacity_de.png",
    "n_test_capacity_per_100_000": OUT_PATH / "n_test_capacity_per_100_000.png",
}


@pytask.mark.depends_on(BLD / "data" / "raw_time_series" / "test_statistics.xlsx")
@pytask.mark.produces(PRODUCTS)
def task_calculate_test_capacity(depends_on, produces):
    df = pd.read_excel(depends_on, sheet_name="2_Testkapazitäten")
    df = _prepare_test_capacity_data(df)

    fig, ax = plot_time_series(
        df, y="n_laboratories", title="Number of Laboratories Reporting"
    )
    fig.savefig(produces["n_laboratories_overall"])

    # restrict to time frame where enough laboratories are participating
    df = df[df["date"] > pd.Timestamp("2020-08-15")]
    df.to_csv(produces["data"])

    fig, ax = plot_time_series(
        df, y="n_laboratories", title="Number of Laboratories Reporting After July"
    )
    fig.savefig(produces["n_laboratories_after_july"])

    fig, ax = plot_time_series(
        df, y="test_capacity", title="Absolute Daily Capacity in Germany"
    )
    fig.savefig(produces["daily_capacity_de"])

    fig, ax = plot_time_series(
        df,
        y="test_capacity_per_100_000",
        title="Tests Available per 100 000",
    )
    fig.savefig(produces["n_test_capacity_per_100_000"])


def _prepare_test_capacity_data(df):
    df["date"] = _create_date(df)
    rename_dict = {
        "Anzahl übermittelnde Labore": "n_laboratories",
        "Reale Testkapazität zum Zeitpunkt der Abfrage": "test_capacity",
    }
    df = df.rename(columns=rename_dict)
    df = df[["date", "n_laboratories", "test_capacity"]]
    df = convert_weekly_to_daily(df, divide_by_7_cols=["test_capacity"])
    df["tests_per_inhabitant"] = df["test_capacity"] / POPULATION_GERMANY
    df["test_capacity_per_100_000"] = 100_000 * df["tests_per_inhabitant"]
    return df


def _create_date(df):
    """Create year and week from string column and convert to true date."""
    time_col = "KW, für die die Angabe prognostisch erfolgt ist"
    year_and_week = df[time_col].str.split(", KW", 1, expand=True)
    year_and_week = year_and_week.astype(int)
    year_and_week.columns = ["year", "week"]
    return year_and_week.apply(get_date_from_year_and_week, axis=1)


def get_date_from_year_and_week(row):
    """Create date from year and week.

    We take the Sunday of each week.

    """
    date = datetime.date.fromisocalendar(
        year=int(row["year"]), week=int(row["week"]), day=7
    )
    return pd.Timestamp(date)


def convert_weekly_to_daily(df, divide_by_7_cols):
    """Convert from a weekly to a daily index.

    Each week is filled with the observation of the end of the week.
    Together with `get_date_from_year_and_week` taking the Sunday of
    each week, this yields the week's values for Mon through Sun to be
    the values of reported for that week.

    Args:
        df (pandas.DataFrame): DataFrame with
        divide_by_7_cols (list): list of columns that have to be
            divided by 7. So for example the number of participating
            laboratories does not change from a weekly to daily
            representation of the data but the available number of
            tests on each day is (ignoring weekends) a seventh of the
            weekly capacity.

    """
    dates = pd.date_range(df["date"].min() - pd.Timedelta(days=6), df["date"].max())
    df = df.set_index("date")
    df = df.reindex(dates)
    df = df.fillna(method="backfill")
    df = df.reset_index()
    df = df.rename(columns={"index": "date"})
    df[divide_by_7_cols] = df[divide_by_7_cols] / 7
    return df


def plot_time_series(df, y, title=""):
    fig, ax = plt.subplots(figsize=(10, 3))
    sns.lineplot(data=df, x="date", y=y)
    ax.set_title(title)
    fig, ax = style_plot(fig, ax)
    fig.tight_layout()
    return fig, ax
