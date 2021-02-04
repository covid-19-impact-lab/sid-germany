import datetime

import matplotlib.pyplot as plt
import numpy as np
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
    df = pd.read_excel(depends_on, sheet_name="Testkapazitäten", header=1)
    df = _process_test_statistics(df)

    fig, ax = plot_time_series(
        df, y="n_laboratories", title="Number of Laboratories Reporting"
    )
    fig.savefig(produces["n_laboratories_overall"])

    # make sure enough laboratories are participating
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


def _process_test_statistics(df):
    df = df.replace("-", np.nan)
    df["date"] = _create_date(df)
    df = expand_to_every_day(df)
    df = df.rename(
        columns={
            "Anzahl übermittelnde Labore": "n_laboratories",
            "Reale Testkapazität zum Zeitpunkt der Abfrage": "weekly_test_capacity",
        }
    )
    df["test_capacity"] = df["weekly_test_capacity"] / 7
    df["tests_per_inhabitant"] = df["test_capacity"] / POPULATION_GERMANY
    df["test_capacity_per_100_000"] = 100_000 * df["tests_per_inhabitant"]

    to_drop = [
        "KW, für die die Angabe prognostisch erfolgt ist:",
        "Testkapazität pro Tag",
        "Theoretische wöchentliche Kapazität anhand von Wochenarbeitstagen",
    ]
    df = df.drop(columns=to_drop)
    return df


def _create_date(df):
    time_col = "KW, für die die Angabe prognostisch erfolgt ist:"
    year_and_week = df[time_col].str.split(", KW", 1, expand=True)
    year_and_week = year_and_week.astype(int)
    year_and_week.columns = ["year", "week"]
    return year_and_week.apply(get_date_from_year_and_week_to_date, axis=1)


def get_date_from_year_and_week_to_date(row):
    date = datetime.date.fromisocalendar(
        year=int(row["year"]), week=int(row["week"]), day=3
    )
    return pd.Timestamp(date)


def expand_to_every_day(df):
    dates = pd.date_range(df["date"].min(), df["date"].max())
    df = df.set_index("date")
    df = df.reindex(dates)
    df = df.interpolate("nearest")
    df = df.reset_index()
    df = df.rename(columns={"index": "date"})
    return df


def plot_time_series(df, y, title=""):
    fig, ax = plt.subplots(figsize=(10, 3))
    sns.lineplot(data=df, x="date", y=y)
    ax.set_title(title)
    fig, ax = style_plot(fig, ax)
    fig.tight_layout()
    return fig, ax
