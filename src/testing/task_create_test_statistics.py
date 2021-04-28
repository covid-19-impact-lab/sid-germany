import matplotlib.pyplot as plt
import pandas as pd
import pytask

from src.config import BLD
from src.config import POPULATION_GERMANY
from src.testing.shared import convert_weekly_to_daily
from src.testing.shared import get_date_from_year_and_week
from src.testing.shared import plot_time_series


OUT_PATH = BLD / "data" / "testing"
PRODUCTS = {"data": OUT_PATH / "test_numbers.csv"}
OUT_COLS = [
    "n_tests",
    "n_positive_tests",
    "share_tests_positive",
    "n_laboratories",
    "n_tests_per_100_000",
    "n_positive_tests_per_100_000",
]
for col in OUT_COLS:
    PRODUCTS[f"{col}_png"] = OUT_PATH / f"{col}.png"


@pytask.mark.depends_on(BLD / "data" / "raw_time_series" / "test_statistics.xlsx")
@pytask.mark.produces(PRODUCTS)
def task_create_test_statistics(depends_on, produces):
    df = pd.read_excel(depends_on, sheet_name="1_Testzahlerfassung")
    df = _prepare_data(df)
    df.to_csv(produces["data"])
    for col in OUT_COLS:
        title = col.replace("_", " ").replace("n ", "Number of ")
        fig, ax = plot_time_series(df=df, y=col, title=title)
        fig.savefig(produces[f"{col}_png"])
        plt.close()


def _prepare_data(df):
    """Prepare the test statistics data.

    In particular:
        - rename columns
        - drop comment and summary rows
        - create a date from the entered calendar week and year
        - Extend the data to have an observation for each day
        - create number of tests and positive tests per 100_000

    """
    translations = {
        "Kalenderwoche": "week_and_year",
        "Anzahl Testungen": "n_tests",
        "Positiv getestet": "n_positive_tests",
        "Positivenanteil (%)": "pct_tests_positive",
        "Anzahl übermittelnder Labore": "n_laboratories",
    }
    df = df.rename(columns=translations)
    df = df[translations.values()]
    # drop rows that contain comments or sums:
    df = df[df["pct_tests_positive"].notnull()]

    df["date"] = _create_date(df)
    df["share_tests_positive"] = df["pct_tests_positive"] / 100
    df = df.drop(columns=["week_and_year"])
    df = convert_weekly_to_daily(df, divide_by_7_cols=["n_tests", "n_positive_tests"])

    df["n_tests_per_100_000"] = 100_000 * df["n_tests"] / POPULATION_GERMANY
    df["n_positive_tests_per_100_000"] = (
        100_000 * df["n_positive_tests"] / POPULATION_GERMANY
    )
    # restrict to time frame where enough laboratories are participating
    df = df[df["date"] > pd.Timestamp("2020-08-15")]
    return df


def _create_date(df):
    """Get week and year from "week_and_year" column and calculate date from it."""
    week_and_year = df["week_and_year"].str.split("/", 1, expand=True)
    week_and_year[1] = week_and_year[1].str.replace("*", "")
    week_and_year = week_and_year.astype(int)
    week_and_year.columns = ["week", "year"]
    return week_and_year.apply(get_date_from_year_and_week, axis=1)
