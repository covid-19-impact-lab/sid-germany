import pandas as pd
import pytask

from src.config import BLD
from src.config import POPULATION_GERMANY
from src.testing.task_calculate_test_capacity import convert_weekly_to_daily
from src.testing.task_calculate_test_capacity import get_date_from_year_and_week
from src.testing.task_calculate_test_capacity import plot_time_series


OUT_PATH = BLD / "data" / "testing"
PRODUCTS = {}
OUT_COLS = [
    "n_tests",
    "n_positive_tests",
    "share_tests_positive",
    "n_laboratories",
    "n_tests_per_100_000",
    "n_positive_tests_per_100_000",
]
for col in OUT_COLS:
    PRODUCTS[col] = OUT_PATH / f"{col}.csv"
    PRODUCTS[f"{col}_png"] = OUT_PATH / f"{col}.png"


@pytask.mark.depends_on(BLD / "data" / "raw_time_series" / "test_statistics.xlsx")
@pytask.mark.produces(PRODUCTS)
def task_create_test_statistics(depends_on, produces):
    df = pd.read_excel(depends_on, sheet_name="1_Testzahlerfassung")
    df = _prepare_data(df)
    for col in OUT_COLS:
        df.set_index("date")[col].to_csv(produces[col])
        fig, ax = plot_time_series(df=df, y=col)
        fig.savefig(produces[f"{col}_png"])


def _prepare_data(df):
    translations = {
        "Kalenderwoche": "week_and_year",
        "Anzahl Testungen": "n_tests",
        "Positiv getestet": "n_positive_tests",
        "Positivenquote (%)": "share_tests_positive",
        "Anzahl übermittelnder Labore": "n_laboratories",
    }
    df = df.rename(columns=translations)
    df = df[translations.values()]
    # drop lines that contain comments or sums:
    df = df[df["share_tests_positive"].notnull()]

    df["date"] = _create_date(df)
    df["share_tests_positive"] = df["share_tests_positive"] / 100
    df = df.drop(columns=["week_and_year"])
    df = convert_weekly_to_daily(df, divide_by_7_cols=["n_tests", "n_positive_tests"])

    df["n_tests_per_100_000"] = 100_000 * df["n_tests"] / POPULATION_GERMANY
    df["n_positive_tests_per_100_000"] = (
        100_000 * df["n_positive_tests"] / POPULATION_GERMANY
    )
    df = df[df["date"] > "2020-08-15"]
    return df


def _create_date(df):
    week_and_year = df["week_and_year"].str.split("/", 1, expand=True)
    week_and_year[1] = week_and_year[1].str.replace("*", "")
    week_and_year = week_and_year.astype(int)
    week_and_year.columns = ["week", "year"]
    return week_and_year.apply(get_date_from_year_and_week, axis=1)
