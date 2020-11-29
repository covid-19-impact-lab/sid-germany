import pandas as pd
import pytask

from src.config import BLD
from src.config import SRC


def _convert_to_correct_dtypes(df):
    df["week"] = df["week"].str.replace("KW", "").astype(int)
    for column in [
        "available_capacity_weekly_theoretical",
        "available_capacity_weekly",
    ]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


def _week_number_to_date(df):
    df["date"] = pd.to_datetime("2020", format="%Y") + pd.to_timedelta(
        df["week"].mul(7).subtract(3).astype(str) + " days"
    )
    assert (df["week"] == df["date"].dt.week).all()
    df = df.drop(columns="week")
    df = df.set_index("date", drop=True)
    return df


def _prepare_capacity(path):
    df = pd.read_excel(
        path,
        sheet_name="Testkapazitäten",
        skiprows=1,
        names=[
            "week",
            "n_labs_capacity",
            "available_capacity_daily",
            "available_capacity_weekly_theoretical",
            "available_capacity_weekly",
        ],
    )
    df = _convert_to_correct_dtypes(df)
    df = _week_number_to_date(df)

    return df


def _prepare_backlog(path):
    df = pd.read_excel(
        path, sheet_name="Probenrückstau", names=["n_labs_w_backlog", "week", "backlog"]
    )
    df["backlog"] = df["backlog"].astype(int)
    df = _week_number_to_date(df)

    return df


def _prepare_test_cases(path):
    df = pd.read_excel(
        path,
        sheet_name="Testzahlen",
        usecols="B:D,F",
        skiprows=3,
        names=["week", "n_tests", "n_tests_positive", "n_labs_tests"],
    ).dropna()
    df = _week_number_to_date(df)
    df["n_labs_tests"] = df["n_labs_tests"].astype(int)

    return df


@pytask.mark.depends_on(SRC / "original_data" / "testing" / "testing.xlsx")
@pytask.mark.produces(BLD / "data" / "testing" / "testing.parquet")
def task_prepare_testing_data(depends_on, produces):
    """Prepare testing data.

    The weekly update file can be found under
    https://www.rki.de/DE/Content/InfAZ/N/Neuartiges_Coronavirus/nCoV_node.html under
    "Daten zum Download" and "Tabellen zu Testzahlen, Testkapazitäten und Probenrückstau
    ...".

    """
    capacity = _prepare_capacity(depends_on)
    backlog = _prepare_backlog(depends_on)
    test_cases = _prepare_test_cases(depends_on)

    df = pd.concat([capacity, backlog, test_cases], axis=1)
    df["week"] = df.index.to_series().dt.week

    df.to_parquet(produces)
