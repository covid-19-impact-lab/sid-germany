import pandas as pd
import pytask

from src.config import BLD
from src.config import SRC
from src.shared import from_timestamps_to_epochs


def _prepare_vacations(path):
    vacations = pd.read_excel(path)
    vacations = (
        vacations.melt(id_vars="Bundesland", var_name="vacation", value_name="date")
        .rename(columns={"Bundesland": "state"})
        .dropna()
    )

    cleaned_str_dates = vacations.date.str.replace(" – ", " - ")
    raw_dates = cleaned_str_dates.str.split(" - ", expand=True)
    dates = raw_dates.rename(columns={0: "start", 1: "end"})
    dates["end"] = dates["end"].fillna(dates["start"])
    dates["start"] = pd.to_datetime(dates["start"], format="%d.%m.%Y")
    dates["end"] = pd.to_datetime(dates["end"], format="%d.%m.%Y")
    vacations = pd.concat([vacations, dates], axis=1).drop(columns="date")
    assert (vacations["start"] <= vacations["end"]).all(), "End date before start date."
    vacations["length"] = vacations["end"] - vacations["start"]
    long_vacations = vacations[vacations["length"] > pd.Timedelta(days=20)]
    assert (long_vacations["vacation"].isin(["Sommerferien", "Sommerferien2021"])).all()
    assert (
        vacations["length"] < pd.Timedelta(days=46)
    ).all(), "No vacation longer than 45 days allowed."
    vacations = vacations.drop(columns=["length"])
    return vacations


def _convert_to_params_format(df):
    long_format = df.melt(
        id_vars=["vacation", "state"], var_name="limit", value_name="date"
    )
    params = long_format.rename(
        columns={
            "vacation": "category",
            "state": "subcategory",
            "limit": "name",
            "date": "value",
        }
    ).set_index(["category", "subcategory", "name"])

    return params


@pytask.mark.depends_on(
    {
        "data": SRC / "original_data" / "vacations" / "vacations_2020.xlsx",
        "shared.py": SRC / "shared.py",
        "config.py": SRC / "config.py",
    }
)
@pytask.mark.produces(BLD / "data" / "vacations.pkl")
def task_prepare_vacations(depends_on, produces):
    """Prepare data on vacations in Germany such that they can be added to params.

    Note that the date is saved in epochs meaning the passed time from 1970-01-01 in
    seconds. This allows to save the dates in a numeric format and keep the "value"
    column in params in a numeric format.

    """
    df = _prepare_vacations(depends_on["data"])
    params = _convert_to_params_format(df)
    # Convert dates to epochs, so that the "value" can stay numeric.
    params["value"] = from_timestamps_to_epochs(params["value"])
    params.to_pickle(produces)
