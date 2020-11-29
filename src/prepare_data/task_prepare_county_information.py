import pandas as pd
import pytask

from src.config import BLD
from src.config import SRC


def _prepare_general_data(paths):
    df = pd.read_csv(
        paths[0],
        skiprows=7,
        skipfooter=4,
        sep=";",
        encoding="latin_1",
        engine="python",
    )

    df = df.rename(
        columns={
            "Unnamed: 0": "id",
            "Unnamed: 1": "name",
            "Insgesamt": "population",
            "männlich": "male",
            "weiblich": "female",
        }
    )

    df_berlin = pd.read_csv(paths[1], sep=";")

    df = df.append(df_berlin)
    df["id"] = df["id"].astype(str)

    return df


def _prepare_federal_states(df):
    states = df.loc[df["id"].str.len() == 2].copy().drop(index=0).reset_index(drop=True)
    states[["population", "male", "female", "id"]] = states[
        ["population", "male", "female", "id"]
    ].astype(int)

    states["weight"] = states.population / states.population.sum()

    return states


def _prepare_counties(df, states):
    # Include Hamburg by converting its state id to Kreisschlüssel.
    df.id = df.id.replace({"02": "02000"})

    counties = df.loc[df["id"].str.len() == 5].copy()
    counties["state"] = counties["id"].str[:2]

    columns = ["population", "male", "female", "id", "state"]
    for col in columns:
        counties[col] = pd.to_numeric(counties[col], errors="coerce")

    counties = counties.dropna()
    counties[columns] = counties[columns].astype(int)
    counties["weight"] = counties.population / counties.population.sum()

    counties = (
        counties.merge(
            states[["id", "name"]],
            left_on="state",
            right_on="id",
            validate="m:1",
            suffixes=("", "_y"),
        )
        .drop(columns=["id_y", "state"])
        .rename(columns={"name_y": "state"})
    )

    counties["name"] = counties["name"].str.strip()
    counties["state"] = counties["state"].str.strip()

    return counties


@pytask.mark.depends_on(
    [
        SRC / "original_data" / "population_structure" / "population.csv",
        SRC / "original_data" / "population_structure" / "population_berlin.csv",
    ]
)
@pytask.mark.produces(
    [BLD / "data" / "federal_states.parquet", BLD / "data" / "counties.parquet"]
)
def task_prepare_geographical_data_de(depends_on, produces):
    df = _prepare_general_data(depends_on)

    states = _prepare_federal_states(df)
    states.to_parquet(produces[0])

    counties = _prepare_counties(df, states)
    counties.to_parquet(produces[1])
