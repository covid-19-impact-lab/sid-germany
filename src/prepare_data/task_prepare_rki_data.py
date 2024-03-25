"""This module prepares the data from RKI.

For validation,

- the sum of "newly_infected" yields the number of total infections
- the sum of "newly_deceased" yields the number of total deaths

in the daily report from RKI for the same date.

Explanation on the coding of the variables

- https://www.arcgis.com/home/item.html?id=f10774f1c63e40168479a1feb6c7ca74
- https://covid19-de-stats.sourceforge.io/rki-fall-tabelle.html

"""

from datetime import datetime
from datetime import timedelta

import numpy as np
import pandas as pd
import pytask

from src.config import BLD


DROPPPED_COLUMNS = [
    "IdBundesland",
    "Landkreis",
    "Geschlecht",
    "Datenstand",
    "NeuGenesen",
    "AnzahlGenesen",
    "Altersgruppe2",
    "Meldedatum",
]

RENAME_COLUMNS = {
    "FID": "id",
    "Altersgruppe": "age_group",
    "IdLandkreis": "county",
    "Bundesland": "state",
    "Refdatum": "date",
    "IstErkrankungsbeginn": "is_date_disease_onset",
    "NeuerFall": "type_case",
    "AnzahlFall": "n_cases",
    "NeuerTodesfall": "type_death",
    "AnzahlTodesfall": "n_deaths",
}

AGE_GROUPS_TO_INTERVALS = {
    "A00-A04": "0-4",
    "A05-A14": "5-14",
    "A15-A34": "15-34",
    "A35-A59": "35-59",
    "A60-A79": "60-79",
    "A80+": "80-100",
    "unbekannt": np.nan,
}

# Translate those states that have an English name to English
# Needed to align with initial states
TRANSLATE_STATES = {
    "Nordrhein-Westfalen": "North Rhine-Westphalia",
    "Rheinland-Pfalz": "Rhineland-Palatinate",
    "Niedersachsen": "Lower Saxony",
    "Sachsen": "Saxony",
    "Bayern": "Bavaria",
    "Sachsen-Anhalt": "Saxony-Anhalt",
    "Thüringen": "Thuringia",
}


@pytask.mark.depends_on(
    {
        "rki": BLD / "data" / "raw_time_series" / "rki.csv",
    }
)
@pytask.mark.produces(
    {
        "data": BLD / "data" / "processed_time_series" / "rki.pkl",
    }
)
def task_prepare_rki_data(depends_on, produces):
    df = pd.read_csv(depends_on["rki"], parse_dates=["Refdatum"])
    df = df.drop(columns=DROPPPED_COLUMNS)
    df = df.rename(columns=RENAME_COLUMNS)
    df["age_group_rki"] = (
        df["age_group"].replace(AGE_GROUPS_TO_INTERVALS).astype("category")
    )
    df = df.drop(columns=["age_group"])

    df["is_date_disease_onset"] = df["is_date_disease_onset"].astype(bool)

    df["newly_infected"] = df["n_cases"] * df["type_case"].isin([0, 1])
    df["newly_deceased"] = df["n_deaths"] * df["type_death"].isin([0, 1])

    county_to_state = (
        df[["county", "state"]]
        .drop_duplicates("county")
        .set_index("county")["state"]
        .to_dict()
    )

    gb = df.groupby(["date", "county", "age_group_rki"])
    summed = gb[["newly_infected", "newly_deceased"]].sum()
    summed = summed.fillna(0)
    today = datetime.now().date()
    one_week_ago = today - timedelta(weeks=1)
    # 2020-02-08 started missing on 2021-02-24.
    cropped = summed.loc[pd.Timestamp("2020-02-09") : one_week_ago]  # noqa: E203
    cropped = cropped.sort_index()
    cropped["state"] = cropped.index.get_level_values("county")
    cropped["state"] = cropped["state"].replace(county_to_state)
    cropped["state"] = cropped["state"].replace(TRANSLATE_STATES)
    assert cropped.notnull().all().all()
    dates = cropped.index.get_level_values("date").unique()
    expected_dates = pd.date_range(dates.min(), dates.max())
    missing_dates = [str(x.date()) for x in expected_dates if x not in dates]
    assert (
        len(missing_dates) == 0
    ), f"There are missing dates in the RKI data: {missing_dates}"

    cropped.to_pickle(produces["data"])
