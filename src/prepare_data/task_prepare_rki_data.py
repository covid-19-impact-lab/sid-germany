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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytask
import seaborn as sns

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


@pytask.mark.depends_on(
    {
        "rki": BLD / "data" / "raw_time_series" / "rki.csv",
        "share_known_cases_until_christmas": BLD
        / "data"
        / "processed_time_series"
        / "share_known_cases_until_christmas.pkl",
        "share_known_cases_since_christmas": BLD
        / "data"
        / "processed_time_series"
        / "share_known_cases_since_christmas.pkl",
    }
)
@pytask.mark.produces(
    {
        "data": BLD / "data" / "processed_time_series" / "rki.pkl",
        "plot": BLD / "data" / "processed_time_series" / "rki_reported_vs_upscaled.png",
    }
)
def task_prepare_rki_data(depends_on, produces):
    df = pd.read_csv(depends_on["rki"], parse_dates=["Refdatum"])
    df = df.drop(columns=DROPPPED_COLUMNS)
    df = df.rename(columns=RENAME_COLUMNS)
    share_known_until_christmas = pd.read_pickle(
        depends_on["share_known_cases_until_christmas"]
    )
    share_known_after_christmas = pd.read_pickle(
        depends_on["share_known_cases_since_christmas"]
    )

    df["age_group_rki"] = (
        df["age_group"].replace(AGE_GROUPS_TO_INTERVALS).astype("category")
    )
    df = df.drop(columns=["age_group"])
    df["is_date_disease_onset"] = df["is_date_disease_onset"].astype(bool)
    df["newly_infected"] = df["n_cases"] * df["type_case"].isin([0, 1])
    df["newly_deceased"] = df["n_deaths"] * df["type_death"].isin([0, 1])

    gb = df.groupby(["date", "county", "age_group_rki"])
    summed = gb[["newly_infected", "newly_deceased"]].sum()
    summed = summed.fillna(0)
    today = datetime.now().date()
    one_week_ago = today - timedelta(weeks=1)
    cropped = summed.loc[:one_week_ago]
    cropped = cropped.sort_index()

    # build share known cases.
    dates = cropped.index.get_level_values("date")
    share_known_cases = pd.Series(index=dates.unique())
    share_known_cases = share_known_cases.fillna(share_known_until_christmas)
    share_known_cases = share_known_cases.fillna(share_known_after_christmas)
    undetected_multiplier = 1 / share_known_cases

    cropped["date"] = dates
    cropped["undetected_multiplier"] = cropped["date"].replace(undetected_multiplier)
    cropped["undetected_multiplier"] = cropped["undetected_multiplier"].astype(float)
    cropped["upscaled_newly_infected"] = (
        cropped["newly_infected"] * cropped["undetected_multiplier"]
    )
    cropped = cropped.drop(columns=["date", "undetected_multiplier"])

    assert cropped["newly_infected"].notnull().all()
    cropped.to_pickle(produces["data"])

    reported_per_day = cropped["newly_infected"].groupby("date").sum()
    upscaled_per_day = cropped["upscaled_newly_infected"].groupby("date").sum()
    fig, ax = plt.subplots(figsize=(8, 3))
    sns.lineplot(
        x=reported_per_day.index, y=reported_per_day, ax=ax, label="Reported Cases"
    )
    sns.lineplot(
        x=upscaled_per_day.index,
        y=upscaled_per_day,
        ax=ax,
        label="Corrected for Undetected Infections",
    )

    ax.set_title("Reported Versus Corrected Cases")
    plt.legend(frameon=False)
    sns.despine()
    fig.tight_layout()
    fig.savefig(produces["plot"])
