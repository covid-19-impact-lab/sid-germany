"""Create the work_multiplier Series.


Notes:
    - Apple mobility data only includes reductions by means of transport
      (e.g. driving, walking) and is therefore not useful for us.
    - the work_multiplier gives the work_contact_priority threshold above wihch
      people still go to work.

"""
import numpy as np
import pandas as pd
import pytask

from src.config import BLD
from src.config import SRC


@pytask.mark.depends_on(
    {
        "mobility_data": SRC / "original_data" / "google_mobility_2021-01-21_DE.csv",
        "hygiene_data": SRC / "original_data" / "cosmo_hygiene_2021-01-28.csv",
    }
)
@pytask.mark.produces(
    {
        "cleaned_mobility": BLD / "policies" / "cleaned_mobility.csv",
        "hygiene_score": BLD / "policies" / "hygiene_score.csv",
        "work_multiplier": BLD / "policies" / "work_multiplier.csv",
    }
)
def task_create_work_multiplier_series(depends_on, produces):
    mobility_data = pd.read_csv(depends_on["mobility_data"])
    hygiene_data = pd.read_csv(depends_on["hygiene_data"])

    mobility_data = _prepare_mobility_data(df=mobility_data)
    mobility_data.to_csv(produces["cleaned_mobility"])

    hygiene_score = _calculate_hygiene_score_from_data(df=hygiene_data)
    hygiene_score.to_csv(produces["hygiene_score"])

    work_multiplier = _calculate_work_multiplier(df=mobility_data)
    work_multiplier.to_csv(produces["work_multiplier"])


def _prepare_mobility_data(df):
    df["date"] = pd.DatetimeIndex(df["date"])
    to_drop = [
        "country_region_code",
        "country_region",
        "sub_region_2",
        "metro_area",
        "iso_3166_2_code",
        "census_fips_code",
    ]
    df = df.drop(columns=to_drop)
    df.columns = [x.replace("_percent_change_from_baseline", "") for x in df.columns]
    return df


def _calculate_hygiene_score_from_data(df):
    assert not df["Datum"].duplicated().any()
    df.index = pd.DatetimeIndex(df["Datum"], name="date")
    # only use mean values
    df = df[[x for x in df if x.endswith("M")]]
    hygiene_score = df.mean(axis=1).dropna()
    start, end = hygiene_score.index.min(), hygiene_score.index.max()
    dates = pd.date_range(start, end, name="date")
    expanded = pd.Series(hygiene_score, index=dates, name="hygiene_score")
    expanded = expanded.interpolate(method="nearest")
    return expanded


def _calculate_work_multiplier(df):
    de_df = df[df["sub_region_1"].isnull()].copy()  # only whole of Germany
    de_df["share_working"] = 1 + (de_df["workplaces"] / 100)
    assert not de_df["date"].duplicated().any()
    work_multiplier = de_df.set_index("date")["share_working"]
    # set weekends to NaN and interpolate because we already handle weekends
    # in the contact models.
    weekends = work_multiplier.index.day_name().isin(["Saturday", "Sunday"])
    work_multiplier[weekends] = np.nan
    work_multiplier = work_multiplier.interpolate(method="nearest")
    return work_multiplier


def _make_hygiene_score_match_our_hygiene_multiplier(hygiene_score):
    """Make hygiene score reflect our multiplier.

    We change scale to have 1 as October value and higher values mean
    riskier behavior. Note that the score can never be interpreted quantitatively.

    """
    hygiene_multiplier = hygiene_score / hygiene_score["2020-10-13"]
    hygiene_multiplier = 2 - hygiene_multiplier
    return hygiene_multiplier
