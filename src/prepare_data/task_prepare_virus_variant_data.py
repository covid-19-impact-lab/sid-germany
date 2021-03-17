import warnings

import numpy as np
import pandas as pd
import pytask
import statsmodels.formula.api as smf

from src.config import BLD
from src.config import SRC
from src.testing.shared import get_date_from_year_and_week


STRAIN_FILES = {
    "rki_strains": BLD / "data" / "virus_strains" / "rki_strains.csv",
    "cologne": BLD / "data" / "virus_strains" / "cologne_strains.csv",
    "final_strain_shares": BLD / "data" / "virus_strains" / "final_strain_shares.pkl",
}


@pytask.mark.depends_on(
    {
        "rki": SRC / "original_data" / "virus_strains_rki.csv",
        "cologne": SRC / "original_data" / "virus_strains_cologne.csv",
    }
)
@pytask.mark.produces(STRAIN_FILES)
def task_prepare_virus_variant_data(depends_on, produces):
    share_cols = ["share_b117", "share_b1351"]

    rki = pd.read_csv(depends_on["rki"])
    rki = _prepare_rki_data(rki)
    rki.to_csv(produces["rki_strains"])

    cologne = pd.read_csv(depends_on["cologne"])
    cologne = _prepare_cologne_data(cologne, share_cols)
    cologne.to_csv(produces["cologne"])

    cologne_smoothed = cologne[share_cols].dropna()
    averaged = _average_over_co_and_rki(rki, cologne_smoothed)

    # extrapolate into the past
    past = pd.DataFrame()
    past["share_b117"] = _extrapolate(
        averaged,
        y="share_b117",
        start="2020-03-01",
        end=averaged.index.min() - pd.Timedelta(days=1),
    )
    past["share_b1351"] = 0
    past["share_p1"] = 0

    # extrapolate into the future
    future = pd.DataFrame()
    future["share_b117"] = _extrapolate(
        averaged,
        y="share_b117",
        start=averaged.index.max() + pd.Timedelta(days=1),
        end=averaged.index.max() + pd.Timedelta(days=28),
    )
    future["share_b1351"] = averaged["share_b1351"].mean()
    future["share_p1"] = 0

    strain_data = pd.concat([past, averaged, future], axis=0).sort_index()
    strain_data.columns = [x.replace("share_", "") for x in strain_data.columns]

    assert strain_data.notnull().all().all()
    expected_dates = pd.date_range(strain_data.index.min(), strain_data.index.max())
    assert (strain_data.index == expected_dates).all()
    strain_data.to_pickle(produces["final_strain_shares"])


def _prepare_rki_data(df):
    # The RKI data also contains info on P.1.
    df = df[df["week"].notnull()].copy(deep=True)
    df["year"] = 2021
    df["date"] = df.apply(get_date_from_year_and_week, axis=1)
    df = df.set_index("date")
    as_float_cols = ["pct_b117", "pct_b1351", "pct_p1", "n_tested_for_variants"]
    df[as_float_cols] = df[as_float_cols].astype(float)
    df["share_b117"] = df["pct_b117"] / 100
    df["share_b1351"] = df["pct_b1351"] / 100
    df["share_p1"] = df["pct_p1"] / 100
    keep_cols = [
        "share_b117",
        "share_b1351",
        "share_p1",
    ]
    df = df[keep_cols]
    dates = pd.date_range(df.index.min(), df.index.max())
    # no division by 7 necessary because the data only contains shares.
    df = df.reindex(dates).interpolate()
    df.index.name = "date"
    return df


def _prepare_cologne_data(df, share_cols):
    keep_cols = ["n_b117_cum", "n_b1351_cum", "n_tests_positive_cum", "date"]
    df = df[keep_cols].dropna().copy(deep=True)
    df["date"] = pd.to_datetime(df["date"], dayfirst=True)
    df = df.set_index("date").astype(int).sort_index()
    # Cologne started screening all PCR positive samples for mutatations at
    # the end of January
    df = df.loc[pd.Timestamp("2021-02-04") :]  # noqa
    for col in df:
        assert (df[col].diff().dropna() >= 0).all(), col
        df[col.replace("_cum", "")] = df[col].diff()
    df["share_b117"] = df["n_b117"] / df["n_tests_positive"]
    df["share_b1351"] = df["n_b1351"] / df["n_tests_positive"]
    df = df[share_cols]
    df.columns = [f"{col}_unsmoothed" for col in df.columns]
    # take 7 day average to remove weekend effects
    df[share_cols] = df.rolling(7, center=True).mean()
    dates = pd.date_range(df.index.min(), df.index.max())
    df = df.reindex(dates)
    df[share_cols] = df[share_cols].interpolate()
    df.index.name = "date"
    return df


def _average_over_co_and_rki(rki, cologne):
    """Merge and average over the Cologne and RKI strains data.

    Args:
        rki (pandas.DataFrame): index are dates. columns are the shares
            of the different virus strains according to the RKI data.
        cologne (pandas.DataFrame): index are dates. columns are the shares
            of the different virus strains according to the cologne data.

    Returns:
        averaged (pandas.DataFrame): index are dates. columns are the
            averaged shares of the different virus strains between RKI
            and cologne

    """
    assert all(
        col.startswith("share_") for col in rki.columns
    ), "RKI has non share columns"
    assert all(
        col.startswith("share_") for col in cologne.columns
    ), "Cologne has non share columns"

    # prepend index levels to identify source dataframe
    cologne["source"] = "cologne"
    rki["source"] = "rki"
    cologne = cologne.set_index("source", append=True)
    rki = rki.set_index("source", append=True)

    merged = pd.concat([rki, cologne]).sort_index()
    averaged = merged.groupby("date").mean()
    # p1 only reported by RKI which only reports weekly. This can lead to NaN
    averaged["share_p1"] = averaged["share_p1"].fillna(method="ffill")
    return averaged


def _extrapolate(df, y, start, end):
    """Use data on the virus strains to extrapolate their incidence into the past."""
    data = df[y].to_frame()
    data["days_since_start"] = (data.index - data.index.min()).days

    model = smf.ols(f"np.log({y}) ~ days_since_start", data=data)
    results = model.fit()
    if results.rsquared <= 0.8:
        warnings.warn(
            f"\n\nYour fit of {y} has worsened to only {results.rsquared.round(2)}.\n\n"
        )

    full_x = pd.DataFrame(index=pd.date_range(start, end))
    full_x["days_since_start"] = (full_x.index - df.index.min()).days
    extrapolated = np.exp(results.predict(full_x))
    extrapolated = extrapolated.round(6).clip(0, 1)
    extrapolated.name = y
    return extrapolated
