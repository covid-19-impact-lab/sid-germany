import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytask
import seaborn as sns
import yaml
from sid.colors import get_colors

from src.config import BLD
from src.config import PLOT_END_DATE
from src.config import PLOT_SIZE
from src.config import PLOT_START_DATE
from src.config import POPULATION_GERMANY
from src.config import SRC
from src.plotting.plotting import style_plot


plt.rcParams.update(
    {
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
    }
)


_BY_AGE_DEPENDENCIES = {
    "june": SRC / "original_data" / "vaccinations_by_age_group" / "2021-06-14.xlsx",
    "july": SRC / "original_data" / "vaccinations_by_age_group" / "2021-07-14.xlsx",
    "aug1": SRC / "original_data" / "vaccinations_by_age_group" / "2021-08-01.xlsx",
    "aug2": SRC / "original_data" / "vaccinations_by_age_group" / "2021-08-12.xlsx",
    "aug3": SRC / "original_data" / "vaccinations_by_age_group" / "2021-08-23.xlsx",
}


@pytask.mark.depends_on(_BY_AGE_DEPENDENCIES)
@pytask.mark.produces(
    {
        "data": BLD / "data" / "vaccinations" / "vaccinations_by_age_group.pkl",
        "fig": BLD / "figures" / "data" / "vaccinations_by_age_group.pdf",
    }
)
def task_create_vaccination_rates_by_age_group(depends_on, produces):
    june = pd.read_excel(
        depends_on["june"], sheet_name="Impfquote_bis_einschl_14.06.21", header=[0, 2]
    )
    # no data at the federal level yet, so take the unweighted mean over the
    # states for which data is available
    june = june["Impfquote mindestens einmal geimpft *"].replace("-", np.nan).mean()
    june = june.rename(index={"<18 Jahre": "12-17 Jahre"})
    june.name = pd.Timestamp("2021-06-14")

    july = pd.read_excel(
        depends_on["july"], sheet_name="Impfquote_bis_einschl_14.07.21", header=[0, 2]
    )
    july = july.loc[17, "Impfquote mindestens einmal geimpft *"]
    july = july.rename(index={"<18 Jahre": "12-17 Jahre"})
    july.name = pd.Timestamp("2021-07-14")

    aug1 = pd.read_excel(
        depends_on["aug1"], sheet_name="Impfquote_bis_einschl_01.08.21", header=[0, 2]
    )
    aug1 = aug1.loc[17, "Impfquote mindestens einmal geimpft "]
    aug1.index = [x.replace("*", "") for x in aug1.index]
    aug1.name = pd.Timestamp("2021-08-01")

    aug2 = pd.read_excel(
        depends_on["aug2"], sheet_name="Impfquote_bis_einschl_12.08.21", header=[0, 2]
    )
    aug2 = aug2.loc[17, "Impfquote mindestens einmal geimpft "]
    aug2.index = [x.replace("*", "") for x in aug2.index]
    aug2.name = pd.Timestamp("2021-08-12")

    aug3 = pd.read_excel(
        depends_on["aug3"], sheet_name="Impfquote_bis_einschl_23.08.21", header=[0, 2]
    )
    aug3 = aug3.loc[17, "Impfquote mindestens einmal geimpft "]
    aug3.index = [x.replace("*", "") for x in aug3.index]
    aug3.name = pd.Timestamp("2021-08-23")

    vacc_shares_by_age = pd.concat([june, july, aug1, aug2, aug3], axis=1) / 100
    vacc_shares_by_age = vacc_shares_by_age.rename(
        index={
            "Gesamt": "overall",
            "12-17 Jahre": "12-17",
            "18-59 Jahre": "18-59",
            "60+ Jahre": ">=60",
        }
    )
    vacc_shares_by_age.to_pickle(produces["data"])

    fig, ax = plt.subplots(figsize=PLOT_SIZE)
    vacc_shares_by_age.T.plot(ax=ax)
    fig, ax = style_plot(fig, ax)
    fig.savefig(produces["fig"])


@pytask.mark.depends_on(
    {
        "data": BLD / "data" / "raw_time_series" / "vaccinations.xlsx",
    }
)
@pytask.mark.produces(
    {
        "vaccination_shares_raw": BLD
        / "data"
        / "vaccinations"
        / "vaccination_shares_raw.pkl",
        "vaccination_shares_extended": BLD
        / "data"
        / "vaccinations"
        / "vaccination_shares_extended.pkl",
        "fig_first_dose": BLD
        / "figures"
        / "data"
        / "share_of_individuals_with_first_vaccine.pdf",
        "fig_vaccination_shares": BLD
        / "figures"
        / "data"
        / "share_receiving_vaccination_per_day.pdf",
        "mean_vacc_share_per_day": BLD
        / "data"
        / "vaccinations"
        / "mean_vacc_share_per_day.yaml",
    }
)
def task_prepare_vaccination_data(depends_on, produces):
    df = pd.read_excel(depends_on["data"], sheet_name="Impfungen_proTag")
    df = _clean_vaccination_data(df)
    # this is for comparing with newspaper sites
    fig, ax = _plot_series(df["share_with_first_dose"], "")
    ax.set_xlim(pd.Timestamp(PLOT_START_DATE), pd.Timestamp(PLOT_END_DATE))
    fig.savefig(produces["fig_first_dose"])
    plt.close()

    vaccination_shares = df["share_with_first_dose"].diff().dropna()

    # the first individuals to be vaccinated were nursing homes which are not
    # in our synthetic data so we exclude the first 1% of vaccinations to
    # be going to them.
    vaccination_shares[vaccination_shares.cumsum() <= 0.01] = 0

    vaccination_shares.to_pickle(produces["vaccination_shares_raw"])

    # family physicians started vaccinating on April 6th (Tue after Easter)
    start_physicians = pd.Timestamp("2021-04-06")

    # for reproduction of vaccination scenarios save the weekday means
    easter_until_july = vaccination_shares.loc[start_physicians:"2021-07-06"]
    dayname_to_mean = easter_until_july.groupby(
        easter_until_july.index.day_name()
    ).mean()
    with open(produces["mean_vacc_share_per_day"], "w") as f:
        yaml.dump(data=dayname_to_mean.to_dict(), stream=f)

    # extrapolate into the future
    last_month_avg = vaccination_shares[-28:].mean()

    end_date = vaccination_shares.index.max() + pd.Timedelta(days=56)
    extended_dates = pd.date_range("2020-03-01", end_date)
    extended = vaccination_shares.copy(deep=True).reindex(extended_dates)
    extended[: vaccination_shares.index.min()] = 0.0
    extrapolated = extended.fillna(last_month_avg)

    labeled = [
        ("raw data", vaccination_shares),
        ("extrapolated", extrapolated),
    ]
    fig, ax = _plot_labeled_series(labeled)
    ax.axvline(
        start_physicians,
        label="Start of family physicians receiving Covid vaccines",
        color="forestgreen",
    )
    plt.legend()

    fig.savefig(produces["fig_vaccination_shares"])
    plt.close()

    _test_extrapolated(extrapolated)
    extrapolated.to_pickle(produces["vaccination_shares_extended"])


def _clean_vaccination_data(df):
    # drop rows below the last date
    first_non_date_loc = df[df["Datum"] == "Gesamt"].index[0]
    df = df.loc[: first_non_date_loc - 1].copy(deep=True)

    df["date"] = pd.to_datetime(df["Datum"], dayfirst=True)

    # check date conversion was correct
    assert df["date"].min() == pd.Timestamp(year=2020, month=12, day=27)
    assert df["date"].max() < pd.Timestamp(year=2021, month=12, day=31)

    # sort_index is super important here because of the cumsum below!
    df = df.set_index("date").sort_index()

    try:
        df["received_first_dose"] = df["mindestens einmal geimpft"].cumsum()
    except KeyError:
        df["received_first_dose"] = df["Erstimpfung"].cumsum()
    df["share_with_first_dose"] = df["received_first_dose"] / POPULATION_GERMANY

    assert df["share_with_first_dose"].sort_index().is_monotonic_increasing
    assert (df.loc["2021-05-01":, "received_first_dose"] > 0.25).all()

    return df


def _plot_series(sr, title, label=None):
    fig, ax = plt.subplots(figsize=PLOT_SIZE)
    sr = sr.loc[PLOT_START_DATE:PLOT_END_DATE]
    sns.lineplot(x=sr.index, y=sr, label=label)
    ax.set_title(title)
    fig, ax = style_plot(fig, ax)
    fig.tight_layout()
    return fig, ax


def _plot_labeled_series(labeled):
    title = "Actual and Extrapolated Share Receiving the Vaccination"
    fig, ax = plt.subplots(figsize=PLOT_SIZE)
    colors = get_colors("categorical", len(labeled))
    for (label, sr), color in zip(labeled, colors):
        sns.lineplot(
            x=sr.loc[PLOT_START_DATE:PLOT_END_DATE].index,
            y=sr.loc[PLOT_START_DATE:PLOT_END_DATE],
            label=label,
            linewidth=2,
            color=color,
        )
    fig, ax = style_plot(fig, ax)
    ax.set_title(title)
    ax.set_ylabel("")
    fig.tight_layout()
    return fig, ax


def _test_extrapolated(sr):
    assert sr.index.is_monotonic, "index is not monotonic."
    assert not sr.index.duplicated().any(), "Duplicate dates in Series."
    assert (sr.index == pd.date_range(start=sr.index.min(), end=sr.index.max())).all()
