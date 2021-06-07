import matplotlib.pyplot as plt
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
from src.plotting.plotting import style_plot


plt.rcParams.update(
    {
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
    }
)


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
    fig, ax = _plot_series(df["share_with_first_dose"], "Share with 1st Dose")
    fig.savefig(produces["fig_first_dose"])
    plt.close()

    vaccination_shares = df["share_with_first_dose"].diff().dropna()
    vaccination_shares.to_pickle(produces["vaccination_shares_raw"])

    # extend data to 2020.
    backward_dates = pd.date_range("2020-01-01", vaccination_shares.index.max())
    vaccination_shares = vaccination_shares.reindex(backward_dates)
    vaccination_shares = vaccination_shares.fillna(0)

    # the first individuals to be vaccinated were nursing homes which are not
    # in our synthetic data so we exclude the first 1% of vaccinations to
    # be going to them.
    vaccination_shares[vaccination_shares.cumsum() <= 0.01] = 0

    # family physicians started vaccinating on April 6th (Tue after Easter)
    # we assume that the number of vaccinations is constant to the weekday's
    # mean when extrapolating into the future.
    start_physicians = pd.Timestamp("2021-04-06")
    after_start = vaccination_shares.loc[start_physicians:]

    dayname_to_mean = after_start.groupby(after_start.index.day_name()).mean()
    with open(produces["mean_vacc_share_per_day"], "w") as f:
        yaml.dump(data=dayname_to_mean.to_dict(), stream=f)

    start_date = vaccination_shares.index.max() + pd.Timedelta(days=1)
    end_date = start_date + pd.Timedelta(weeks=12)
    future_dates = pd.date_range(start_date, end_date)
    future_day_names = future_dates.day_name()
    future_values = future_day_names.to_series().replace(dayname_to_mean)
    extension = pd.Series(future_values.values, index=future_dates)

    labeled = [
        ("raw data", vaccination_shares),
        ("extension", extension),
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

    extended = pd.concat([vaccination_shares, extension])
    _test_extended(extended)
    extended.to_pickle(produces["vaccination_shares_extended"])


def _clean_vaccination_data(df):
    # drop last two rows (empty and total vaccinations)
    df = df[df["Datum"].isnull().cumsum() == 0].copy(deep=True)
    df["date"] = pd.to_datetime(df["Datum"], format="%m/%d/%yyyy")
    # check date conversion was correct
    assert df["date"].min() == pd.Timestamp(year=2020, month=12, day=27)
    df = df.set_index("date")
    df["received_first_dose"] = df["Begonnene Impfserie"].cumsum()
    df["share_with_first_dose"] = df["received_first_dose"] / POPULATION_GERMANY
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


def _test_extended(sr):
    assert sr.index.is_monotonic, "index is not monotonic."
    assert not sr.index.duplicated().any(), "Duplicate dates in Series."
    assert (sr.index == pd.date_range(start=sr.index.min(), end=sr.index.max())).all()
