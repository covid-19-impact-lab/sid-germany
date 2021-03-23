import matplotlib.pyplot as plt
import pandas as pd
import pytask
import seaborn as sns

from src.config import BLD
from src.config import POPULATION_GERMANY
from src.simulation.plotting import style_plot


plt.rcParams.update(
    {
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
    }
)


@pytask.mark.depends_on(BLD / "data" / "raw_time_series" / "vaccinations.xlsx")
@pytask.mark.produces(
    {
        "fig_first_dose": BLD / "data" / "processed_time_series" / "first_dose.png",
        "share_becoming_immune": BLD
        / "data"
        / "processed_time_series"
        / "share_becoming_immune.pkl",
        "fig_share_becoming_immune": BLD
        / "data"
        / "processed_time_series"
        / "share_becoming_immune.png",
    }
)
def task_prepare_vaccination_data(depends_on, produces):
    df = pd.read_excel(depends_on, sheet_name="Impfungen_proTag")
    df = _clean_vaccination_data(df)
    # this is for comparing with newspaper sites
    fig, ax = _plot_series(df["share_with_first_dose"], "Share with 1st Dose")
    fig.savefig(produces["fig_first_dose"], dpi=200, transparent=False, facecolor="w")

    share_becoming_immune = _calculate_share_receiving_immunization_from_vaccination(df)
    share_becoming_immune.to_pickle(produces["share_becoming_immune"])
    fig, ax = _plot_series(
        share_becoming_immune, "Share Becoming Immune Through Vaccination"
    )
    fig.savefig(
        produces["fig_share_becoming_immune"], dpi=200, transparent=False, facecolor="w"
    )


def _clean_vaccination_data(df):
    # drop last two rows (empty and total vaccinations)
    df = df[df["Datum"].isnull().cumsum() == 0].copy(deep=True)
    df["date"] = pd.to_datetime(df["Datum"], format="%m/%d/%yyyy")
    # check date conversion was correct
    assert df["date"].min() == pd.Timestamp(year=2020, month=12, day=27)
    df = df.set_index("date")
    df["received_first_dose"] = df["Erstimpfung"].cumsum()
    df["share_with_first_dose"] = df["received_first_dose"] / POPULATION_GERMANY
    return df


def _calculate_share_receiving_immunization_from_vaccination(df):
    """Calculate the share of individuals becoming immune from vaccination.

    We ignore booster shots and simply assume immunity will start
    21 days after the first shot (Hunter2021) with 75% probability
    (90% for Pfizer but AstraZeneca is lower and protection against
    mutations is lower)

    """
    share_immune = 0.75 * df["share_with_first_dose"]
    share_immune.index = share_immune.index + pd.Timedelta(weeks=3)
    share_becoming_immune = share_immune.diff().dropna()
    return share_becoming_immune


def _plot_series(sr, title):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(x=sr.index, y=sr)
    ax.set_title(title)
    fig, ax = style_plot(fig, ax)
    fig.tight_layout()
    return fig, ax
