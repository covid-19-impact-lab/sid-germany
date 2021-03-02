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
        "share_immune": BLD / "data" / "processed_time_series" / "share_immune.pkl",
        "fig": BLD / "data" / "processed_time_series" / "share_immune.png",
    }
)
def task_prepare_vaccination_data(depends_on, produces):
    df = pd.read_excel(depends_on, sheet_name="Impfungen_proTag")
    df = _clean_vaccination_data(df)
    share_immune = _calculate_share_immune_from_vaccination(df)
    share_immune.to_pickle(produces["share_immune"])
    fig, ax = _plot_share_immune(share_immune)
    fig.savefig(produces["fig"], dpi=200, transparent=False, facecolor="w")


def _clean_vaccination_data(df):
    # drop last two rows (empty and total vaccinations)
    df = df[:-2]
    df["date"] = pd.to_datetime(df["Datum"], format="%m/%d/%yyyy")
    # check date conversion was correct
    assert df["date"].min() == pd.Timestamp(year=2020, month=12, day=27)
    df = df.set_index("date")
    df["received_first_dose"] = df["Erstimpfung"].cumsum()
    df["share_with_first_dose"] = df["received_first_dose"] / POPULATION_GERMANY
    return df


def _calculate_share_immune_from_vaccination(df):
    """Calculate the share of individuals immune from vaccination.

    We ignore booster shots and simply assume immunity will start
    21 days after the first shot (Hunter2021) with 75% probability
    (90% for Pfizer but AstraZeneca is lower and protection against
    mutations is lower)

    """
    share_immune = 0.75 * df["share_with_first_dose"]
    share_immune.index = share_immune.index + pd.Timedelta(weeks=3)
    return share_immune


def _plot_share_immune(sr):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(x=sr.index, y=sr)
    ax.set_title("Share Immune Through Vaccination Over Time")
    fig, ax = style_plot(fig, ax)
    fig.tight_layout()
    return fig, ax
