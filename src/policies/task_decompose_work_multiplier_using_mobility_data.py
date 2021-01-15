""""""
import matplotlib.pyplot as plt
import pandas as pd
import pytask
import seaborn as sns
from matplotlib.dates import AutoDateLocator
from matplotlib.dates import DateFormatter
from sid.colors import get_colors

from src.config import BLD
from src.config import SRC

plt.rcParams.update(
    {
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
    }
)

sns.set_palette(get_colors("categorical", 12))


@pytask.mark.depends_on(
    {"data": SRC / "original_data" / "google_mobility_2021-01-08_DE.csv"}
)
@pytask.mark.produces(
    {
        "by_state": BLD / "policies" / "work_mobility_reduction_by_state.png",
        "de": BLD / "policies" / "work_mobility_reduction_aggregated.png",
        "de_weekdays": BLD / "policies" / "work_mobility_reduction_only_weekdays.png",
    }
)
def task_decompose_work_multiplier(depends_on, produces):
    df = pd.read_csv(depends_on["data"])
    df = _prepare_mobility_data(df)
    fig, ax = _visualize_reductions_by_state(df)
    fig.savefig(produces["by_state"], dpi=200, transparent=False, facecolor="w")

    # only whole of Germany
    df = df[df["sub_region_1"].isnull()]
    df["workplaces_smoothed"] = df["workplaces"].rolling(window=7).mean()
    assert not df["date"].duplicated().any()
    fig, ax = _plot_time_series(df, y="workplaces", x="date")
    fig.savefig(produces["de"], dpi=200, transparent=False, facecolor="w")

    work_days = df[~df["date"].dt.day_name().isin(["Saturday", "Sunday"])]
    fig, ax = _plot_time_series(work_days)
    fig.savefig(produces["de_weekdays"], dpi=200, transparent=False, facecolor="w")


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


def decompose_work_multiplier(work_multiplier, start_date, end_date, google_data):
    """Decompose the work multiplier into participation and hygiene multipliers.

    Derivation:
        The work_multiplier is implemented as the share of non-systemically
        relevant workers that go to work and have (full risk) work contacts.
        Who goes to work is independent of how many work contacts someone has.

        => the work_multiplier can be interpreted as the share of (full risk)
           contacts that take place among the non-systemically relevant workers.

        => share_risk_contacts_still_happening = 0.33 + 0.66 * work_multiplier

        Another way to look at it is:

            share_risk_contacts_still_happening =
                participation_multiplier * hygiene_multiplier

        Combining the two ways of writing this, we get:

        0.33 + 0.66 * work_multiplier = participation_multiplier * hygiene_multiplier

        <=>  hygiene_multiplier = (0.33 + 0.66 * work_multiplier) /
                                   participation_multiplier

        the participation_multiplier can be proxied by the reduction in workplace
        mobility in the google mobility data from the respective time period.

    Args:
        work_multiplier (float): the work multiplier
        start_date (str): start date of the period
        end_date (str): end date of the period
        google_data (pandas.DataFrame): Data with the google mobility
            reductions. The index are the dates, weekends removed.

    Returns:
        participation_multiplier (float)
        hygiene_multiplier (float)

    """
    data = google_data.loc[start_date:end_date]
    participation_multiplier = 1 - data["workplaces"].mean() / 100
    hygiene_multiplier = (0.33 + 0.66 * work_multiplier) / participation_multiplier
    return participation_multiplier, hygiene_multiplier


def _visualize_reductions_by_state(df):
    states = ["Bavaria", "North Rhine-Westphalia", "Saxony", "Mecklenburg-Vorpommern"]
    subset = df[
        df["sub_region_1"].isin(states)
        & ~df["date"].dt.day_name().isin(["Saturday", "Sunday"])
    ]
    fig, ax = _plot_time_series(data=subset, hue="sub_region_1")
    title = "Reduction in Workplace Mobility Acc. to Google Mobility Data by State"
    ax.set_title(title)
    return fig, ax


def _plot_time_series(data, y="workplaces", x="date", hue=None, fig=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 3))
    sns.lineplot(data=data, y=y, x=x, hue=hue, ax=ax)
    ax.axhline(0, color="k", linewidth=1)
    date_form = DateFormatter("%m/%Y")
    ax.xaxis.set_major_formatter(date_form)
    fig.autofmt_xdate()
    loc = AutoDateLocator(minticks=5, maxticks=12)
    ax.xaxis.set_major_locator(loc)
    return fig, ax
