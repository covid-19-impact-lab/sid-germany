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
from src.contact_models.get_contact_models import get_all_contact_models
from src.policies.combine_policies_over_periods import get_october_to_christmas_policies


plt.rcParams.update(
    {
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
    }
)

sns.set_palette(get_colors("categorical", 12))


@pytask.mark.depends_on(
    {
        "data": SRC / "original_data" / "google_mobility_2021-01-21_DE.csv",
        "policy_py": SRC / "policies" / "combine_policies_over_periods.py",
        "contacts_py": SRC / "contact_models" / "get_contact_models.py",
    }
)
@pytask.mark.produces(
    {
        "by_state": BLD / "policies" / "work_mobility_reduction_by_state.png",
        "de": BLD / "policies" / "work_mobility_reduction_aggregated.png",
        "de_weekdays": BLD / "policies" / "work_mobility_reduction_only_weekdays.png",
        "de_weekdays_since_dec": BLD
        / "policies"
        / "work_mobility_reduction_only_weekdays_since_dec.png",
        "decomposition_table": BLD / "policies" / "decomposition_table.csv",
        "work_days": BLD / "policies" / "google_workday_data.csv",
    }
)
def task_decompose_work_multiplier(depends_on, produces):
    df = pd.read_csv(depends_on["data"])
    df = _prepare_mobility_data(df)
    fig, ax = _visualize_reductions_by_state(df)
    fig.savefig(produces["by_state"], dpi=200, transparent=False, facecolor="w")

    de_df = df[df["sub_region_1"].isnull()].copy()  # only whole of Germany
    de_df["share_working"] = 1 + (de_df["workplaces"] / 100)
    assert not de_df["date"].duplicated().any()
    fig, ax = _plot_time_series(de_df, title="Share Going to Work")
    fig.savefig(produces["de"], dpi=200, transparent=False, facecolor="w")

    work_days = de_df[~de_df["date"].dt.day_name().isin(["Saturday", "Sunday"])]
    work_days.to_csv(produces["work_days"])
    fig, ax = _plot_time_series(work_days, title="Share Going to Work (Workdays Only)")
    fig.savefig(produces["de_weekdays"], dpi=200, transparent=False, facecolor="w")

    fig, ax = _plot_time_series(
        work_days[work_days["date"] > "2020-11-30"],
        title="Share Working (Workdays Only)",
    )
    ax.axvline(pd.Timestamp("2020-12-27"))
    ax.axvline(pd.Timestamp("2021-01-04"))
    ax.axvline(pd.Timestamp("2021-01-12"))
    fig.savefig(
        produces["de_weekdays_since_dec"], dpi=200, transparent=False, facecolor="w"
    )

    contact_models = get_all_contact_models(None, None)
    policies = get_october_to_christmas_policies(contact_models=contact_models)
    google_data = work_days.set_index("date")

    decomposition = _build_decomposition_table(
        policies=policies, google_data=google_data
    )
    decomposition.to_csv(produces["decomposition_table"])


def _build_decomposition_table(policies, google_data):
    cols = [
        "period",
        "contact_type",
        "work_multiplier",
        "participation_multiplier",
        "hygiene_multiplier",
    ]
    df = pd.DataFrame(columns=cols).set_index(["period", "contact_type"])

    for name, policy_block in policies.items():
        if "work" in name:
            kwargs = policy_block["policy"].keywords
            if "multiplier" in kwargs:
                (
                    participation_multiplier,
                    hygiene_multiplier,
                ) = decompose_work_multiplier(
                    google_data=google_data,
                    work_multiplier=kwargs["multiplier"],
                    start_date=policy_block["start"],
                    end_date=policy_block["end"],
                )
                row = pd.Series(
                    {
                        "work_multiplier": kwargs["multiplier"],
                        "participation_multiplier": participation_multiplier,
                        "hygiene_multiplier": hygiene_multiplier,
                    }
                )
                df.loc[tuple(name.split("_work_"))] = row
    return df


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

        => the work_multiplier is approximately the share of (full risk)
           contacts that take place among workers. Only approximately because
           the work multiplier selects individuals who stop / continue to go
           to work where they have "full risk" contacts. If the hygiene multiplier
           is close to 1 the difference between the extensive and intensive margin
           should be negligible.

        Another way to look at it is:

            share_risk_contacts_still_happening =
                participation_multiplier * hygiene_multiplier

        Combining the two ways of writing this, we get:

        work_multiplier = participation_multiplier * hygiene_multiplier

        <=>  hygiene_multiplier = work_multiplier / participation_multiplier

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
    mobility_increase_to_baseline = data["workplaces"].mean() / 100
    participation_multiplier = 1 + mobility_increase_to_baseline
    hygiene_multiplier = work_multiplier / participation_multiplier
    return participation_multiplier, hygiene_multiplier


def _visualize_reductions_by_state(df):
    states = ["Bavaria", "North Rhine-Westphalia", "Saxony", "Mecklenburg-Vorpommern"]
    subset = df[
        df["sub_region_1"].isin(states)
        & ~df["date"].dt.day_name().isin(["Saturday", "Sunday"])
    ]
    fig, ax = _plot_time_series(data=subset, y="workplaces", hue="sub_region_1")
    title = "Reduction in Workplace Mobility Acc. to Google Mobility Data by State"
    ax.set_title(title)
    return fig, ax


def _plot_time_series(
    data,
    y="share_working",
    x="date",
    title="",
    hue=None,
    fig=None,
    ax=None,
):
    data = data.copy()
    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 5))

    sns.lineplot(data=data, y=y, x=x, hue=hue, ax=ax)
    date_form = (
        DateFormatter("%d/%m/%Y") if len(data) < 100 else DateFormatter("%d/%m/%Y")
    )
    ax.xaxis.set_major_formatter(date_form)
    fig.autofmt_xdate()
    loc = AutoDateLocator(minticks=5, maxticks=12)
    ax.xaxis.set_major_locator(loc)
    ax.grid(axis="y")
    ax.set_title(title)
    return fig, ax
