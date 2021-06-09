import matplotlib.pyplot as plt
import pandas as pd
import pytask
import seaborn as sns

from src.config import BLD
from src.config import PLOT_END_DATE
from src.config import PLOT_SIZE
from src.config import PLOT_START_DATE
from src.config import SRC
from src.plotting.plotting import ORANGE
from src.plotting.plotting import PURPLE
from src.plotting.plotting import style_plot
from src.plotting.plotting import YELLOW
from src.policies.enacted_policies import HYGIENE_MULTIPLIER
from src.policies.enacted_policies import OTHER_MULTIPLIER_SPECS
from src.simulation.seasonality import create_seasonality_series
from src.testing.shared import get_piecewise_linear_interpolation


_DEPENDENCIES = {
    # modules
    "plotting.py": SRC / "plotting" / "plotting.py",
    "testing_shared.py": SRC / "testing" / "shared.py",
    # data
    "enacted_policies.py": SRC / "policies" / "enacted_policies.py",
    "stringency_data": BLD / "data" / "raw_time_series" / "stringency_data.csv",
    "work_multiplier": BLD / "policies" / "work_multiplier.csv",
    "params": BLD / "params.pkl",
}

_PRODUCTS = {
    "school_comparison": BLD / "figures" / "data" / "school_multiplier_comparison.pdf",
    "0_no_seasonality": BLD / "figures" / "data" / "stringency_no_seasonality.pdf",
    "1_no_seasonality": BLD / "figures" / "data" / "stringency2_no_seasonality.pdf",
    "0_with_seasonality": BLD / "figures" / "data" / "stringency_with_seasonality.pdf",
    "1_with_seasonality": BLD / "figures" / "data" / "stringency2_with_seasonality.pdf",
    "data": BLD / "tables" / "multiplier_data.csv",
}


@pytask.mark.depends_on(_DEPENDENCIES)
@pytask.mark.produces(_PRODUCTS)
def task_plot_multipliers_and_stringency_index(depends_on, produces):
    home_office_share = pd.read_csv(
        depends_on["work_multiplier"], parse_dates=["date"], index_col="date"
    )
    stringency_data = pd.read_csv(
        depends_on["stringency_data"], low_memory=False, parse_dates=["Date"]
    )
    params = pd.read_pickle(depends_on["params"])

    stringency, doubled_stringency = _prepare_stringency(
        stringency_data, PLOT_START_DATE, PLOT_END_DATE
    )

    work_multiplier = home_office_share.loc[PLOT_START_DATE:PLOT_END_DATE, "Germany"]
    work_multiplier["2020-11-01":] = HYGIENE_MULTIPLIER * work_multiplier["2020-11-01":]
    scaled_work_multiplier = work_multiplier / work_multiplier[0]
    other_multiplier = _get_other_multiplier(
        PLOT_START_DATE, PLOT_END_DATE, OTHER_MULTIPLIER_SPECS
    )
    scaled_other_multiplier = other_multiplier / other_multiplier[0]
    (
        school_multiplier_without_vacations,
        school_multiplier_with_vacations,
    ) = _get_school_multipliers(PLOT_START_DATE)
    school_multiplier_without_vacations = school_multiplier_without_vacations[
        PLOT_START_DATE:PLOT_END_DATE
    ]
    school_multiplier_with_vacations = school_multiplier_with_vacations[
        PLOT_START_DATE:PLOT_END_DATE
    ]

    fig, ax = plt.subplots(figsize=PLOT_SIZE)
    for sr, label in [
        (school_multiplier_with_vacations, "with vacations"),
        (school_multiplier_without_vacations, "without vacations"),
    ]:
        sns.lineplot(x=sr.index, y=sr, label=label, alpha=0.6, linewidth=4)
    fig, ax = style_plot(fig, ax)
    fig.tight_layout()
    fig.savefig(produces["school_comparison"])

    our_stringency = pd.concat(
        [work_multiplier, other_multiplier, school_multiplier_with_vacations], axis=1
    ).mean(axis=1)

    weak_seasonality, mean_seasonality, strong_seasonality = _get_seasonalities(
        params, PLOT_START_DATE, PLOT_END_DATE
    )

    for i, oxford_stringency in enumerate([stringency, doubled_stringency]):
        fig = _create_multiplier_plot(
            oxford_stringency=oxford_stringency,
            work_multiplier=scaled_work_multiplier,
            other_multiplier=scaled_other_multiplier,
            school_multiplier=school_multiplier_with_vacations,
            our_stringency=our_stringency,
        )
        fig.savefig(produces[f"{i}_no_seasonality"])

    work_multiplier_seasonal = scaled_work_multiplier * weak_seasonality
    work_multiplier_seasonal = work_multiplier_seasonal / work_multiplier_seasonal[0]
    other_multiplier_seasonal = scaled_other_multiplier * strong_seasonality
    other_multiplier_seasonal = other_multiplier_seasonal / other_multiplier_seasonal[0]
    school_multiplier_seasonal = school_multiplier_with_vacations * weak_seasonality
    school_multiplier_seasonal = (
        school_multiplier_seasonal / school_multiplier_seasonal[0]
    )
    our_stringency_seasonal = our_stringency * mean_seasonality
    our_stringency_seasonal = our_stringency_seasonal / our_stringency_seasonal[0]

    for i, oxford_stringency in enumerate([stringency, doubled_stringency]):
        fig = _create_multiplier_plot(
            oxford_stringency=oxford_stringency,
            work_multiplier=work_multiplier_seasonal,
            other_multiplier=other_multiplier_seasonal,
            school_multiplier=school_multiplier_seasonal,
            our_stringency=our_stringency_seasonal,
        )
        fig.savefig(produces[f"{i}_with_seasonality"])
    plt.close()

    df = pd.DataFrame(
        {
            # seasonal
            "work_multiplier_seasonal": work_multiplier_seasonal,
            "work_multiplier_seasonal": work_multiplier_seasonal,
            "other_multiplier_seasonal": other_multiplier_seasonal,
            "other_multiplier_seasonal": other_multiplier_seasonal,
            "school_multiplier_seasonal": school_multiplier_seasonal,
            "school_multiplier_seasonal": school_multiplier_seasonal,
            # oxford stringency indices
            "stringency": stringency,
            "doubled_stringency": doubled_stringency,
            # multipliers without seasonality
            "work_multiplier": scaled_work_multiplier,
            "other_multiplier": scaled_other_multiplier,
            "school_multiplier": school_multiplier_with_vacations,
            # for completeness
            "school_multiplier_without_vacations": school_multiplier_without_vacations,
            "our_mean_stringency": our_stringency,
            "our_mean_stringency_seasonal": our_stringency_seasonal,
        }
    )
    df.round(3).to_csv(produces["data"])


def _prepare_stringency(df, start_date, end_date):
    """Prepare the Oxford stringency data.

    Documentation of the data can be found at https://bit.ly/3cgBwOQ
    The citation is Hale2020.

    """
    df = df.query("CountryName == 'Germany'")
    df = df.set_index("Date")
    df = df[start_date:end_date]
    stringency = 1 - df["StringencyIndex"] / 100
    doubled_stringency = 2 * stringency
    return stringency, doubled_stringency


def _get_other_multiplier(start_date, end_date, multiplier_spec):
    date_range = pd.date_range(start_date, end_date)
    sr = pd.Series(index=date_range, dtype=float)
    for _, end_date, multiplier in multiplier_spec:
        sr[start_date:] = multiplier
        start_date = end_date
    return sr


def _get_school_multipliers(start_date):
    share_age_for_emergency_care = 0.5
    share_in_graduating_classes = 0.25  # 16, 17 and 18 year olds
    share_in_primary = 0.3
    a_b_multiplier = 0.5

    share_getting_strict_emergency_care = 0.2
    share_getting_generous_emergency_care = 0.3

    strict_emergency_care_multiplier = (
        share_age_for_emergency_care
        * share_getting_strict_emergency_care
        * HYGIENE_MULTIPLIER
    )

    generous_emergency_care_multiplier = (
        share_in_graduating_classes * a_b_multiplier
        + share_age_for_emergency_care * share_getting_generous_emergency_care
    ) * HYGIENE_MULTIPLIER

    feb_to_march_a_b_share = (
        share_in_primary + share_in_graduating_classes
    ) * a_b_multiplier
    feb_to_march_emergency_share = (
        share_age_for_emergency_care
        * share_getting_generous_emergency_care
        * a_b_multiplier
    )
    feb_to_march_multiplier = (
        feb_to_march_a_b_share + feb_to_march_emergency_share
    ) * HYGIENE_MULTIPLIER

    a_b_for_most_multiplier = (
        a_b_multiplier + feb_to_march_emergency_share * a_b_multiplier
    )

    school_multiplier_without_vacations = pd.Series(
        {
            start_date: 1.0,
            "2020-11-01": 1.0,
            "2020-11-02": HYGIENE_MULTIPLIER,
            "2020-12-15": HYGIENE_MULTIPLIER,
            # strict emergency care
            "2020-12-16": strict_emergency_care_multiplier,
            "2021-01-10": strict_emergency_care_multiplier,
            # generous emergency care
            "2021-01-11": generous_emergency_care_multiplier,
            "2021-02-21": generous_emergency_care_multiplier,
            # primary and graduating in A / B
            "2021-02-22": feb_to_march_multiplier,
            "2021-03-14": feb_to_march_multiplier,
            # mid March until Easter: A / B for most
            "2021-03-15": a_b_for_most_multiplier,
            "2021-04-05": a_b_for_most_multiplier,
            # easter until may:
            "2021-04-06": generous_emergency_care_multiplier,
            "2021-04-30": generous_emergency_care_multiplier,
            # may
            "2021-05-01": a_b_for_most_multiplier,
            "2021-05-31": a_b_for_most_multiplier,
        }
    )

    school_multiplier_without_vacations = get_piecewise_linear_interpolation(
        school_multiplier_without_vacations
    )

    school_multiplier_with_vacations = pd.Series(
        {
            start_date: 1.0,
            # fall vacation:
            # first start on 2020-10-05. last end 2020-11-06
            # on average from 2020-10-13 to 2020-10-23
            # there was no overlap in the fall vacation dates between states.
            "2020-10-05": 1.0,
            "2020-10-13": 0.3,  # number to cover that many states had fall vacation
            "2020-10-23": 0.3,
            "2020-11-06": HYGIENE_MULTIPLIER,
            # strict emergency care started in the week before Christmas
            "2020-12-15": HYGIENE_MULTIPLIER,
            "2020-12-16": strict_emergency_care_multiplier,
            # Christmas vacations started on 2002-12-21 in most states.
            # Christmas vacations ended between 2021-01-02 and 2021-01-10.
            "2020-12-20": strict_emergency_care_multiplier,
            "2020-12-21": 0.0,
            "2021-01-02": 0.0,
            "2021-01-10": strict_emergency_care_multiplier,
            # generous emergency care
            # winter vacations were from 2021-01-25 until 2021-03-12 depending on the
            # state and only short so we ignore them here.
            "2021-01-11": generous_emergency_care_multiplier,
            "2021-02-21": generous_emergency_care_multiplier,
            # primary and graduating in A / B
            "2021-02-22": feb_to_march_multiplier,
            "2021-03-14": feb_to_march_multiplier,
            # mid March until Easter: A / B for most
            "2021-03-15": a_b_for_most_multiplier,
            "2021-03-26": a_b_for_most_multiplier,
            # Easter vacations started on 2021-03-29 in most states
            # and ended between 2021-04-05 and 2021-04-16, for most on 2021-04-10
            "2021-03-29": 0.0,
            "2021-04-06": 0.0,
            "2021-04-10": 0.1,  # some states started school before 2021-04-10
            # easter until may:
            "2021-04-11": generous_emergency_care_multiplier,
            "2021-04-30": generous_emergency_care_multiplier,
            # may
            # we ignore Pentecoast vacation because it was <4 days on average
            "2021-05-01": a_b_for_most_multiplier,
            "2021-05-31": a_b_for_most_multiplier,
        }
    )
    school_multiplier_with_vacations = get_piecewise_linear_interpolation(
        school_multiplier_with_vacations
    )

    return school_multiplier_without_vacations, school_multiplier_with_vacations


def _create_multiplier_plot(
    oxford_stringency,
    work_multiplier,
    other_multiplier,
    school_multiplier,
    our_stringency,  # noqa: U100
):
    fig, ax = plt.subplots(figsize=PLOT_SIZE)

    named_lines = [
        # (our_stringency, "mean of our multiplier", 1.0, 4, RED),  # noqa: E800
        (oxford_stringency, "rescaled Oxford stringency index", 1.0, 3, "k"),
        (work_multiplier, "Work", 0.8, 3, PURPLE),
        (school_multiplier, "School", 0.8, 3, ORANGE),
        (other_multiplier, "Other", 0.8, 3, YELLOW),
    ]

    for multiplier, label, alpha, linewidth, color in named_lines:
        sns.lineplot(
            x=multiplier.index,
            y=multiplier.rolling(7).mean(),
            label=label,
            linewidth=linewidth,
            alpha=alpha,
            color=color,
        )

    style_plot(fig, ax)
    fig.tight_layout()
    return fig


def _get_seasonalities(params, start_date, end_date):
    dates = pd.date_range(start_date, end_date)
    weak_val = params.loc[("seasonality_effect", "seasonality_effect", "weak"), "value"]
    strong_val = params.loc[
        ("seasonality_effect", "seasonality_effect", "strong"), "value"
    ]
    weak_seasonality = create_seasonality_series(dates, weak_val)
    mean_seasonality = create_seasonality_series(dates, 0.5 * (weak_val + strong_val))
    strong_seasonality = create_seasonality_series(dates, strong_val)
    return weak_seasonality, mean_seasonality, strong_seasonality
