import warnings

import pandas as pd
from sid import get_date


def find_people_to_vaccinate(
    receives_vaccine,  # noqa: U100
    states,
    params,
    seed,  # noqa: U100
    vaccination_shares,
    init_start,
):
    """Find people that have to be vaccinated on a given day.

    On the init_start date all individuals that should have been vaccinated until
    that day get vaccinated. Since vaccinations take effect within three weeks
    and the burn in period is four weeks this does not lead to jumps in the
    simulation period.

    Args:
        states (pandas.DataFrame): States DataFrame that must contain the
            column vaccination_rank. This column is a float with values
            between zero and one. Low values mean that people get
            vaccinated early.
        params (pandas.DataFrame): not used.
        seed (int): not used.
        vaccination_shares (pandas.Series): Series with a date index. For each
            day the value indicates the share of people who get vaccinated on
            that day.
        no_vaccination_share (float): Share of people who refuse to get
            vaccinated.
        init_start (pd.Timestamp): start date of the burn in period. On the
            init_start all vaccinations that have been done until then are
            handed out on that day.

    """
    date = get_date(states)
    no_vaccination_share = params.loc[
        ("vaccinations", "share_refuser", "share_refuser"), "value"
    ]

    if not (vaccination_shares < 0.05).all():
        warnings.warn(
            "The vaccination shares imply that >=5% of people get vaccinated per day.",
            "If this was intended, simply ignore the warning."
        )

    cutoffs = vaccination_shares.sort_index().cumsum()
    # set all cutoffs before the init_start to 0.
    # that way on the init_start date everyone who should have been vaccinated
    # until that day gets vaccinated.
    cutoffs[: init_start - pd.Timedelta(days=1)] = 0

    lower_candidate = cutoffs[date - pd.Timedelta(days=1)]
    upper_candidate = cutoffs[date]
    lower = min(lower_candidate, no_vaccination_share)
    upper = min(upper_candidate, no_vaccination_share)

    to_vaccinate = (lower <= states["vaccination_rank"]) & (
        states["vaccination_rank"] < upper
    )
    return to_vaccinate
