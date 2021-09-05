import warnings

import numpy as np
import pandas as pd
from sid import get_date


def find_people_to_vaccinate(
    receives_vaccine,  # noqa: U100
    states,
    params,
    seed,
    vaccination_shares,
    init_start,
):
    """Find people that have to be vaccinated on a given day.

    On the init_start date all individuals that should have been vaccinated
    until that day get vaccinated. Since vaccinations take effect within three
    weeks and the burn in period is four weeks this does not lead to jumps in
    the simulation period.

    On June 7 the vaccination priorization was lifted in Germany. On May 28 the
    EMA approved the first vaccine for youths from 12 to 15. However, until July
    15 less than 5% of 12 to 17 year olds were vaccinated and the STIKO only
    recommended the CoViD vaccine for healthy 12 to 17 year olds in August.

    We model this by vaccinating individuals 12 and older that are not refusers
    randomly starting on July 15.

    Sources:
        - https://bit.ly/2WpIegM
        - https://bit.ly/3zlzIhe

    Args:
        states (pandas.DataFrame): States DataFrame that must contain the
            column vaccination_rank. This column is a float with values between zero
            and one. Low values mean that people get vaccinated early.
        params (pandas.DataFrame): params DataFrame that contains a row labeled
            ("vaccinations", "share_refuser", "adult").
        seed (int): used after 2021-06-07.
        vaccination_shares (pandas.Series): Series with a date index. For each day the
            value indicates the share of people who get vaccinated on that day.
        init_start (pd.Timestamp): start date of the burn in period. On the init_start
            all vaccinations that have been done until then are handed out on that day.

    """
    np.random.seed(seed)
    date = get_date(states)
    no_vaccination_share_adult = params.loc[
        ("vaccinations", "share_refuser", "adult"), "value"
    ]

    if not (vaccination_shares < 0.05).all():
        warnings.warn(
            "The vaccination shares imply that >=5% of people get vaccinated per day. "
            "If this was intended, simply ignore the warning.",
        )

    if date < pd.Timestamp("2021-07-15"):  # priorisation. only >= 16 get vaccinated.
        cutoffs = vaccination_shares.sort_index().cumsum()
        # set all cutoffs before the init_start to 0.
        # that way on the init_start date everyone who should have been vaccinated
        # until that day gets vaccinated.
        cutoffs[: init_start - pd.Timedelta(days=1)] = 0

        lower_candidate = cutoffs[date - pd.Timedelta(days=1)]
        upper_candidate = cutoffs[date]
        lower = min(lower_candidate, 1 - no_vaccination_share_adult)
        upper = min(upper_candidate, 1 - no_vaccination_share_adult)

        to_vaccinate = (lower <= states["vaccination_rank"]) & (
            states["vaccination_rank"] < upper
        )
    else:  # no priorisation. >= 12 get vaccinated
        unvaccinated = ~states["ever_vaccinated"]
        eligible = states["age"] >= 12

        # the highest value denotes refusers
        willing = (
            states["vaccination_group_with_refuser_group"]
            < states["vaccination_group_with_refuser_group"].max()
        )
        pool = states[unvaccinated & eligible & willing].index
        n_to_draw = int(vaccination_shares[date] * len(states))
        sampled = np.random.choice(pool, n_to_draw, replace=False)
        to_vaccinate = pd.Series(states.index.isin(sampled), index=states.index)

    return to_vaccinate
