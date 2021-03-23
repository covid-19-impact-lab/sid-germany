import pandas as pd
from sid import get_date


def find_people_to_vaccinate(
    states, params, seed, vaccination_shares, no_vaccination_share  # U100
):
    """Finde peopl that have to be vaccinated on a given day.

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


    """
    date = get_date(states)
    cutoffs = vaccination_shares.sort_index().cumsum()
    lower_candidate = cutoffs.get(date - pd.Timedelta(1, unit="day"), 0)
    upper_candidate = cutoffs[date]
    lower = min(lower_candidate, no_vaccination_share)
    upper = min(upper_candidate, no_vaccination_share)

    to_vaccinate = pd.Series(
        data=lower <= states["vaccination_rank"] < upper,
        index=states.index,
    )
    return to_vaccinate
