"""Define policy functions and helper functions.

All public functions have the same first arguments which will not be documented in
individual docstrings:

- states (pandas.DataFrame): A sid states DataFrame
- contacts (pandas.Series): A Series with the same index as states.
- seed (int): A seed for the random state.

Moreover, all public functions return a pandas.Series with the same index as states.

All other arguments must be documented.


"""
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sid.shared import get_date


def shut_down_model(states, contacts, seed):
    """Set all contacts to zero independent of incoming contacts."""
    return pd.Series(0, index=states.index)


def reopen_educ_model_germany(
    states,
    contacts,
    seed,
    start_multiplier,
    end_multiplier,
    switching_date,
    reopening_dates,
    is_recurrent,
):
    """Reopen an educ model at state specific dates

    - Keep the model closed until local reopening date
    - Work with strongly reduced contacts until summer vacation
    - Work with slightly reduced contact after summer vacation

    The reopening dates are very coarse, based on a very simplified map covering
    first openings and only focusing on schools: https://tinyurl.com/yyrsmfp2
    Following it, we err on the side of making kids start attending too early.

    Args:
        start_multiplier (float): Activity multiplier after reopening but before
            summer vacations. Typically stricter than after summer vacation.
        end_multiplier (float): Activity multiplier after summer vacation.
        switching_date (str or pandas.Timestamp): Date at which multipliers are switched
        reopening_dates (dict): Maps German federal states to dates
        is_recurrent (bool): If the affected contact models is recurrent.

    """
    default_reopening_dates = {
        # https://tinyurl.com/y2733xkl
        # https://tinyurl.com/yywha63j
        "Baden-Württemberg": pd.Timestamp("2020-05-04"),
        "Bayern": pd.Timestamp("2020-04-27"),
        "Berlin": pd.Timestamp("2020-04-20"),
        "Brandenburg": pd.Timestamp("2020-05-04"),
        "Bremen": pd.Timestamp("2020-05-04"),
        "Hamburg": pd.Timestamp("2020-05-04"),
        "Hessen": pd.Timestamp("2020-05-04"),
        "Mecklenburg-Vorpommern": pd.Timestamp("2020-04-27"),
        "Niedersachsen": pd.Timestamp("2020-04-27"),
        "Nordrhein-Westfalen": pd.Timestamp("2020-04-20"),
        "Rheinland-Pfalz": pd.Timestamp("2020-04-27"),
        "Saarland": pd.Timestamp("2020-05-04"),  # https://tinyurl.com/y4p55nh9
        "Sachsen": pd.Timestamp("2020-04-20"),
        "Sachsen-Anhalt": pd.Timestamp("2020-04-20"),
        "Schleswig-Holstein": pd.Timestamp("2020-04-20"),
        "Thüringen": pd.Timestamp("2020-04-27"),
    }

    date = get_date(states)
    if reopening_dates is None:
        assert pd.Timestamp("2020-04-22") <= date <= pd.Timestamp("2020-09-30"), (
            "Default reopening dates assume that policy is applied between 2020-04-22 "
            "and 2020-09-30"
        )
        reopening_dates = default_reopening_dates
    else:
        assert set(reopening_dates) == set(default_reopening_dates), (
            "You need to provide reopening dates for all german states:\n"
            f"{set(default_reopening_dates)}"
        )

    closed_states = []
    for state, start_date in reopening_dates.items():
        if date < start_date:
            closed_states.append(state)
    still_closed = states["state"].isin(closed_states)
    contacts[still_closed] = 0

    switching_date = pd.Timestamp(switching_date)
    multiplier = start_multiplier if date < switching_date else end_multiplier

    if is_recurrent:
        contacts = reduce_recurrent_model(
            states=states,
            contacts=contacts,
            seed=seed,
            multiplier=multiplier,
        )
    else:
        contacts = contacts * multiplier

    return contacts


def reduce_recurrent_model(states, contacts, seed, multiplier):
    """Reduce the number of recurrent contacts taking place by a multiplier.

    For recurrent contacts only whether the contacts Series is > 0 plays a role.
    Therefore, simply multiplying the number of contacts with it would not have
    an effect on the number of contacts taking place. Instead we make a random share of
    individuals scheduled to participate not participate.

    Args:
        multiplier (float): Must be smaller or equal to one.

    """
    np.random.seed(seed)
    contacts = contacts.to_numpy()
    resampled_contacts = np.random.choice(
        [1, 0], size=len(states), p=[multiplier, 1 - multiplier]
    )
    reduced = np.where(contacts > 0, resampled_contacts, contacts)
    return pd.Series(reduced, index=states.index)


def implement_a_b_school_system_above_age(
    states, contacts, seed, age_cutoff
):  # noqa: U100
    """Classes are split in two for children above age cutoff.

    Args:
        age_cutoff (float): Minimum age to which the policy applies. Set to 0 to
            implement the policy for all.

    """
    assert set(states["school_group_a"].unique()) == {0, 1}
    assert "teacher" in states["occupation"].cat.categories
    date = get_date(states)
    attending_half = contacts.where(
        (states["age"] < age_cutoff)
        | (states["occupation"] == "teacher")
        | (states["school_group_a"] == date.week % 2),
        0,
    )
    return attending_half


def shut_down_work_model(states, contacts, seed):  # noqa: U100
    return contacts.where(states["systemically_relevant"], 0)


def reduce_work_model(states, contacts, seed, multiplier):  # noqa: U100
    """Reduce contacts for the non essential working population.

    Contacts of essential workers are never reduced.

    Args:
        multiplier (float): multiplier of non-essential workers that have work contacts.

    """
    assert 0 <= multiplier <= 1
    threshold = 1 - multiplier
    reduced_contacts = contacts.where(states["work_contact_priority"] > threshold, 0)
    return reduced_contacts


def reopen_work_model(
    states, contacts, seed, start_multiplier, end_multiplier, start_date, end_date
):
    """Reduce work contacts to active people in gradual opening or closing phase.

    This is for example used to model the gradual reopening after the first lockdown
    in Germany (End of April 2020 to beginning of October 2020).

    Work contacts require special treatment because workers are differ persistently in
    their "work_contact_priority", i.e. some workers are essential, others are not,
    with a continuum in between.

    Args:
        start_multiplier (float): Activity at start.
        end_multiplier (float): Activity level at end.
        start_date (str or pandas.Timestamp): Date at which the interpolation phase
            starts.
        end_date (str or pandas.Timestamp): Date at which the interpolation phase ends.

    """
    date = get_date(states)

    multiplier = _interpolate_activity_level(
        date=date,
        start_multiplier=start_multiplier,
        end_multiplier=end_multiplier,
        start_date=start_date,
        end_date=end_date,
    )
    contacts = reduce_work_model(
        states=states, contacts=contacts, seed=seed, multiplier=multiplier
    )

    return contacts


def reopen_other_model(
    states,
    contacts,
    seed,
    start_multiplier,
    end_multiplier,
    start_date,
    end_date,
    is_recurrent,
):
    """Reduce non-work contacts to active people in gradual opening or closing phase.

    This is for example used to model the gradual reopening after the first lockdown
    in Germany (End of April 2020 to beginning of October 2020).

    Args:
        start_multiplier (float): Activity at start.
        end_multiplier (float): Activity level at end.
        start_date (str or pandas.Timestamp): Date at which the interpolation phase
            starts.
        end_date (str or pandas.Timestamp): Date at which the interpolation phase ends.

    """
    date = get_date(states)
    multiplier = _interpolate_activity_level(
        date=date,
        start_multiplier=start_multiplier,
        end_multiplier=end_multiplier,
        start_date=start_date,
        end_date=end_date,
    )

    if is_recurrent:
        reduced = reduce_recurrent_model(states, contacts, seed, multiplier)
    else:
        reduced = multiplier * contacts
    return reduced


def _interpolate_activity_level(
    date, start_multiplier, end_multiplier, start_date, end_date
):
    """Calculate an activity level in a gradual reopening or closing phase.

    Args:
        date (str or pandas.Timestamp): Date at which activity level is calculated.
        start_multiplier (float): Activity at start.
        end_multiplier (float): Activity level at end.
        start_date (str or pandas.Timestamp): Date at which the interpolation phase
            starts.
        end_date (str or pandas.Timestamp): Date at which the interpolation phase ends.

    Returns:
        float: The interpolated activity level.

    """
    date = pd.Timestamp(date)
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    assert date >= start_date
    assert date <= end_date
    assert 0 <= start_multiplier <= 1
    assert 0 <= end_multiplier <= 1

    interpolator = interp1d(
        x=[start_date.dayofyear, end_date.dayofyear],
        y=[start_multiplier, end_multiplier],
        kind="linear",
    )
    activity = interpolator(date.dayofyear)
    return activity
