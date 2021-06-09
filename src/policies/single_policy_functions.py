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
from sid.shared import boolean_choices
from sid.time import get_date


def shut_down_model(states, contacts, seed, is_recurrent, params=None):  # noqa: U100
    """Set all contacts to zero independent of incoming contacts."""
    if is_recurrent:
        return pd.Series(False, index=states.index)
    else:
        return pd.Series(0, index=states.index)


def reduce_recurrent_model(
    states, contacts, seed, multiplier, params=None  # noqa: U100
):
    """Reduce the number of recurrent contacts taking place by a multiplier.

    For recurrent contacts the contacts Series is boolean.
    Therefore, simply multiplying the number of contacts with it would not have
    an effect on the number of contacts taking place. Instead we make a random share of
    individuals scheduled to participate not participate.

    Args:
        multiplier (float or pd.Series): Must be smaller or equal to one. If a
            Series is supplied the index must be dates.


    Returns:
        reduced (pandas.Series): same index as states. For a *multiplier* fraction
            of the population the contacts have been set to False. The more individuals
            already had a False there, the smaller the effect.

    """
    np.random.seed(seed)
    if isinstance(multiplier, pd.Series):
        date = get_date(states)
        multiplier = multiplier[date]

    contacts = contacts.to_numpy()
    resampled_contacts = boolean_choices(np.full(len(states), multiplier))

    reduced = np.where(contacts, resampled_contacts, contacts)
    return pd.Series(reduced, index=states.index)


def reduce_work_model(
    states,
    contacts,
    seed,
    attend_multiplier,
    is_recurrent,
    hygiene_multiplier,
    params=None,  # noqa: U100
):
    """Reduce contacts for the working population.

    Args:
        attend_multiplier (float, pandas.Series, pandas.DataFrame):
            share of workers that have work contacts.
            If it is a Series or DataFrame, the index must be dates.
            If it is a DataFrame the columns must be the values of
            the "state" column in the states.
        hygiene_multiplier (float, or pandas.Series): Degree to
            which contacts at work can still lead to infection.
            Must be smaller or equal to one. If a Series is supplied
            the index must be dates.
        is_recurrent (bool): True if the contact model is recurernt

    """
    attend_multiplier = _process_multiplier(states, attend_multiplier, "attend")
    hygiene_multiplier = _process_multiplier(states, hygiene_multiplier, "hygiene")

    threshold = 1 - attend_multiplier
    if isinstance(threshold, pd.Series):
        threshold = states["state"].map(threshold.get).astype(float)
        # this assert could be skipped because we check in
        # task_check_initial_states that the federal state names overlap.
        assert threshold.notnull().all()

    above_threshold = states["work_contact_priority"] > threshold
    if is_recurrent:
        reduced_contacts = contacts.where(above_threshold, False)
        if hygiene_multiplier < 1:
            reduced_contacts = reduce_recurrent_model(
                states, contacts, seed, hygiene_multiplier, params=params
            )
    else:
        reduced_contacts = contacts.where(above_threshold, 0)
        reduced_contacts = hygiene_multiplier * reduced_contacts
    return reduced_contacts


def _process_multiplier(states, multiplier, name):
    if isinstance(multiplier, (pd.Series, pd.DataFrame)):
        date = get_date(states)
        multiplier = multiplier.loc[date]
    msg = f"Work {name} multiplier not in [0, 1] on {get_date(states)}"
    if isinstance(multiplier, (float, int)):
        assert 0 <= multiplier <= 1, msg
    else:
        assert (multiplier >= 0).all(), msg
        assert (multiplier <= 1).all(), msg
    return multiplier


def reopen_other_model(
    states,
    contacts,
    seed,
    start_multiplier,
    end_multiplier,
    start_date,
    end_date,
    is_recurrent,
    params=None,  # noqa: U100
):
    """Reduce non-work contacts to active people in gradual opening or closing phase.

    This is for example used to model the gradual reopening after the first lockdown
    in Germany (End of April 2020 to beginning of October 2020).

    Args:
        start_multiplier (float): Activity level at start.
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


# ----------------------------------------------------------------------------


def mixed_educ_policy(
    states,
    contacts,
    seed,
    group_id_column,
    always_attend_query,
    a_b_query,
    non_a_b_attend,
    hygiene_multiplier,
    a_b_rhythm="weekly",
    params=None,  # noqa: U100
):
    """Apply a education policy, including potential emergency care and A/B mode.

    Args:
        group_id_column (str): name of the column identifying which indivdiuals
            attend class together, i.e. the assort by column of the current
            contact model. We assume that the column identifying which
            individuals belong to the A or B group is group_id_column + "_a_b".
        always_attend_query (str, optional): query string that identifies
            children always going to school. This allows to model emergency
            care. If None is given no emergency care is implemented.
        a_b_query (str or bool): pandas query string identifying the
            children that are taught in split classes. If True, all children
            are covered by A/B schooling, if False, no A/B schooling is in order.
            If a string, it is interpreted as a query string identifying the
            children that are subject to A/B schooling
        non_a_b_attend (bool): if True, children not selected by the a_b_query
            attend school normally. If False, children not selected by
            the a_b_query and not among the always attend children stay home.
        hygiene_multiplier (float): Applied to all children that still attend
            educational facilities.
        a_b_rhythm (str, optional): one of "weekly" or "daily". Default is weekly.
            If weekly, A/B students rotate between attending and not attending on
            a weekly basis. If daily, A/B students rotate between attending and
            not attending on a daily basis.

    """
    np.random.seed(seed)
    contacts = contacts.copy(deep=True)

    attends_always = states["educ_worker"] | states.eval(always_attend_query)
    attends_because_of_a_b_schooling = _identify_who_attends_because_of_a_b_schooling(
        states=states,
        a_b_query=a_b_query,
        a_b_rhythm=a_b_rhythm,
    )
    attends_for_any_reason = attends_always | attends_because_of_a_b_schooling
    if non_a_b_attend:
        attends_for_any_reason = attends_for_any_reason | ~states.eval(a_b_query)

    staying_home = ~attends_for_any_reason
    contacts[staying_home] = False

    contacts = reduce_recurrent_model(
        states,
        contacts,
        seed=seed,
        multiplier=hygiene_multiplier,
    )

    return contacts


def _identify_who_attends_because_of_a_b_schooling(states, a_b_query, a_b_rhythm):
    """Identify who attends school because (s)he is a student in A/B mode.

    We can ignore educ workers here because they are already covered in attends_always.
    Same for children coverey by emergency care.

    Returns:
        attends_because_of_a_b_schooling (pandas.Series): True for individuals that
            are in rotating split classes and whose half of class is attending today.

    """
    if isinstance(a_b_query, bool):
        attends_because_of_a_b_schooling = pd.Series(a_b_query, index=states.index)
    elif isinstance(a_b_query, str):
        date = get_date(states)
        a_b_eligible = states.eval(a_b_query)
        if a_b_rhythm == "weekly":
            in_attend_group = states["educ_a_b_identifier"] == (date.week % 2 == 1)
        elif a_b_rhythm == "daily":
            in_attend_group = states["educ_a_b_identifier"] == (date.day % 2 == 1)
        attends_because_of_a_b_schooling = a_b_eligible & in_attend_group
    else:
        raise ValueError(
            f"a_b_query must be either bool or str, you supplied a {type(a_b_query)}"
        )
    return attends_because_of_a_b_schooling
