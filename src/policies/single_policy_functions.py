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
from sid.time import get_date
from src.config import BLD


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
    - Use start_multiplier until switching date (e.g. end of summer vacation)
    - Us end_multiplier after switching date.

    The default reopening dates are very coarse, based on a very simplified map covering
    first openings and only focusing on schools: https://tinyurl.com/yyrsmfp2
    Following it, we err on the side of making kids start attending too early.

    Args:
        start_multiplier (float): Activity multiplier after reopening but before
            switching date. Typically stricter than after switching date.
        end_multiplier (float): Activity multiplier after switching date.
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
        assert pd.Timestamp("2020-04-22") <= date <= pd.Timestamp("2020-10-30"), (
            "Default reopening dates assume that policy is applied between 2020-04-22 "
            "and 2020-10-30"
        )
        reopening_dates = default_reopening_dates
    assert (
        states["state"].isin(reopening_dates.keys()).all()
    ), "Not all federal states are included in the reopening dates."

    closed_states = []
    for state, start_date in reopening_dates.items():
        if date < start_date:
            closed_states.append(state)
    still_closed = states["state"].isin(closed_states)
    contacts[still_closed] = 0

    switching_date = pd.Timestamp(switching_date)
    assert start_multiplier >= 0.0, "Multipliers must be greater or equal to zero."
    assert end_multiplier >= 0.0, "Multipliers must be greater or equal to zero."
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

    This function returns a Series of 0s and 1s.

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
    date = get_date(states)
    attending_half = contacts.where(
        (states["age"] < age_cutoff)
        | states["educ_worker"]
        | (states["school_group_a"] == date.week % 2),
        0,
    )
    return attending_half


def shut_down_work_model(states, contacts, seed):  # noqa: U100
    return contacts.where(states["systemically_relevant"], 0)


def reduce_work_model(states, contacts, seed, multiplier):  # noqa: U100
    """Reduce contacts for the non systemically relevant working population.

    Contacts of systemically relevant workers are never reduced.

    Args:
        multiplier (float): share of non-systemically relevant workers
            that have work contacts.

    """
    assert 0 <= multiplier <= 1
    threshold = 1 - multiplier
    reduced_contacts = contacts.where(
        states["systemically_relevant"] | (states["work_contact_priority"] > threshold),
        0,
    )
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


def reduce_contacts_through_private_contact_tracing(
    contacts, states, seed, multiplier, group_ids, is_recurrent, path
):
    today = get_date(states)
    days_since_christmas = (today - pd.Timestamp("2020-12-26")).days
    test_condition = f"-{days_since_christmas} <= cd_received_test_result_true <= 0"
    symptom_condition = f"-{days_since_christmas} <= cd_symptoms_true <= 0"
    risk_condition = f"({symptom_condition}) | ({test_condition})"

    reduced = reduce_contacts_when_condition_among_recurrent_contacts(
        contacts=contacts,
        states=states,
        seed=seed,
        multiplier=multiplier,
        group_ids=group_ids,
        condition=risk_condition,
        is_recurrent=is_recurrent,
        path=path,
    )
    return reduced


def reduce_contacts_when_condition_among_recurrent_contacts(
    contacts, states, seed, multiplier, group_ids, condition, is_recurrent, path=None
):
    """Reduce contacts when one of your contacts fulfills a condition.

    This is akin to private contact tracing, i.e. coworkers or friends informing
    their contacts that they are symptomatic or tested positive.

    Args:
        contacts (pandas.Series)
        states (pandas.DataFrame)
        seed (int)
        multiplier (float): Multiplier, i.e. the share of people that still participate
            in contact models or the multiplier on the non-recurrent contacts.
        group_ids (list): list of columns identifying group memberships.
        condition (str): query/eval string. If any member of any group of an
            individual fulfills the condition, the individual is marked as having
            had a risk contact (unless (s)he herself fulfills the condition).
        is_recurrent (bool): Whether the contact model is recurrent or not.
        path (str or pathlib.Path): Path to a folder in which information on the
            contact tracing is stored.

    Returns:
        reduced (pandas.Series): reduced contacts.

    """
    with_risk_contacts = _identify_individuals_with_risk_contacts(
        states, group_ids, condition, path
    )

    if is_recurrent:
        all_reduced = reduce_recurrent_model(states, contacts, seed, multiplier)
    else:
        all_reduced = multiplier * contacts

    reduced = all_reduced.where(with_risk_contacts, contacts)

    return reduced


def _identify_individuals_with_risk_contacts(states, group_ids, condition, path=None):
    """Identify those in whose groups someone fulfills the condition.

    .. warning::
        This potentially identifies much more people than those who
        actually had risk contacts. For example, people who work
        from home but have a sick co-worker would still be identified
        as having had a risk contact!

    .. warning::
        This modifies states inplace!

    Args:
        states (pandas.DataFrame)
        group_ids (list): list of columns identifying group memberships.
        condition (str): query string. If any member of any group
            fulfills the condition, an individual is marked as having
            had a risk contact (unless (s)he herself fulfills the condition).
        path (str or pathlib.Path): Path to a folder in which information on the
            contact tracing is stored.

    Returns:
        risk_in_any_group (pandas.Series): boolean Series with same index
            as states. True for individuals who don't fulfill the condition
            but have a contact in one of their groups who does.

    """
    today = get_date(states)
    risk_col = f"has_known_risk_contact_{today.date()}"
    if risk_col not in states.columns:
        old_col = f"has_known_risk_contact_{(today - pd.Timedelta(days=1)).date()}"
        if old_col in states.columns:
            states.drop(
                columns=[old_col],
                inplace=True,
            )

        helper = states[group_ids].copy()
        helper["is_risk_contact"] = states.eval(condition)

        risk_in_any_group = pd.Series(False, index=states.index)
        for col in group_ids:
            risk_in_this_group = helper.groupby(col)["is_risk_contact"].transform("any")
            # those in the -1 group have no contacts
            risk_in_this_group = risk_in_this_group.where(helper[col] != -1, False)
            risk_in_any_group = risk_in_any_group | risk_in_this_group

        # individuals who are themselves affected (e.g. symptomatic)
        # reduce behavior through a different function. We don't want to "double" this.
        risk_in_any_group = risk_in_any_group.where(~states.eval(condition), False)

        states[risk_col] = risk_in_any_group
        if path is not None:
            risk_in_any_group.to_pickle(path / f"{risk_col}.pkl")
    else:
        risk_in_any_group = states[risk_col]

    return risk_in_any_group
