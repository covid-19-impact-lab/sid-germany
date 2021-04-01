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


def shut_down_model(states, contacts, seed, is_recurrent):
    """Set all contacts to zero independent of incoming contacts."""
    if is_recurrent:
        return pd.Series(False, index=states.index)
    else:
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
    resampled_contacts = np.random.choice(
        [True, False], size=len(states), p=[multiplier, 1 - multiplier]
    )
    reduced = np.where(contacts, resampled_contacts, contacts)
    return pd.Series(reduced, index=states.index)


def reduce_work_model(states, contacts, seed, multiplier, is_recurrent):  # noqa: U100
    """Reduce contacts for the working population.

    Args:
        multiplier (float, pandas.Series, pandas.DataFrame):
            share of workers that have work contacts.
            If it is a Series or DataFrame, the index must be dates.
            If it is a DataFrame the columns must be the values of
            the "state" column in the states.
        is_recurrent (bool): True if the contact model is recurernt

    """
    if isinstance(multiplier, (pd.Series, pd.DataFrame)):
        date = get_date(states)
        multiplier = multiplier.loc[date]

    msg = f"Work multiplier not in [0, 1] on {get_date(states)}"
    if isinstance(multiplier, (float, int)):
        assert 0 <= multiplier <= 1, msg
    else:
        assert (multiplier >= 0).all(), msg
        assert (multiplier <= 1).all(), msg

    threshold = 1 - multiplier
    if isinstance(threshold, pd.Series):
        threshold = states["state"].map(threshold.get)
        # this assert could be skipped because we check in
        # task_check_initial_states that the federal state names overlap.
        assert threshold.notnull().all()

    above_threshold = states["work_contact_priority"] > threshold
    if not is_recurrent:
        reduced_contacts = contacts.where(above_threshold, 0)
    if is_recurrent:
        reduced_contacts = contacts.where(above_threshold, False)
    return reduced_contacts


def reopen_work_model(
    states,
    contacts,
    seed,
    start_multiplier,
    end_multiplier,
    start_date,
    end_date,
    is_recurrent,
):
    """Reduce work contacts to active people in gradual opening or closing phase.

    This is for example used to model the gradual reopening after the first lockdown
    in Germany (End of April 2020 to beginning of October 2020).

    Work contacts require special treatment because workers are differ persistently in
    their "work_contact_priority".

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
        states=states,
        contacts=contacts,
        seed=seed,
        multiplier=multiplier,
        is_recurrent=is_recurrent,
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


# ----------------------------------------------------------------------------


def apply_educ_policy(
    states,
    contacts,
    seed,
    group_id_column,
    always_attend_query,
    a_b_query,
    non_a_b_attend,
    hygiene_multiplier,
    a_b_rhythm="weekly",
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
        group_column=group_id_column + "_a_b",
        a_b_query=a_b_query,
        a_b_rhythm=a_b_rhythm,
    )
    attends_for_any_reason = attends_always | attends_because_of_a_b_schooling
    if non_a_b_attend:
        attends_for_any_reason = attends_for_any_reason | ~states.eval(a_b_query)

    staying_home = ~attends_for_any_reason
    contacts[staying_home] = False

    # since our educ models are all recurrent and educ_workers must always attend
    # we only apply the hygiene multiplier to the students
    all_reduced = reduce_recurrent_model(
        states,
        contacts,
        seed=seed,
        multiplier=hygiene_multiplier,
    )
    contacts = contacts.where(states["educ_worker"], other=all_reduced)

    teachers_with_0_students = _find_educ_workers_with_zero_students(
        contacts, states, group_id_column
    )
    contacts[teachers_with_0_students] = False
    return contacts


def _identify_who_attends_because_of_a_b_schooling(
    states, group_column, a_b_query, a_b_rhythm
):
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
            in_attend_group = states[group_column] == date.week % 2
        elif a_b_rhythm == "daily":
            in_attend_group = states[group_column] == date.day % 2
        attends_because_of_a_b_schooling = a_b_eligible & in_attend_group
    else:
        raise ValueError(
            f"a_b_query must be either bool or str, you supplied a {type(a_b_query)}"
        )
    return attends_because_of_a_b_schooling


def _find_educ_workers_with_zero_students(contacts, states, group_id_column):
    """Return educ_workers whose classes / groups don't have any children in them.

    Returns:
        has_no_class (pandas.Series): boolean Series with the
            same index as states. True for educ_workers whose classes / groups
            don't have any children in them.

    """
    size_0_classes = _find_size_zero_classes(contacts, states, group_id_column)
    has_no_class = states["educ_worker"] & states[group_id_column].isin(size_0_classes)
    return has_no_class


def _find_size_zero_classes(contacts, states, col):
    students_group_ids = states[col][~states["educ_worker"]]
    students_contacts = contacts[~states["educ_worker"]]
    # the .drop(-1) is needed because we use -1 instead of NaN to identify
    # individuals not participating in a recurrent contact model
    class_sizes = students_contacts.groupby(students_group_ids).sum().drop(-1)
    size_zero_classes = class_sizes[class_sizes == 0].index
    return size_zero_classes
