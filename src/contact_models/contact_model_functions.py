import numba as nb
import numpy as np
import pandas as pd
from sid.shared import boolean_choices
from sid.time import get_date

from src.shared import from_epochs_to_timestamps


IS_POSITIVE_CASE = (
    "knows_infectious | (knows_immune & symptomatic) "
    "| (knows_immune & (cd_received_test_result_true >= -13))"
)
"""str: Condition for a positive test case.

The individual either ...

- knows that she is infectious.
- knows she is immune but still symptomatic.
- knows she is immune but 14 days since infection have not passed.

"""


def go_to_weekly_meeting(
    states, params, group_col_name, day_of_week, seed  # noqa: U100
):  # noqa: U100
    """Return who participates in a weekly meeting.

    Args:
        states (pandas.DataFrame): sid states DataFrame
        params (pandas.DataFrame): DataFrame with two index levels,
            subcategory and name.
        group_col_name (str): name of the column identifying this contact model's
            group column.
        day_of_week (str): day of the week on which this model takes place.

    Returns:
        attends_meeting (pandas.Series): same index as states. 1 for workers that
            go to the weekly meeting today.

    """
    date = get_date(states)
    day = date.day_name()
    if day != day_of_week:
        attends_meeting = pd.Series(data=False, index=states.index)
    else:
        attends_meeting = states[group_col_name] != -1
        for params_entry, condition in [
            ("symptomatic_multiplier", states["symptomatic"]),
            ("positive_test_multiplier", states["knows_currently_infected"]),
        ]:
            attends_meeting = reduce_contacts_on_condition(
                attends_meeting,
                states,
                params.loc[(params_entry, params_entry), "value"],
                condition,
                is_recurrent=True,
            )
    return attends_meeting


def go_to_daily_work_meeting(states, params, seed):  # noqa: U100
    """Return which people go to work.

    Args:
        states (pandas.DataFrame): sid states DataFrame
        params (pandas.DataFrame): DataFrame with two index levels,
            subcategory and name. has a "value" column that contains the probabilities
            to the number of possible columns in the "name" index level.

    Returns:
        attends_work (pandas.Series): same index as states. 1 for workers that go to
            work this period, 0 for everyone else.

    """
    date = get_date(states)
    day = date.day_name()

    attends_work = (states["occupation"] == "working") & (
        states["work_daily_group_id"] != -1
    )

    if day in ["Saturday", "Sunday"]:
        attends_work = attends_work & states[f"work_{day.lower()}"]
    else:
        for params_entry, condition in [
            ("symptomatic_multiplier", states["symptomatic"]),
            ("positive_test_multiplier", states["knows_currently_infected"]),
        ]:
            attends_work = reduce_contacts_on_condition(
                attends_work,
                states,
                params.loc[(params_entry, params_entry), "value"],
                condition,
                is_recurrent=True,
            )
    return attends_work


def meet_daily_other_contacts(states, params, group_col_name, seed):  # noqa: U100
    attends_meeting = states[group_col_name] != -1
    for params_entry, condition in [
        ("symptomatic_multiplier", states["symptomatic"]),
        ("positive_test_multiplier", states["knows_currently_infected"]),
    ]:
        attends_meeting = reduce_contacts_on_condition(
            attends_meeting,
            states,
            params.loc[(params_entry, params_entry), "value"],
            condition,
            is_recurrent=True,
        )
    return attends_meeting


def attends_educational_facility(states, params, id_column, seed):  # noqa: U100
    """Indicate which children go to an educational facility.

    Children go to an educational facility on weekdays.
    During vacations, all children do not go to educational facilities.
    Furthermore, there is a probability that children stay at home when they experience
    symptoms or receive a positive test result.

    Args:
        states (pandas.DataFrame): The states given by sid.
        params (pandas.DataFrame): DataFrame with three category levels,
        id_column (str): name of the column in *states* that identifies
            which pupils and adults belong to a group.

    Returns:
        attends_facility (pandas.Series): It is a series with the same index as states.
            The values are one for children that go to the facility and zero for those
            who do not.

    """
    facility, _, _, digit = id_column.split("_")
    model_name = f"educ_{facility}_{digit}"

    date = get_date(states)
    day = date.day_name()
    if day in ["Saturday", "Sunday"]:
        attends_facility = pd.Series(data=False, index=states.index)
    else:
        attends_facility = states[id_column] != -1
        attends_facility = _pupils_having_vacations_do_not_attend(
            attends_facility, states, params
        )
        for params_entry, condition in [
            ("symptomatic_multiplier", states["symptomatic"]),
            ("positive_test_multiplier", states["knows_currently_infected"]),
        ]:
            attends_facility = reduce_contacts_on_condition(
                attends_facility,
                states,
                params.loc[(model_name, params_entry, params_entry), "value"],
                condition,
                is_recurrent=True,
            )
    return attends_facility


def meet_hh_members(states, params, seed):  # noqa: U100
    """Meet household members.

    As single person households have unique household ids, everyone meets their
    household unless they are symptomatic. In that case the sick household member
    don't meet the others with a certain probability.

    Args:
        states (pandas.DataFrame): The states.
        params (pandas.DataFrame): DataFrame with two index levels,
            subcategory and name. has a "value" column that contains the probabilities
            to the number of possible columns in the "name" index level.

    """
    date = get_date(states)
    if date in pd.date_range("2020-12-24", "2020-12-26"):
        meet_hh = pd.Series(0, index=states.index)
    else:
        meet_hh = states["hh_model_group_id"] != -1
        for params_entry, condition in [
            ("symptomatic_multiplier", states["symptomatic"]),
            ("positive_test_multiplier", states["knows_currently_infected"]),
        ]:
            meet_hh = reduce_contacts_on_condition(
                meet_hh,
                states,
                params.loc[(params_entry, params_entry), "value"],
                condition,
                is_recurrent=True,
            )
    return meet_hh


def meet_other_non_recurrent_contacts(states, params, seed):
    """Meet other non recurrent contacts.

    Individuals in households with educ_workers, retired and children have
    additional contacts during vacations.

    """
    contacts = calculate_non_recurrent_contacts_from_empirical_distribution(
        states=states,
        params=params.loc["other_non_recurrent"],
        seed=seed,
        on_weekends=True,
        query=None,
        reduce_on_condition=False,
    )
    affected_by_vacation = _identify_ppl_affected_by_vacation(states)

    date = get_date(states)
    state_to_vacation = get_states_w_vacations(date, params)
    potential_vacation_contacts = _draw_potential_vacation_contacts(
        states, params, state_to_vacation, seed
    )
    vacation_contacts = potential_vacation_contacts.where(affected_by_vacation, 0)
    contacts = contacts + vacation_contacts

    for params_entry, condition in [
        ("symptomatic_multiplier", states["symptomatic"]),
        ("positive_test_multiplier", states["knows_currently_infected"]),
    ]:
        contacts = reduce_contacts_on_condition(
            contacts,
            states,
            params.loc[("other_non_recurrent", params_entry, params_entry), "value"],
            condition,
            is_recurrent=False,
        )

    contacts = contacts.astype(int)
    return contacts


def _identify_ppl_affected_by_vacation(states):
    vacation_cols = ["school", "preschool", "nursery", "retired"]
    has_school_vacation = (
        states["occupation"].isin(vacation_cols) | states["educ_worker"]
    )
    # ~60% of individuals are in a household where someone has school vacations
    in_hh_with_vacation = has_school_vacation.groupby(states["hh_id"]).transform(np.any)
    return in_hh_with_vacation


def _draw_potential_vacation_contacts(states, params, state_to_vacation, seed):
    np.random.seed(seed)
    fed_state_to_p_contact = {fed_state: 0 for fed_state in states["state"].unique()}
    for fed_state, vacation in state_to_vacation.items():
        tup = ("additional_other_vacation_contact", "probability", vacation)
        fed_state_to_p_contact[fed_state] = params.loc[tup, "value"]
    p_contact = states["state"].map(fed_state_to_p_contact.get)
    vacation_contact = pd.Series(boolean_choices(p_contact), index=states.index)
    vacation_contact = vacation_contact.astype(int)
    return vacation_contact


def calculate_non_recurrent_contacts_from_empirical_distribution(
    states, params, on_weekends, seed, query=None, reduce_on_condition=True
):
    """Draw how many non recurrent contacts each person will have today.

    Args:
        states (pandas.DataFrame): sid states DataFrame.
        params (pandas.DataFrame): DataFrame with two index levels,
            subcategory and name. has a "value" column that contains the probabilities
            to the number of possible columns in the "name" index level.
        on_weekends (bool or str): whether to meet on weekends or not. If it's a string
            it's interpreted as the prefix of columns identifying who participates
            in this contact model on weekends. Then, columns of the form
            "{on_weekends}_saturday" and "{on_weekends}_sunday" must be in states.
        query (str): query string to identify the subset of individuals to which this
            contact model applies.

    Returns:
        contacts (pandas.Series): index is the same as states. values is the number of
            contacts.

    """
    date = get_date(states)
    day = date.day_name()
    contacts = pd.Series(0, index=states.index)

    if not on_weekends and day in ["Saturday", "Sunday"]:
        pass
    else:
        if isinstance(on_weekends, str) and day in ["Saturday", "Sunday"]:
            participating_today = states[f"{on_weekends}_{day.lower()}"]
            is_participating = states.eval(query) & participating_today
        else:
            if query is not None:
                is_participating = states.eval(query)
            else:
                is_participating = pd.Series(True, index=states.index)

        distribution = params.query("~subcategory.str.contains('multiplier')")["value"]
        contacts[is_participating] = _draw_nr_of_contacts(
            distribution=distribution,
            is_participating=is_participating,
            states=states,
            seed=seed,
        )

        if reduce_on_condition:
            for params_entry, condition in [
                ("symptomatic_multiplier", states["symptomatic"]),
                ("positive_test_multiplier", states["knows_currently_infected"]),
            ]:
                contacts = reduce_contacts_on_condition(
                    contacts,
                    states,
                    params.loc[(params_entry, params_entry), "value"],
                    condition,
                    is_recurrent=False,
                )
    contacts = contacts.astype(float)
    return contacts


def _draw_nr_of_contacts(distribution, is_participating, states, seed):
    """Draw the number of contacts for everyone in a is_participating.

    Args:
        distribution (pandas.Series): slice of the params DataFrame with
            the distribution. The `subcategory` level of the index either
            identifies the age group specific distribution or must be the
            same for the whole slice. The `name` index level gives the support
            and the values of the Series give the probabilities.
        is_participating (pandas.Series): same index as states. True for the individuals
            that participate in the current contact model, i.e. for which the
            number of contacts should be drawn.
        states (pandas.DataFrame): sid states DataFrame.

    Returns:
        nr_of_contacts (pandas.Series): Same index as the states, values are
            the number of contacts for each person.

    """
    is_age_varying = distribution.index.get_level_values("subcategory").nunique() > 1

    if is_age_varying:
        age_labels = [f"{i}-{i + 9}" for i in range(0, 71, 10)] + ["80-100"]
        age_dtype = pd.CategoricalDtype(categories=age_labels, ordered=True)
        age_group = states["age_group"].astype(age_dtype)
        age_codes = age_group.cat.codes.to_numpy()

        probs_df = distribution.unstack().reindex(age_labels).fillna(0)
        support = probs_df.columns.to_numpy().astype(int)
        probs = probs_df.to_numpy()
        cum_probs = probs.cumsum(axis=1)

        nr_of_contacts_arr = _draw_age_varying_nr_of_contacts_numba(
            support=support,
            cum_probs=cum_probs,
            age_codes=age_codes,
            is_participating=is_participating.to_numpy(),
            seed=seed,
        )

    else:
        np.random.seed(seed)
        support = distribution.index.get_level_values("name").to_numpy()
        probs = distribution.to_numpy()
        nr_of_contacts_arr = np.where(
            is_participating.to_numpy(),
            np.random.choice(support, p=probs, size=len(states)),
            0,
        )

    return pd.Series(nr_of_contacts_arr, index=states.index)


@nb.njit
def _draw_age_varying_nr_of_contacts_numba(
    support, cum_probs, age_codes, is_participating, seed
):
    np.random.seed(seed)
    n_obs = len(age_codes)
    out = np.zeros(n_obs)
    for i in range(n_obs):
        if is_participating[i]:
            out[i] = _fast_choice(support, cum_probs[age_codes[i]])
    return out


@nb.njit
def _fast_choice(arr, cdf):
    u = np.random.uniform(0, 1)
    i = 0
    highest_i = len(cdf) - 1

    while cdf[i] < u and i < highest_i:
        i += 1
    return arr[i]


# -------------------------------------------------------------------------------------


def reduce_contacts_on_condition(contacts, states, multiplier, condition, is_recurrent):
    """Reduce contacts for share of population for which condition is fulfilled.

    The subset of contacts for which contacts are reduced is specified by the condition
    and whoever has a positive number of contacts. Then, a share of individuals in the
    subset is sampled and the contacts are set to 0.

    Args:
        contacts (pandas.Series): The series with contacts.
        states (pandas.DataFrame): The states of one day passed by sid.
        multiplier (float): The share of people who maintain their contacts
            despite condition.
        condition (str, numpy.ndarray or pandas.Series): Condition or boolean array
            or Series which defines the subset of individuals who potentially reduce
            their contacts.
        seed (int)

    """
    if isinstance(condition, str):
        is_condition_true = states.eval(condition)
    elif isinstance(condition, pd.Series):
        is_condition_true = condition.to_numpy()
    elif isinstance(condition, np.ndarray):
        is_condition_true = condition
    else:
        raise ValueError

    refuser = states["quarantine_compliance"] <= multiplier
    no_change = refuser | ~is_condition_true

    if is_recurrent:
        reduced = contacts.where(cond=no_change, other=False)
    else:
        reduced = contacts.where(cond=no_change, other=0)

    return reduced


# =============================================================================


def _pupils_having_vacations_do_not_attend(attends_facility, states, params):
    """Make pupils stay away from school if their state has vacations."""
    attends_facility = attends_facility.copy(deep=True)
    date = get_date(states)
    states_w_vacations = get_states_w_vacations(date, params).keys()
    has_vacation = states.state.isin(states_w_vacations)
    attends_facility.loc[attends_facility & has_vacation] = False

    return attends_facility


def get_states_w_vacations(date: pd.Timestamp, params: pd.DataFrame) -> dict:
    """Get states which currently have vacations for pupils.

    Returns:
        state_to_vacation_name (dict): keys are the states that have vacations
            on the current date. Values are the names of the vacation.

    """
    vacations = params.filter(like="ferien", axis=0).copy()
    if vacations.empty:
        raise ValueError("'params' does not contain any information about vacations.")

    # Dates are stored as epochs so that value can be a numeric column.
    vacations["value"] = from_epochs_to_timestamps(vacations["value"])
    vacations = vacations.groupby(vacations.index.names)["value"].first().unstack()
    latest_vacation_date = vacations["end"].max()
    assert (
        date <= latest_vacation_date
    ), f"Vacations are only known until {latest_vacation_date}"

    has_vacations = (vacations["start"] <= date) & (date <= vacations["end"])
    state_to_vacation = {
        state: name for name, state in has_vacations[has_vacations].index
    }
    return state_to_vacation
