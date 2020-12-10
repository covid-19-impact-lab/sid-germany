from typing import List

import numba as nb
import numpy as np
import pandas as pd
from sid.shared import get_date

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


def go_to_weekly_meeting(states, contact_params, group_col_name, day_of_week, seed):
    """Return who participates in a weekly meeting.

    Args:
        states (pandas.DataFrame): sid states DataFrame
        contact_params (pandas.DataFrame): DataFrame with two category levels,
            subcategory and name. has a "value" column that contains the probabilities
            to the number of possible columns in the "name" index level.
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
        attends_meeting = pd.Series(data=0, index=states.index)
    else:
        attends_meeting = (states[group_col_name] != -1).astype(int)
        for params_entry, condition in [
            ("reduction_when_symptomatic", "symptomatic"),
            ("reduction_when_positive", IS_POSITIVE_CASE),
        ]:
            attends_meeting = reduce_recurrent_contacts_on_condition(
                attends_meeting,
                states,
                contact_params.loc[(params_entry, params_entry), "value"],
                condition,
                seed=seed,
            )
    return attends_meeting


def go_to_work(states, contact_params, seed):
    """Return which people go to work.

    Adults go to work if they are workers, it is a weekday, and they do not show any
    symptoms.

    Args:
        states (pandas.DataFrame): sid states DataFrame
        contact_params (pandas.DataFrame): DataFrame with two category levels,
            subcategory and name. has a "value" column that contains the probabilities
            to the number of possible columns in the "name" index level.

    Returns:
        attends_work (pandas.Series): same index as states. 1 for workers that go to
            work this period, 0 for everyone else.

    """
    date = get_date(states)
    day = date.day_name()
    if day in ["Saturday", "Sunday"]:
        attends_work = pd.Series(data=0, index=states.index)
    else:
        attends_work = states.eval(
            "(occupation == 'working') & (daily_work_group_id != -1)"
        ).astype(int)
        for params_entry, condition in [
            ("reduction_when_symptomatic", "symptomatic"),
            ("reduction_when_positive", IS_POSITIVE_CASE),
        ]:
            attends_work = reduce_recurrent_contacts_on_condition(
                attends_work,
                states,
                contact_params.loc[(params_entry, params_entry), "value"],
                condition,
                seed=seed,
            )
    return attends_work


def meet_daily_other_contacts(states, params, seed):
    going = pd.Series(data=1, index=states.index)
    for params_entry, condition in [
        ("reduction_when_symptomatic", "symptomatic"),
        ("reduction_when_positive", IS_POSITIVE_CASE),
    ]:
        going = reduce_recurrent_contacts_on_condition(
            going,
            states,
            params.loc[("other_recurrent", params_entry, params_entry), "value"],
            condition,
            seed=seed,
        )
    return going


def attends_educational_facility(states, params, id_column, seed):
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
    facility = id_column.split("_")[0]

    date = get_date(states)
    day = date.day_name()
    if day in ["Saturday", "Sunday"]:
        attends_facility = pd.Series(data=0, index=states.index)
    else:
        attends_facility = (states[id_column] != -1).astype(int)
        attends_facility = _pupils_having_vacations_do_not_attend(
            attends_facility, states, params
        )
        for params_entry, condition in [
            ("reduction_when_symptomatic", "symptomatic"),
            ("reduction_when_positive", IS_POSITIVE_CASE),
        ]:
            attends_facility = reduce_recurrent_contacts_on_condition(
                attends_facility,
                states,
                params.loc[(facility, params_entry, params_entry), "value"],
                condition,
                seed=seed,
            )
    return attends_facility


def meet_hh_members(states, contact_params, seed):
    """Meet household members.

    As single person households have unique household ids, everyone meets their
    household unless they are symptomatic. In that case the sick household member
    don't meet the others with a certain probability.

    Args:
        states (pandas.DataFrame): The states.
        contact_params (pandas.DataFrame): DataFrame with two category levels,
            subcategory and name. has a "value" column that contains the probabilities
            to the number of possible columns in the "name" index level.

    """
    meet_hh = pd.Series(1, index=states.index)
    for params_entry, condition in [
        ("reduction_when_symptomatic", "symptomatic"),
        ("reduction_when_positive", IS_POSITIVE_CASE),
    ]:
        meet_hh = reduce_recurrent_contacts_on_condition(
            meet_hh,
            states,
            contact_params.loc[(params_entry, params_entry), "value"],
            condition,
            seed=seed,
        )
    return meet_hh


def calculate_non_recurrent_contacts_from_empirical_distribution(
    states, contact_params, on_weekends, seed, query=None
):
    """Draw how many non recurrent contacts each person will have today.

    Args:
        states (pandas.DataFrame): sid states DataFrame.
        contact_params (pandas.DataFrame): DataFrame with two category levels,
            subcategory and name. has a "value" column that contains the probabilities
            to the number of possible columns in the "name" index level.
        on_weekends (bool): whether to meet on weekends or not.
        query (str): query string to identify the subset of individuals to which this
            contact model applies.

    Returns:
        contacts (pandas.Series): index is the same as states. values is the number of
            contacts.

    """
    date = get_date(states)
    contacts = pd.Series(0, index=states.index)

    if not on_weekends and (date.day_name() in ["Saturday", "Sunday"]):
        pass

    else:
        if query is not None:
            is_participating = states.eval(query)
        else:
            is_participating = pd.Series(True, index=states.index)

        distribution = contact_params.query("~subcategory.str.contains('reduction')")[
            "value"
        ]
        contacts[is_participating] = _draw_nr_of_contacts(
            distribution=distribution,
            is_participating=is_participating,
            states=states,
            seed=seed,
        )
        for params_entry, condition in [
            ("reduction_when_symptomatic", "symptomatic"),
            ("reduction_when_positive", IS_POSITIVE_CASE),
        ]:
            contacts = reduce_non_recurrent_contacts_on_condition(
                contacts,
                states,
                contact_params.loc[(params_entry, params_entry), "value"],
                condition,
            )

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


def reduce_non_recurrent_contacts_on_condition(contacts, states, factor, condition):
    """Reduce the number of contacts by *factor* for symptomatic indivdiuals.

    Args:
        contacts (pandas.Series): number of contacts of healthy individuals.
            Index is the same as states.
        states (pandas.DataFrame)
        factor (float): Factor by which the number of contacts are reduced.
            1 means no contacts for sick individuals, 0 means no change in contacts.

    Returns
        contacts (pandas.Series): number of contacts, reduced for symptomatic
            individuals.

    """
    contacts = contacts.copy(deep=True)
    factor_sr = 1 - factor * (states.eval(condition))
    reduced_contacts = factor_sr * contacts
    return reduced_contacts


def reduce_recurrent_contacts_on_condition(contacts, states, share, condition, seed):
    """Reduce contacts for share of population for which condition is fulfilled.

    The subset of contacts for which contacts are reduced is specified by the condition
    and whoever has a positive number of contacts. Then, a share of individuals in the
    subset is sampled and the contacts are set to 0.

    Args:
        contacts (pandas.Series): The series with contacts.
        states (pandas.DataFrame): The states of one day passed by sid.
        share (float): The share of people who will reduce their contacts if the
            condition is true.
        condition (str): Condition which defines the subset of individuals who
            potentially reduce their contacts.
        seed (int)

    """
    contacts = contacts.copy(deep=True)
    np.random.seed(seed)
    is_condition_true = states.eval(condition) & (contacts > 0)
    is_complier = np.random.choice(
        [True, False],
        size=is_condition_true.sum(),
        p=[share, 1 - share],
    )
    complies = is_condition_true.copy(deep=True)
    complies[complies] = is_complier
    contacts.loc[complies] = 0
    return contacts


def reduce_contacts_when_symptomatic_case_among_recurrent_contacts(
    contacts, states, share, group_id, seed
):
    """Reduce contacts when there is a symptomatic case among the recurrent contacts.

    Individuals react only to symptomatic cases among their recurrent contacts because
    it is more likely that they will be notified. (For random contacts, the contact
    tracing app would be helpful.)

    Since this function is computationally demanding, it is not used at the moment.

    """
    np.random.seed(seed)
    contacts = contacts.copy(deep=True)
    has_symptomatic_case = pd.Series(index=states.index, data=False)
    s = (
        states.loc[states.group_id != -1, "symptomatic"]
        .groupby(group_id)
        .transform("any")
    )

    has_symptomatic_case.loc[s.index] = has_symptomatic_case.loc[s.index] | s

    compliers = np.random.choice(
        [True, False], size=has_symptomatic_case.sum(), p=[share, 1 - share]
    )
    contacts.loc[has_symptomatic_case[compliers].index] = 0

    return contacts


def reduce_contacts_when_positive_case_among_recurrent_contacts(
    contacts, states, share, group_id, seed
):
    """Reduce contacts when there is a positive case among the recurrent contacts.

    Individuals react only to positive cases among their recurrent contacts because
    it is more likely that they will be notified. (For random contacts, the contact
    tracing app would be helpful.)

    Since this function is computationally demanding, it is not used at the moment.

    """
    np.random.seed(seed)
    contacts = contacts.copy(deep=True)
    is_positive_case = states.eval(IS_POSITIVE_CASE)
    has_positive_case = pd.Series(index=states.index, data=False)
    valid_group = states.loc[states[group_id] != -1, "group_id"]
    s = is_positive_case.loc[valid_group.index].groupby(valid_group).transform("any")

    has_positive_case.loc[s.index] = has_positive_case.loc[s.index] | s

    compliers = np.random.choice(
        [True, False], size=has_positive_case.sum(), p=[share, 1 - share]
    )
    contacts.loc[has_positive_case[compliers].index] = 0

    return contacts


def _pupils_having_vacations_do_not_attend(attends_facility, states, params):
    """Make pupils stay away from school if their state has vacations."""
    date = get_date(states)
    states_w_vacations = _get_states_w_vacations(date, params)

    attends_facility.loc[attends_facility & states.state.isin(states_w_vacations)] = 0

    return attends_facility


def _get_states_w_vacations(date: pd.Timestamp, params: pd.DataFrame) -> List[str]:
    """Get states which currently have vacations for pupils."""
    vacations = params.filter(like="ferien", axis=0).copy()
    if vacations.empty:
        raise ValueError("'params' does not contain any information about vacations.")
    else:
        # Dates are stored as epochs so that value can be a numeric column.
        vacations["value"] = from_epochs_to_timestamps(vacations["value"])
        vacations = vacations.groupby(vacations.index.names)["value"].first().unstack()
        has_vacations = (vacations["start"] <= date) & (date <= vacations["end"])
        states = (
            vacations.loc[has_vacations].index.get_level_values("subcategory").unique()
        ).tolist()

    return states
