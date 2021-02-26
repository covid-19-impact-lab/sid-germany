import numpy as np
from sid.time import get_date

from src.policies.single_policy_functions import reduce_recurrent_model


def a_b_education(
    states,
    contacts,
    seed,
    group_id_column,
    others_attend,
    hygiene_multiplier,
    subgroup_query=None,
):
    """Implement education with split groups for some children.

    This does not support state specific yet. Once state specific policies
    are supported, subgroup_query, others_attend and the hygiene multiplier
    can be dictionaries where the keys are the names of the German states:
    'Baden-Württemberg', 'Bavaria', 'Berlin', 'Brandenburg', 'Bremen',
    'Hamburg', 'Hessen', 'Lower Saxony', 'Mecklenburg-Vorpommern',
    'North Rhine-Westphalia', 'Rhineland-Palatinate', 'Saarland', 'Saxony',
    'Saxony-Anhalt', 'Schleswig-Holstein' and 'Thuringia' and the values are
    the state specific subgroup_query, others_attend and hygiene multipliers.

    Args:
        group_id_column (str): name of the column identifying which indivdiuals
            attend class together, i.e. the assort by column of the current
            contact model. We assume that the column identifying which
            individuals belong to the A or B group is group_id_column + "_a_b".
        others_attend (bool): if True, children not selected by the subgroup
            query attend school normally. If False, children not selected by
            the subgroup query stay home.
        hygiene_multiplier (float): Applied to all children that still attend
            educational facilities.
        subgroup_query (str, optional): string identifying the children that
            are taught in split classes. If None, all children in all education
            facilities attend in split classes.

    """
    np.random.seed(seed)
    contacts = contacts.copy(deep=True)
    if subgroup_query is None:
        # create a query string that is True for everyone
        subgroup_query = "educ_worker == educ_worker"
    date = get_date(states)

    a_b_children_staying_home = _get_a_b_children_staying_home(
        states=states,
        subgroup_query=subgroup_query,
        group_column=group_id_column + "_a_b",
        date=date,
    )
    contacts[a_b_children_staying_home] = 0

    # ~ not supported for categorical columns which could appear in subgroup_query
    children_not_in_a_b = states.eval(f"~educ_worker & not ({subgroup_query})")
    if not others_attend:
        contacts[children_not_in_a_b] = 0

    # since our educ models are all recurrent and educ_workers must always attend
    # we only apply the hygiene multiplier to the students
    contacts[~states["educ_worker"]] = reduce_recurrent_model(
        states[~states["educ_worker"]],
        contacts[~states["educ_worker"]],
        seed,
        hygiene_multiplier,
    )

    # educ_workers of classes with 0 participants don't go to school
    size_0_classes = _find_size_zero_classes(contacts, states, group_id_column)
    has_no_students = states.query("educ_worker")[group_id_column].isin(size_0_classes)
    teachers_with_0_students = states.query("educ_worker")[has_no_students].index
    contacts[teachers_with_0_students] = 0

    return contacts


def _get_a_b_children_staying_home(states, subgroup_query, group_column, date):
    a_b_children = states.eval(f"~educ_worker & ({subgroup_query})")
    in_attend_group = states[group_column] == date.week % 2
    a_b_children_staying_home = a_b_children & ~in_attend_group
    return a_b_children_staying_home


def _find_size_zero_classes(contacts, states, col):
    students_group_ids = states[~states["educ_worker"]][col]
    students_contacts = contacts[~states["educ_worker"]]
    class_sizes = students_contacts.groupby(students_group_ids).sum().drop(-1)
    size_zero_classes = class_sizes[class_sizes == 0].index
    return size_zero_classes
