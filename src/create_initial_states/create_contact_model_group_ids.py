import itertools as it

import numpy as np

from src.create_initial_states.add_weekly_ids import add_weekly_ids
from src.create_initial_states.make_educ_group_columns import make_educ_group_columns
from src.shared import create_groups_from_dist


def add_contact_model_group_ids(
    df, work_daily_dist, work_weekly_dist, other_daily_dist, other_weekly_dist, seed
):
    """Create and add contact model group ids to the states.

    For children between 6 and 18 we assume everyone who is not working goes to a school
    class with 23 individuals of the same age. Classes meet their peers and 3 pairs of
    two teachers per school day.

    For children between 3 and 6 we assume that everyone (officially 93%) attends
    pre-school with a group size of 9 and mixed ages. Mean group sizes vary between 7
    and 13 between German states. Every preschool group meets its peers and
    the same two adults every school day.

    For children below 3 we assume that 35% attend a nursery with a group size of 4 and
    mixed ages. This hides substantial heterogeneity in attendance with respect to age
    (older children are much more likely to be in day care) and with respect to state
    (East German children are much more likely to attend nurseries). Mean group sizes
    vary between 3 and 6 between German states. Every group meets their peers and
    one adult every school day.

    Sources:

    - percent of children in day care: https://tinyurl.com/yefhpney
    - percent of children below age 3 in day care: https://tinyurl.com/y6zft8sx
    - group sizes in early education: https://tinyurl.com/y6tjju6r
    - group sizes for school classes: source: https://tinyurl.com/y3xrgxaw
        (21.2 for primaries, 24.5 for secondary)


    Args:
        df (pandas.DataFrame): the states DataFrame
        work_daily_dist (pandas.Series): share of workers reporting a certain
            number of daily repeating contacts. Index is the number of contacts,
            the values are the shares.
        work_weekly_dist (pandas.Series): same as work_daily_dist but with
            weekly repeating contacts.
        other_daily_dist (pandas.Series): share of individuals reportang a
            certain number of daily repeating contacts. Index is the number of
            contacts. The values are the shares.
        other_weekly_dist (pandas.Series): same as other_daily_dist but with
            weekly repeating contacts.
        seed (int)

    Returns:
        df (pandas.DataFrame): states with an updated occupation column and
            expanded by the following contact model group ids:
                -  "school_group_id_0"

    """
    seed = it.count(seed)
    df = df.copy(deep=True)

    school_class_ids, updated_occupation = make_educ_group_columns(
        states=df,
        query="occupation == 'school' & age >= 6",
        group_size=23,
        strict_assort_by=["state", "age"],
        weak_assort_by=["county"],
        adults_per_group=2,
        n_contact_models=3,
        column_prefix="school_group_id",
        occupation_name="school_teacher",
        seed=next(seed),
    )
    df = df.merge(school_class_ids, left_index=True, right_index=True, validate="1:1")
    df["occupation"] = updated_occupation

    preschool_class_ids, updated_occupation = make_educ_group_columns(
        states=df,
        query="3 <= age < 6",
        group_size=9,
        strict_assort_by=["state"],
        weak_assort_by=["county"],
        adults_per_group=2,
        n_contact_models=1,
        column_prefix="preschool_group_id",
        occupation_name="preschool_teacher",
        seed=next(seed),
    )
    df = df.merge(
        preschool_class_ids, left_index=True, right_index=True, validate="1:1"
    )
    df["occupation"] = updated_occupation

    df["attends_nursery"] = _draw_who_attends_nursery(df, next(seed))
    nursery_class_ids, updated_occupation = make_educ_group_columns(
        states=df,
        query="attends_nursery & (age < 3)",
        group_size=4,
        strict_assort_by=["state"],
        weak_assort_by=["county"],
        adults_per_group=1,
        n_contact_models=1,
        column_prefix="nursery_group_id",
        occupation_name="nursery_teacher",
        seed=next(seed),
    )
    df = df.merge(nursery_class_ids, left_index=True, right_index=True, validate="1:1")
    df["occupation"] = updated_occupation

    work_daily_group_sizes = work_daily_dist.copy(deep=True)
    work_daily_group_sizes.index += 1
    df["work_daily_group_id"] = create_groups_from_dist(
        initial_states=df,
        group_distribution=work_daily_group_sizes,
        query="occupation == 'working'",
        assort_bys=["county"],
        seed=next(seed),
    )

    other_daily_group_sizes = other_daily_dist.copy(deep=True)
    other_daily_group_sizes.index += 1
    df["other_daily_group_id"] = create_groups_from_dist(
        initial_states=df,
        group_distribution=other_daily_group_sizes,
        query=None,
        assort_bys=["county", "age_group"],
        seed=next(seed),
    )

    weekly_work_ids = add_weekly_ids(
        states=df,
        weekly_dist=work_weekly_dist,
        query="occupation =='working'",
        seed=9958,
        col_prefix="work_weekly_group_id",
        county_assortativeness=0.8,
    )

    df = df.merge(
        right=weekly_work_ids,
        left_index=True,
        right_index=True,
        validate="1:1",
    )

    weekly_other_ids = add_weekly_ids(
        states=df,
        weekly_dist=other_weekly_dist,
        seed=4748,
        query=None,
        col_prefix="other_weekly_group_id",
        county_assortativeness=0.8,
    )

    df = df.merge(
        right=weekly_other_ids,
        left_index=True,
        right_index=True,
        validate="1:1",
    )

    return df


def _draw_who_attends_nursery(df, seed):
    np.random.seed(seed)
    below_3_yos = df.query("age < 3").index
    share_of_children_in_nursery = 0.35
    n_kids_to_draw = int(share_of_children_in_nursery * len(below_3_yos))
    attend_nursery_indices = np.random.choice(
        below_3_yos, size=n_kids_to_draw, replace=False
    )
    attends_nursery = df.index.isin(attend_nursery_indices)
    return attends_nursery
