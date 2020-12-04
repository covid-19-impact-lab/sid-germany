import itertools as it

import pandas as pd

from src.create_initial_states.add_weekly_ids import add_weekly_ids
from src.shared import create_groups_from_dist


def create_contact_model_group_ids(
    df, work_daily_dist, work_weekly_dist, other_daily_dist, other_weekly_dist, seed
):
    seed = it.count(seed)
    group_ids = pd.DataFrame(index=df.index)

    # FIRST TEACHER AND STUDENTS
    # -> new occupation column!

    work_daily_group_sizes = work_daily_dist.copy(deep=True)
    work_daily_group_sizes.index += 1
    group_ids["work_daily_group_id"] = create_groups_from_dist(
        initial_states=df,
        group_distribution=work_daily_group_sizes,
        query="occupation == 'working'",
        assort_bys=["county"],
        seed=next(seed),
    )

    other_daily_group_sizes = other_daily_dist.copy(deep=True)
    other_daily_group_sizes.index += 1
    group_ids["other_daily_group_id"] = create_groups_from_dist(
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

    group_ids = pd.merge(
        left=group_ids,
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

    group_ids = pd.merge(
        left=group_ids,
        right=weekly_other_ids,
        left_index=True,
        right_index=True,
        validate="1:1",
    )

    return group_ids
