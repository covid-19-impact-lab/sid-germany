import itertools

import numpy as np
import pandas as pd

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
    vary between 3 and 6 between German states. Every group meets their peers and one
    adult every school day.

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
            expanded by the contact model group ids and helper columns:
                - school_group_id_0
                - school_group_id_1
                - school_group_id_2
                - preschool_group_id_0
                - nursery_group_id_0
                - updated occupation column
                - educ_worker
                - work_contact_priority
                - stays_home_when_schools_close

    """
    seed = itertools.count(seed)
    df = df.copy(deep=True)

    df = _add_educ_group_ids(df, seed)

    df["work_contact_priority"] = _draw_work_contact_priority(
        df["occupation"], next(seed)
    )
    df = _add_work_group_ids(df, work_daily_dist, work_weekly_dist, seed)

    df = _add_other_group_ids(df, other_daily_dist, other_weekly_dist, seed)

    hh_sizes = df.groupby("hh_id")["one"].transform("size")
    df["hh_model_group_id"] = (
        df["hh_id"].astype(int).where(hh_sizes > 1, -1).astype("category")
    )

    educ_group_cols = [
        "school_group_id_0",
        "school_group_id_1",
        "school_group_id_2",
        "preschool_group_id_0",
        "nursery_group_id_0",
    ]
    df[educ_group_cols] = df[educ_group_cols].astype("category")

    # the stays_home_when_schools_close variable is not used yet.
    df[
        "stays_home_when_schools_close"
    ] = _identify_adult_staying_home_when_schools_close(df)

    df = df.drop(columns=["pos_in_group", "one", "group_size"])
    return df


def _add_educ_group_ids(df, seed):
    df = df.copy(deep=True)
    school_class_ids, updated_occupation = make_educ_group_columns(
        states=df,
        query="occupation == 'school'",
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

    df["one"] = 1
    gb = df.groupby("school_group_id_0")
    df["pos_in_group"] = gb["one"].cumsum() - 1
    df["group_size"] = gb["one"].transform("size")
    df["educ_a_b_identifier"] = df.eval("pos_in_group < group_size / 2")

    preschool_class_ids, updated_occupation = make_educ_group_columns(
        states=df,
        query="occupation == 'preschool'",
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

    nursery_class_ids, updated_occupation = make_educ_group_columns(
        states=df,
        query="occupation == 'nursery'",
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
    df["educ_worker"] = df["occupation"].str.endswith("_teacher")
    return df


def _draw_work_contact_priority(occupation, seed):
    np.random.seed(seed)

    is_worker = occupation == "working"
    values = np.random.uniform(low=0, high=1, size=len(occupation))
    work_contact_priority = pd.Series(
        values, index=occupation.index, name="work_contact_priority"
    )
    work_contact_priority = work_contact_priority.where(is_worker, other=-1)
    return work_contact_priority


def _sample_household_groups(df, seed, assort_by, same_group_probability=None, n_hhs=3):
    """Put groups of households together into groups.

    Args:
        df (pandas.DataFrame): states DataFrame
        seed (int)
        assort_by (str, optional): variable on which to assort by
        assortativeness (float, optional): values of assortativeness
        n_hhs (int): Number of households to group together.

    Returns:
        id_col (pandas.Series): Series with the same index as df.
            Individuals of households that were grouped together
            have the same value. The dtype is categorical.

    """
    if assort_by is not None:
        assert (
            0 <= same_group_probability <= 1
        ), "same_group_probability must be a between 0 and 1."
    seed = itertools.count(seed)

    hh_ids = df.query("private_hh")["hh_id"].unique().astype(float)
    if assort_by is None:
        id_col = pd.Series(np.nan, index=df.index)
        # this is inplace
        np.random.shuffle(hh_ids)

        if not len(hh_ids) % n_hhs == 0:
            to_append = [np.nan] * (n_hhs - len(hh_ids) % n_hhs)
            hh_ids = np.append(hh_ids, to_append)

        target_shape = (int(len(hh_ids) / n_hhs), n_hhs)
        grouped_hh_ids = np.reshape(hh_ids, target_shape)
        for i, group_hhs in enumerate(grouped_hh_ids):
            selection = df["hh_id"].isin(group_hhs)
            id_col[selection] = i

    else:
        id_counter = 0
        result = pd.Series(np.nan, index=hh_ids, name="new_group_id")

        group_to_hh_ids = df.query("private_hh").groupby(assort_by)["hh_id"].unique()
        group_to_hh_ids = {
            group: hh_ids.tolist() for group, hh_ids in group_to_hh_ids.items()
        }
        # shuffle is inplace
        for hh_ids in group_to_hh_ids.values():
            np.random.shuffle(hh_ids)

        group_to_hh_ids = {
            group: iter(hh_ids) for group, hh_ids in group_to_hh_ids.items()
        }

        all_groups = sorted(group_to_hh_ids.keys())

        for i, group in enumerate(all_groups):
            hh_ids = group_to_hh_ids[group]
            if i + 1 < len(all_groups):
                groups_to_choose_from = all_groups[i:]
                n_other_groups = len(groups_to_choose_from) - 1
                other_prob = (1 - same_group_probability) / n_other_groups
                group_probs = [same_group_probability] + [other_prob] * n_other_groups
            else:
                groups_to_choose_from = [group]
                group_probs = [1]
            for base_hh in hh_ids:
                result[base_hh] = id_counter
                for _ in range(n_hhs - 1):
                    other_hh = None
                    # to avoid an endless loop
                    tries = 0
                    while other_hh is None and tries < 20:
                        other_group = np.random.choice(
                            groups_to_choose_from, p=group_probs
                        )
                        other_hh = next(group_to_hh_ids[other_group], None)
                        tries += 1
                    # this handles if we could not find another hh to match
                    if other_hh is not None:
                        result[other_hh] = id_counter
                id_counter += 1

        expanded = pd.merge(
            df[["hh_id"]], result, left_on="hh_id", right_index=True, how="left"
        )
        id_col = expanded["new_group_id"]

    id_col = id_col.where(df["private_hh"], -1)
    assert id_col.notnull().all(), "NaN remain in the group id column."
    id_col = id_col.astype("category")

    return id_col


def _add_work_group_ids(df, work_daily_dist, work_weekly_dist, seed):
    df = df.copy(deep=True)

    work_daily_group_sizes = work_daily_dist.copy(deep=True)
    work_daily_group_sizes.index += 1
    df["work_daily_group_id"] = create_groups_from_dist(
        initial_states=df,
        group_distribution=work_daily_group_sizes,
        query="occupation == 'working'",
        assort_bys=["county"],
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

    cols_with_non_parquet_compatible_categories = [
        "work_daily_group_id"
    ] + weekly_work_ids.columns.tolist()
    for col in cols_with_non_parquet_compatible_categories:
        simplified = pd.Series(pd.factorize(df[col])[0], index=df.index)
        # -1 has a special meaning so it needs to remain
        df[col] = simplified.where(df[col] != -1, -1)
        df[col] = df[col].astype("category")

    return df


def _add_other_group_ids(df, other_daily_dist, other_weekly_dist, seed):
    other_daily_group_sizes = other_daily_dist.copy(deep=True)
    other_daily_group_sizes.index += 1
    df["other_daily_group_id"] = create_groups_from_dist(
        initial_states=df,
        group_distribution=other_daily_group_sizes,
        query=None,
        assort_bys=["county", "age_group"],
        seed=next(seed),
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

    cols_with_non_parquet_compatible_categories = [
        "other_daily_group_id"
    ] + weekly_other_ids.columns.tolist()
    for col in cols_with_non_parquet_compatible_categories:
        simplified = pd.Series(pd.factorize(df[col])[0], index=df.index)
        # -1 has a special meaning so it needs to remain
        df[col] = simplified.where(df[col] != -1, -1)
        df[col] = df[col].astype("category")
    return df


def _identify_adult_staying_home_when_schools_close(df):
    df = df.copy(deep=True)

    df["is_under_14"] = df["age"] < 14
    hh_has_young_child = df.groupby("hh_id")["is_under_14"].transform(np.any)

    not_working = df["occupation"].isin(["retired", "stays_home"])
    df["non_working_adult"] = not_working & (df["age"] > 18)
    has_non_working_adult = df.groupby("hh_id")["non_working_adult"].transform(np.any)

    oldest_adult_in_hh = df["age"] == df.groupby("hh_id")["age"].transform("max")
    oldest_in_hh_with_kids = oldest_adult_in_hh & hh_has_young_child
    stays_home_when_schools_close = oldest_in_hh_with_kids & ~has_non_working_adult

    return stays_home_when_schools_close
