import itertools

import numpy as np
import pandas as pd
from sid.contacts import _sum_preserving_round


# ---------------------------------- Contact Models ----------------------------------


def draw_groups(df, group_name, query, assort_bys, n_per_group):
    """Assign individuals to random groups based on their characteristics.

    Args:
        df (pandas.DataFrame): sid states DataFrame
        group_name (str): name of the column to be created
        query (str): identify who gets a group. All others are assigned -1. Make sure
            your contact model assigns these people a 0 so they do not meet.
        assort_bys (list): columns by which to group individuals, such that in every
            group people share all characteristics in the assort_by variables.
        n_per_group (int): number of people per group.

    Returns:
        drawn_groups (pandas.Series): Series with the group ids.
            It has the same index as **df**. It's -1 for individuals without a group.

    """
    counter = itertools.count()
    drawn_groups = pd.Series(-1, index=df.index, name=group_name)
    grouped_people_to_get_group = df.query(query).groupby(assort_bys, as_index=False)
    for _, indices in grouped_people_to_get_group.groups.items():
        drawn_groups[indices] = _create_groups(df.loc[indices], counter, n_per_group)

    drawn_groups = pd.Categorical(
        values=drawn_groups, categories=drawn_groups.unique(), ordered=False
    )
    return drawn_groups


def _create_groups(df, counter, n_per_group):
    n = len(df)

    n_groupes = int(np.ceil(n / n_per_group))
    group_ids = [next(counter) for _ in range(n_groupes)]

    groups = np.random.choice(group_ids, size=n, replace=True)

    return groups


def create_groups_from_dist(
    initial_states, group_name, group_distribution, query, assort_bys
):
    """Assign individuals to random groups to match a group size distribution.

    Notes:
        - This could be made faster by not creating single member groups.
        - Group assignment is completely random (within each assort_by value
            combination). This means there is either perfect or zero assortativeness
            in the contacts with respect to characteristics. E.g. if age is not given
            in the assort_bys, groups are completely randomly assigned with respect to
            age. On the other hand if age is in the assort_bys all members of each group
            have the exact same age.

    Args:
        initial_states (pandas.DataFrame): SID initial states DataFrame.
        group_name (str): name of the Series to be created.
        group_distribution (pandas.Series): the index is the support of the group sizes,
            the values is the share of the group size we are aiming for.
        query (str): query string to identify the subpopulation for which we want to
            create group ids. Note that group_distribution must describe the
            distribution of group sizes in this subpopulation.
        assort_bys (list): columns by which to group individuals, such that in every
            group people share all charackteristics in the assort_by variables.

    Returns:
        group_sr (pandas.Series): index is the same as the initial_states. Values are
            identifiers (strings) of each group.

    """
    assert 0 not in group_distribution.index, "Group sizes must be greater than 0."
    df = initial_states.query(query)
    group_sr = pd.Series(-1, index=initial_states.index, name=group_name)
    size_sr = pd.Series(-1, index=df.index, name="group_size")

    grouped_people_to_get_group = df.groupby(assort_bys, as_index=False).groups

    for assort_by_vals, indices in grouped_people_to_get_group.items():
        nr_of_groups = _determine_number_of_groups(len(indices), group_distribution)
        ids = _create_group_ids(nr_of_groups, assort_by_vals)
        ids = np.random.choice(ids, size=len(ids), replace=False)
        ids = _expand_or_contract_ids(ids, len(indices), assort_by_vals)
        group_sr[indices] = ids
        id_to_size = pd.Series(ids).value_counts()
        size_sr[indices] = [id_to_size[x] for x in ids]

    _check_created_groups(
        group_sr.loc[df.index], size_sr.loc[df.index], group_distribution
    )
    group_sr = group_sr.astype("category")

    return group_sr


def _determine_number_of_groups(nobs, dist):
    nr_of_inds_per_group = _sum_preserving_round(nobs * dist.to_numpy())
    exact_nr_of_groups = nr_of_inds_per_group / dist.index
    rounded_nr_of_groups = _sum_preserving_round(exact_nr_of_groups.to_numpy()).astype(
        int
    )
    nr_of_groups = pd.Series(rounded_nr_of_groups, index=dist.index)
    return nr_of_groups


def _create_group_ids(nr_of_groups, assort_by_vals):
    ids = []
    for size, nr_groups in nr_of_groups.items():
        for id_ in range(nr_groups):
            ids += [f"{assort_by_vals}_{size}_{id_}"] * size
    return ids


def _expand_or_contract_ids(ids, nobs, assort_by_vals):
    nr_to_add = nobs - len(ids)
    if nr_to_add < 0:
        # purposefully drop people in single work groups
        ids = ids[-nr_to_add:]
    elif nr_to_add > 0:
        ids = np.concatenate([ids, [f"{assort_by_vals}_{nr_to_add}_rest"] * nr_to_add])
    return ids


def _check_created_groups(group_sr, size_sr, group_distribution):
    assert (group_sr != -1).all(), "Did not add a group for every individual"
    resulting_dist = size_sr.value_counts(normalize=True).sort_index()
    resulting_dist.name = "actual_size"
    to_compare = pd.concat([group_distribution, resulting_dist], axis=1)
    # resulting_dist takes some values that dist does not.
    # Comparing the cdfs allows us to compare them more easily
    cdfs = to_compare.fillna(0).cumsum()
    assert np.abs(cdfs["actual_size"] - cdfs[group_distribution.name]).max() < 0.01, (
        "Difference between target and actual distribution too large."
        + cdfs.to_string()
    )


def format_thousands_with_comma(value, pos):  # noqa: U100
    return f"{value:,.0f}"


def draw_from_distribution_for_subset(states, distribution, query, outside_val):
    """Draw for all workers from a distribution how many they are going to meet.

    Args:
        states (pandas.DataFrame): sid states DataFrame
        distribution (pandas.Series): index is the support, values are the
            probabilities.
        query (str): query to identify the subset of the states for which to
            draw the number of contacts.
        outside_val: value the output Series should draw for indivdiuals who
            do not fulfill the query condition.

    Returns:
        contacts (pandas.Series): index is the same as states,
            the values are outside_val for anyone outside the query and
            drawn from the distribution for the rest.
    """
    contacts = pd.Series(outside_val, index=states.index)
    to_pair = states.query(query).index

    contacts[to_pair] = np.random.choice(
        a=distribution.index, size=len(to_pair), p=distribution
    )
    return contacts


def create_age_groups(age_sr):
    bins = list(range(0, 81, 10)) + [100]
    labels = [f"{i}-{i + 9}" for i in range(0, 71, 10)] + ["80-100"]
    return pd.cut(age_sr, bins=bins, right=False, labels=labels)


def create_age_groups_rki(df):
    intervals = pd.IntervalIndex.from_tuples(
        [(0, 4), (5, 14), (15, 34), (35, 59), (60, 79), (80, 100)], closed="both"
    )
    age_groups = pd.cut(df["age"], intervals)
    age_groups = relabel_age_groups_rki_for_parquet(age_groups)
    return age_groups


def relabel_age_groups_rki_for_parquet(sr):
    def convert_interval_to_string(interval):
        return f"{interval.left}-{interval.right}"

    new_sr = sr.cat.rename_categories(convert_interval_to_string)
    return new_sr


def from_timestamps_to_epochs(timestamps):
    """Convert timestamps to epochs.

    https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html
    #from-timestamps-to-epoch

    """
    return (timestamps - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")


def from_epochs_to_timestamps(epochs):
    """Convert epochs to timestamps.

    https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html
    #epoch-timestamps

    """
    return pd.to_datetime(epochs, unit="s")


def load_dataset(path):
    """Infer data type from suffix and load the data with pandas."""
    if path.suffix == ".csv":
        df = pd.read_csv(path, low_memory=False)
    elif path.suffix == ".pkl":
        df = pd.read_pickle(path)
    elif path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unknown suffix for {path}")
    return df
