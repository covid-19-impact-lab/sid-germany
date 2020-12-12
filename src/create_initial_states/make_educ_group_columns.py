import numpy as np
import pandas as pd


def make_educ_group_columns(
    states,
    query,
    group_size,
    strict_assort_by,
    weak_assort_by,
    adults_per_group,
    n_contact_models,
    column_prefix,
    occupation_name,
    seed,
):
    """Generate contact model columns for education contacts.

    This generates raw group ids using create_balanced_group_column. It then
    replicates this column n_contact_models times and mixes in some individuals
    for which occupation == "working" (e.g. to simulate teachers).

    Args:
        states (pandas.DataFrame): DataFrame with background variables, including
            all assort_by variables.
        query (str): Query that selects which individuals are part of a group.
        group_size (int): Target group size that will be achieved approximately.
        strict_assort_by (list or str): Groups only contain individuals that have
            the same value in all ``strict_assort_by`` variables.
        weak_assort_by (list or str): Individuals that have the same value in all
            weak_assort_by variables are more likely to be matched into one group.
            Adults are taken from the modal weak group.
        adults_per_group (int): Number of teachers added to each class.
        n_contact_models (int): Number of contact models for which group ids
            are generated. This is also the average number of classes each
            teacher teaches.
        column_prefix (str): Prefix for column names.
        occupation_name (str): Value to which the
        seed (int): Random seed.

    Returns:
        pd.DataFrame: The generated id columns. Column names are
            f"{prefix}_{number}" where number counts the contact models.
        pd.Series: Modified occupation column where "working" was changed
            to occupation_name in some cases.

    """
    # create raw id column (without adults)
    np.random.seed(seed)
    states = states.copy()
    raw_id = create_balanced_group_column(
        states=states,
        query=query,
        group_size=group_size,
        strict_assort_by=strict_assort_by,
        weak_assort_by=weak_assort_by,
    )

    # create helpers
    states["__weak_group_id"], _ = pd.factorize(
        states.groupby(weak_assort_by).grouper.group_info[0]
    )
    participants, non_participants = _split_data_by_query(states, query)
    id_to_weak_group = _get_id_to_weak_group(participants, raw_id)

    # initialize results
    occupation = states["occupation"].copy().cat.add_categories(occupation_name)
    id_cols = pd.DataFrame()
    for col in [f"{column_prefix}_{i}" for i in range(n_contact_models)]:
        id_cols[col] = raw_id.copy(deep=True)

    # modify results
    for weak_group, indices in states.groupby("__weak_group_id").groups.items():
        reduced_states = states.loc[indices].copy()
        group_ids = id_to_weak_group[id_to_weak_group == weak_group].index.tolist()
        n_groups = len(group_ids)
        # skip groups with no participants
        if n_groups > 0:
            n_adults = n_groups * adults_per_group
            candidate_query = "(occupation == 'working') & (25 <= age <= 68)"
            adult_candidates = reduced_states.query(candidate_query).index
            adults = np.random.choice(adult_candidates, size=n_adults, replace=False)
            occupation.loc[adults] = occupation_name
            for contact_model in id_cols.columns:
                urn = np.array(group_ids * adults_per_group)
                id_cols.loc[adults, contact_model] = np.random.choice(
                    urn, size=n_adults, replace=False
                )

    id_cols = id_cols.astype(int)
    return id_cols, occupation


def create_balanced_group_column(
    states, query, group_size, strict_assort_by, weak_assort_by
):
    """Create a group id for a recurrent contact model with equally sized groups.

    This is a low level function that will probably rather be called via
    get_educ_group_column.

    When reading the code it is helpful to distinguish four types of groups of
    individuals:
    1. The group whose ID column we want to generate, called just "group"
    2. The groups induced by the strict_assort_by variables, called "strong_group"
    3. The groups induced by the weak_assort_by_variables, called "weak_group"
    4. Participants and non participants. Participants are those selected by query

    The algorithm is deterministic but might depend on the order of states.

    Args:
        states (pandas.DataFrame): DataFrame with background variables, including
            all assort_by variables.
        query (str): Query that selects which individuals are part of a group.
        group_size (int): Target group size that will be achieved approximately.
        strict_assort_by (list or str): Groups only contain individuals that have
            the same value in all ``strict_assort_by`` variables.
        weak_assort_by (list or sttr): Individuals that have the same value in all
            weak_assort_by variables are more likely to be matched into one group.

    Returns:
        pandas.Series: The group_id with same index as states.

    """
    states = states.copy(deep=True).reset_index()
    participants, non_participants = _split_data_by_query(states, query)
    id_participants = _create_group_id_for_participants(
        participants, group_size, strict_assort_by, weak_assort_by
    )
    id_non_participants = _create_group_id_for_non_participants(
        non_participants, id_participants.max() + 1
    )
    # sorting brings this in same order as states because we reset the index above
    group_id = pd.concat([id_participants, id_non_participants]).sort_index()
    group_id.index = states.index
    group_id = group_id.astype(int)
    return group_id


def _get_id_to_weak_group(participants, raw_id):
    """Create a mapping from groups to weak_assort_by groups

    This is not a unique mapping since each group can have members from
    multiple weak assort by groups. We make it unique by just assigning
    the weak_assort_by group of the first group member to the whole
    group.

    Args:
        participants (pandas.DataFrame): DataFrame of participating individuals.
            It has to have the "__weak_group_id" column.
        raw_id (pandas.Series): column giving the groups which are to be mapped
            to __weak_group_ids.

    Returns:
        id_to_weak_group (pandas.Series): the index are the group ids in
            participants, the values are the first weak group ids of each group.

    """
    participants = participants.copy()
    participants["__raw_id"] = raw_id.loc[participants.index]
    id_to_weak_group = participants.groupby("__raw_id")["__weak_group_id"].first()
    return id_to_weak_group


def _split_data_by_query(df, query):
    """Split data into those selected by query and the rest."""
    locs = df.query(query).index
    boolean = pd.Series(False, index=df.index)
    boolean[locs] = True
    selected = df[boolean].copy(deep=True)
    others = df[~boolean].copy(deep=True)
    return selected, others


def _create_group_id_for_participants(df, group_size, strict_assort_by, weak_assort_by):
    """Create the group id for those selected by query.

    The main work is done in _create_group_id_for_one_strict_assort_by_group.

    """
    df = df.copy(deep=True)
    to_concat = []
    max_id = 0
    for _, indices in df.groupby(strict_assort_by).groups.items():
        id_col, max_id = _create_group_id_for_one_strict_assort_by_group(
            df=df.loc[indices],
            group_size=group_size,
            weak_assort_by=weak_assort_by,
            start_id=max_id + 1,
        )
        to_concat.append(id_col)

    return pd.concat(to_concat)


def _determine_group_sizes(target_size, population_size):
    """Calculate group sizes given a target size and a population size.

    Args:
        target_size (int): Target group size
        population_size (int): Number of people that are split into groups.


    Returns:
        list: List of integers. The length is the number of groups. The entries
            are the group sizes. Not all groups have the same size but they differ
            at most by one.

    """
    number = max(1, int(np.round(population_size / target_size, 0)))
    small_size = int(np.floor(population_size / number))
    large_size = small_size + 1
    n_large_classes = population_size % number
    n_small_classes = number - n_large_classes
    sizes = [large_size] * n_large_classes + [small_size] * n_small_classes
    assert np.sum(sizes) == population_size
    return sizes


def _create_group_id_for_one_strict_assort_by_group(
    df, group_size, weak_assort_by, start_id
):
    """Create group id for all people of the same strict_assort_by group.

    To make matching as assortative as possible with respect to the
    weak_assort_by variables, for each group we first try to fill it with
    members of only one group (i.e. we start with the largest remaining
    weak_assort_by_group). If this is not enough, we fill the
    group by members of the smallest remaining weak_assort_by group.

    Args:
        df (pandas.DataFrame): DataFrame that only contains people from one
            strict_assort_by_group.
        group_size (int): The target group size.
        weak_assort_by (str or list): Variable or list of variables according to which
            group matching should be assortative.
        start_id (int): The id of the first group.

    Returns:
        pd.Series: The index is the same as df. The values are the group_ids.

    """
    sizes = _determine_group_sizes(group_size, len(df))
    df = df.copy()
    # factorize is necessary to start counting at zero even if categoricals with
    # unused categories are among the weak_assort_by variables.
    df["__weak_group_id"], _ = pd.factorize(
        df.groupby(weak_assort_by).grouper.group_info[0]
    )
    weak_group_indices = {
        i: list(val) for i, val in df.groupby("__weak_group_id").groups.items()
    }

    id_to_indices = {}
    for i, size in enumerate(sizes):
        group_id = i + start_id
        largest = _get_key_with_longest_value(weak_group_indices)
        if len(weak_group_indices[largest]) > size:
            id_to_indices[group_id] = weak_group_indices[largest][:size]
            weak_group_indices[largest] = weak_group_indices[largest][size:]
        elif len(weak_group_indices[largest]) == size:
            id_to_indices[group_id] = weak_group_indices[largest]
            del weak_group_indices[largest]
        else:
            indices = weak_group_indices.pop(largest)
            rest_size = size - len(indices)
            while len(indices) < size and weak_group_indices:
                smallest = _get_key_with_shortest_value(weak_group_indices)
                if len(weak_group_indices[smallest]) > rest_size:
                    indices += weak_group_indices[smallest][:rest_size]
                    weak_group_indices[smallest] = weak_group_indices[smallest][
                        rest_size:
                    ]
                else:
                    indices += weak_group_indices[smallest]
                    rest_size = size - len(indices)
                    del weak_group_indices[smallest]
            id_to_indices[group_id] = indices

    for id_, indices in id_to_indices.items():
        df.loc[indices, "group_id"] = id_

    return df["group_id"], start_id + len(sizes)


def _get_key_with_longest_value(dict_):
    """Get the key from dict_ that has the longest value."""
    sorted_keys = sorted(dict_, key=lambda k: len(dict_[k]))
    return sorted_keys[-1]


def _get_key_with_shortest_value(dict_):
    """Get the key from dict_ that has the shortest value"""
    sorted_keys = sorted(dict_, key=lambda k: len(dict_[k]))
    return sorted_keys[0]


def _create_group_id_for_non_participants(df, start_id):
    """Create group_id for those not selected by query."""
    return pd.Series(np.arange(start_id, start_id + len(df)), index=df.index)
