import itertools as it

import numba as nb
import numpy as np
import pandas as pd
from sid.contacts import boolean_choice
from sid.contacts import choose_other_group
from sid.contacts import choose_other_individual
from sid.contacts import create_group_indexer
from sid.matching_probabilities import create_group_transition_probs
from sid.shared import factorize_assortative_variables

from src.shared import draw_from_distribution_for_subset


def add_weekly_ids(
    states, weekly_dist, seed, query, col_prefix, county_assortativeness
):
    """Add a column for every possible weekly contact.

    We draw from the number of weekly contacts distribution in how
    many weekly contact models a person participates and then randomly
    choose in which column she'll be paired with someone. Lastly, for each
    column we match people who participate in the respective contact model
    with a specified geographic assortativeness.

    Args:
        states (pandas.DataFrame): sid states DataFrame
        weekly_dist (pandas.Series): the index is the support of the
            number of weekly contacts that are possible, the values are
            the frequencies in the synthetic population we aim for of each
            number of weekly contacts. One pair id column will be created
            for every possible contact.
        seed (int): seed.
        query (str): query which subset of the population participates.
            If None, everyone is grouped. (e.g. "occupation =='working'")
        col_prefix (str): prefix for the columns to be created.
        county_assortativeness (float): share of weekly contacts that
            should belong to the same county.

    Returns:
        states (pandas.DataFrame): sit states DataFrame with additional columns
            specifying the weekly work group pairs.

    """
    seed = it.count(seed)
    weekly_ids = pd.DataFrame(index=states.index)
    max_contacts = weekly_dist.index.max()

    for i in range(max_contacts):
        weekly_ids[f"{col_prefix}_{i}"] = -1

    nr_of_weekly_contacts = draw_from_distribution_for_subset(
        states=states,
        distribution=weekly_dist,
        query=query,
        seed=next(seed),
        outside_val=0,
    ).to_numpy()

    weekly_id_cols = []
    for i in range(max_contacts):
        col_name = f"{col_prefix}_{i}"
        weekly_ids[col_name] = np.nan
        weekly_id_cols.append(col_name)

    weekly_ids[weekly_id_cols] = _create_pairs(
        states=states,
        nr_of_weekly_contacts=nr_of_weekly_contacts,
        county_assortativeness=county_assortativeness,
        seed=next(seed),
    )
    return weekly_ids


def _create_pairs(states, nr_of_weekly_contacts, county_assortativeness, seed):
    group_codes_per_individual, _ = factorize_assortative_variables(states, ["county"])
    indexer = create_group_indexer(states, ["county"])
    fake_params = pd.DataFrame(
        data=county_assortativeness,
        columns=["value"],
        index=pd.MultiIndex.from_tuples(
            [("assortative_matching", "fake_model", "county")]
        ),
    )
    model_name = "fake_model"
    first_stage_cum_probs = create_group_transition_probs(
        states, ["county"], fake_params, model_name
    )
    to_match = _create_participation_array(nr_of_weekly_contacts, seed=seed + 1)
    pair_array = _create_pairs_numba(
        to_match=to_match,
        indexer=indexer,
        first_stage_cum_probs=first_stage_cum_probs,
        group_codes_per_individual=group_codes_per_individual,
        seed=seed,
    )
    return pair_array


@nb.njit
def _create_pairs_numba(
    to_match, indexer, first_stage_cum_probs, group_codes_per_individual, seed
):
    """
    Args:
        to_match (np.ndarry): 2d boolean array with one row per individual
            and one column sub-contact model.
        indexer (numba.List): Numba list that maps id of county to a numpy array
            with the row positions of all individuals from that county.
        first_stage_cum_probs(numpy.ndarray): Array of shape n_group, n_groups.
            cum_probs[i, j] is the probability that an individual from group i
            meets someone from group j or lower.
        group (np.ndarray): 1d array with assortative matching group ids,
            coded as integers.

    Returns:
        pairs_of_workers (np.ndarray): 2d integer array with meeting ids.

    """
    np.random.seed(seed)
    unique_group_codes = np.arange(len(first_stage_cum_probs))
    to_match = to_match.copy()
    out = np.full(to_match.shape, -1)
    n_obs, n_models = to_match.shape
    for m in range(n_models):
        meeting_id = 0
        for i in range(n_obs):
            if to_match[i, m]:
                group_i = group_codes_per_individual[i]
                group_j = choose_other_group(
                    unique_group_codes, first_stage_cum_probs[group_i]
                )
                group_j_indices = indexer[group_j]
                weights = to_match[group_j_indices, m].astype(np.float64)
                j = choose_other_individual(group_j_indices, weights)
                if j != -1:
                    to_match[i, m] = False
                    to_match[j, m] = False
                    out[i, m] = meeting_id
                    out[j, m] = meeting_id
                    meeting_id += 1
    return out


@nb.njit
def _create_participation_array(nr_of_contacts, seed):
    """Draw randomly in which pairs an individual participates.

    Args:
        nr_of_contacts (pandas.Series): number of contacts, i.e. number of pairs
            in which every individual is supposed to participate. The specific
            pair columns will be randomly drawn here.
        seed (int): seed

    Returns:
        participation (numpy.ndarray): boolean array of shape
            (len(nr_of_contacts), nr_of_contacts.max()).
            If participation[i, mod] is True, individual i was drawn
            to participate in mod.

    """
    np.random.seed(seed)
    n_models = nr_of_contacts.max()
    n_obs = len(nr_of_contacts)
    success_prob = nr_of_contacts / n_models
    participation_array = np.full((n_obs, n_models), False)
    for i in range(n_obs):
        prob_i = success_prob[i]
        for m in range(n_models):
            participation_array[i, m] = boolean_choice(prob_i)

    return participation_array
