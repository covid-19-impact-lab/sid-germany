import functools
import os
import shutil

import pandas as pd
from sid import get_msm_func
from sid import get_simulate_func
from sid.msm import get_diag_weighting_matrix

from src.calculate_moments import smoothed_outcome_per_hundred_thousand_rki
from src.calculate_moments import smoothed_outcome_per_hundred_thousand_sim
from src.config import BLD
from src.manfred.shared import hash_array


def get_parallelizable_msm_criterion(simulate_kwargs, prefix):
    """Get a parallelizable msm criterion function."""
    pmsm = functools.partial(
        _build_and_evaluate_msm_func,
        simulate_kwargs=simulate_kwargs,
        prefix=prefix,
    )
    return pmsm


def get_index_bundles(params):
    """Get indices of parameters that are constrained to be equal."""
    base_query = "category == 'infection_prob' & subcategory.str.contains('{}')"
    queries = {
        "school": base_query.format("_school"),
        "young_educ": base_query.format("educ")
        + " & ~subcategory.str.contains('_school')",
        "hh": base_query.format("households"),
        "work": base_query.format("work"),
        "other": base_query.format("other"),
    }

    out = {key: params.query(q, engine="python").index for key, q in queries.items()}
    return out


def _build_and_evaluate_msm_func(params, seed, prefix, simulate_kwargs):
    params_hash = hash_array(params["value"].to_numpy())
    path = BLD / "exploration" / f"{prefix}_{params_hash}_{os.getpid()}"

    sim_start = simulate_kwargs["duration"]["start"]
    sim_end = simulate_kwargs["duration"]["end"]

    simulate = get_simulate_func(
        **simulate_kwargs,
        params=params,
        path=path,
        seed=seed,
    )

    calc_moments = _get_calc_moments()
    rki_data = pd.read_pickle(BLD / "data" / "processed_time_series" / "rki.pkl")
    rki_data = rki_data.loc[sim_start:sim_end]

    age_group_info = pd.read_pickle(
        BLD / "data" / "population_structure" / "age_groups_rki.pkl"
    )

    state_info = pd.read_parquet(
        BLD / "data" / "population_structure" / "federal_states.parquet"
    )
    state_sizes = state_info.set_index("name")["population"]

    empirical_moments = _get_empirical_moments(
        rki_data,
        age_group_sizes=age_group_info["n"],
        state_sizes=state_sizes,
    )

    weight_mat = _get_weighting_matrix(
        empirical_moments=empirical_moments,
        age_weights=age_group_info["weight"],
        state_weights=state_sizes / state_sizes.sum(),
    )

    msm_func = get_msm_func(
        simulate=functools.partial(_simulate_wrapper, simulate=simulate),
        calc_moments=calc_moments,
        empirical_moments=empirical_moments,
        replace_nans=lambda x: x * 1,
        weighting_matrix=weight_mat,
    )

    res = msm_func(params)
    shutil.rmtree(path)
    return res


def _simulate_wrapper(params, simulate):
    return simulate(params)["time_series"]


def _get_calc_moments():
    kwargs = {"window": 7, "min_periods": 1}

    calc_moments = {
        "infections_by_age_group": functools.partial(
            smoothed_outcome_per_hundred_thousand_sim,
            outcome="new_known_case",
            groupby="age_group_rki",
            take_logs=True,
            **kwargs,
        ),
        "aggregated_deaths": functools.partial(
            smoothed_outcome_per_hundred_thousand_sim,
            outcome="newly_deceased",
            take_logs=True,
            **kwargs,
        ),
        "infections_by_state": functools.partial(
            smoothed_outcome_per_hundred_thousand_sim,
            outcome="new_known_case",
            groupby="state",
            take_logs=True,
            **kwargs,
        ),
        "aggregated_infections": functools.partial(
            smoothed_outcome_per_hundred_thousand_sim,
            outcome="new_known_case",
            take_logs=False,
            **kwargs,
        ),
    }
    return calc_moments


def _get_empirical_moments(df, age_group_sizes, state_sizes):

    kwargs = {"window": 7, "min_periods": 1}

    empirical_moments = {
        "infections_by_age_group": smoothed_outcome_per_hundred_thousand_rki(
            df=df,
            outcome="newly_infected",
            groupby="age_group_rki",
            group_sizes=age_group_sizes,
            take_logs=True,
            **kwargs,
        ),
        "aggregated_deaths": smoothed_outcome_per_hundred_thousand_rki(
            df=df,
            outcome="newly_deceased",
            take_logs=True,
            **kwargs,
        ),
        "infections_by_state": smoothed_outcome_per_hundred_thousand_rki(
            df=df,
            outcome="newly_infected",
            groupby="state",
            group_sizes=state_sizes,
            take_logs=True,
            **kwargs,
        ),
        "aggregated_infections": smoothed_outcome_per_hundred_thousand_rki(
            df=df,
            outcome="newly_infected",
            take_logs=False,
            **kwargs,
        ),
    }
    return empirical_moments


def _get_weighting_matrix(empirical_moments, age_weights, state_weights):
    """Get a weighting matrix for msm estimation."""
    infections_by_age_weights = _get_grouped_weight_series(
        group_weights=age_weights,
        moment_series=empirical_moments["infections_by_age_group"],
        scaling_factor=1,
    )

    infections_by_state_weights = _get_grouped_weight_series(
        group_weights=state_weights,
        moment_series=empirical_moments["infections_by_state"],
        scaling_factor=0.2,
    )

    weights = {
        "infections_by_age_group": infections_by_age_weights,
        "aggregated_deaths": 0.1,
        "infections_by_state": infections_by_state_weights,
        # extremely low weight because not in logs
        "aggregated_infections": 1e-8,
    }

    weight_mat = get_diag_weighting_matrix(
        empirical_moments=empirical_moments,
        weights=weights,
    )
    return weight_mat


def _get_grouped_weight_series(group_weights, moment_series, scaling_factor=1):
    """Create a weight Series for a moment defined on a group level.

    group_weights (pd.Series or dict): Dict or series with group
        labels as index or keys and group weights as values.
    moment_series (pd.Series): The empirical moment for which the
        weights are constructed. It is assumed that the group is
        indicated by the second index level.
    """
    assert 0.99 <= group_weights.sum() <= 1, "Group weights should sum to 1."

    if isinstance(group_weights, pd.Series):
        group_weights = group_weights.to_dict()

    df = moment_series.to_frame()
    df["group"] = df.index.get_level_values(1)
    weight_sr = df["group"].replace(group_weights) * scaling_factor

    return weight_sr
