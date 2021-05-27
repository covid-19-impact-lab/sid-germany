import functools
import os
import shutil

import numpy as np
import pandas as pd
from sid import get_msm_func
from sid import get_simulate_func
from sid.msm import get_diag_weighting_matrix
from sid.plotting import prepare_data_for_infection_rates_by_contact_models

from src.calculate_moments import aggregate_and_smooth_period_outcome_sim
from src.calculate_moments import calculate_period_outcome_sim
from src.calculate_moments import smoothed_outcome_per_hundred_thousand_rki
from src.config import BLD
from src.manfred.shared import hash_array
from src.simulation.load_simulation_inputs import load_simulation_inputs


def get_parallelizable_msm_criterion(
    prefix,
    fall_start_date,
    fall_end_date,
    winter_start_date,
    winter_end_date,
    spring_start_date,
    spring_end_date,
    mode,
    debug,
):
    """Get a parallelizable msm criterion function."""
    pmsm = functools.partial(
        _build_and_evaluate_msm_func,
        prefix=prefix,
        fall_start_date=fall_start_date,
        fall_end_date=fall_end_date,
        winter_start_date=winter_start_date,
        winter_end_date=winter_end_date,
        spring_start_date=spring_start_date,
        spring_end_date=spring_end_date,
        mode=mode,
        debug=debug,
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


def _build_and_evaluate_msm_func(
    params,
    seed,
    prefix,
    fall_start_date,
    fall_end_date,
    winter_start_date,
    winter_end_date,
    spring_start_date,
    spring_end_date,
    mode,
    debug,
):
    """ """
    params_hash = hash_array(params["value"].to_numpy())
    winter_path = BLD / "exploration" / f"winter_share_known_{params_hash}_{seed}.pkl"
    spring_path = BLD / "exploration" / f"spring_share_known_{params_hash}_{seed}.pkl"
    if mode in ["fall", "combined"]:
        res_fall = _build_and_evaluate_msm_func_one_season(
            params=params,
            seed=seed,
            prefix=prefix,
            start_date=fall_start_date,
            end_date=fall_end_date,
            debug=debug,
        )
        res_fall["share_known_cases"].to_pickle(winter_path)
    if mode in ["winter", "combined"]:
        res_winter = _build_and_evaluate_msm_func_one_season(
            params=params,
            seed=seed,
            prefix=prefix,
            start_date=winter_start_date,
            end_date=winter_end_date,
            debug=debug,
            group_share_known_case_path=winter_path,
        )
        res_winter["share_known_cases"].to_pickle(spring_path)
    if mode in ["spring", "combined"]:
        res_spring = _build_and_evaluate_msm_func_one_season(
            params=params,
            seed=seed + 84587,
            prefix=prefix,
            start_date=spring_start_date,
            end_date=spring_end_date,
            debug=debug,
            group_share_known_case_path=spring_path,
        )
    if mode == "fall":
        res = res_fall
    elif mode == "winter":
        res = res_winter
    elif mode == "spring":
        res = res_spring
    else:
        results = [res_fall, res_winter, res_spring]
        raw_weights = np.array(
            [
                (fall_end_date - fall_start_date).days,
                (winter_end_date - winter_start_date).days,
                (spring_end_date - spring_start_date).days,
            ]
        )
        weights = raw_weights / raw_weights.sum()
        res = _combine_results(results, weights)

    return res


def _combine_results(results, weights):
    combined = {}
    res0 = results[0]
    for key in res0:
        if key == "value":
            values = np.array([res["value"] for res in results])
            combined[key] = values @ weights
        elif key in ["empirical_moments", "simulated_moments"]:
            combined[key] = _concatenate_pd_objects_from_dicts(
                [res[key] for res in results]
            )
        else:
            combined[key] = pd.concat([res[key] for res in results])
    return combined


def _concatenate_pd_objects_from_dicts(dicts):
    combined = {}
    for key in dicts[0]:
        combined[key] = pd.concat([d[key] for d in dicts])
    return combined


def _build_and_evaluate_msm_func_one_season(
    params,
    seed,
    prefix,
    start_date,
    end_date,
    debug,
    group_share_known_case_path=None,
):
    """Build and evaluate a msm criterion function.

    Building the criterion function freshly for each run is necessary for it to be
    parallelizable.

    """
    simulate_kwargs = load_simulation_inputs(
        "baseline",
        start_date=start_date,
        end_date=end_date,
        group_share_known_case_path=group_share_known_case_path,
        debug=debug,
    )
    params_hash = hash_array(params["value"].to_numpy())
    path = BLD / "exploration" / f"{prefix}_{params_hash}_{os.getpid()}"

    sim_start = simulate_kwargs["duration"]["start"]
    sim_end = simulate_kwargs["duration"]["end"]
    period_outputs = _get_period_outputs_for_simulate()

    simulate = get_simulate_func(
        **simulate_kwargs,
        params=params,
        path=path,
        seed=seed,
        period_outputs=period_outputs,
        return_last_states=False,
        return_time_series=False,
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

    additional_outputs = {
        "infection_channels": _aggregate_infection_channels,
        "share_known_cases": _calculate_share_known_cases,
    }

    msm_func = get_msm_func(
        simulate=simulate,
        calc_moments=calc_moments,
        empirical_moments=empirical_moments,
        replace_nans=lambda x: x * 1,
        weighting_matrix=weight_mat,
        additional_outputs=additional_outputs,
    )

    res = msm_func(params)
    shutil.rmtree(path)
    return res


def _aggregate_infection_channels(simulate_result):
    """Aggregate the infection channel data that was calculated in each period."""
    return pd.concat(simulate_result["period_outputs"]["infection_channels"])


def _get_period_outputs_for_simulate():
    """Construct the period_outputs argument for ``get_simulate_func``.

    All estimation moments as well as the infection channel data are calculated
    as per period outcomes. This needs much less memory than calculating those outcomes
    from the full time series.

    """
    additional_outputs = {
        "infections_by_age_group": functools.partial(
            calculate_period_outcome_sim,
            outcome="new_known_case",
            groupby="age_group_rki",
        ),
        "aggregated_deaths": functools.partial(
            calculate_period_outcome_sim,
            outcome="newly_deceased",
        ),
        "infections_by_state": functools.partial(
            calculate_period_outcome_sim,
            outcome="new_known_case",
            groupby="state",
        ),
        "aggregated_infections": functools.partial(
            calculate_period_outcome_sim,
            outcome="new_known_case",
        ),
        "infection_channels": prepare_data_for_infection_rates_by_contact_models,
        "currently_infected_by_age_group": functools.partial(
            calculate_period_outcome_sim,
            outcome="currently_infected",
            groupby="age_group_rki",
        ),
        "knows_currently_infected_by_age_group": functools.partial(
            calculate_period_outcome_sim,
            outcome="knows_currently_infected",
            groupby="age_group_rki",
        ),
    }
    return additional_outputs


def _get_calc_moments():
    """Construct the ``calc_moments`` argument for ``get_msm_func``.

    Instead of calculating those moments from the full time series we provide functions
    that simply aggregate and smooth the per period outcomes that are calculated on
    each simulated day.

    """
    calc_moments = {
        "infections_by_age_group": functools.partial(
            aggregate_and_smooth_period_outcome_sim,
            outcome="infections_by_age_group",
            groupby="age_group_rki",
            take_logs=True,
        ),
        "aggregated_deaths": functools.partial(
            aggregate_and_smooth_period_outcome_sim,
            outcome="aggregated_deaths",
            take_logs=True,
        ),
        "infections_by_state": functools.partial(
            aggregate_and_smooth_period_outcome_sim,
            outcome="infections_by_state",
            groupby="state",
            take_logs=True,
        ),
        "aggregated_infections_not_log": functools.partial(
            aggregate_and_smooth_period_outcome_sim,
            outcome="aggregated_infections",
            take_logs=False,
        ),
        "aggregated_infections": functools.partial(
            aggregate_and_smooth_period_outcome_sim,
            outcome="aggregated_infections",
            take_logs=True,
        ),
    }
    return calc_moments


def _calculate_share_known_cases(sim_out):
    currently_infected = aggregate_and_smooth_period_outcome_sim(
        simulate_result=sim_out,
        outcome="currently_infected_by_age_group",
        groupby="age_group_rki",
        take_logs=False,
    )
    knows_currently_infected = aggregate_and_smooth_period_outcome_sim(
        simulate_result=sim_out,
        outcome="knows_currently_infected_by_age_group",
        groupby="age_group_rki",
        take_logs=False,
    )

    share_known = (knows_currently_infected / currently_infected).unstack()
    end_date = share_known.index.max()
    avg_share_known = share_known[end_date - pd.Timedelta(days=28) :].mean()

    return avg_share_known


def _get_empirical_moments(df, age_group_sizes, state_sizes):
    """Construct the ``empirical_moments`` argument for ``get_msm_func``."""
    empirical_moments = {
        "infections_by_age_group": smoothed_outcome_per_hundred_thousand_rki(
            df=df,
            outcome="newly_infected",
            groupby="age_group_rki",
            group_sizes=age_group_sizes,
            take_logs=True,
        ),
        "aggregated_deaths": smoothed_outcome_per_hundred_thousand_rki(
            df=df,
            outcome="newly_deceased",
            take_logs=True,
        ),
        "infections_by_state": smoothed_outcome_per_hundred_thousand_rki(
            df=df,
            outcome="newly_infected",
            groupby="state",
            group_sizes=state_sizes,
            take_logs=True,
        ),
        "aggregated_infections_not_log": smoothed_outcome_per_hundred_thousand_rki(
            df=df,
            outcome="newly_infected",
            take_logs=False,
        ),
        "aggregated_infections": smoothed_outcome_per_hundred_thousand_rki(
            df=df,
            outcome="newly_infected",
            take_logs=True,
        ),
    }
    return empirical_moments


def _get_weighting_matrix(empirical_moments, age_weights, state_weights):
    """Get a weighting matrix for msm estimation."""
    # set the weight of the oldest group a bit lower because we do not have
    # old age homes in our model and expect not to match that moment very well.
    age_weights = age_weights.copy(deep=True)
    age_weights["80-100"] = age_weights["80-100"] * 0.5
    age_weights = age_weights / age_weights.sum()
    infections_by_age_weights = _get_grouped_weight_series(
        group_weights=age_weights,
        moment_series=empirical_moments["infections_by_age_group"],
        scaling_factor=1,
    )

    infections_by_state_weights = _get_grouped_weight_series(
        group_weights=state_weights,
        moment_series=empirical_moments["infections_by_state"],
        scaling_factor=1,
    )

    weights = {
        "infections_by_age_group": infections_by_age_weights,
        # lower weight because not a primary target
        "aggregated_deaths": 0.1,
        "infections_by_state": infections_by_state_weights,
        # not used for estimation because not in logs
        "aggregated_infections_not_log": 1e-10,
        # strong weight because very important
        "aggregated_infections": 2.5,
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
