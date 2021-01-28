#!/usr/bin/env python
import functools
import os
import shutil

import pandas as pd
from estimagic import minimize
from estimagic.optimization.process_constraints import process_constraints
from sid import get_msm_func
from sid import get_simulate_func
from sid.msm import get_diag_weighting_matrix

from src.calculate_moments import smoothed_outcome_per_hundred_thousand_rki
from src.calculate_moments import smoothed_outcome_per_hundred_thousand_sim
from src.config import BLD
from src.config import SRC
from src.contact_models.get_contact_models import get_all_contact_models
from src.create_initial_states.create_initial_conditions import (  # noqa
    create_initial_conditions,
)
from src.manfred.minimize_manfred_estimagic import minimize_manfred_estimagic
from src.manfred.shared import hash_array
from src.policies.combine_policies_over_periods import get_october_to_christmas_policies


def _get_free_params(params, constraints):
    pc, pp = process_constraints(constraints, params)
    return pp.query("_internal_free")


ESTIMATION_START = pd.Timestamp("2020-08-15")
ESTIMATION_END = pd.Timestamp("2020-12-05")

INIT_START = ESTIMATION_START - pd.Timedelta(31, unit="D")
INIT_END = ESTIMATION_START - pd.Timedelta(1, unit="D")
initial_states = pd.read_parquet(BLD / "data" / "initial_states.parquet")
share_known_cases = pd.read_pickle(
    BLD / "data" / "processed_time_series" / "share_known_cases.pkl"
)


initial_conditions = create_initial_conditions(
    start=INIT_START, end=INIT_END, seed=3484
)


contact_models = get_all_contact_models()


def parallelizable_msm_func(
    params, initial_states, initial_conditions, prefix, share_known_cases
):

    params_hash = hash_array(params["value"].to_numpy())
    path = SRC / "exploration" / f"{prefix}_{params_hash}_{os.getpid()}"

    contact_models = get_all_contact_models()

    estimation_policies = get_october_to_christmas_policies(contact_models)

    simulate = get_simulate_func(
        params=params,
        initial_states=initial_states,
        contact_models=contact_models,
        contact_policies=estimation_policies,
        duration={"start": ESTIMATION_START, "end": ESTIMATION_END},
        initial_conditions=initial_conditions,
        share_known_cases=share_known_cases,
        path=path,
        saved_columns={
            "initial_states": ["age_group_rki"],
            "disease_states": ["newly_infected"],
            "time": ["date"],
            "other": ["new_known_case"],
        },
    )

    calc_moments = {
        "infections_by_age_group": functools.partial(
            smoothed_outcome_per_hundred_thousand_sim,
            outcome="new_known_case",
            groupby="age_group_rki",
        ),
    }

    data = pd.read_pickle(BLD / "data" / "processed_time_series" / "rki.pkl")
    data = data.loc[ESTIMATION_START:ESTIMATION_END]
    age_group_info = pd.read_pickle(
        BLD / "data" / "population_structure" / "age_groups_rki.pkl"
    )

    empirical_moments = {
        "infections_by_age_group": smoothed_outcome_per_hundred_thousand_rki(
            df=data,
            outcome="newly_infected",
            groupby="age_group_rki",
            window=7,
            min_periods=1,
            group_sizes=age_group_info["n"],
        )
    }

    age_weights = age_group_info["weight"].to_dict()

    temp = empirical_moments["infections_by_age_group"].to_frame().copy(deep=True)
    temp["age_group"] = temp.index.get_level_values(1)
    temp["weights"] = temp["age_group"].replace(age_weights)

    weights = {"infections_by_age_group": temp["weights"]}

    weight_mat = get_diag_weighting_matrix(
        empirical_moments=empirical_moments,
        weights=weights,
    )

    def simulate_wrapper(params, simulate):
        return simulate(params)["time_series"]

    msm = get_msm_func(
        simulate=functools.partial(simulate_wrapper, simulate=simulate),
        calc_moments=calc_moments,
        empirical_moments=empirical_moments,
        replace_nans=lambda x: x * 1,
        weighting_matrix=weight_mat,
    )

    res = msm(params)
    shutil.rmtree(path)
    return res


pmsm = functools.partial(
    parallelizable_msm_func,
    initial_states=initial_states,
    initial_conditions=initial_conditions,
    prefix="estimation",
    share_known_cases=share_known_cases,
)


params = pd.read_pickle(BLD / "start_params.pkl")

hh_probs = ("infection_prob", "households", "households")
educ_models = [cm for cm in contact_models if "educ" in cm]
educ_probs = params.query(
    f"category == 'infection_prob' & subcategory in {educ_models}"
).index
work_models = [cm for cm in contact_models if "work" in cm]
work_probs = params.query(
    f"category == 'infection_prob' & subcategory in {work_models}"
).index
other_rec_models = [
    cm for cm in contact_models if "other" in cm and "non_recurrent" not in cm
]
other_rec_probs = params.query(
    f"category == 'infection_prob' & subcategory in {other_rec_models}"
).index

other_non_rec_probs = ("infection_prob", "other_non_recurrent", "other_non_recurrent")
school_models = [
    cm
    for cm in contact_models
    if "educ" in cm and "school" in cm and "preschool" not in cm
]
school_probs = params.query(
    f"category == 'infection_prob' & subcategory in {school_models}"
).index

other_educ_probs = [
    ("infection_prob", "educ_nursery_0", "educ_nursery_0"),
    ("infection_prob", "educ_preschool_0", "educ_preschool_0"),
]

params.loc[educ_probs, "value"] = 0.015
params.loc[school_probs, "value"] = 0.01
params.loc[other_rec_probs, "value"] = 0.08
params.loc[other_non_rec_probs, "value"] = 0.05
params.loc[work_probs, "value"] = 0.08
params.loc[hh_probs, "value"] = 0.1

params.loc["infection_prob", "lower_bound"] = 0.02
params.loc["infection_prob", "upper_bound"] = 0.12
params.loc[educ_probs, "lower_bound"] = 0.001
params.loc[educ_probs, "upper_bound"] = 0.02
params.loc[hh_probs, "lower_bound"] = 0.03
params.loc[hh_probs, "upper_bound"] = 0.15


constraints = [
    {"query": "category != 'infection_prob'", "type": "fixed"},
    {"loc": other_educ_probs, "type": "equality"},
    {"loc": other_rec_probs, "type": "equality"},
    {"loc": school_probs, "type": "equality"},
    {"loc": work_probs, "type": "equality"},
]


free_index = _get_free_params(params, constraints).index
params["group"] = None
params.loc[free_index, "group"] = "Free Probabilities"
params.loc[free_index]


algo_options = {
    "step_sizes": [0.2, 0.1, 0.05],
    "max_step_sizes": [0.4, 0.2, 0.1],
    "linesearch_n_points": 8,
    "gradient_weight": 0.5,
    "noise_n_evaluations_per_x": [10, 15, 20],
    "convergence_relative_params_tolerance": 0.001,
    "direction_window": 3,
    "convergence_direct_search_mode": "fast",
    "batch_evaluator_options": {"n_cores": 24},
}


res = minimize(
    criterion=pmsm,
    params=params,
    algorithm=minimize_manfred_estimagic,
    algo_options=algo_options,
    logging="second_manfred_attempt.db",
    constraints=constraints,
)
