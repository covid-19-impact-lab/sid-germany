import pandas as pd
import pytask
import sid

from src.config import BLD
from src.contact_models.get_contact_models import get_all_contact_models
from src.create_initial_states.create_initial_conditions import (
    create_initial_conditions,
)
from src.policies.combine_policies_over_periods import get_december_to_feb_policies


SIMULATION_START = pd.Timestamp("2020-12-01")
INITIAL_START = SIMULATION_START - pd.Timedelta(weeks=2)
SIMULATION_END = SIMULATION_START + pd.Timedelta(weeks=4)


def _create_parametrization():
    parametrizations = []
    for christmas_mode in ["full", "same_group", "meet_twice"]:
        for contact_tracing_multiplier in [None, 0.1]:
            ctm_str = (
                "wo_ct"
                if contact_tracing_multiplier is None
                else f"w_ct_{contact_tracing_multiplier}"
            )
            path = (
                BLD / "simulation" / f"simulation_christmas_mode_{christmas_mode}_"
                f"{ctm_str}"
            )
            produces = path / "last_states"

            single_run = (christmas_mode, contact_tracing_multiplier, path, produces)

            parametrizations.append(single_run)

    return parametrizations


PARAMETRIZATIONS = _create_parametrization()


@pytask.mark.depends_on(
    {
        "params": BLD / "start_params.pkl",
        "share_known_cases": BLD
        / "data"
        / "processed_time_series"
        / "share_known_cases.pkl",
        # !!! Replace with real states.
        "initial_states": BLD / "data" / "debug_initial_states.parquet",
    }
)
@pytask.mark.parametrize(
    "christmas_mode, contact_tracing_multiplier, path, produces", PARAMETRIZATIONS
)
def task_simulation_christmas_scenarios(
    depends_on, christmas_mode, contact_tracing_multiplier, path
):
    params = pd.read_pickle(depends_on["params"])

    share_known_cases = pd.read_pickle(depends_on["share_known_cases"])
    share_known_cases = share_known_cases[~share_known_cases.index.duplicated()]

    df = pd.read_parquet(depends_on["initial_states"])

    contact_models = get_all_contact_models(
        christmas_mode=christmas_mode, n_extra_contacts_before_christmas=2
    )

    policies = get_december_to_feb_policies(
        contact_models=contact_models,
        pre_christmas_other_multiplier=0.3,
        christmas_other_multiplier=0.0,
        post_christmas_multiplier=0.3,
        contact_tracing_multiplier=contact_tracing_multiplier,
    )

    infection_probs = _build_infection_probs(contact_models.keys())

    params = pd.concat([infection_probs, params])

    initial_conditions = create_initial_conditions(
        start=INITIAL_START,
        end=SIMULATION_START - pd.Timedelta(days=1),
        reporting_delay=7,
        seed=99,
    )

    simulate = sid.get_simulate_func(
        params=params,
        initial_states=df,
        contact_models=contact_models,
        duration={"start": SIMULATION_START, "end": SIMULATION_END},
        contact_policies=policies,
        initial_conditions=initial_conditions,
        share_known_cases=share_known_cases,
        path=path,
        events=None,
        seed=384,
    )

    simulate(params)


def _build_infection_probs(names):
    index_tuples = [("infection_prob", mod_name, mod_name) for mod_name in names]
    df = pd.DataFrame(index=pd.MultiIndex.from_tuples(index_tuples))
    df.index.names = ["category", "subcategory", "name"]
    df = df.reset_index()
    prob_dict = {
        "educ": 0.02,
        "work": 0.1,
        "household": 0.2,
        "other": 0.1,
        "christmas": 0.2,
        "holiday_preparation": 0.1,
    }
    full_prob_dict = {}
    for mod_name in names:
        for k, v in prob_dict.items():
            if k in mod_name:
                full_prob_dict[mod_name] = v
        assert (
            mod_name in full_prob_dict
        ), f"No infection probability for {mod_name} specified."

    df["value"] = df["name"].map(full_prob_dict.get)
    df = df.set_index(["category", "subcategory", "name"])

    return df
