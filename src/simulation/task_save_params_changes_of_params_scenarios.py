from inspect import getmembers
from inspect import isfunction

import pandas as pd
import pytask

from src.config import BLD
from src.config import SRC
from src.simulation import params_scenarios


def get_params_scenarios():
    all_functions = getmembers(params_scenarios, isfunction)
    public_functions = [
        (name, func) for name, func in all_functions if not name.startswith("_")
    ]
    imported_funcs = [
        "get_piecewise_linear_interpolation",
        "get_piecewise_linear_interpolation_for_one_day",
    ]
    named_params_scenarios = [
        (name, func) for name, func in public_functions if name not in imported_funcs
    ]
    return named_params_scenarios


_PARAMETRIZATION = [
    (func, BLD / "simulation" / "param_scenarios" / f"{name}.csv")
    for name, func in get_params_scenarios()
]


@pytask.mark.depends_on(
    {
        "params": BLD / "params.pkl",
        "params_scenarios.py": SRC / "simulation" / "params_scenarios.py",
    }
)
@pytask.mark.parametrize("func, produces", _PARAMETRIZATION)
def task_save_params_changes_of_params_scenarios(depends_on, func, produces):
    old_params = pd.read_pickle(depends_on["params"])
    params = pd.read_pickle(depends_on["params"])

    new_params = func(params)

    comparison = create_comparison_df(new_params, old_params)
    comparison.to_csv(produces)


def create_comparison_df(new_params, old_params):
    new_params, old_params = _make_indices_identical(new_params, old_params)
    changed = (new_params != old_params)["value"]
    to_concat = {
        "before": old_params[changed]["value"],
        "after": new_params[changed]["value"],
    }
    comparison = pd.concat(to_concat, axis=1)
    return comparison


def _make_indices_identical(new_params, old_params):
    added_locs = new_params.index.difference(old_params.index)
    added = pd.DataFrame(
        data="added by the scenario function", index=added_locs, columns=["value"]
    )
    old_params = old_params.append(added).sort_index()

    deleted_locs = old_params.index.difference(new_params.index)
    deleted = pd.DataFrame(
        data="deleted by the scenario function", index=deleted_locs, columns=["value"]
    )
    new_params = new_params.append(deleted).sort_index()
    return new_params, old_params
