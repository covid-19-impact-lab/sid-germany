import warnings

import matplotlib.pyplot as plt
import pandas as pd
import pytask
import seaborn as sns
import sid

from src.config import BLD
from src.config import SRC
from src.create_initial_states.create_initial_conditions import (
    create_group_specific_share_known_cases,
)
from src.plotting.plotting import style_plot
from src.simulation.scenario_config import (
    create_path_to_initial_group_share_known_cases,
)
from src.simulation.scenario_config import SPRING_START
from src.testing.shared import get_piecewise_linear_interpolation


@pytask.mark.depends_on(
    {
        "group_share_known_cases": create_path_to_initial_group_share_known_cases(
            "fall_baseline", pd.Timestamp("2020-12-23")
        ),
        "rki_age_groups": BLD / "data" / "population_structure" / "age_groups_rki.pkl",
        "create_initial_conditions": SRC
        / "create_initial_states"
        / "create_initial_conditions.py",
        "simulation_shared": SRC / "simulation" / "scenario_config.py",
        "plotting": SRC / "plotting" / "plotting.py",
        "params": BLD / "params.pkl",
    }
)
@pytask.mark.produces(BLD / "figures" / "share_known_cases_prediction.png")
def task_plot_group_specific_share_known_cases(depends_on, produces):
    group_share_known_cases = pd.read_pickle(depends_on["group_share_known_cases"])
    params = pd.read_pickle(depends_on["params"])
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="indexing past lexsort depth may impact performance."
        )
        params_slice = params.loc[("share_known_cases", "share_known_cases")]
    overall_share_known_cases = get_piecewise_linear_interpolation(params_slice)
    group_weights = pd.read_pickle(depends_on["rki_age_groups"])["weight"]
    start_date = pd.Timestamp(SPRING_START)
    init_start = start_date - pd.Timedelta(31, unit="D")
    init_end = start_date - pd.Timedelta(1, unit="D")

    to_plot = (
        create_group_specific_share_known_cases(
            group_share_known_cases=group_share_known_cases,
            overall_share_known_cases=overall_share_known_cases,
            group_weights=group_weights,
            date_range=pd.date_range(start=init_start, end=init_end, name="date"),
        )
        .stack()
        .reset_index()
    )

    n_groups = to_plot["age_group_rki"].nunique()
    colors = sid.get_colors("ordered", n_groups)
    sns.set_palette(colors)

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.lineplot(data=to_plot, x="date", y=0, hue="age_group_rki")
    fig, ax = style_plot(fig, ax)
    fig.tight_layout()
    fig.savefig(produces)
