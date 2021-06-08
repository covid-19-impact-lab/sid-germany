import matplotlib.pyplot as plt
import pandas as pd
import pytask
import seaborn as sns

from src.config import BLD
from src.config import FAST_FLAG
from src.config import PLOT_SIZE
from src.config import SRC
from src.plotting.plotting import BLUE
from src.plotting.plotting import ORANGE
from src.plotting.plotting import OUTCOME_TO_EMPIRICAL_LABEL
from src.plotting.plotting import RED
from src.plotting.plotting import style_plot
from src.plotting.plotting import TEAL
from src.simulation.scenario_config import create_path_for_weekly_outcome_of_scenario
from src.simulation.scenario_config import (
    create_path_to_scenario_outcome_time_series as get_dependency,
)

_DEPENDENCIES = {
    "ever_vaccinated": get_dependency("combined_baseline", "ever_vaccinated"),
    "share_rapid_test_in_last_week": get_dependency(
        "combined_baseline", "share_rapid_test_in_last_week"
    ),
    "empirical": BLD / "data" / "empirical_data_for_plotting.pkl",
    "plotting.py": SRC / "plotting" / "plotting.py",
    "scenario_config.py": SRC / "simulation" / "scenario_config.py",
}


@pytask.mark.depends_on(_DEPENDENCIES)
@pytask.mark.produces(
    create_path_for_weekly_outcome_of_scenario(
        comparison_name="combined_fit",
        fast_flag=FAST_FLAG,
        outcome="share_rapid_test_in_last_week_and_vaccinated",
        suffix="pdf",
    )
)
def task_plot_share_rapid_test_last_week_and_vaccinations(depends_on, produces):
    empirical_df = pd.read_pickle(depends_on["empirical"])
    vaccinated = pd.read_pickle(depends_on["ever_vaccinated"])
    rapid_test_share = pd.read_pickle(depends_on["share_rapid_test_in_last_week"])

    rapid_test_label = (
        "share of people who did a rapid test\nwithin the last 7 days in the simulation"
    )

    fig, ax = plt.subplots(figsize=PLOT_SIZE)

    plot_spec = [
        (
            empirical_df[["ever_vaccinated"]],
            OUTCOME_TO_EMPIRICAL_LABEL["ever_vaccinated"],
            BLUE,
        ),
        (
            empirical_df[["share_rapid_test_in_last_week"]],
            OUTCOME_TO_EMPIRICAL_LABEL["share_rapid_test_in_last_week"],
            RED,
        ),
        (vaccinated, "simulated share of vaccinated individuals", TEAL),
        (rapid_test_share, rapid_test_label, ORANGE),
    ]

    for df, label, color in plot_spec:
        sns.lineplot(
            x=df.index,
            y=df.mean(axis=1),
            ax=ax,
            color=color,
            label=label,
            linewidth=3.0,
            alpha=0.6,
        )

    x, y, width, height = 0.0, -0.3, 1, 0.2
    ax.legend(loc="upper center", bbox_to_anchor=(x, y, width, height), ncol=2)
    fig.tight_layout()

    fig, ax = style_plot(fig, ax)
    fig.savefig(produces)
