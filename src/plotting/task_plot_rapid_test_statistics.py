import matplotlib.pyplot as plt
import pandas as pd
import pytask
import seaborn as sns

from src.config import BLD
from src.config import PLOT_END_DATE
from src.config import PLOT_SIZE
from src.plotting.plotting import BLUE
from src.plotting.plotting import GREEN
from src.plotting.plotting import PURPLE
from src.plotting.plotting import RED
from src.plotting.plotting import style_plot
from src.simulation.scenario_config import create_path_to_scenario_outcome_time_series
from src.simulation.task_process_rapid_test_statistics import ALL_CHANNELS
from src.simulation.task_process_rapid_test_statistics import DEMAND_SHARE_COLS
from src.simulation.task_process_rapid_test_statistics import OTHER_COLS
from src.simulation.task_process_rapid_test_statistics import SHARE_INFECTED_COLS
from src.simulation.task_process_rapid_test_statistics import TYPES


def _create_rapid_test_plot_parametrization():
    signature = "depends_on, column, color, plot_single_runs, ylabel, produces"

    column_color_label = []
    for column in DEMAND_SHARE_COLS:
        ylabel = "share of the population demanding a rapid test"
        column_color_label.append((column, BLUE, ylabel))
    for column in SHARE_INFECTED_COLS:
        ylabel = "share of rapid tests demanded by infected individuals"
        column_color_label.append((column, RED, ylabel))
    for typ in TYPES:
        columns = [f"{typ}_rate_overall"] + [f"{typ}_rate_in_{c}" for c in ALL_CHANNELS]
        for column in columns:
            color = GREEN if "true" in typ else PURPLE
            column_color_label.append((column, color, None))
    for column in OTHER_COLS:
        column_color_label.append((column, "k", None))

    parametrization = []
    for column, color, ylabel in column_color_label:
        for plot_single_runs in [True, False]:
            spec = _create_spec(
                column=column,
                plot_single_runs=plot_single_runs,
                color=color,
                ylabel=ylabel,
            )
            parametrization.append(spec)

    return signature, parametrization


def _create_spec(column, plot_single_runs, color, ylabel=None):
    depends_on = create_path_to_scenario_outcome_time_series(
        "combined_baseline", column
    )
    if ylabel is None:
        ylabel = column.replace("_", " ")
    file_name = (
        f"{column}_with_single_runs.pdf" if plot_single_runs else f"{column}.pdf"
    )
    produces = BLD / "figures" / "rapid_test_statistics" / file_name
    spec = (depends_on, column, color, plot_single_runs, ylabel, produces)
    return spec


_PARAMETRIZATION = _create_rapid_test_plot_parametrization()


@pytask.mark.parametrize(*_PARAMETRIZATION)
def task_plot_rapid_test_statistics(
    depends_on, column, color, plot_single_runs, ylabel, produces
):
    df = pd.read_pickle(depends_on)
    fig = _plot_df(
        df=df,
        column=column,
        color=color,
        plot_single_runs=plot_single_runs,
        ylabel=ylabel,
    )
    fig.savefig(produces)
    plt.close()


def _plot_df(df, column, color, plot_single_runs, ylabel):
    fig, ax = plt.subplots(figsize=PLOT_SIZE)

    sns.lineplot(
        x=df[column].index,
        y=df[column],
        ax=ax,
        linewidth=4,
        color=color,
        alpha=0.8,
    )

    if plot_single_runs:
        for col in df.columns:
            if col != column:
                sns.lineplot(
                    x=df.index,
                    y=df[col].rolling(window=7, min_periods=1, center=False).mean(),
                    ax=ax,
                    linewidth=2.5,
                    color=color,
                    alpha=0.6,
                )
    ax.set_xlim(pd.Timestamp("2021-03-15"), pd.Timestamp(PLOT_END_DATE))
    fig, ax = style_plot(fig, ax)

    ax.set_ylabel(ylabel)
    fig.tight_layout()
    return fig
