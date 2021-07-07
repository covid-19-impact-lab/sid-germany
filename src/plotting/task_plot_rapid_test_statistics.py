import matplotlib.pyplot as plt
import pandas as pd
import pytask
import seaborn as sns

from src.config import BLD
from src.config import PLOT_END_DATE
from src.config import PLOT_SIZE
from src.plotting.plotting import BLUE
from src.plotting.plotting import GREEN
from src.plotting.plotting import RED
from src.plotting.plotting import style_plot
from src.simulation.scenario_config import (
    create_path_to_rapid_test_statistic_time_series,
)
from src.simulation.task_process_rapid_test_statistics import CHANNELS
from src.simulation.task_process_rapid_test_statistics import DEMAND_SHARE_COLS
from src.simulation.task_process_rapid_test_statistics import SHARE_INFECTED_COLS
from src.simulation.task_process_rapid_test_statistics import TYPES


def _create_rapid_test_plot_parametrization():
    signature = "depends_on, plot_single_runs, ylabel, produces"

    columns_and_label = [
        (
            DEMAND_SHARE_COLS + ["share_with_rapid_test"],
            "share of the population demanding a rapid test",
        ),
        (SHARE_INFECTED_COLS, "share of rapid tests demanded by infected individuals"),
        (
            ["false_positive_rate_in_the_population"],
            "false positive rate in the population",
        ),
        (
            ["n_rapid_tests_overall_in_germany"],
            "number of rapid tests per day \n(upscaled to the German population)",
        ),
    ]
    for typ in TYPES:
        column_names = [f"{typ}_rate_in_{channel}" for channel in CHANNELS]
        columns_and_label.append((column_names, f"{typ.replace('_', ' ')} rate"))

    parametrization = []
    for columns, ylabel in columns_and_label:
        for plot_single_runs in [True, False]:
            spec = _create_spec(
                columns=columns,
                plot_single_runs=plot_single_runs,
                ylabel=ylabel,
            )
            parametrization.append(spec)

    return signature, parametrization


def _create_spec(columns, plot_single_runs, ylabel):
    depends_on = {
        col: create_path_to_rapid_test_statistic_time_series("combined_baseline", col)
        for col in columns
    }
    if plot_single_runs:
        file_name = f"{ylabel.replace(' ', '_')}_with_single_runs.pdf"
    else:
        file_name = f"{ylabel.replace(' ', '_')}.pdf"
    produces = BLD / "figures" / "rapid_test_statistics" / file_name
    spec = (depends_on, plot_single_runs, ylabel, produces)
    return spec


_PARAMETRIZATION = _create_rapid_test_plot_parametrization()


@pytask.mark.parametrize(*_PARAMETRIZATION)
def task_plot_rapid_test_statistics(depends_on, plot_single_runs, ylabel, produces):
    dfs = {col: pd.read_pickle(path) for col, path in depends_on.items()}

    fig, ax = plt.subplots(figsize=PLOT_SIZE)
    for col, df in dfs.items():
        color, label = _get_channel_color_and_label(col)
        ax = _plot_df(
            df=df,
            column=col,
            color=color,
            plot_single_runs=plot_single_runs,
            ax=ax,
            label=label,
        )

    ax.set_xlim(pd.Timestamp("2021-03-01"), pd.Timestamp(PLOT_END_DATE))
    fig, ax = style_plot(fig, ax)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(produces)
    plt.close()


def _plot_df(
    df,
    column,
    color,
    label,
    ax,
    plot_single_runs,
):
    sns.lineplot(
        x=df[column].index,
        y=df[column],
        ax=ax,
        linewidth=4,
        color=color,
        label=label,
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
    return ax


def _get_channel_color_and_label(col):
    if "work" in col:
        color = BLUE
        label = "work"
    elif "educ" in col:
        color = GREEN
        label = "educ"
    elif "private" in col:
        color = RED
        label = "private"
    else:
        color = "k"
        label = None
    return color, label
