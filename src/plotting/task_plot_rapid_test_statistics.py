import matplotlib.pyplot as plt
import pandas as pd
import pytask
import seaborn as sns

from src.config import BLD
from src.config import PLOT_END_DATE
from src.config import PLOT_SIZE
from src.plotting.plotting import BLUE
from src.plotting.plotting import GREEN
from src.plotting.plotting import style_plot
from src.plotting.plotting import YELLOW
from src.simulation.scenario_config import (
    create_path_to_rapid_test_statistic_time_series,
)
from src.simulation.task_process_rapid_test_statistics import CHANNELS
from src.simulation.task_process_rapid_test_statistics import OUTCOMES
from src.simulation.task_process_rapid_test_statistics import SHARE_TYPES


def _create_rapid_test_plot_parametrization():
    signature = "depends_on, plot_single_runs, ylabel, produces"

    label_templates = {
        "number": "number of {nice_outcome} per million",
        "popshare": "share of the population with {nice_outcome}",
        "testshare": "share of {nice_outcome}",
    }
    nice_outcomes = {
        "false_negative": "false negative tests",
        "false_positive": "false positive tests",
        "tested_negative": "negative tests",
        "tested_positive": "positive tests",
        "true_negative": "true negative tests",
        "true_positive": "true positive tests",
        "tested": "tests",
    }

    column_and_label = []
    for outcome in OUTCOMES:
        for share_type in SHARE_TYPES:
            column = f"{share_type}_{outcome}"
            label = label_templates[share_type].format(
                nice_outcome=nice_outcomes[outcome]
            )
            column_and_label.append((column, label))

    parametrization = []
    for column, ylabel in column_and_label:
        for plot_single_runs in [True, False]:
            spec = _create_spec(
                column=column,
                plot_single_runs=plot_single_runs,
                ylabel=ylabel,
            )
            parametrization.append(spec)

    return signature, parametrization


def _create_spec(column, plot_single_runs, ylabel):
    depends_on = {}
    for channel in CHANNELS:
        channel_column = column + f"_by_{channel}"
        depends_on[channel_column] = create_path_to_rapid_test_statistic_time_series(
            "spring_baseline", channel_column
        )

    if plot_single_runs:
        file_name = f"{column}_with_single_runs.pdf"
    else:
        file_name = f"{column}.pdf"
    produces = BLD / "figures" / "rapid_test_statistics" / file_name
    spec = (depends_on, plot_single_runs, ylabel, produces)
    return spec


_PARAMETRIZATION = _create_rapid_test_plot_parametrization()


@pytask.mark.parametrize(*_PARAMETRIZATION)
def task_plot_rapid_test_statistics(depends_on, plot_single_runs, ylabel, produces):
    dfs = {col: pd.read_pickle(path) for col, path in depends_on.items()}

    fig, ax = plt.subplots(figsize=PLOT_SIZE)
    for col, df in dfs.items():
        if "number_" in col:
            # scale from cases in Germany to cases per million
            df = df / 83

        color, label = _get_channel_color_and_label(col)
        ax = _plot_df(
            df=df.loc["2021-03-15":],
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
        color = GREEN
        label = "work"
    elif "educ" in col:
        color = YELLOW
        label = "educ"
    elif "private" in col:
        color = BLUE
        label = "private"
    else:
        color = "k"
        label = "aggregate"
    return color, label
