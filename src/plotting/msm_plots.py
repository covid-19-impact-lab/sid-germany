import matplotlib.dates as dt
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import seaborn as sns
from estimagic.visualization.colors import get_colors


plt.rcParams.update(
    {
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
    }
)


def plot_estimation_moment(results, name):
    moment_name, group_name = _split_name(name)

    if group_name is not None:
        simulated, empirical = _extract_grouped_moment(results, name)
        raw_groups = simulated.index.get_level_values("group").unique()
        if group_name == "age_group":
            groups = _sort_age_groups(raw_groups)
        else:
            groups = sorted(raw_groups)

        n_groups = len(groups)
        fig, axes = plt.subplots(figsize=(6, n_groups * 3), nrows=n_groups)
        for group, ax in zip(groups, axes):
            _plot_simulated_and_empirical_moment(
                simulated=simulated.loc[group],
                empirical=empirical.loc[group],
                ax=ax,
            )
    else:
        simulated, empirical = _extract_aggregated_moment(results, name)
        fig, ax = plt.subplots(figsize=(6, 3))
        axes = [ax]
        _plot_simulated_and_empirical_moment(
            simulated=simulated, empirical=empirical, ax=ax
        )

    for ax in axes:
        ax.set_ylabel(moment_name)
        ax.xaxis.set_major_locator(dt.MonthLocator())
        ax.xaxis.set_major_formatter(dt.DateFormatter("%b %Y"))
        ax.xaxis.set_minor_locator(dt.DayLocator())
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())

    fig.tight_layout()
    plt.close()

    return fig


def _extract_grouped_moment(results, name):
    """Extract a moment defined on subgroups from list of msm results.

    Args:
        results (list): List of msm results
        name (str): Key of the moment in the moment dictionary

    Returns:
        pd.DataFrame: simulated moment per age group
        pd.Series: empirical moment per age_group

    """
    to_concat = []
    for i, res in enumerate(results):
        df = res["simulated_moments"][name].to_frame().copy(deep=True)
        df.index.names = ["date", "group"]
        df = df.reset_index()
        df["run"] = i
        to_concat.append(df)

    simulated = pd.concat(to_concat).set_index(["group", "date", "run"]).unstack()

    empirical = results[0]["empirical_moments"][name].copy(deep=True)
    empirical.index.names = ["date", "group"]
    empirical = empirical.reset_index().set_index(["group", "date"])[0]

    return simulated, empirical


def _extract_aggregated_moment(results, name):
    """Extract a moment defined on the population level from list of msm results.

    Args:
        results (list): List of msm results
        name (str): Key of the moment in the moment dictionary

    Returns:
        pd.DataFrame: simulated moment per age group
        pd.Series: empirical moment per age_group

    """
    simulated = pd.DataFrame()
    for i, res in enumerate(results):
        simulated[i] = res["simulated_moments"][name]

    empirical = results[0]["empirical_moments"][name]

    return simulated, empirical


def _plot_simulated_and_empirical_moment(simulated, empirical, ax=None):
    """Plot moments into axis."""
    if ax is None:
        _, ax = plt.subplots()

    sim_color, emp_color = get_colors("categorical", 2)

    dates = simulated.index

    for run in simulated:
        sns.lineplot(x=dates, y=simulated[run], ax=ax, color=sim_color, alpha=0.3)

    sns.lineplot(
        x=dates,
        y=simulated.mean(axis=1),
        ax=ax,
        color=sim_color,
        lw=2.5,
        label="simulated",
    )

    sns.lineplot(
        x=empirical.index,
        y=empirical,
        ax=ax,
        color=emp_color,
        lw=2.5,
        label="empirical",
    )


def _split_name(name):
    if name.startswith("aggregated_"):
        moment_name = name.replace("aggregated_", "")
        group_name = None
    else:
        moment_name, group_name = name.split("_by_")
    return moment_name, group_name


def _sort_age_groups(age_groups):
    return sorted(age_groups, key=lambda x: int(x.split("-")[0]))
