import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from estimagic.visualization.colors import get_colors
from sid.plotting import plot_infection_rates_by_contact_models

from src.plotting.plotting import format_date_axis


plt.rcParams.update(
    {
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
    }
)


def plot_estimation_moment(results, name):
    """Visualize an estimation moment over several runs.

    It is assumed that the entries in results only differ by their random seed.

    Args:
        results (list): List of msm criterion outputs
        name (str): Name of the estimation moment.

    """
    moment_name, group_name = _split_name(name)

    if group_name is not None:
        simulated, empirical = _extract_grouped_moment(results, name)
        raw_groups = simulated.index.get_level_values("group").unique()
        if group_name == "age_group":
            groups = _sort_age_groups(raw_groups)
        else:
            groups = sorted(raw_groups)

        n_groups = len(groups)

        n_rows = int(np.ceil(n_groups / 2))
        fig, axes = plt.subplots(figsize=(13.5, n_rows * 6), nrows=n_rows, ncols=2)
        axes = axes.flatten()
        for group, ax in zip(groups, axes):
            _plot_simulated_and_empirical_moment(
                simulated=simulated.loc[group],
                empirical=empirical.loc[group],
                ax=ax,
            )
            ax.set_title(f"{group_name}: {group}")
    else:
        simulated, empirical = _extract_aggregated_moment(results, name)
        fig, ax = plt.subplots(figsize=(10, 8))
        axes = [ax]
        _plot_simulated_and_empirical_moment(
            simulated=simulated, empirical=empirical, ax=ax
        )

    for ax in axes:
        ax.set_ylabel(moment_name)
        ax = format_date_axis(ax)

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
        plot_line_with_gaps(
            x=dates, y=simulated[run], ax=ax, color=sim_color, alpha=0.15
        )

    plot_line_with_gaps(
        x=dates,
        y=simulated.mean(axis=1),
        ax=ax,
        color=sim_color,
        lw=2.5,
        label="simulated",
    )

    plot_line_with_gaps(
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


def plot_infection_channels(results, aggregate=False, unit="incidence"):
    """Plot average infection channels over several runs.

    It is assumed that the entries in results only differ by their random seed.

    Args:
        results (list): List of msm criterion outputs
        aggregate (bool): Whether contact models are aggregated over the domains
            work, households, school, young_educ and other.

    """
    to_concat = []
    for i, res in enumerate(results):
        df = res["infection_channels"].copy()
        df["run"] = i
        to_concat.append(df)

    raw_channels = pd.concat(to_concat)

    channels = (
        raw_channels.groupby(
            [pd.Grouper(key="date", freq="D"), "channel_infected_by_contact"]
        )["share"]
        .mean()
        .dropna(how="any")
        .reset_index()
    )

    if aggregate:
        channels = _aggregate_models_over_domain(channels)

    plot = plot_infection_rates_by_contact_models(channels, unit=unit)
    return plot


def _aggregate_models_over_domain(df):
    df = df.copy(deep=True)
    categories = df["channel_infected_by_contact"].unique()
    replace_dict = {}
    for cat in categories:
        if "work" in cat:
            replace_dict[cat] = "work"
        elif "other" in cat:
            replace_dict[cat] = "other"
        elif "_school" in cat:
            replace_dict[cat] = "school"
        elif "educ" in cat:
            replace_dict[cat] = "young_educ"
        elif "households" in cat:
            replace_dict[cat] = "households"
        else:
            raise ValueError(f"Invalid category: {cat}")

    df["channel_infected_by_contact"] = df["channel_infected_by_contact"].replace(
        replace_dict
    )
    df = (
        df.groupby([pd.Grouper(key="date", freq="D"), "channel_infected_by_contact"])[
            "share"
        ]
        .sum()
        .reset_index()
    )

    return df


def plot_line_with_gaps(x, y, ax, **kwargs):
    """ "Lineplot that does skips where there are no observations."""
    kwargs = kwargs.copy()

    skip_points = x[pd.Series(x).diff() > pd.Timedelta(days=1)].tolist()
    skip_points.append(x.max())

    start_loc = 0
    for end in skip_points:
        end_loc = x.get_loc(end)
        current_x = x[start_loc : end_loc - 1]
        current_y = y[start_loc : end_loc - 1]
        ax = sns.lineplot(x=current_x, y=current_y, ax=ax, **kwargs)

        start_loc = end_loc
        if "label" in kwargs.keys():
            kwargs["label"] = None

    return ax
