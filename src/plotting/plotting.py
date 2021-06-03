from typing import Optional

import matplotlib.dates as dt
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

from src.calculate_moments import smoothed_outcome_per_hundred_thousand_rki
from src.config import BLD
from src.config import SRC


plt.rcParams.update(
    {
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
    }
)

# Colors
BLUE = "#4e79a7"
ORANGE = "#f28e2b"
RED = "#e15759"
TEAL = "#76b7b2"
GREEN = "#59a14f"
YELLOW = "#edc948"
PURPLE = "#b07aa1"
BROWN = "#9c755f"

ORDERED_COLORS = [BLUE, TEAL, YELLOW, ORANGE, RED, PURPLE]


def calculate_virus_strain_shares(results):
    """Create the weekly incidences from a list of simulation runs.

    Args:
        results (list): list of DataFrames with the time series data from sid
            simulations.

    Returns:
        virus_strain_shares (pandas.DataFrame): every column is the
            weekly incidence over time for one simulation run.
            The index are the dates of the simulation period.

    """
    to_concat = []
    for res in results:
        new_known_case = res[res["new_known_case"]]
        n_infected_per_day = new_known_case["date"].value_counts().compute()
        grouped = new_known_case.groupby("date")
        # date and strain as MultiIndex
        n_strain_per_day = grouped["virus_strain"].value_counts()
        n_strain_per_day = n_strain_per_day.compute()
        n_strain_per_day = n_strain_per_day.unstack()
        share_strain_per_day = n_strain_per_day.divide(n_infected_per_day, axis=0)
        to_concat.append(share_strain_per_day.stack())
    strain_shares = pd.concat(to_concat, axis=1).unstack()
    strain_shares = strain_shares.swaplevel(axis=1)
    strain_shares.index.name = "date"
    return strain_shares


def plot_incidences(
    incidences,
    title,
    name_to_label,
    colors,
    n_single_runs: Optional[int] = 0,
    empirical=False,
    scenario_starts=None,
    fig=None,
    ax=None,
):
    """Plot incidences.

    Args:
        incidences (dict): keys are names of the scenarios, values are dataframes where
            each column is the incidence of interest of one run
        title (str): plot title.
        name_to_label (dict): keys must contain the ones in *incidences*. Values will be
            plotted as labels of the scenarios in the figure's legend.
        n_single_runs (Optional[int or None]): Number of individual runs with
            different seeds visualize to show statistical uncertainty. Passing ``None``
            will plot all runs.
        empirical (str or bool): if str name of the empirical outcome to be added to
            the plot.
        scenario_starts (list, optional): the scenario start points. Each consists of a
            tuple of the date and a label.

    Returns:
        fig, ax

    """
    if fig is None and ax is None:
        if len(incidences) <= 4:
            fig, ax = plt.subplots(figsize=(8, 4))
        else:
            fig, ax = plt.subplots(figsize=(8, 6))

    dates = list(incidences.values())[0].index
    if empirical in ["new_known_case", "newly_deceased"]:
        rki_data = pd.read_pickle(BLD / "data" / "processed_time_series" / "rki.pkl")
        rki_dates = rki_data.index.get_level_values("date")
        dates = rki_dates.intersection(dates).unique().sort_values()

    if colors is None:
        colors = [BLUE, ORANGE, RED, TEAL, GREEN, YELLOW, PURPLE, BROWN]
    for (name, df), color in zip(incidences.items(), colors):
        sns.lineplot(
            x=dates,
            y=df.loc[dates].mean(axis=1),
            ax=ax,
            color=color,
            label=name_to_label[name] if name in name_to_label else name,
            linewidth=2.0,
            alpha=0.6,
        )
        # plot individual runs to visualize statistical uncertainty
        for run in df.columns[:n_single_runs]:
            sns.lineplot(
                x=dates,
                y=df.loc[dates, run],
                ax=ax,
                color=color,
                linewidth=1.0,
                alpha=0.2,
            )
    if empirical:
        if empirical in ["new_known_case", "newly_deceased"]:
            rki_data = rki_data.groupby("date").sum()
            if empirical == "new_known_case":
                rki_col = "newly_infected"
            else:
                rki_col = empirical
            label = "official case numbers"

            weekly_smoothed = (
                smoothed_outcome_per_hundred_thousand_rki(
                    df=rki_data,
                    outcome=rki_col,
                    take_logs=False,
                    window=7,
                )
                * 7
            )
            sns.lineplot(
                x=dates, y=weekly_smoothed[dates], ax=ax, color="k", label=label
            )
        else:
            cosmo_share, cosmo_label = _get_cosmo_share(empirical)
            sns.lineplot(
                x=cosmo_share.index,
                y=cosmo_share,
                label=cosmo_label,
                ax=ax,
                color="k",
            )

    if scenario_starts is not None:
        if isinstance(scenario_starts, (str, pd.Timestamp)):
            scenario_starts = [(scenario_starts, "scenario start")]

        for date, label in scenario_starts:
            ax.axvline(
                pd.Timestamp(date),
                label=label,
                color="darkgrey",
            )

    fig, ax = style_plot(fig, ax)
    ax.set_title(title)
    if "New Cases" in title or "New Deaths" in title:
        ax.set_ylabel("smoothed weekly incidence")
    elif "share" in title.lower():
        ax.set_ylabel("share")
    elif "Effective" in title:
        ax.set_ylabel("$R_t$")
    x, y, width, height = 0.0, -0.3, 1, 0.2
    ax.legend(loc="upper center", bbox_to_anchor=(x, y, width, height), ncol=2)
    fig.tight_layout()
    return fig, ax


def plot_share_known_cases(share_known_cases, title, plot_single_runs=False):
    sns.set_palette(ORDERED_COLORS)
    fig, ax = plt.subplots(figsize=(10, 5))

    if plot_single_runs:
        for col in share_known_cases:
            alpha = 0.6 if col == "mean" else 0.2
            linewidth = 2.5 if col == "mean" else 1
            sns.lineplot(
                data=share_known_cases.reset_index(),
                x="date",
                y=col,
                hue="age_group_rki",
                linewidth=linewidth,
                alpha=alpha,
            )

    else:
        sns.lineplot(
            data=share_known_cases.reset_index(),
            x="date",
            y="mean",
            hue="age_group_rki",
            linewidth=2.5,
            alpha=0.6,
        )

    fig, ax = style_plot(fig, ax)
    ax.set_title(title)

    # Reduce legend to have each age group only once and move it to below the plot
    x, y, width, height = 0.0, -0.3, 1, 0.2
    n_groups = share_known_cases.index.get_level_values("age_group_rki").nunique()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles[:n_groups],
        labels[:n_groups],
        loc="upper center",
        bbox_to_anchor=(x, y, width, height),
        ncol=n_groups,
        title=None,
    )

    fig.tight_layout()

    # undo side effect on color palette
    sns.color_palette()
    return fig, ax


def plot_group_time_series(df, title, rki=None):
    """Plot a time series by group with more than one run.

    Args:
        df (pandas.DataFrame): index levels are dates and group identifiers.
            There is one column for each simulation run.
        title (str): the title of the plot
        rki (pandas.Series, optional): Series with the RKI data. Must have the same
            index as df.

    """
    df = df.swaplevel().copy(deep=True)
    groups = df.index.levels[0].unique()
    dates = df.index.levels[1].unique()

    n_rows = int(np.ceil(len(groups) / 2))
    fig, axes = plt.subplots(
        figsize=(12, n_rows * 3), nrows=n_rows, ncols=2, sharey=True
    )
    axes = axes.flatten()

    if "0-4" in groups:
        colors = ORDERED_COLORS
    else:
        colors = [BLUE] * len(groups)

    for group, ax in zip(groups, axes):
        plot_incidences(
            incidences={group: df.loc[group]},
            title=title.format(group=group),
            name_to_label={group: "simulated"},
            empirical=False,
            colors=colors,
            scenario_starts=None,
            fig=fig,
            ax=ax,
        )

        if rki is not None:
            rki_data = rki.loc[dates, group].reset_index()
            sns.lineplot(
                x=rki_data["date"],
                y=rki_data[0],
                ax=ax,
                color="k",
                label="official case numbers",
            )

    return fig, axes


def style_plot(fig, axes):
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    for ax in axes:
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.grid(axis="y")
        ax = format_date_axis(ax)
    sns.despine()
    return fig, ax


def format_date_axis(ax):
    ax.xaxis.set_major_locator(dt.MonthLocator())
    # for month and year use "%b %Y"
    ax.xaxis.set_major_formatter(dt.DateFormatter("%B"))
    ax.xaxis.set_minor_locator(dt.DayLocator())
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    return ax


def shorten_dfs(dfs, empirical, plot_start=None):
    """Shorten all incidence DataFrames.

    All DataFrames are shortened to the shortest. In addition, if plot_start is given
    all DataFrames start at or after plot_start.

    Args:
        dfs (dict): keys are the names of the scenarios, values are the incidence
            DataFrames.
        empirical (str or bool): if not False and empirical is the name of an
            outcome that is in the RKI data, reduce the DataFrames to the time
            for which RKI data is available.
        plot_start (pd.Timestamp or None): earliest allowed start date for the plot

    Returns:
        shortened (dict): keys are the names of the scenarios, values are the shortened
            DataFrames.

    """
    shortened = {}

    start_date = max(df.index.min() for df in dfs.values())
    end_date = min(df.index.max() for df in dfs.values())
    if plot_start is not None and plot_start < end_date:
        start_date = max(plot_start, start_date)
    if empirical and empirical in ["new_known_case", "newly_deceased"]:
        rki_data = pd.read_pickle(BLD / "data" / "processed_time_series" / "rki.pkl")
        end_date = min(end_date, rki_data.index.get_level_values("date").max())

    for name, df in dfs.items():
        shortened[name] = df.loc[start_date:end_date].copy(deep=True)

    return shortened


def create_automatic_labels(names):
    name_to_label = {}
    for name in names:
        name_to_label[name] = make_name_nice(name)
    return name_to_label


def make_name_nice(name):
    """Make a scenario name nice.

    Args:
        name (str): name of the scenario

    Returns:
        nice_name (str): nice name of the scenario

    """
    replacements = [
        ("_", " "),
        (" with", "\n with"),
        ("fall", ""),
        ("spring", ""),
        ("summer", ""),
    ]
    nice_name = name
    for old, new in replacements:
        nice_name = nice_name.replace(old, new)
    nice_name = nice_name.lstrip("\n")
    return nice_name


def make_nice_outcome(outcome):
    outcome_to_nice_outcome = {
        "new_known_case": "Observed New Cases",
        "newly_infected": "Total New Cases",
        "newly_deceased": "New Deaths",
        "share_ever_rapid_test": "\nShare of People who Have Ever Done a Rapid Test\n",
        "share_rapid_test_in_last_week": "\nShare of People who Have Done a Rapid Test"
        + "\nin the Last Week",
        "r_effective": "the Effective Reproduction Number",
    }

    nice_outcome = outcome_to_nice_outcome.get(
        outcome, outcome.replace("_", " ").title()
    )
    return nice_outcome


def _get_cosmo_share(empirical):
    if empirical == "share_ever_rapid_test":
        cosmo_share = pd.read_csv(
            SRC / "original_data" / "testing" / "cosmo_share_ever_had_a_rapid_test.csv",
            parse_dates=["date"],
            index_col="date",
        )
        label = "share of Germans reporting to have\n" "ever done a rapid test"
        cosmo_share = cosmo_share["share_ever_had_a_rapid_test"]
    elif empirical == "share_rapid_test_in_last_week":
        cosmo_share = pd.read_csv(
            SRC
            / "original_data"
            / "testing"
            / "cosmo_selftest_frequency_last_four_weeks.csv",
            parse_dates=["date"],
            index_col="date",
        )
        weekly_or_more_cols = [
            "share_more_than_5_tests_per_week",
            "share_5_tests_per_week",
            "share_2-4_tests_per_week",
            "share_weekly",
        ]
        cosmo_share = cosmo_share[weekly_or_more_cols].sum(axis=1)
        label = (
            "share of Germans reporting to have\n"
            "done at least one self-administered\n"
            "rapid test per week within the last 4 weeks"
        )
    else:
        raise ValueError(f"No known empirical equivalent for {empirical}.")
    return cosmo_share, label
