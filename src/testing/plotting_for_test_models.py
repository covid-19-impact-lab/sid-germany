import matplotlib.pyplot as plt
import seaborn as sns
from sid.colors import get_colors

from src.simulation.plotting import plot_incidences
from src.simulation.plotting import style_plot
from src.simulation.plotting import weekly_incidences_from_results


def plot_known_vs_total_cases(res, title):
    new_known_case = weekly_incidences_from_results([res], "new_known_case")
    newly_infected = weekly_incidences_from_results([res], "newly_infected")

    to_plot = {"newly_infected": newly_infected, "new_known_cases": new_known_case}
    name_to_label = {
        "newly_infected": "newly infected",
        "new_known_case": "new known case",
    }

    fig, ax = plot_incidences(
        to_plot, title=title, n_single_runs=0, name_to_label=name_to_label
    )
    return fig, ax


def plot_share_known_cases(res, title):
    new_known_case = weekly_incidences_from_results([res], "new_known_case")
    newly_infected = weekly_incidences_from_results([res], "newly_infected")
    share_known_case = new_known_case / newly_infected

    colors = get_colors("ordered", 4)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.lineplot(
        x=share_known_case.index,
        y=share_known_case[0],
        label="Share Known Cases",
        color=colors[2],
        linewidth=2.5,
        alpha=0.8,
        ax=ax,
    )
    fig, ax = style_plot(fig, ax)
    ax.set_title(title)
    return fig, ax
