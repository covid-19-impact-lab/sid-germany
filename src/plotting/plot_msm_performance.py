import matplotlib.pyplot as plt
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


def plot_msm_performance(iteration):
    """Plot the moment performance contrasting empirical and simulated moments.

    Args:
        iteration (dict): estimagic optimization iteration

    Returns:
        fig, axes

    """
    colors = get_colors(palette="categorical", number=2)
    sim_mom = iteration["simulated_moments"]
    emp_mom = iteration["empirical_moments"]

    by_age = ["infections_by_age_group"]
    n_age_groups = 6
    n_plots = n_age_groups * len(by_age)

    fig, axes = plt.subplots(n_plots, figsize=(10, 20), sharex=False)

    pos = 0

    for key in by_age:
        nice_key = key.replace("_", " ")

        sim_age_time_series = _convert_to_dataframe_with_age_groups_as_columns(
            sim_mom[key]
        )
        emp_age_time_series = _convert_to_dataframe_with_age_groups_as_columns(
            emp_mom[key]
        )
        for age_group in sorted(sim_age_time_series.columns):
            sns.lineplot(
                x=sim_age_time_series.index,
                y=sim_age_time_series[age_group],
                label=f"simulated {nice_key} in {age_group}",
                color=colors[0],
                ax=axes[pos],
            )
            sns.lineplot(
                x=emp_age_time_series.index,
                y=emp_age_time_series[age_group],
                label=f"empirical {nice_key} in {age_group}",
                color=colors[1],
                ax=axes[pos],
            )
            axes[pos].set_title(
                f"{nice_key.title()} among {age_group} Year Olds", fontsize=15
            )
            axes[pos].set_ylabel(f"{nice_key} per 100 000")
            pos += 1

    for ax in axes:
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))

    fig.tight_layout()

    return fig, axes


def _convert_to_dataframe_with_age_groups_as_columns(sr):
    sr = sr.copy()
    sr.name = "value"
    df = sr.to_frame()
    df["date"] = list(map(lambda x: x.split("'", 2)[1], df.index))
    df["date"] = pd.to_datetime(df["date"])
    df["group"] = list(map(lambda x: x.rsplit(",", 1)[1].strip("') "), df.index))
    df.set_index(["date", "group"], inplace=True)
    return df["value"].unstack()
