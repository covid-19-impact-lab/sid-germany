from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytask

from src.config import BLD
from src.config import SRC
from src.plotting.plotting import BLUE
from src.plotting.plotting import PLOT_SIZE
from src.plotting.plotting import style_plot
from src.policies.find_people_to_vaccinate import find_people_to_vaccinate


@pytask.mark.depends_on(
    {
        "states": BLD / "data" / "initial_states.parquet",
        "vaccination_shares": BLD
        / "data"
        / "vaccinations"
        / "vaccination_shares_extended.pkl",
        "params": BLD / "params.pkl",
        "empirical_by_age": BLD
        / "data"
        / "vaccinations"
        / "vaccinations_by_age_group.pkl",
        "vaccination_func": SRC / "policies" / "find_people_to_vaccinate.py",
    }
)
@pytask.mark.produces(
    BLD / "figures" / "vaccinations" / "simulated_vs_empirical_rates_by_age_group.pdf"
)
def task_compare_vaccination_rates_by_age_group(depends_on, produces):
    vaccination_shares = pd.read_pickle(depends_on["vaccination_shares"])
    empirical_by_age = pd.read_pickle(depends_on["empirical_by_age"]).T
    params = pd.read_pickle(depends_on["params"])
    states = pd.read_parquet(depends_on["states"])
    states["ever_vaccinated"] = False

    start = pd.Timestamp("2020-12-15")
    end = vaccination_shares.index.max()
    dates = pd.date_range(start, end)

    vaccination_func = partial(
        find_people_to_vaccinate,
        vaccination_shares=vaccination_shares,
        init_start=start,
        params=params,
    )

    modeled_vaccination_shares = _simulate_vaccinations_alone(
        states, dates, vaccination_func
    )
    fig = _plot_simulated_vs_empirical_vaccination_shares(
        modeled_vaccination_shares=modeled_vaccination_shares,
        empirical_by_age=empirical_by_age,
        overall_vaccination_shares=vaccination_shares,
    )
    fig.savefig(produces)


def _simulate_vaccinations_alone(states, dates, vaccination_func):
    """Simulate just the vaccinations over a time frame.

    Args:
        states (pandas.DataFrame): states DataFrame, must have "age" and
            "ever_vaccinated".
        dates (iterable): list of dates.
        vaccination_func (callable): vaccination function that expects the
            arguments "states", "receives_vaccine" and "seed"

    Returns:
        pandas.DataFrame: columns are "overall", "12-17", "18-59", ">=60".
            Each row is a date. Each cell contains the share of the respective
            age group on the particular date.

    """
    states = states.copy(deep=True)

    out = pd.DataFrame(
        np.nan, columns=["overall", "12-17", "18-59", ">=60"], index=dates
    )

    for seed, date in enumerate(dates):
        states["date"] = date
        to_vaccinate = vaccination_func(
            receives_vaccine=pd.Series(False, index=states.index),
            states=states,
            seed=seed,
        )
        states.loc[to_vaccinate, "ever_vaccinated"] = True
        out.loc[date, "overall"] = states["ever_vaccinated"].mean()
        out.loc[date, "12-17"] = states.query("12 <= age <= 17")[
            "ever_vaccinated"
        ].mean()
        out.loc[date, "18-59"] = states.query("18 <= age <= 59")[
            "ever_vaccinated"
        ].mean()
        out.loc[date, ">=60"] = states.query("60 <= age")["ever_vaccinated"].mean()
    return out


def _plot_simulated_vs_empirical_vaccination_shares(
    modeled_vaccination_shares, empirical_by_age, overall_vaccination_shares
):
    fig, axes = plt.subplots(
        nrows=2, ncols=2, figsize=(2 * PLOT_SIZE[0], 2 * PLOT_SIZE[1])
    )
    for col, ax in zip(empirical_by_age, axes.flatten()):
        modeled_vaccination_shares[col].plot(
            label="simulated",
            alpha=0.6,
            ax=ax,
            lw=2.5,
            color=BLUE,
        )
        if col != "overall":
            empirical_by_age[col].plot(
                label="empirical", alpha=0.6, ax=ax, lw=2.5, color="k"
            )
        else:
            overall_vaccination_shares["2021-01-01":].cumsum().plot(
                label="empirical overall",
                alpha=0.6,
                ax=ax,
                lw=2.5,
                color="k",
            )

        ax.set_title(col)
        ax.legend()
        ax.set_ylim(-0.1, 1.1)
        fig, ax = style_plot(fig, ax)
    return fig
