import warnings

import matplotlib.pyplot as plt
import pandas as pd
import pytask
import seaborn as sns

from src.config import BLD
from src.config import POPULATION_GERMANY
from src.simulation.plotting import style_plot
from src.testing.shared import get_piecewise_linear_interpolation


@pytask.mark.depends_on(
    {
        "params": BLD / "params.pkl",
        "rki": BLD / "data" / "processed_time_series" / "rki.pkl",
    }
)
@pytask.mark.produces(
    BLD / "data" / "testing" / "share_of_hh_members_with_rapid_test_after_hh_event.png"
)
def task_plot_share_of_workers_receiving_test_offer(depends_on, produces):
    rki = pd.read_pickle(BLD / "data" / "processed_time_series" / "rki.pkl")
    rki = rki.groupby("date")["newly_infected"].sum()
    rki = rki.rolling(7).sum().dropna()
    rki = 100_000 * rki / POPULATION_GERMANY
    rki = rki["2021-03-01":]

    params = pd.read_pickle(depends_on["params"])
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="indexing past lexsort depth may impact performance."
        )
        params_slice = params.loc[("rapid_test_demand", "hh_member_demand")]
    share_hh_members_demanding_test = get_piecewise_linear_interpolation(params_slice)
    share_hh_members_demanding_test = share_hh_members_demanding_test.loc[
        "2021-03-01" : rki.index.max()
    ]

    fig, axes = plt.subplots(2, figsize=(10, 10), sharex=True)
    sns.lineplot(x=rki.index, y=rki, ax=axes[0])
    axes[0].grid()
    axes[0].set_title("Incidence")

    sns.lineplot(
        x=share_hh_members_demanding_test.index,
        y=share_hh_members_demanding_test,
        ax=axes[1],
    )
    axes[1].grid()
    axes[1].set_title(
        "Share of Household Members Demanding Rapid Test\n"
        "When Household Member Tests Positive Or Becomes Symptomatic"
    )
    fig, axes = style_plot(fig, axes)
    fig.tight_layout()

    fig.savefig(produces)
    plt.close()
