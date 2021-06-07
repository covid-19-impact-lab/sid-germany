import warnings

import matplotlib.pyplot as plt
import pandas as pd
import pytask
import seaborn as sns

from src.config import BLD
from src.config import PLOT_END_DATE
from src.config import PLOT_SIZE
from src.config import PLOT_START_DATE
from src.config import SRC
from src.plotting.plotting import style_plot
from src.testing.shared import get_piecewise_linear_interpolation


@pytask.mark.depends_on(
    {
        "params": BLD / "params.pkl",
        "rki": BLD / "data" / "processed_time_series" / "rki.pkl",
        # "plotting.py": SRC / "plotting" / "plotting.py",
        "testing_shared.py": SRC / "testing" / "shared.py",
    }
)
@pytask.mark.produces(
    BLD / "figures" / "data" / "testing" / "private_test_demand_shares.pdf"
)
def task_plot_private_test_demand_shares(depends_on, produces):
    params = pd.read_pickle(depends_on["params"])
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="indexing past lexsort depth may impact performance."
        )
        params_slice = params.loc[("rapid_test_demand", "private_demand")]

    private_demand_shares = get_piecewise_linear_interpolation(params_slice)
    private_demand_shares = private_demand_shares.loc[PLOT_START_DATE:PLOT_END_DATE]

    fig, ax = plt.subplots(figsize=PLOT_SIZE)
    sns.lineplot(
        x=private_demand_shares.index,
        y=private_demand_shares,
        ax=ax,
    )
    ax.set_title(
        "Private Rapid Test Demand\n"
        "(Share of Individuals who Do a Rapid Test \n"
        "When a Household Member Tests Positive Or Becomes Symptomatic Or \n"
        "When Developing Symptoms but not Receiving a Rapid Test Or \n"
        "When Participating in Some Private Events)"
    )
    fig, ax = style_plot(fig, ax)
    fig.tight_layout()

    fig.savefig(produces)
    plt.close()
