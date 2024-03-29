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
        "plotting.py": SRC / "plotting" / "plotting.py",
        "shared.py": SRC / "testing" / "shared.py",
    }
)
@pytask.mark.produces(
    BLD
    / "figures"
    / "data"
    / "testing"
    / "share_of_workers_with_rapid_test_offer_at_work.pdf"
)
def task_plot_share_of_workers_receiving_test_offer(depends_on, produces):
    params = pd.read_pickle(depends_on["params"])
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="indexing past lexsort depth may impact performance."
        )
        params_slice = params.loc[
            ("rapid_test_demand", "share_workers_receiving_offer")
        ]
    share_workers_receiving_offer = get_piecewise_linear_interpolation(params_slice)
    share_workers_receiving_offer = share_workers_receiving_offer.loc[
        PLOT_START_DATE:PLOT_END_DATE
    ]

    fig, ax = plt.subplots(figsize=PLOT_SIZE)
    sns.lineplot(
        x=share_workers_receiving_offer.index,
        y=share_workers_receiving_offer,
        ax=ax,
    )
    ax.set_title("Share Workers With Work Contacts With Rapid Test Offer At Work")
    fig, ax = style_plot(fig, ax)
    fig.tight_layout()

    fig.savefig(produces)
    plt.close()
