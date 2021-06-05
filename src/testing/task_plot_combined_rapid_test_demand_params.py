import warnings

import matplotlib.pyplot as plt
import pandas as pd
import pytask
import seaborn as sns

from src.config import BLD
from src.config import PLOT_END_DATE
from src.config import PLOT_START_DATE
from src.config import SRC
from src.plotting.plotting import BLUE
from src.plotting.plotting import GREEN
from src.plotting.plotting import PURPLE
from src.plotting.plotting import RED
from src.plotting.plotting import style_plot
from src.testing.shared import get_piecewise_linear_interpolation

_DEPENDENCIES = {
    "params": BLD / "params.pkl",
    "plotting.py": SRC / "plotting" / "plotting.py",
    "testing_shared.py": SRC / "testing" / "shared.py",
}


@pytask.mark.depends_on(_DEPENDENCIES)
@pytask.mark.produces(
    BLD / "figures" / "data" / "testing" / "rapid_test_demand_shares.pdf"
)
def task_plot_combined_rapid_test_demand_params(depends_on, produces):
    params = pd.read_pickle(depends_on["params"])
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="indexing past lexsort depth may impact performance."
        )
        educ_workers_params = params.loc[("rapid_test_demand", "educ_worker_shares")]
        students_params = params.loc[("rapid_test_demand", "student_shares")]
        work_offer_params = params.loc[
            ("rapid_test_demand", "share_workers_receiving_offer")
        ]
        private_demand_params = params.loc[("rapid_test_demand", "private_demand")]

    # educ demand
    share_educ_workers = get_piecewise_linear_interpolation(educ_workers_params)
    share_educ_workers = share_educ_workers.loc[PLOT_START_DATE:PLOT_END_DATE]
    share_students = get_piecewise_linear_interpolation(students_params)
    share_students = share_students.loc[PLOT_START_DATE:PLOT_END_DATE]

    # worker demand
    share_workers_receiving_offer = get_piecewise_linear_interpolation(
        work_offer_params
    )
    share_workers_receiving_offer = share_workers_receiving_offer.loc[
        PLOT_START_DATE:PLOT_END_DATE
    ]
    worker_demand = params.loc[
        ("rapid_test_demand", "work", "share_accepting_offer"), "value"
    ]
    share_workers = share_workers_receiving_offer * worker_demand

    # private demand
    private_demand_shares = get_piecewise_linear_interpolation(private_demand_params)
    private_demand_shares = private_demand_shares.loc[PLOT_START_DATE:PLOT_END_DATE]

    fig = _plot_rapid_test_demand_shares(
        share_educ_workers=share_educ_workers,
        share_students=share_students,
        share_workers=share_workers,
        private_demand_shares=private_demand_shares,
    )
    fig.savefig(produces)
    plt.close()


def _plot_rapid_test_demand_shares(
    share_educ_workers, share_students, share_workers, private_demand_shares
):
    fig, ax = plt.subplots(figsize=(8, 3))
    sns.lineplot(
        x=share_educ_workers.index,
        y=share_educ_workers,
        ax=ax,
        label="Teachers (School, Preschool, Nursery)",
        color=PURPLE,
        alpha=0.8,
    )
    sns.lineplot(
        x=share_students.index,
        y=share_students,
        ax=ax,
        label="School Students",
        color=RED,
        alpha=0.8,
    )
    sns.lineplot(
        x=share_workers.index,
        y=share_workers,
        ax=ax,
        label="Workers",
        color=BLUE,
        alpha=0.8,
    )
    sns.lineplot(
        x=private_demand_shares.index,
        y=private_demand_shares,
        ax=ax,
        label="Share of household member of positively tested,\n"
        "of symptomatic individuals without PCR test\n and individuals with planned "
        "weekly leisure meeting ",
        color=GREEN,
        alpha=0.8,
    )

    fig, ax = style_plot(fig, ax)
    fig.tight_layout()
    return fig
