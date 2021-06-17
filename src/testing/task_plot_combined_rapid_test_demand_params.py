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
    {
        "figure": BLD / "figures" / "data" / "testing" / "rapid_test_demand_shares.pdf",
        "data": BLD / "tables" / "rapid_test_demand_shares.csv",
    }
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
        work_accept_params = params.loc[
            ("rapid_test_demand", "share_accepting_work_offer")
        ]

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
    share_workers_accepting_offer = get_piecewise_linear_interpolation(
        work_accept_params
    )
    share_workers = share_workers_receiving_offer * share_workers_accepting_offer

    # private demand
    private_demand_shares = get_piecewise_linear_interpolation(private_demand_params)
    private_demand_shares = private_demand_shares.loc[PLOT_START_DATE:PLOT_END_DATE]

    fig = _plot_rapid_test_demand_shares(
        share_educ_workers=share_educ_workers,
        share_students=share_students,
        share_workers=share_workers,
        private_demand_shares=private_demand_shares,
    )
    fig.savefig(produces["figure"])
    plt.close()

    df = pd.DataFrame(
        {
            "share_educ_workers": share_educ_workers,
            "share_students": share_students,
            "share_workers": share_workers,
            "private_demand_shares": private_demand_shares,
        }
    )
    df.round(3).to_csv(produces["data"])


def _plot_rapid_test_demand_shares(
    share_educ_workers,
    share_students,
    share_workers,
    private_demand_shares,
):
    named_lines = [
        (share_educ_workers, "Teachers (School, Preschool, Nursery)", PURPLE),
        (share_students, "School Students", RED),
        (share_workers, "Workers", BLUE),
        (private_demand_shares, "Private Reasons", GREEN),
    ]

    fig, ax = plt.subplots(figsize=PLOT_SIZE)

    for sr, label, color in named_lines:
        sns.lineplot(
            x=sr.index,
            y=sr,
            ax=ax,
            label=label,
            color=color,
            alpha=0.8,
            linewidth=3,
        )
    fig, ax = style_plot(fig, ax)
    fig.tight_layout()
    return fig
