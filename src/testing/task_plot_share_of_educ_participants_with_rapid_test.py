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
        # "plotting.py": SRC / "plotting" / "plotting.py",
        "testing_shared.py": SRC / "testing" / "shared.py",
    }
)
@pytask.mark.produces(
    BLD
    / "figures"
    / "data"
    / "testing"
    / "share_of_educ_participants_with_rapid_test.pdf"
)
def task_plot_share_of_educ_participants_with_rapid_test(depends_on, produces):
    params = pd.read_pickle(depends_on["params"])
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="indexing past lexsort depth may impact performance."
        )
        educ_workers_params = params.loc[("rapid_test_demand", "educ_worker_shares")]
        students_params = params.loc[("rapid_test_demand", "student_shares")]

    share_educ_workers = get_piecewise_linear_interpolation(educ_workers_params)
    share_educ_workers = share_educ_workers.loc[PLOT_START_DATE:PLOT_END_DATE]

    share_students = get_piecewise_linear_interpolation(students_params)
    share_students = share_students.loc[PLOT_START_DATE:PLOT_END_DATE]

    fig, ax = plt.subplots(figsize=PLOT_SIZE)
    sns.lineplot(
        x=share_educ_workers.index,
        y=share_educ_workers,
        ax=ax,
        label="Teachers (School, Preschool, Nursery)",
    )
    sns.lineplot(
        x=share_students.index,
        y=share_students,
        ax=ax,
        label="School Students",
    )
    ax.set_title("Share of Students and Teachers Receiving Rapid Tests")
    fig, ax = style_plot(fig, ax)
    fig.tight_layout()

    fig.savefig(produces)
    plt.close()
