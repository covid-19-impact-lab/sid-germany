import warnings

import matplotlib.pyplot as plt
import pandas as pd
import pytask
import seaborn as sns

from src.config import BLD
from src.config import SRC
from src.plotting.plotting import style_plot
from src.testing.shared import get_piecewise_linear_interpolation


@pytask.mark.depends_on(
    {
        "params": BLD / "params.pkl",
        "config.py": SRC / "config.py",
        "plotting.py": SRC / "plotting" / "plotting.py",
        "testing_shared.py": SRC / "testing" / "shared.py",
    }
)
@pytask.mark.produces(
    BLD / "data" / "testing" / "share_of_educ_participants_with_rapid_test.png"
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
    share_educ_workers = share_educ_workers.loc["2021-01-01":"2021-07-01"]

    share_students = get_piecewise_linear_interpolation(students_params)
    share_students = share_students.loc["2021-01-01":"2021-07-01"]

    fig, ax = plt.subplots(figsize=(8, 3))
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
    ax.set_title("Share Educ Participants Getting Rapid Test")
    fig, ax = style_plot(fig, ax)
    fig.tight_layout()

    fig.savefig(produces)
    plt.close()
