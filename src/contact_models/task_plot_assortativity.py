import matplotlib.pyplot as plt
import pandas as pd
import pytask
import seaborn as sns

from src.config import BLD

_PARAMETRIZATION = [
    (
        BLD / "contact_models" / "cell_counts" / "other_non_recurrent.pkl",
        BLD / "figures" / "data" / "assortativity_other_non_recurrent.pdf",
    ),
    (
        BLD / "contact_models" / "cell_counts" / "work_non_recurrent.pkl",
        BLD / "figures" / "data" / "assortativity_work_non_recurrent.pdf",
    ),
]


@pytask.mark.parametrize("depends_on, produces", _PARAMETRIZATION)
def task_create_assortativity_heatmap(depends_on, produces):
    cell_counts = pd.read_pickle(depends_on)
    fig, ax = _create_heatmap(cell_counts)
    fig.tight_layout()
    fig.savefig(produces)
    plt.close()


def _create_heatmap(cell_counts):
    cell_counts.index.name = "age group"
    cell_counts.columns.name = "age group of contact"

    fig, ax = plt.subplots(figsize=(10, 8))
    ax = sns.heatmap(
        cell_counts, annot=True, fmt=".0f", cbar=False, cmap="coolwarm", center=0, ax=ax
    )
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    return fig, ax
