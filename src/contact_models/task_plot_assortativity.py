import matplotlib.pyplot as plt
import pandas as pd
import pytask
import seaborn as sns

from src.config import BLD

_PARAMETRIZATION = [
    (
        BLD / "contact_models" / "age_assort_params" / "other_non_recurrent.pkl",
        "assortative_matching_other_non_recurrent_age_group",
        BLD / "figures" / "data" / "assortativity_other_non_recurrent.pdf",
    ),
    (
        BLD / "contact_models" / "age_assort_params" / "work_non_recurrent.pkl",
        "assortative_matching_work_non_recurrent_age_group",
        BLD / "figures" / "data" / "assortativity_work_non_recurrent.pdf",
    ),
]


@pytask.mark.parametrize("depends_on, loc, produces", _PARAMETRIZATION)
def task_create_assortativity_heatmap(depends_on, loc, produces):
    sr = pd.read_pickle(depends_on)
    fig, ax = _create_heatmap(sr, loc)
    fig.tight_layout()
    fig.savefig(produces)


def _create_heatmap(sr, loc):
    probs = sr.unstack().loc[loc]
    probs.index.name = "age group"
    probs.columns.name = "age group of contact"

    fig, ax = plt.subplots(figsize=(10, 6))
    ax = sns.heatmap(
        probs, annot=True, fmt=".2f", cbar=False, cmap="coolwarm", center=0, ax=ax
    )
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    return fig, ax
