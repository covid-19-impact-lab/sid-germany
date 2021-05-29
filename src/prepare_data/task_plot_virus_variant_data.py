import matplotlib.pyplot as plt
import pandas as pd
import pytask
import seaborn as sns
from sid import get_colors

from src.config import BLD
from src.config import SRC
from src.plotting.plotting import style_plot
from src.prepare_data.task_prepare_virus_variant_data import STRAIN_FILES

_MODULE_DEPENDENCIES = {
    "plotting.py": SRC / "plotting" / "plotting.py",
}


@pytask.mark.depends_on(_MODULE_DEPENDENCIES)
@pytask.mark.depends_on(STRAIN_FILES)
@pytask.mark.produces(BLD / "data" / "virus_strains" / "b117.png")
def task_plot_virus_variant_data(depends_on, produces):
    rki_b117 = pd.read_csv(
        depends_on["rki_strains"],
        parse_dates=["date"],
        index_col="date",
    )["share_b117"]

    our_b117 = pd.read_pickle(depends_on["virus_shares_dict"])["b117"]

    title = "Share of Virus Variants Over Time"

    fig, ax = plt.subplots()

    for sr, label, style in zip(
        [rki_b117, our_b117], ["rki", "extrapolated"], ["-", "--"]
    ):
        sr = sr["2020-11-01":"2021-05-01"]
        sns.lineplot(x=sr.index, y=sr, label=label, ax=ax, linestyle=style)

    fig.savefig(produces, dpi=200, transparent=False, facecolor="w")
    plt.close()
