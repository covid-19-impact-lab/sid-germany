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
@pytask.mark.produces(
    BLD / "data" / "virus_strains" / "figures" / "averaged_strain_shares.png"
)
def task_plot_virus_variant_data(depends_on, produces):
    rki_strains = pd.read_csv(
        depends_on["rki_strains"],
        parse_dates=["date"],
        index_col="date",
    )
    cologne = pd.read_csv(
        depends_on["cologne"],
        parse_dates=["date"],
        index_col="date",
    )
    final_strain_shares = pd.read_pickle(depends_on["final_strain_shares"])
    final_strain_shares = final_strain_shares.loc["2020-12-15":]

    title = "Share of Virus Variants Over Time"
    fig, ax = _plot_shares(final_strain_shares, title=title)
    fig.savefig(produces)
    plt.close()

    for col in final_strain_shares:
        to_plot = [(final_strain_shares[col], "extrapolated")]
        if f"share_{col}" in rki_strains:
            to_plot.append((rki_strains[f"share_{col}"], "RKI Data"))
        if f"share_{col}" in cologne:
            to_plot.append((cologne[f"share_{col}"], "Cologne Data"))
            to_plot.append(
                (cologne[f"share_{col}_unsmoothed"], "Unsmoothed Cologne Data")
            )
        title = f"Share of {col.title()} Over Time"
        fig, ax = _plot_shares(to_plot, title)
        fig.savefig(
            produces.parent / f"{col}.png", dpi=200, transparent=False, facecolor="w"
        )
        plt.close()


def _plot_shares(shares, title):
    """Plot a list of Series and names."""
    if isinstance(shares, pd.DataFrame):
        shares = [(shares[col], col.title()) for col in shares]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = get_colors("categorical", len(shares))
    line_styles = ["dotted", "--", "-.", "dotted"]
    for i, (color, (sr, label), style) in enumerate(zip(colors, shares, line_styles)):
        sns.lineplot(
            x=sr.index,
            y=sr,
            color=color,
            linewidth=2,
            label=label,
            linestyle=style,
            alpha=1 - i * 0.2,
        )
    ax.set_title(title)
    fig, ax = style_plot(fig, ax)
    return fig, ax
