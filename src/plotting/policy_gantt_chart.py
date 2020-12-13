import matplotlib.pyplot as plt
import pandas as pd
from estimagic.visualization.colors import get_colors


def make_gantt_chart_of_policy_dict(
    policies, title=None, bar_height=0.8, bar_color=None, edge_color=None, alpha=1
):
    cm_names = sorted({pol["affected_contact_model"] for pol in policies.values()})
    positions = dict(zip(cm_names, range(len(cm_names))))
    fig, ax = plt.subplots(figsize=(12, len(cm_names)))
    edge_color = get_colors("categorical", 1)[0] if edge_color is None else edge_color
    bar_color = "#ffffff00" if bar_color is None else bar_color

    for pol in policies.values():
        affected_model = pol["affected_contact_model"]
        start = pd.Timestamp(pol["start"])
        end = pd.Timestamp(pol["end"])
        ax.broken_barh(
            xranges=[(start, end - start)],
            yrange=(positions[affected_model] - 0.5 * bar_height, bar_height),
            edgecolors=edge_color,
            facecolors=bar_color,
            alpha=alpha,
        )
    ax.set_yticks(range(len(cm_names)))
    ax.set_yticklabels(cm_names)
    if title is not None:
        ax.set_title(title.replace("_", " ").title())
    return fig, ax
