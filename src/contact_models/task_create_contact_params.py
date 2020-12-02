import matplotlib.pyplot as plt
import pandas as pd
import pytask
import seaborn as sns

from src.config import BLD


SUBSET_SPECS = {
    "work_non_recurrent": {
        "places": ["work"],
        "recurrent": False,
        "frequency": None,
        "weekend": False,
    },
    "other_non_recurrent": {
        "places": ["otherplace", "leisure"],
        "recurrent": False,
        "frequency": None,
        "weekend": None,
    },
    "work_recurrent_daily": {
        "places": ["work"],
        "recurrent": True,
        "frequency": "(almost) daily",
        "weekend": False,
    },
    "work_recurrent_weekly": {
        "places": ["work"],
        "recurrent": True,
        "frequency": "1-2 times a week",
        "weekend": False,
    },
    "other_recurrent": {
        "places": ["otherplace", "leisure"],
        "recurrent": True,
        "frequency": None,
        "weekend": None,
    },
}


OUT_PATH = BLD / "contact_models" / "empirical_distributions"

FIG_SPECS = [
    (spec, [OUT_PATH / "figures" / f"{name}.png", OUT_PATH / f"{name}_raw.pkl"])
    for name, spec in SUBSET_SPECS.items()
]


@pytask.mark.depends_on(BLD / "data" / "mossong_2008" / "contact_data.pkl")
@pytask.mark.parametrize("specs, produces", FIG_SPECS)
def task_calculate_and_plot_nr_of_contacts(depends_on, specs, produces):
    contacts = pd.read_pickle(depends_on)
    fig_title = produces[0].stem.replace("_", " ").title()
    n_contacts = _create_n_contacts(contacts, **specs)
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.countplot(n_contacts, ax=ax, color="goldenrod", alpha=0.7)
    ax.set_title(fig_title)
    sns.despine()
    fig.savefig(produces[0])
    n_contacts.to_pickle(produces[1])


def _create_n_contacts(contacts, places, recurrent, frequency, weekend):
    if places != ["work"]:
        relevant_ids = contacts["id"].unique()
    else:
        query = "participant_occupation == 'working'"
        relevant_ids = contacts.query(query)["id"].unique()
    df = _get_relevant_contacts_subset(contacts, places, recurrent, frequency, weekend)
    n_contacts = _calculate_n_of_contacts(df, relevant_ids)
    return n_contacts


def _get_relevant_contacts_subset(contacts, places, recurrent, frequency, weekend):
    """Reduce the contacts data to that relevant for the current contact model.

    Args:
        contacts (pandas.DataFrame)
        places (list): list of the places belonging to the current contact model
        recurrent (bool): whether to keep recurrent or non-recurrent contacts
        frequency (str): which frequency to keep. None means all frequencies are kept
        weekend (bool): whether to use weekday or weekend data.
            None means both are kept.

    Returns:
        df (pandas.DataFrame): reduced DataFrame that only contains longer contacts of
            the specified places, frequency and day of week. Reduce to only workers
            if looking at work contacts.

    """
    df = contacts.copy(deep=True)
    df = df[df["phys_contact"] | (df["duration"] > "<5min")]
    df = df[df["place"].isin(places)]
    df = df.query(f"recurrent == {recurrent}")
    if "work" in places:
        df = df[df["participant_occupation"] == "working"]
    if frequency is not None:
        df = df[df["frequency"] == frequency]
    if weekend is not None:
        df = df[df["weekend"] == weekend]
    return df


def _calculate_n_of_contacts(df, relevant_ids):
    """Sum up over individuals to get the number of contacts.

    Args:
        df (pandas.DataFrame): DataFrame reduced to relevant people and contacts.
        relevant_ids (numpy.ndarray): ids of individuals that were in the original
            dataset. If we wouldn't expand the contacts back to this we would miss
            the zero contact individuals.

    Returns:
        n_contacts (pandas.Series): index are the ids of the individuals, values
            are the number of contacts the individuals had in the relevant category.

    """
    n_contacts = df.groupby("id").size()
    missing = [x for x in relevant_ids if x not in n_contacts.index]
    to_append = pd.Series(0, index=missing)
    n_contacts = n_contacts.append(to_append)
    return n_contacts
