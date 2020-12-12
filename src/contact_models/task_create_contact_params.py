from functools import partial

import matplotlib.pyplot as plt
import pandas as pd
import pytask
import seaborn as sns
from estimagic import minimize

from src.config import BLD


def _create_parametrization():
    """Create the pytask parametrization.

    Each entry includes criteria for which contacts count to a contact type,
    specified in the data_selection_criteria that maps the type to the
    criteria that a contact must fulfill to belong to that contact type,
    as well as potentially the maximal number of contacts allowed.

    The parametrization is a list of tuples where the first entry is the
    data_selection_criteria entry and the second the list of expected
    outputs.

    """
    data_selection_criteria = {
        "work_non_recurrent": {
            "places": ["work"],
            "recurrent": False,
            "frequency": None,
            "weekend": False,
            "max_contacts": None,
        },
        "work_recurrent_daily": {
            "places": ["work"],
            "recurrent": True,
            "frequency": "(almost) daily",
            "weekend": False,
            "max_contacts": 15,
        },
        "work_recurrent_weekly": {
            "places": ["work"],
            "recurrent": True,
            "frequency": "1-2 times a week",
            "weekend": False,
            "max_contacts": 14,
        },
        "other_non_recurrent": {
            "places": ["otherplace", "leisure"],
            "recurrent": False,
            "frequency": None,
            "weekend": None,
            "max_contacts": None,
        },
        "other_recurrent_daily": {
            "places": ["otherplace", "leisure"],
            "recurrent": True,
            "frequency": "(almost) daily",
            "weekend": None,
            "max_contacts": 5,
        },
        "other_recurrent_weekly": {
            "places": ["otherplace", "leisure"],
            "recurrent": True,
            "frequency": "1-2 times a week",
            "weekend": None,
            "max_contacts": 4,
        },
    }

    out_path = BLD / "contact_models" / "empirical_distributions"

    parametrization = []
    for name, criteria in data_selection_criteria.items():
        produce_paths = [out_path / "figures" / f"{name}.png", out_path / f"{name}.pkl"]
        if not criteria["recurrent"]:
            produce_paths.append(
                BLD / "contact_models" / "age_assort_params" / f"{name}.pkl"
            )
        parametrization.append((criteria, produce_paths))
    return parametrization


PARAMETRIZATION = _create_parametrization()


@pytask.mark.depends_on(BLD / "data" / "mossong_2008" / "contact_data.pkl")
@pytask.mark.parametrize("specs, produces", PARAMETRIZATION)
def task_calculate_and_plot_nr_of_contacts(depends_on, specs, produces):
    name = produces[0].stem.replace("_", " ").title()
    regression_criterion_values = {
        "Other Recurrent Weekly": 137_000,
        "Other Recurrent Daily": 1385,
        "Work Recurrent Weekly": 90,
        "Work Recurrent Daily": 1285,
    }
    max_contacts = specs.pop("max_contacts")
    contacts = pd.read_pickle(depends_on)

    n_contacts = _create_n_contacts(contacts, **specs)
    empirical_distribution = n_contacts.value_counts().sort_index()
    if max_contacts is not None and max_contacts < empirical_distribution.index.max():
        approx_dist = _reduce_empirical_distribution_to_max_contacts(
            empirical_distribution, max_contacts, regression_criterion_values[name]
        )
        approx_dist.name = name
    else:
        approx_dist = empirical_distribution

    pct_non_zero = (empirical_distribution / empirical_distribution.sum())[1:].sum()
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.lineplot(x=approx_dist.index, y=approx_dist, ax=ax)
    ax.set_title(name)
    ax.set_xlabel(
        f"\n{int(100 * pct_non_zero)}% individuals reported non-zero contacts."
    )
    sns.despine()
    fig.tight_layout()
    fig.savefig(produces[0])

    shares = approx_dist / approx_dist.sum()
    shares.to_pickle(produces[1])

    if not specs["recurrent"]:
        assort_params = _create_assort_params(produces[0].stem, contacts, **specs)
        assort_params.to_pickle(produces[2])


def _create_assort_params(model_name, contacts, places, recurrent, frequency, weekend):
    df = _get_relevant_contacts_subset(contacts, places, recurrent, frequency, weekend)
    normalized_cell_counts = pd.crosstab(
        df["part_age_group"],
        df["age_group_of_contact"],
        dropna=False,
        normalize="index",
    )
    meeting_prob = normalized_cell_counts.fillna(0).round(4)
    meeting_prob = meeting_prob.where(
        meeting_prob.sum(axis=1) != 0.0, 1 / len(meeting_prob.columns)
    )
    meeting_prob = meeting_prob / meeting_prob.sum(axis=1).to_numpy().reshape(-1, 1)
    assert (meeting_prob.sum(axis=1) > 0.9999).all() & (
        meeting_prob.sum(axis=1) < 1.0001
    ).all(), "meeting probabilities do not add up to 1 in every row."
    assert (
        meeting_prob.index == meeting_prob.columns
    ).all(), "meeting probabilities are not square."

    assort_params = meeting_prob.stack()

    first_level = f"assortative_matching_{model_name}_age_group"
    assort_params = pd.concat([assort_params], keys=[first_level], names=["category"])
    assort_params.index.names = ["category", "subcategory", "name"]
    assort_params.name = "value"
    return assort_params


def _create_n_contacts(contacts, places, recurrent, frequency, weekend):
    """Calculate the reported number of contacts for the given specification.

    Args:
        contacts (pandas.DataFrame)
        places (list): list of the places belonging to the current contact model.
            If places is ["work"], only workers are counted as reporting contacts.
        recurrent (bool): whether to keep recurrent or non-recurrent contacts
        frequency (str): which frequency to keep. None means all frequencies are kept
        weekend (bool): whether to use weekday or weekend data.
            None means both are kept.

    Returns:
        n_contacts (pandas.Series): index are reporting individuals
            (or reporting workers if places == ["work"]) and the values are the
            number of condition meeting contacts (with respect to
            recurrent, placse, frequency and weekend) individuals reported.

    """
    if places != ["work"]:
        relevant_ids = contacts["id"].unique()
    else:
        query = "participant_occupation == 'working'"
        relevant_ids = contacts.query(query)["id"].unique()
    df = _get_relevant_contacts_subset(contacts, places, recurrent, frequency, weekend)

    n_contacts = df.groupby("id").size()
    missing = [x for x in relevant_ids if x not in n_contacts.index]
    to_append = pd.Series(0, index=missing)
    n_contacts = n_contacts.append(to_append)
    n_contacts = n_contacts.sort_index()
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


def _reduce_empirical_distribution_to_max_contacts(
    empirical_distribution, max_contacts, assert_below_this
):
    """Find the closest distribution that has no more than max_contacts.

    Args:
        empirical_distribution (pandas.Series): value counts of the
            number of reported contacts.
        max_contacts (int): maximal allowed numbers of reported contacts

    Returns:
        closest_distribution (pandas.Series): approximated value counts
            of the number of reported contacts. Adjusted so that no one
            has too many contacts, the number of individuals reporting
            contacts stayed the same and the total number of reported
            contacts over all individuals is close to before.

    """
    desired_total = empirical_distribution @ empirical_distribution.index
    nobs = empirical_distribution.sum()

    truncated = empirical_distribution.copy(deep=True)
    n_above_contacts = truncated[max_contacts + 1 :].sum()  # noqa
    truncated[max_contacts] += n_above_contacts
    truncated = truncated[: max_contacts + 1]

    assert truncated.index[0] == 0, "No individuals reporting 0 contacts."

    params = _make_decreasing(truncated).to_frame()

    constraints = [
        # fix number of people without contacts
        {"loc": 0, "type": "fixed", "value": truncated[0]},
        # decreasing
        {"loc": params.index[1:], "type": "decreasing"},
        # fix number of individuals reporting contacts
        {
            "loc": params.index[1:],
            "type": "linear",
            "weights": 1,
            "value": nobs - truncated[0],
        },
    ]

    criterion_func = partial(
        measure_of_diff_btw_distributions,
        old_distribution=truncated,
        desired_total=desired_total,
    )

    start_crit = criterion_func(params)

    res = minimize(
        criterion=criterion_func,
        params=params,
        constraints=constraints,
        algorithm="nag_pybobyqa",
        logging=False,
    )
    assert res["success"]
    assert res["solution_criterion"] <= start_crit
    assert res["solution_criterion"] < assert_below_this
    closest_distribution = res["solution_params"]["value"].astype(int)
    return closest_distribution


def measure_of_diff_btw_distributions(params, old_distribution, desired_total):
    """Cost function for the minimization problem.

    We try to minimize the difference between the truncated value counts and the
    proposed distribution while keeping the number of total contacts as close as
    possible to the original distribution.

    Args:
        params (pandas.DataFrame): estimagic params DataFrame. The index is the
            support of the distribuiton, the "value" column the number of
            individuals reporting the respective number.

        old_distribution (pandas.Series): The distribution to be approximated.
            The index is the support of the distribution. It must be the same
            as that of params.

        desired_total (int): desired value of the dot product of params. In the
            context of number of reported contacts this means the total number
            of reported contacts by all individuals.

    Returns:
        cost (float): sum of the parameter distance penalty and a penalty for
            distance between the desired and achieved total.

    """
    assert (params.index == old_distribution.index).all()
    params_deviation = params["value"].to_numpy() - old_distribution.to_numpy()
    params_penalty = (params_deviation ** 2).sum()
    actual_total = params.index.to_numpy() @ params["value"].to_numpy()
    total_contacts_penalty = (desired_total - actual_total) ** 2
    cost = total_contacts_penalty + params_penalty
    return cost


def _make_decreasing(sr):
    """Make a pandas.Series decreasing."""
    out = sr.copy(deep=True)
    out.name = "value"

    to_add_to_2nd_entry = 0
    for loc, val in out[1:].items():
        before = out[loc - 1]
        if before < val:
            out[loc] = before
            to_add_to_2nd_entry += val - before
    out[1] += to_add_to_2nd_entry
    return out
