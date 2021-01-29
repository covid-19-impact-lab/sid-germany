import pandas as pd
import pytask
from sid import load_epidemiological_parameters

from src.config import BLD
from src.contact_models.get_contact_models import get_all_contact_models
from src.contact_models.get_contact_models import get_christmas_contact_models
from src.policies.policy_tools import combine_dictionaries


@pytask.mark.depends_on(
    {
        "dist_other_non_recurrent": BLD
        / "contact_models"
        / "empirical_distributions"
        / "other_non_recurrent.pkl",
        "dist_work_non_recurrent": BLD
        / "contact_models"
        / "empirical_distributions"
        / "work_non_recurrent.pkl",
        "assort_other_non_recurrent": BLD
        / "contact_models"
        / "age_assort_params"
        / "other_non_recurrent.pkl",
        "assort_work_non_recurrent": BLD
        / "contact_models"
        / "age_assort_params"
        / "work_non_recurrent.pkl",
        "vacations": BLD / "data" / "vacations.pkl",
    }
)
@pytask.mark.produces(BLD / "start_params.pkl")
def task_create_full_params(depends_on, produces):
    epi_params = load_epidemiological_parameters()
    vacations = pd.read_pickle(depends_on.pop("vacations"))

    distributions = {
        name[5:]: path for name, path in depends_on.items() if name.startswith("dist_")
    }
    dist_params = []
    for category, path in distributions.items():
        dist = pd.read_pickle(path)
        dist = _make_mergable_with_params(dist, category)
        dist_params.append(dist)
    dist_params = pd.concat(dist_params, axis=0)

    age_assort_params = {
        name[7:]: pd.read_pickle(path)
        for name, path in depends_on.items()
        if name.startswith("assort")
    }

    non_christmas_contact_models, all_contact_models = _get_contact_models_for_params()

    assort_params = _build_assort_params(
        non_christmas_contact_models, age_assort_params
    )
    # assortative matching of the holiday preparation by county and age_group
    # This is supposed to cover holiday shopping and traveling.
    assort_params.loc[
        ("assortative_matching", "holiday_preparation", "county"), "value"
    ] = 0.3
    assort_params.loc[
        ("assortative_matching", "holiday_preparation", "age_group"), "value"
    ] = 0.3

    reaction_params = _build_reaction_params(all_contact_models)
    param_slices = [
        reaction_params,
        dist_params,
        assort_params,
        epi_params,
        vacations,
    ]
    params = pd.concat(param_slices, axis=0)
    params.to_pickle(produces)


def _get_contact_models_for_params():
    non_christmas_contact_models = get_all_contact_models()
    full_christmas = get_christmas_contact_models("full", 2)
    same_group_christmas = get_christmas_contact_models("same_group", 2)
    del same_group_christmas["holiday_preparation"]
    meet_twice_christmas = get_christmas_contact_models("meet_twice", 2)
    del meet_twice_christmas["holiday_preparation"]
    christmas_contact_models = combine_dictionaries(
        [full_christmas, same_group_christmas, meet_twice_christmas]
    )
    all_contact_models = combine_dictionaries(
        [non_christmas_contact_models, christmas_contact_models]
    )
    return non_christmas_contact_models, all_contact_models


def _make_mergable_with_params(dist, category):
    """Change the index and Series name to easily merge it to params.

    Args:
        dist (pandas.Series): distribution of number of contacts. The
            index is the support, the values the probabilities.
        category (str): name of the contact model to which the distribution
            belongs. This is set as the category index level of the
            returned Series.
    Returns:
        pandas.Series: Series with triple index category, subcategory, name.
            the name index level is the support. the value column contains
            the probabilities.

    """
    dist.name = "value"
    dist = dist.to_frame()
    dist["category"] = category
    dist["subcategory"] = "n_contacts"
    dist["name"] = dist.index
    dist = dist.set_index(["category", "subcategory", "name"], drop=True)
    return dist


def _build_assort_params(contact_models, age_assort_params):
    df = pd.DataFrame(columns=["category", "subcategory", "name", "value"])
    sr = df.set_index(["category", "subcategory", "name"])["value"]
    for name, model in contact_models.items():
        if not model["is_recurrent"]:
            for var in model["assort_by"]:
                if var == "county":
                    sr[("assortative_matching", name, var)] = 0.8
                else:
                    sr = pd.concat([sr, age_assort_params[name]], axis=0)
    return sr.to_frame()


def _build_reaction_params(contact_models):
    df = pd.DataFrame(columns=["category", "subcategory", "name", "value"])
    df = df.set_index(["category", "subcategory", "name"])
    multipliers = [
        ("symptomatic_multiplier", 0.15, 0.7),
        ("positive_test_multiplier", 0.05, 0.5),
    ]
    for name, multiplier, hh_multiplier in multipliers:
        for cm in contact_models:
            if "household" in cm:
                df.loc[(cm, name, name)] = hh_multiplier
            else:
                # this includes the holiday preparation and christmas models
                df.loc[(cm, name, name)] = multiplier
    return df
