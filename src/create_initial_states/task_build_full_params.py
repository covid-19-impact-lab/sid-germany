import pandas as pd
import pytask
from sid import get_epidemiological_parameters

from src.config import BLD
from src.contact_models.get_contact_models import get_all_contact_models


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
    }
)
@pytask.mark.produces(BLD / "contact_models" / "params.pkl")
def task_create_full_params(depends_on, produces):
    epi_params = get_epidemiological_parameters()

    distributions = {
        name[5:]: path for name, path in depends_on.items() if name.startswith("dist_")
    }
    dist_params = []
    for category, path in distributions.items():
        dist = pd.read_pickle(path)
        dist = _make_mergable_with_params(dist, category)
        dist_params.append(dist)
    dist_params = pd.concat(dist_params, axis=0)

    contact_models = get_all_contact_models()
    infection_probs = _build_infection_probs(contact_models.keys())

    age_assort_params = {
        name[7:]: pd.read_pickle(path)
        for name, path in depends_on.items()
        if name.startswith("assort")
    }
    assort_params = _build_assort_params(contact_models, age_assort_params)

    reaction_params = _build_reaction_params(contact_models)
    param_slices = [
        epi_params,
        dist_params,
        infection_probs,
        assort_params,
        reaction_params,
    ]
    params = pd.concat(param_slices, axis=0)
    params.to_pickle(produces)


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


def _build_infection_probs(names):
    index_tuples = [("infection_prob", mod_name, mod_name) for mod_name in names]
    df = pd.DataFrame(index=pd.MultiIndex.from_tuples(index_tuples))
    df.index.names = ["category", "subcategory", "name"]
    prob_dict = {
        "educ": 0.008,
        "work": 0.025,
        "household": 0.05,
        "other": 0.025,
    }
    full_prob_dict = {}
    for mod_name in names:
        for k, v in prob_dict.items():
            if k in mod_name:
                full_prob_dict[mod_name] = v
        assert (
            mod_name in full_prob_dict
        ), f"No infection probability for {mod_name} specified."

    value_sr = df.index.get_level_values("name").to_series().map(full_prob_dict.get)
    value_sr.name = "value"
    df.join(value_sr)
    return df


def _build_assort_params(contact_models, age_assort_params):
    df = pd.DataFrame(columns=["category", "subcategory", "name", "value"])
    df = df.set_index(["category", "subcategory", "name"])
    for name, model in contact_models.items():
        if not model["is_recurrent"]:
            for var in model["assort_by"]:
                if var == "county":
                    df.loc[("assortative_matching", name, var)] = 0.8
                else:
                    df = pd.concat([df, age_assort_params[name]], axis=0)
    return df


def _build_reaction_params(contact_models):
    df = pd.DataFrame(columns=["category", "subcategory", "name", "value"])
    df = df.set_index(["category", "subcategory", "name"])
    multipliers = [
        ("symptomatic_multiplier", 0.15, 0.5),
        ("positive_test_multiplier", 0.05, 0.5),
    ]
    for cm in contact_models:
        for name, multiplier, hh_multiplier in multipliers:
            if "educ" in cm or "work" in cm or "other" in cm:
                df.loc[(cm, name, name)] = multiplier
            else:
                df.loc[(cm, name, name)] = hh_multiplier
    return df
