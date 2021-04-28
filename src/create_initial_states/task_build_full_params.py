import pandas as pd
import pytask
from sid import load_epidemiological_parameters

from src.config import BLD
from src.config import SRC
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
        "vacations": BLD / "data" / "vacations.pkl",
        "infection_probs": SRC / "simulation" / "infection_probs.pkl",
        "susceptibility": SRC / "original_data" / "susceptibility.csv",
    }
)
@pytask.mark.produces(BLD / "params.pkl")
def task_create_full_params(depends_on, produces):
    epi_params = load_epidemiological_parameters()
    vacations = pd.read_pickle(depends_on["vacations"])
    infection_probs = pd.read_pickle(depends_on["infection_probs"])
    susceptibility = pd.read_csv(
        depends_on["susceptibility"], index_col=["category", "subcategory", "name"]
    )

    distributions = {
        name[5:]: path for name, path in depends_on.items() if name.startswith("dist_")
    }
    dist_params = []
    for category, path in distributions.items():
        dist = pd.read_pickle(path)
        dist = _make_mergable_with_params(dist, category)
        dist_params.append(dist)
    dist_params = pd.concat(dist_params, axis=0)

    age_assort_params = {}
    for name, path in depends_on.items():
        if name.startswith("assort"):
            age_assort_params[name[7:]] = pd.read_pickle(path)

    contact_models = get_all_contact_models()

    assort_params = _build_assort_params(contact_models, age_assort_params)

    reaction_params = _build_reaction_params(contact_models)
    share_known_cases_params = _build_share_known_cases_params()
    param_slices = [
        infection_probs,
        reaction_params,
        dist_params,
        assort_params,
        epi_params,
        vacations,
        share_known_cases_params,
        susceptibility,
    ]
    params = pd.concat(param_slices, axis=0)

    # number of available tests is implemented in the test demand model.
    # therefore, we set the "sid" limit, which is time invariant to one test
    # per individual
    params.loc[("testing", "allocation", "rel_available_tests")] = 100_000
    params.loc[("testing", "processing", "rel_available_capacity")] = 100_000

    # Testing parameters governing test demand
    params.loc[("test_demand", "symptoms", "share_symptomatic_requesting_test")] = 0.5

    # Testing parameters governing rapid test demand
    # -----------------------------------------------

    # source: https://bit.ly/3gHlcKd (section 3.5, 2021-03-09, accessed 2021-04-28)
    params.loc[
        ("test_demand", "shares", "share_w_positive_rapid_test_requesting_test"),
    ] = 0.85

    # Only 60% of workers receiving a test offer accept it regularly
    # source: https://bit.ly/3t1z0lf (COSMO, 2021-04-21)
    params.loc[("rapid_test_demand", "work", "share_accepting_offer")] = 0.6

    # assume start of rapid tests in firms in Jan 01
    offer_loc = ("rapid_test_demand", "share_workers_receiving_offer")
    params.loc[(*offer_loc, "2020-01-01"), "value"] = 0.0
    params.loc[(*offer_loc, "2021-01-01"), "value"] = 0.0

    # 2021-03-20: 20% of workers get weekly test
    # source: https://bit.ly/3eu0meK, https://bit.ly/3gANaan
    params.loc[(*offer_loc, "2021-03-20"), "value"] = 0.2
    # 2021-04-05: 60% of workers get weekly test
    # source: https://bit.ly/2RWCDMz
    params.loc[(*offer_loc, "2021-04-05"), "value"] = 0.6
    # 2021-04-15: 70% of workers are expected to get weekly tests
    # source: https://bit.ly/32BqKhd
    # COSMO (https://bit.ly/3t1z0lf, 2021-04-20) report <2/3 of people having
    # work contacts receiving a test offer
    params.loc[(*offer_loc, "2021-04-15"), "value"] = 0.66
    # 2021-04-19: employers are required by law to offer weekly tests
    # source: https://bit.ly/3tJNUh1, https://bit.ly/2QfNctJ
    # receive test offers at work.
    # There is no data available on compliance yet
    params.loc[(*offer_loc, "2021-04-15"), "value"] = 0.7

    # seasonality parameter
    params.loc[("seasonality_effect", "seasonality_effect", "seasonality_effect")] = 0.2

    params = _convert_index_to_int_where_possible(params)
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
                df.loc[(cm, name, name)] = multiplier
    return df


def _convert_index_to_int_where_possible(params):
    params = params.reset_index().copy(deep=True)
    params["name"] = params["name"].apply(_convert_to_int_if_possible)
    params = params.set_index(["category", "subcategory", "name"])
    return params


def _build_share_known_cases_params():
    params_slice = pd.Series(
        {
            # from dunkelzifferradar
            "2020-01-01": 0.07,
            "2020-03-01": 0.07,
            "2020-03-17": 0.2,
            "2020-06-15": 0.2,
            "2020-07-10": 0.46,
            "2020-09-01": 0.67,
            "2020-09-25": 0.6,
            "2020-12-23": 0.22,
            # free parameters
            "2021-02-28": 0.25,
            "2021-04-30": 0.35,
            "2021-05-15": 0.35,
        },
        name="value",
    ).to_frame()
    params = pd.concat([params_slice], keys=["share_known_cases"])
    params = pd.concat(
        [params], keys=["share_known_cases"], names=["category", "subcategory", "name"]
    )
    return params


def _convert_to_int_if_possible(x):
    """pd.to_numeric did not correctly work."""
    try:
        return int(x)
    except ValueError:
        return x
