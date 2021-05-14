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
    params.loc[("testing", "allocation", "rel_available_tests"), "value"] = 100_000
    params.loc[("testing", "processing", "rel_available_capacity"), "value"] = 100_000

    params = _add_virus_strain_params(params)
    params = _add_vacation_model_distribution_params(params)

    # Share of individuals refusing to be vaccinated.
    # 80% of Germans are somewhat or definitely willing to be vaccinated.
    # 12% are undecided. 8% are opposed to being vaccinated.
    # We assume that 15% will refuse to be vaccinated.
    # source: https://bit.ly/3c9mTgX (publication date: 2021-03-02)
    params.loc[("vaccinations", "share_refuser", "share_refuser"), "value"] = 0.15

    # source: https://bit.ly/3gHlcKd (section 3.5, 2021-03-09, accessed 2021-04-28)
    loc = ("test_demand", "shares", "share_w_positive_rapid_test_requesting_test")
    params.loc[loc, "value"] = 0.4

    params = _add_work_rapid_test_params(params)
    params = _add_educ_rapid_test_fade_in_params(params)
    params = _add_hh_rapid_test_fade_in_params(params)
    params = _add_rapid_test_reaction_params(params)

    # seasonality parameter
    params.loc[("seasonality_effect", "seasonality_effect", "weak"), "value"] = 0.15
    params.loc[("seasonality_effect", "seasonality_effect", "strong"), "value"] = 0.25

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
        # source: The COSMO Study of 2021-03-09: 85% of individuals would isolate
        # after a positive rapid test.
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


def _add_virus_strain_params(params):
    """Add parameters governing the infectiousness of the virus strains.

    source: https://doi.org/10.1101/2020.12.24.20248822
    "We estimate that this variant has a 43–90% (range of 95% credible
    intervals 38–130%) higher reproduction number than preexisting variants"

    We take the midpoint of 67%.

    """
    params = params.copy(deep=True)
    params.loc[("virus_strain", "base_strain", "factor"), "value"] = 1.0
    params.loc[("virus_strain", "b117", "factor"), "value"] = 1.67
    return params


def _add_vacation_model_distribution_params(params):
    params = params.copy(deep=True)
    loc = ("additional_other_vacation_contact", "probability")
    # 2020
    params.loc[(*loc, "Winterferien"), "value"] = 0.275
    params.loc[(*loc, "Osterferien"), "value"] = 0.275
    params.loc[(*loc, "Pfingstferien"), "value"] = 0.275
    params.loc[(*loc, "Sommerferien"), "value"] = 0.275
    params.loc[(*loc, "Herbstferien"), "value"] = 0.275
    params.loc[(*loc, "Weihnachtsferien"), "value"] = 0.275
    # 2021
    params.loc[(*loc, "Winterferien2021"), "value"] = 0.275
    params.loc[(*loc, "Osterferien2021"), "value"] = 0.275
    params.loc[(*loc, "Pfingstferien2021"), "value"] = 0.275
    params.loc[(*loc, "Sommerferien2021"), "value"] = 0.275
    return params


def _add_work_rapid_test_params(params):
    """Add parameters governing the rapid test demand at work.

    Only 60% of workers receiving a test offer accept it regularly
    (https://bit.ly/3t1z0lf (COSMO, 2021-04-21))

    We assume rapid tests in firms on Jan 01 2021.

    2021-03-17-19: 20% of employers offer weekly test (https://bit.ly/3eu0meK)
    second half of March: 23% of workers report test offer (https://bit.ly/3gANaan)

    2021-04-05: 60% of workers get weekly test (https://bit.ly/2RWCDMz)

    2021-04-15: 70% of workers expected to get weekly tests (https://bit.ly/32BqKhd)
    COSMO (https://bit.ly/3t1z0lf, 2021-04-20) report <2/3 of people having
    work contacts receiving a test offer.

    2021-04-19: employers are required by law to offer two weekly tests
    (https://bit.ly/3tJNUh1, https://bit.ly/2QfNctJ)
    There is no data available on compliance or take-up yet.

    """
    params = params.copy(deep=True)

    params.loc[("rapid_test_demand", "work", "share_accepting_offer"), "value"] = 0.6

    offer_loc = ("rapid_test_demand", "share_workers_receiving_offer")
    params.loc[(*offer_loc, "2020-01-01"), "value"] = 0.0
    params.loc[(*offer_loc, "2021-01-01"), "value"] = 0.0
    params.loc[(*offer_loc, "2021-03-17"), "value"] = 0.2
    params.loc[(*offer_loc, "2021-04-05"), "value"] = 0.6
    params.loc[(*offer_loc, "2021-04-15"), "value"] = 0.66
    params.loc[(*offer_loc, "2021-04-15"), "value"] = 0.7
    params.loc[(*offer_loc, "2021-06-15"), "value"] = 0.7
    return params


def _add_educ_rapid_test_fade_in_params(params):
    """Add the shares how many people with educ contacts get a rapid test.

    Sources:
        17-24 of March 2021 (Mon, 2021-03-22):
            - NRW had 80% tests for students before Easter (https://bit.ly/3u7z8Rx)
            - BY: test offers to educ_workers (https://bit.ly/3tbVX5u)
            - BW: only tests for educ workers (https://bit.ly/2S7251M)

            - federal level:
                "In Kitas und Schulen sollen die Testmöglichkeiten "mit der
                steigenden Verfügbarkeit von Schnell- und Selbsttests"
                ausgebaut werden" (https://bit.ly/3nuCSKi)
            - Some KiTa workers are being tested (https://bit.ly/3nyGyus)
            - Self tests for students in Berlin (https://bit.ly/2ScGu8m)
            - Schleswig-Holstein: test offer (https://bit.ly/3eVfkuv)
            - mandatory tests in Saxony (https://bit.ly/3eEQGhn)
            - no tests yet for students in Hessia, but already ordered
              (https://bit.ly/3gMGJB4)
            - Niedersachsen had one test week before Easter (https://bit.ly/3gOOC96)

            => assume 90% of teachers and 30% of students do rapid tests

        After Easter (2021-04-07):
            - NRW: tests are mandatory for all
            - Bavaria: tests are mandatory for all (https://bit.ly/3nz5fXS,
              https://bit.ly/2QHilX3)
            - BW: voluntary tests for students (https://bit.ly/3vuetaD)
            - Brandenburg starts with tests (https://bit.ly/3xAihZB)
            - Schleswig-Holstein: mandatory tests (https://bit.ly/3eVfkuv)

            => assume 95% of teachers and 75% of students get tested

        - BW: tests mandatory starting 2021-04-19 (https://bit.ly/3vuetaD)

            => assume 95% of teachers and 95% of students get tested

    """
    params = params.copy(deep=True)

    loc = ("rapid_test_demand", "educ_worker_shares")
    params.loc[(*loc, "2020-01-01")] = 0.0
    params.loc[(*loc, "2021-01-01")] = 0.0
    # this is arbitrary to have a more convex shape
    params.loc[(*loc, "2021-03-01")] = 0.3
    params.loc[(*loc, "2021-03-22")] = 0.9
    params.loc[(*loc, "2021-04-07")] = 0.95
    params.loc[(*loc, "2021-04-19")] = 0.95
    params.loc[(*loc, "2021-06-01")] = 0.95

    loc = ("rapid_test_demand", "student_shares")
    params.loc[(*loc, "2020-01-01")] = 0.0
    params.loc[(*loc, "2021-01-01")] = 0.0
    params.loc[(*loc, "2021-03-22")] = 0.3
    params.loc[(*loc, "2021-04-07")] = 0.75
    params.loc[(*loc, "2021-04-19")] = 0.95
    params.loc[(*loc, "2021-06-01")] = 1.0

    return params


def _add_hh_rapid_test_fade_in_params(params):
    """Add the share of people demanding a rapid test after a Covid household event.

    Bürgertests started in mid March but demand was very low initially
    (https://bit.ly/3ehmGcj). Anecdotally, the demand continues to be limited.

    First tests to self-administer became available starting March 6.
    However, supply was very limited in the beginning (https://bit.ly/3xJCIn8).

    All values are arbitrary.

    We assume that for Easter visits many people demanded tests for the first
    time and are more likely to test themselves after knowing where to get them.

    """
    params = params.copy(deep=True)
    loc = ("rapid_test_demand", "hh_member_demand")
    params.loc[(*loc, "2020-01-01"), "value"] = 0
    params.loc[(*loc, "2021-03-10"), "value"] = 0
    params.loc[(*loc, "2021-03-25"), "value"] = 0.1
    params.loc[(*loc, "2021-04-05"), "value"] = 0.3
    params.loc[(*loc, "2021-05-01"), "value"] = 0.4
    params.loc[(*loc, "2021-05-15"), "value"] = 0.5
    params.loc[(*loc, "2021-06-01"), "value"] = 0.75
    params.loc[(*loc, "2021-10-01"), "value"] = 0.75

    return params


def _add_rapid_test_reaction_params(params):
    """Add rapid test reaction params.

    source: The COSMO Study of 2021-03-09 (https://bit.ly/3gHlcKd)
    In section 3.5 "Verhalten nach positivem Selbsttest"
    85% claim they would isolate ("isoliere mich und beschränke meine Kontakte
    bis zur Klärung")
        => We use this multiplier of 0.15 here.

    We assume households are only reduced by 30%, i.e. have a multiplier of 0.7.

    """
    params = params.copy(deep=True)
    params.loc[
        ("rapid_test_demand", "reaction", "hh_contacts_multiplier"), "value"
    ] = 0.7
    params.loc[
        ("rapid_test_demand", "reaction", "not_hh_contacts_multiplier"), "value"
    ] = 0.15
    return params


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
            "2020-06-10": 0.2,
            "2020-07-05": 0.46,
            "2020-08-15": 0.56,
            "2020-09-01": 0.56,
            "2020-11-05": 0.36,
            "2020-12-24": 0.31,
            "2020-12-25": 0.2,
            # free parameters
            "2021-01-01": 0.2,
            "2021-01-30": 0.31,
            "2021-06-15": 0.31,
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
