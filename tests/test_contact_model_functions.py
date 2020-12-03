from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal

from src.contact_models.contact_model_functions import _draw_nr_of_contacts
from src.contact_models.contact_model_functions import (
    calculate_non_recurrent_contacts_from_empirical_distribution,
)
from src.contact_models.contact_model_functions import (
    reduce_non_recurrent_contacts_on_condition,
)
from src.contact_models.contact_model_functions import (
    reduce_recurrent_contacts_on_condition,
)
from src.shared import draw_groups


@pytest.fixture
def params():
    params = pd.DataFrame()
    params["category"] = ["work_non_recurrent"] * 2 + ["other_non_recurrent"] * 2
    params["subcategory"] = [
        "reduction_when_symptomatic",
        "reduction_when_positive",
    ] * 2
    params["name"] = ["reduction_when_symptomatic", "reduction_when_positive"] * 2
    params["value"] = [1.0, 1.0, 1.0, 1.0]
    params.set_index(["category", "subcategory", "name"], inplace=True)
    return params


@pytest.fixture
def states():
    """states DataFrame for testing purposes.

    Columns:
        - date: 2020-04-01 - 2020-04-30
        - id: 50 individuals, with 30 observations each. id goes from 0 to 49.
        - immune: bool
        - infectious: bool
        - age_group: ordered Categorical, either 10-19 or 40-49.
        - region: unordered Categorical, ['Overtjssel', 'Drenthe', 'Gelderland']
        - n_has_infected: int, 0 to 3.
        - cd_infectious_false: int, -66 to 8.
        - occupation: Categorical. "working" or "in school".
        - cd_symptoms_false: int, positive for the first 20 individuals, negative after.

    """
    this_modules_path = Path(__file__).resolve()
    states = pd.read_parquet(this_modules_path.parent / "1.parquet")

    old_to_new = {old: i for i, old in enumerate(sorted(states["id"].unique()))}
    states["id"].replace(old_to_new, inplace=True)
    states["age_group"] = pd.Categorical(
        states["age_group"], ["10 - 19", "40 - 49"], ordered=True
    )
    states["age_group"] = states["age_group"].cat.rename_categories(
        {"10 - 19": "10-19", "40 - 49": "40-49"}
    )
    states["region"] = pd.Categorical(
        states["region"], ["Overtjssel", "Drenthe", "Gelderland"], ordered=False
    )
    states["date"] = pd.to_datetime(states["date"], format="%Y-%m-%d", unit="D")
    states["n_has_infected"] = states["n_has_infected"].astype(int)
    states["cd_infectious_false"] = states["cd_infectious_false"].astype(int)
    states["occupation"] = states["age_group"].replace(
        {"10-19": "in school", "40-49": "working"}
    )
    states["cd_symptoms_false"] = list(range(1, 21)) + list(range(-len(states), -20))
    states["symptomatic"] = states["cd_symptoms_false"] >= 0
    states["knows_infectious"] = False
    states["knows_immune"] = False
    states["cd_received_test_result_true"] = -100
    return states


@pytest.fixture
def a_thursday(states):
    a_thursday = states[states["date"] == "2020-04-30"].copy()
    a_thursday["cd_symptoms_false"] = list(range(1, 21)) + list(
        range(-len(a_thursday), -20)
    )
    a_thursday["symptomatic"] = a_thursday["cd_symptoms_false"] >= 0

    a_thursday["work_recurrent_weekly"] = draw_groups(
        df=a_thursday,
        query="occupation == 'working'",
        assort_bys=["region"],
        n_per_group=20,
        seed=484,
    )

    return a_thursday


# --------------------------- Non Recurrent Contact Models ---------------------------


def test_non_recurrent_work_contacts_weekend(states, params):
    a_saturday = states[states["date"] == pd.Timestamp("2020-04-04 00:00:00")]
    res = calculate_non_recurrent_contacts_from_empirical_distribution(
        states=a_saturday,
        contact_params=params.loc["work_non_recurrent"],
        on_weekends=False,
        query="occupation == 'working'",
        seed=494,
    )
    assert_series_equal(res, pd.Series(data=0, index=a_saturday.index))


@pytest.fixture
def params_with_positive():
    params = pd.DataFrame.from_dict(
        {
            "category": ["work_non_recurrent"] * 3,
            "subcategory": [
                "all",
                "reduction_when_symptomatic",
                "reduction_when_positive",
            ],
            "name": [
                2,
                "reduction_when_symptomatic",
                "reduction_when_positive",
            ],  # nr of contacts
            "value": [1.0, 1.0, 1.0],  # probability
        }
    )
    params = params.set_index(["category", "subcategory", "name"])
    return params


def test_non_recurrent_work_contacts_no_random_no_sick(
    a_thursday, params_with_positive
):
    a_thursday["symptomatic"] = False

    res = calculate_non_recurrent_contacts_from_empirical_distribution(
        states=a_thursday,
        contact_params=params_with_positive.loc["work_non_recurrent"],
        on_weekends=False,
        query="occupation == 'working'",
        seed=433,
    )

    expected = a_thursday["age_group"].replace({"10-19": 0.0, "40-49": 2.0})

    assert_series_equal(res, expected, check_names=False, check_dtype=False)


def test_non_recurrent_work_contacts_no_random_with_sick(
    a_thursday, params_with_positive
):
    res = calculate_non_recurrent_contacts_from_empirical_distribution(
        states=a_thursday,
        contact_params=params_with_positive.loc["work_non_recurrent"],
        on_weekends=False,
        query="occupation == 'working'",
        seed=448,
    )
    expected = a_thursday["age_group"].replace({"10-19": 0.0, "40-49": 2.0})
    expected[:20] = 0.0
    assert_series_equal(res, expected, check_names=False, check_dtype=False)


def test_non_recurrent_work_contacts_random_with_sick(a_thursday):
    np.random.seed(77)
    params = pd.DataFrame.from_dict(
        {
            "category": ["work_non_recurrent"] * 4,
            "subcategory": ["all"] * 2
            + ["reduction_when_symptomatic", "reduction_when_positive"],
            "name": [
                3,
                2,
                "reduction_when_symptomatic",
                "reduction_when_positive",
            ],  # nr of contacts
            "value": [0.5, 0.5, 1.0, 1.0],  # probability
        }
    ).set_index(["category", "subcategory", "name"])

    res = calculate_non_recurrent_contacts_from_empirical_distribution(
        states=a_thursday,
        contact_params=params.loc["work_non_recurrent"],
        on_weekends=False,
        query="occupation == 'working'",
        seed=338,
    )
    assert (res[:20] == 0).all()  # symptomatics
    assert (res[a_thursday["occupation"] != "working"] == 0).all()  # non workers
    healthy_workers = (a_thursday["occupation"] == "working") & (
        a_thursday["cd_symptoms_false"] < 0
    )
    assert res[healthy_workers].isin([2, 3]).all()


# ------------------------------------------------------------------------------------


def test_non_recurrent_other_contacts_no_random_no_sick(a_thursday):
    a_thursday["symptomatic"] = False
    params = pd.DataFrame.from_dict(
        {
            "category": ["other_non_recurrent"] * 3,
            "subcategory": [
                "all",
                "reduction_when_symptomatic",
                "reduction_when_positive",
            ],
            "name": [
                2,
                "reduction_when_symptomatic",
                "reduction_when_positive",
            ],  # nr of contacts
            "value": [1.0, 1.0, 1.0],  # probability
        }
    ).set_index(["category", "subcategory", "name"])

    res = calculate_non_recurrent_contacts_from_empirical_distribution(
        states=a_thursday,
        contact_params=params.loc["other_non_recurrent"],
        on_weekends=True,
        query=None,
        seed=334,
    )
    expected = pd.Series(data=2, index=a_thursday.index)
    assert_series_equal(res, expected, check_names=False, check_dtype=False)


def test_non_recurrent_other_contacts_no_random_with_sick(a_thursday):
    params = pd.DataFrame.from_dict(
        {
            "category": ["other_non_recurrent"] * 3,
            "subcategory": [
                "all",
                "reduction_when_symptomatic",
                "reduction_when_positive",
            ],
            "name": [
                2,
                "reduction_when_symptomatic",
                "reduction_when_positive",
            ],  # nr of contacts
            "value": [1.0, 1.0, 1.0],  # probability
        }
    ).set_index(["category", "subcategory", "name"])

    res = calculate_non_recurrent_contacts_from_empirical_distribution(
        states=a_thursday,
        contact_params=params.loc["other_non_recurrent"],
        on_weekends=True,
        query=None,
        seed=332,
    )
    expected = pd.Series(data=2, index=a_thursday.index)
    expected[:20] = 0
    assert_series_equal(res, expected, check_names=False, check_dtype=False)


def test_non_recurrent_other_contacts_random_with_sick(a_thursday):
    np.random.seed(770)
    params = pd.DataFrame.from_dict(
        {
            "category": ["other_non_recurrent"] * 4,
            "subcategory": ["all"] * 2
            + ["reduction_when_symptomatic", "reduction_when_positive"],
            "name": [
                3,
                2,
                "reduction_when_symptomatic",
                "reduction_when_positive",
            ],  # nr of contacts
            "value": [0.5, 0.5, 1.0, 1.0],  # probability
        }
    ).set_index(["category", "subcategory", "name"])

    res = calculate_non_recurrent_contacts_from_empirical_distribution(
        states=a_thursday,
        contact_params=params.loc["other_non_recurrent"],
        on_weekends=True,
        query=None,
        seed=474,
    )
    assert (res[:20] == 0).all()  # symptomatics
    assert res[a_thursday["cd_symptoms_false"] < 0].isin([2, 3]).all()


# --------------------------------- General Functions ---------------------------------


def test_draw_nr_of_contacts_always_five(states):
    dist = pd.DataFrame(
        data=[[4, 0, "all"], [5, 1, "all"]], columns=["name", "value", "subcategory"]
    ).set_index(["subcategory", "name"])["value"]
    pop = pd.Series(data=True, index=states.index)
    res = _draw_nr_of_contacts(dist, pop, states, seed=939)
    expected = pd.Series(5.0, index=states.index)
    assert_series_equal(res, expected, check_dtype=False)


def test_draw_nr_of_contacts_mean_5(states):
    # this relies on the law of large numbers
    np.random.seed(3499)
    dist = pd.DataFrame(
        [[4, 0.5, "all"], [6, 0.5, "all"]], columns=["name", "value", "subcategory"]
    ).set_index(["subcategory", "name"])["value"]
    pop = pd.Series(data=True, index=states.index)
    res = _draw_nr_of_contacts(dist, pop, states, seed=939)
    assert res.isin([4, 6]).all()
    assert res.mean() == pytest.approx(5, 0.01)


def test_draw_nr_of_contacts_differ_btw_ages(states):
    dist = pd.DataFrame.from_dict(
        {"name": [0, 6], "value": [1, 1], "subcategory": ["10-19", "40-49"]}
    ).set_index(["subcategory", "name"])["value"]
    pop = pd.Series(data=True, index=states.index)

    res = _draw_nr_of_contacts(dist, pop, states, seed=939)

    assert (res[states["age_group"] == "10-19"] == 0).all()
    assert (res[states["age_group"] == "40-49"] == 6).all()


def test_draw_nr_of_contacts_differ_btw_ages_random(states):
    np.random.seed(24)
    dist = pd.DataFrame(
        data=[
            [0, 0.5, "10-19"],
            [1, 0.5, "10-19"],
            [6, 0.5, "40-49"],
            [7, 0.5, "40-49"],
        ],
        columns=["name", "value", "subcategory"],
    ).set_index(["subcategory", "name"])["value"]
    pop = pd.Series(data=True, index=states.index)

    res = _draw_nr_of_contacts(dist, pop, states, seed=24)

    young = res[states["age_group"] == "10-19"]
    old = res[states["age_group"] == "40-49"]

    assert young.isin([0, 1]).all()
    assert old.isin([6, 7]).all()

    assert young.mean() == pytest.approx(0.5, 0.05)
    assert old.mean() == pytest.approx(6.5, 0.05)


# ------------------------------------------------------------------------------------


def test_reduce_non_recurrent_contacts_on_condition(states):
    nr_of_contacts = pd.Series(data=10, index=states.index)
    states["symptomatic"] = [True, True, True] + [False] * (len(states) - 3)
    factor = 0.5
    expected = pd.Series([5, 5, 5] + [10] * (len(states) - 3))
    res = reduce_non_recurrent_contacts_on_condition(
        contacts=nr_of_contacts, states=states, factor=factor, condition="symptomatic"
    )
    assert_series_equal(res, expected, check_dtype=False)


# ------------------------------------------------------------------------------------


def test_reduce_recurrent_contacts_on_condition(states):
    nr_of_contacts = pd.Series(data=10, index=states.index)
    states["symptomatic"] = [True, True, True] + [False] * (len(states) - 3)
    factor = 0.5
    res = reduce_recurrent_contacts_on_condition(
        contacts=nr_of_contacts,
        states=states,
        share=factor,
        seed=8388,
        condition="symptomatic",
    )
    expected = pd.Series([10, 0, 0] + [10] * (len(states) - 3))
    assert_series_equal(res, expected, check_dtype=False)
