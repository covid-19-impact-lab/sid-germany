from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal

from src.contact_models.contact_model_functions import _draw_nr_of_contacts
from src.contact_models.contact_model_functions import _draw_potential_vacation_contacts
from src.contact_models.contact_model_functions import (
    _identify_ppl_affected_by_vacation,
)
from src.contact_models.contact_model_functions import (
    calculate_non_recurrent_contacts_from_empirical_distribution,
)
from src.contact_models.contact_model_functions import go_to_daily_work_meeting
from src.contact_models.contact_model_functions import go_to_weekly_meeting
from src.contact_models.contact_model_functions import meet_daily_other_contacts
from src.contact_models.contact_model_functions import reduce_contacts_on_condition
from src.shared import draw_groups


@pytest.fixture
def params():
    params = pd.DataFrame()
    params["category"] = ["work_non_recurrent"] * 2 + ["other_non_recurrent"] * 2
    params["subcategory"] = [
        "symptomatic_multiplier",
        "positive_test_multiplier",
    ] * 2
    params["name"] = ["symptomatic_multiplier", "positive_test_multiplier"] * 2
    params["value"] = [0.0, 0.0, 0.0, 0.0]
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
    states["knows_currently_infected"] = states.eval(
        "knows_infectious | (knows_immune & symptomatic) "
        "| (knows_immune & (cd_received_test_result_true >= -13))"
    )
    states["quarantine_compliance"] = 1.0

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


@pytest.fixture
def no_reduction_params():
    params = pd.DataFrame()
    params["subcategory"] = ["symptomatic_multiplier", "positive_test_multiplier"]
    params["name"] = params["subcategory"]
    params["value"] = 1.0
    params = params.set_index(["subcategory", "name"])
    return params


# ----------------------------------------------------------------------------


def test_go_to_weekly_meeting_wrong_day(a_thursday):
    a_thursday["group_col"] = [1, 2, 1, 2, 3, 3, 3] + [-1] * (len(a_thursday) - 7)
    contact_params = pd.DataFrame()
    group_col_name = "group_col"
    day_of_week = "Saturday"
    seed = 3931
    res = go_to_weekly_meeting(
        a_thursday, contact_params, group_col_name, day_of_week, seed
    )
    expected = pd.Series(False, index=a_thursday.index)
    assert_series_equal(res, expected, check_names=False)


def test_go_to_weekly_meeting_right_day(a_thursday, no_reduction_params):
    a_thursday["group_col"] = [1, 2, 1, 2, 3, 3, 3] + [-1] * (len(a_thursday) - 7)

    res = go_to_weekly_meeting(
        states=a_thursday,
        params=no_reduction_params,
        group_col_name="group_col",
        day_of_week="Thursday",
        seed=3931,
    )
    expected = pd.Series(False, index=a_thursday.index)
    expected[:7] = True
    assert_series_equal(res, expected, check_names=False)


def test_go_to_daily_work_meeting_weekend(states, no_reduction_params):
    a_saturday = states[states["date"] == pd.Timestamp("2020-04-04")].copy()
    a_saturday["work_saturday"] = [True, True] + [False] * (len(a_saturday) - 2)
    a_saturday["work_daily_group_id"] = 333
    res = go_to_daily_work_meeting(a_saturday, no_reduction_params, seed=None)
    expected = pd.Series(False, index=a_saturday.index)
    expected[:2] = True
    assert_series_equal(res, expected, check_names=False)


def test_go_to_daily_work_meeting_weekday(a_thursday, no_reduction_params):
    a_thursday["work_daily_group_id"] = [1, 2, 1, 2, 3, 3, 3] + [-1] * (
        len(a_thursday) - 7
    )
    res = go_to_daily_work_meeting(a_thursday, no_reduction_params, seed=None)
    expected = pd.Series(False, index=a_thursday.index)
    # not every one we assigned a group id is a worker
    expected.iloc[:7] = [True, True, False, True, True, False, True]
    assert_series_equal(res, expected, check_names=False)


def test_go_to_daily_work_meeting_weekday_with_reduction(
    a_thursday, no_reduction_params
):
    reduction_params = no_reduction_params
    reduction_params["value"] = 0.0
    a_thursday["work_daily_group_id"] = [1, 2, 1, 2, 3, 3, 3, 3, 3] + [-1] * (
        len(a_thursday) - 9
    )
    a_thursday.loc[1450:1458, "symptomatic"] = [
        False,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        False,
    ]
    res = go_to_daily_work_meeting(a_thursday, no_reduction_params, seed=None)
    expected = pd.Series(False, index=a_thursday.index)
    # not every one we assigned a group id is a worker
    expected[:9] = [True, True, False, True, False, False, True, False, True]
    assert_series_equal(res, expected, check_names=False)


# --------------------------- Non Recurrent Contact Models ---------------------------


def test_non_recurrent_work_contacts_weekend(states, params):
    a_saturday = states[states["date"] == pd.Timestamp("2020-04-04")]
    res = calculate_non_recurrent_contacts_from_empirical_distribution(
        states=a_saturday,
        params=params.loc["work_non_recurrent"],
        on_weekends=False,
        query="occupation == 'working'",
        seed=494,
    )
    assert_series_equal(res, pd.Series(data=0, index=a_saturday.index, dtype=float))


@pytest.fixture
def params_with_positive():
    params = pd.DataFrame.from_dict(
        {
            "category": ["work_non_recurrent"] * 3,
            "subcategory": [
                "all",
                "symptomatic_multiplier",
                "positive_test_multiplier",
            ],
            "name": [
                2,
                "symptomatic_multiplier",
                "positive_test_multiplier",
            ],  # nr of contacts
            "value": [1.0, 0.0, 0.0],  # probability
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
        params=params_with_positive.loc["work_non_recurrent"],
        on_weekends=False,
        query="occupation == 'working'",
        seed=433,
    )

    expected = a_thursday["age_group"].replace({"10-19": 0.0, "40-49": 2.0})

    assert_series_equal(res, expected, check_names=False, check_dtype=False)


def test_non_recurrent_work_contacts_no_random_no_sick_sat(
    states, params_with_positive
):
    a_saturday = states[states["date"] == pd.Timestamp("2020-04-04")].copy()
    a_saturday["symptomatic"] = False
    a_saturday["participates_saturday"] = [True, True, True] + [False] * (
        len(a_saturday) - 3
    )

    res = calculate_non_recurrent_contacts_from_empirical_distribution(
        states=a_saturday,
        params=params_with_positive.loc["work_non_recurrent"],
        on_weekends="participates",
        query="occupation == 'working'",
        seed=433,
    )

    expected = pd.Series(0, index=a_saturday.index)
    expected[:2] = 2

    assert_series_equal(res, expected, check_names=False, check_dtype=False)


def test_non_recurrent_work_contacts_no_random_with_sick(
    a_thursday, params_with_positive
):
    res = calculate_non_recurrent_contacts_from_empirical_distribution(
        states=a_thursday,
        params=params_with_positive.loc["work_non_recurrent"],
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
            + ["symptomatic_multiplier", "positive_test_multiplier"],
            "name": [
                3,
                2,
                "symptomatic_multiplier",
                "positive_test_multiplier",
            ],  # nr of contacts
            "value": [0.5, 0.5, 0.0, 0.0],  # probability
        }
    ).set_index(["category", "subcategory", "name"])

    res = calculate_non_recurrent_contacts_from_empirical_distribution(
        states=a_thursday,
        params=params.loc["work_non_recurrent"],
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
                "symptomatic_multiplier",
                "positive_test_multiplier",
            ],
            "name": [
                2,
                "symptomatic_multiplier",
                "positive_test_multiplier",
            ],  # nr of contacts
            "value": [1.0, 0.0, 0.0],  # probability
        }
    ).set_index(["category", "subcategory", "name"])

    res = calculate_non_recurrent_contacts_from_empirical_distribution(
        states=a_thursday,
        params=params.loc["other_non_recurrent"],
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
                "symptomatic_multiplier",
                "positive_test_multiplier",
            ],
            "name": [
                2,
                "symptomatic_multiplier",
                "positive_test_multiplier",
            ],  # nr of contacts
            "value": [1.0, 0.0, 0.0],  # probability
        }
    ).set_index(["category", "subcategory", "name"])

    res = calculate_non_recurrent_contacts_from_empirical_distribution(
        states=a_thursday,
        params=params.loc["other_non_recurrent"],
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
            + ["symptomatic_multiplier", "positive_test_multiplier"],
            "name": [
                3,
                2,
                "symptomatic_multiplier",
                "positive_test_multiplier",
            ],  # nr of contacts
            "value": [0.5, 0.5, 0.0, 0.0],  # probability
        }
    ).set_index(["category", "subcategory", "name"])

    res = calculate_non_recurrent_contacts_from_empirical_distribution(
        states=a_thursday,
        params=params.loc["other_non_recurrent"],
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
    multiplier = 0.5
    states.loc[:1, "quarantine_compliance"] = 0.3
    expected = pd.Series([10, 10, 0] + [10] * (len(states) - 3))
    res = reduce_contacts_on_condition(
        contacts=nr_of_contacts,
        states=states,
        multiplier=multiplier,
        condition="symptomatic",
        is_recurrent=False,
    )
    assert_series_equal(res, expected, check_dtype=False)


def test_reduce_recurrent_contacts_on_condition(states):
    participating = pd.Series(data=True, index=states.index)
    states["symptomatic"] = [True, True, True] + [False] * (len(states) - 3)
    states.loc[:0, "quarantine_compliance"] = 0.3
    multiplier = 0.5
    res = reduce_contacts_on_condition(
        contacts=participating,
        states=states,
        multiplier=multiplier,
        condition="symptomatic",
        is_recurrent=True,
    )
    expected = pd.Series([True, False, False] + [True] * (len(states) - 3))
    assert_series_equal(res, expected, check_dtype=False)


# ------------------------------------------------------------------------------------


def test_meet_daily_other_contacts():
    states = pd.DataFrame()
    states["symptomatic"] = [False, False, False, True]
    states["knows_infectious"] = [False, False, False, False]
    states["knows_immune"] = False
    states["cd_received_test_result_true"] = -20
    states["daily_meeting_id"] = [-1, 2, 2, 2]
    states["knows_currently_infected"] = states.eval(
        "knows_infectious | (knows_immune & symptomatic) "
        "| (knows_immune & (cd_received_test_result_true >= -13))"
    )
    states["quarantine_compliance"] = 1.0

    params = pd.DataFrame()
    params["value"] = [0.0, 0.0]
    params["subcategory"] = ["symptomatic_multiplier", "positive_test_multiplier"]
    params["name"] = params["subcategory"]
    params = params.set_index(["subcategory", "name"])

    res = meet_daily_other_contacts(
        states, params, group_col_name="daily_meeting_id", seed=None
    )
    expected = pd.Series([False, True, True, False])
    assert_series_equal(res, expected, check_names=False)


def test_identify_ppl_affected_by_vacation():
    states = pd.DataFrame()
    # 0: unaffected
    # 1: with child
    # 2: with educ worker
    # 3: with retired
    states["hh_id"] = [0, 0, 1, 1, 1, 2, 2, 3, 3]
    states["occupation"] = [
        # 0
        "working",
        "stays_home",
        # 1
        "school",
        "working",
        "stays_home",
        # 2
        "nursery_teacher",
        "working",
        # 3
        "retired",
        "stays_home",
    ]
    states["educ_worker"] = [False] * 5 + [True] + [False] * 3

    res = _identify_ppl_affected_by_vacation(states)
    expected = pd.Series([False, False] + [True] * 7)
    assert_series_equal(res, expected, check_names=False)


def test_draw_potential_vacation_contacts_not_random():
    state_to_vacation = {"A": "Easter", "B": "Spring"}  # C has no vacation
    states = pd.DataFrame()
    states["state"] = ["A", "A", "B", "B", "C"]

    params = pd.DataFrame()
    params["value"] = [1.0, 0.0]
    params["name"] = ["Easter", "Spring"]
    params["category"] = "additional_other_vacation_contact"
    params["subcategory"] = "probability"
    params = params.set_index(["category", "subcategory", "name"])
    seed = 3300
    res = _draw_potential_vacation_contacts(states, params, state_to_vacation, seed)
    expected = pd.Series([1, 1, 0, 0, 0])
    assert_series_equal(res, expected, check_names=False)
