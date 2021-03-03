import numpy as np
import pandas as pd
import pytest

from src.policies.single_policy_functions import _find_educ_workers_with_zero_students
from src.policies.single_policy_functions import _find_size_zero_classes
from src.policies.single_policy_functions import _get_a_b_children_staying_home
from src.policies.single_policy_functions import _get_non_a_b_children_staying_home
from src.policies.single_policy_functions import a_b_education


@pytest.fixture
def fake_states():
    states = pd.DataFrame(index=np.arange(10))
    states["state"] = ["Bayern", "Berlin"] * 5
    # date at which schools are open in Berlin but closed in Bavaria
    # date with uneven week number, i.e. where group a attends school
    states["date"] = pd.Timestamp("2020-04-23")
    states["school_group_id_0_a_b"] = [0, 1] * 5
    states["occupation"] = pd.Categorical(
        ["school"] * 8 + ["preschool_teacher", "school_teacher"]
    )
    states["school_group_id_0"] = [-1] + [22] * 9
    states["educ_worker"] = [False] * 8 + [True] * 2
    states["age"] = np.arange(10)
    return states


@pytest.fixture
def contacts(fake_states):
    return pd.Series(1, index=fake_states.index)


def test_a_b_school_system_above_age_0(fake_states, contacts):
    calculated = a_b_education(
        states=fake_states,
        contacts=contacts,
        seed=123,
        group_id_column="school_group_id_0",
        subgroup_query="occupation == 'school'",
        others_attend=True,
        hygiene_multiplier=1.0,
    )
    expected = pd.Series([0, 1] * 4 + [1, 1])
    pd.testing.assert_series_equal(calculated, expected)


def test_a_b_school_system_above_age_5(fake_states, contacts):
    calculated = a_b_education(
        states=fake_states,
        contacts=contacts,
        seed=123,
        group_id_column="school_group_id_0",
        subgroup_query="occupation == 'school' & age > 5",
        others_attend=True,
        hygiene_multiplier=1.0,
    )
    expected = pd.Series([1] * 6 + [0] + [1] * 3)
    pd.testing.assert_series_equal(calculated, expected)


def test_a_b_school_system_below_age_5(fake_states, contacts):
    calculated = a_b_education(
        states=fake_states,
        contacts=contacts,
        seed=123,
        group_id_column="school_group_id_0",
        subgroup_query="occupation == 'school' & age < 5",
        others_attend=False,
        hygiene_multiplier=1.0,
    )
    expected = pd.Series([0, 1, 0, 1, 0, 0, 0, 0, 1, 1])
    pd.testing.assert_series_equal(calculated, expected)


def test_a_b_education_others_home_no_hygiene():
    states = pd.DataFrame()
    states["county"] = [1, 1, 2, 2, 2, 2, 2, 2]
    states["educ_worker"] = [True, False, True, False, False, False, False, False]
    states["school_group_id_0"] = [11, 11, 22, 22, 22, 22, 22, -1]
    states["school_group_id_0_a_b"] = [0, 1, 1, 1, 0, 1, 0, 1]
    states["date"] = pd.Timestamp("2021-01-04")  # week 1

    contacts = pd.Series([1] * 6 + [0] * 2, index=states.index)
    seed = 333
    res = a_b_education(
        states=states,
        contacts=contacts,
        seed=seed,
        group_id_column="school_group_id_0",
        subgroup_query="county == 2",
        others_attend=False,
        hygiene_multiplier=1.0,
    )

    # zero class, closed county, teacher despite wrong week,
    # wrong week, right week, wrong week, right week but not attending, not in school
    expected = pd.Series([0, 0, 1, 1, 0, 1, 0, 0])
    pd.testing.assert_series_equal(res, expected)


def test_a_b_education_no_contacts():
    states = pd.DataFrame()
    states["educ_worker"] = [True, False, True, False, False, False, False]
    states["school_group_id_0_a_b"] = [0, 1, 0, 1, 0, 1, 0]
    states["school_group_id_0"] = [11, 11, 22, 22, 22, 22, -1]
    states["date"] = pd.Timestamp("2021-01-04")  # week 1

    contacts = pd.Series(0, index=states.index)
    seed = 333
    res = a_b_education(
        states=states,
        contacts=contacts,
        seed=seed,
        group_id_column="school_group_id_0",
        subgroup_query=None,
        others_attend=True,
        hygiene_multiplier=1.0,
    )

    pd.testing.assert_series_equal(res, contacts)


def test_get_a_b_children_staying_home():
    states = pd.DataFrame()
    states["educ_worker"] = [False, False, True, False, False]
    states["county"] = [1, 1, 2, 2, 2]
    states["group_col"] = [0, 1, 0, 1, 0]
    # wrong county, wrong county, educator, attending, WIN
    expected = pd.Series([False, False, False, False, True])

    subgroup_query = "county == 2"
    date = pd.Timestamp("2021-01-04")  # week number 1
    res = _get_a_b_children_staying_home(
        states,
        subgroup_query,
        group_column="group_col",
        date=date,
        always_attend_query=None,
        rhythm="weekly",
    )
    pd.testing.assert_series_equal(res, expected)


def test_get_a_b_children_staying_home_daily():
    states = pd.DataFrame()
    states["educ_worker"] = [False, False, True, False, False]
    states["county"] = 2
    states["group_col"] = [0, 1, 0, 1, 0]
    states["always_attend"] = [False, False, False, False, True]
    # non-school day, school day, educator, non-school day, always attend
    expected = pd.Series([True, False, False, False, False])

    subgroup_query = "county == 2"
    date = pd.Timestamp("2021-01-05")
    res = _get_a_b_children_staying_home(
        states,
        subgroup_query,
        group_column="group_col",
        date=date,
        always_attend_query="always_attend",
        rhythm="daily",
    )
    pd.testing.assert_series_equal(res, expected)


def test_get_a_b_children_staying_home_daily2():
    states = pd.DataFrame()
    states["educ_worker"] = [False, False, True, False, False]
    states["county"] = 2
    states["group_col"] = [0, 1, 0, 1, 0]
    states["always_attend"] = [False, False, False, False, True]
    # school day, non-school day, educator, non school day, always attend
    expected = pd.Series([False, True, False, True, False])

    subgroup_query = "county == 2"
    date = pd.Timestamp("2021-01-12")
    res = _get_a_b_children_staying_home(
        states,
        subgroup_query,
        group_column="group_col",
        date=date,
        always_attend_query="always_attend",
        rhythm="daily",
    )
    pd.testing.assert_series_equal(res, expected)


def test_get_non_a_b_children_staying_home_no_always_attend():
    states = pd.DataFrame()
    states["educ_worker"] = [False, False, True, False, False]
    states["county"] = [1, 1, 2, 2, 2]
    res = _get_non_a_b_children_staying_home(states, "county == 1", None)
    # under A/B under A/B, educator, affected, affected
    expected = pd.Series([False, False, False, True, True])
    pd.testing.assert_series_equal(res, expected)


def test_get_non_a_b_children_staying_home_with_always_attend():
    states = pd.DataFrame()
    states["educ_worker"] = [False, False, True, False, False]
    states["county"] = [1, 1, 2, 2, 2]
    states["always_attend"] = [False, True, False, False, True]
    res = _get_non_a_b_children_staying_home(states, "county == 1", "always_attend")
    # under A/B under A/B, educator, affected, always_attend
    expected = pd.Series([False, False, False, True, False])
    pd.testing.assert_series_equal(res, expected)


def test_find_size_zero_classes():
    col = "school_group_id_0"
    states = pd.DataFrame()
    states["educ_worker"] = [False, False, True, False, False, True, False]
    states[col] = [11, 11, 11, 22, 22, 22, -1]
    contacts = pd.Series([1, 1, 1, 0, 0, 1, 0])
    res = _find_size_zero_classes(contacts, states, col)
    expected = [22]
    assert res.tolist() == expected


def test_find_educ_workers_with_zero_students():
    col = "school_group_id_0"
    states = pd.DataFrame()
    states["educ_worker"] = [False, False, True, False, False, True, False]
    states[col] = [11, 11, 11, 22, 22, 22, -1]
    contacts = pd.Series([1, 1, 1, 0, 0, 1, 0])
    res = _find_educ_workers_with_zero_students(contacts, states, col)
    expected = pd.Series([False, False, False, False, False, True, False])
    pd.testing.assert_series_equal(res, expected)
