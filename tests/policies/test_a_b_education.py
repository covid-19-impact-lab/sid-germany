import numpy as np
import pandas as pd
import pytest

from src.policies.single_policy_functions import (
    _identify_who_attends_because_of_a_b_schooling,
)
from src.policies.single_policy_functions import mixed_educ_policy


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
    return pd.Series(True, index=fake_states.index)


def test_a_b_school_system_above_age_0(fake_states, contacts):
    calculated = mixed_educ_policy(
        states=fake_states,
        contacts=contacts,
        seed=123,
        group_id_column="school_group_id_0",
        a_b_query="occupation == 'school'",
        non_a_b_attend=True,
        hygiene_multiplier=1.0,
        always_attend_query="state == 'Niedersachsen'",  # no one
        params=None,
    )
    expected = pd.Series([False, True] * 4 + [True, True])
    pd.testing.assert_series_equal(calculated, expected)


def test_a_b_school_system_above_age_5(fake_states, contacts):
    calculated = mixed_educ_policy(
        states=fake_states,
        contacts=contacts,
        seed=123,
        group_id_column="school_group_id_0",
        a_b_query="occupation == 'school' & age > 5",
        non_a_b_attend=True,
        hygiene_multiplier=1.0,
        always_attend_query="state == 'Niedersachsen'",  # no one
        params=None,
    )
    expected = pd.Series([True] * 6 + [False] + [True] * 3)
    pd.testing.assert_series_equal(calculated, expected)


def test_a_b_school_system_below_age_5(fake_states, contacts):
    calculated = mixed_educ_policy(
        states=fake_states,
        contacts=contacts,
        seed=123,
        group_id_column="school_group_id_0",
        a_b_query="occupation == 'school' & age < 5",
        non_a_b_attend=False,
        hygiene_multiplier=1.0,
        always_attend_query="state == 'Niedersachsen'",  # no one
        params=None,
    )
    expected = pd.Series(
        [False, True, False, True, False, False, False, False, True, True]
    )
    pd.testing.assert_series_equal(calculated, expected)


def test_mixed_educ_policy_others_home_no_hygiene():
    states = pd.DataFrame()
    states["county"] = [1, 1, 2, 2, 2, 2, 2, 2]
    states["educ_worker"] = [True, False, True, False, False, False, False, False]
    states["school_group_id_0"] = [11, 11, 22, 22, 22, 22, 22, -1]
    states["school_group_id_0_a_b"] = [0, 1, 1, 1, 0, 1, 0, 1]
    states["date"] = pd.Timestamp("2021-01-04")  # week 1

    contacts = pd.Series([True] * 6 + [False] * 2, index=states.index)
    seed = 333
    res = mixed_educ_policy(
        states=states,
        contacts=contacts,
        seed=seed,
        group_id_column="school_group_id_0",
        a_b_query="county == 2",
        non_a_b_attend=False,
        hygiene_multiplier=1.0,
        always_attend_query="county == 55",  # no one
        params=None,
    )

    # zero class, closed county, teacher despite wrong week,
    # wrong week, right week, wrong week, right week but not attending, not in school
    expected = pd.Series([True, False, True, True, False, True, False, False])
    pd.testing.assert_series_equal(res, expected)


def test_mixed_educ_policy_no_contacts():
    states = pd.DataFrame()
    states["educ_worker"] = [True, False, True, False, False, False, False]
    states["school_group_id_0_a_b"] = [0, 1, 0, 1, 0, 1, 0]
    states["school_group_id_0"] = [11, 11, 22, 22, 22, 22, -1]
    states["date"] = pd.Timestamp("2021-01-04")  # week 1
    states["county"] = 33

    contacts = pd.Series(False, index=states.index)
    seed = 333
    res = mixed_educ_policy(
        states=states,
        contacts=contacts,
        seed=seed,
        group_id_column="school_group_id_0",
        a_b_query=True,
        non_a_b_attend=True,
        hygiene_multiplier=1.0,
        always_attend_query="county == 55",  # no one
        params=None,
    )

    pd.testing.assert_series_equal(res, contacts)


def test_identify_who_attends_because_of_a_b_schooling():
    states = pd.DataFrame()
    states["county"] = [1, 1, 2, 2, 2]
    states["group_col"] = [0, 1, 0, 1, 0]
    states["date"] = pd.Timestamp("2021-01-04")  # week number 1
    # wrong county, wrong county, wrong week, right week, wrong week
    expected = pd.Series([False, False, False, True, False])

    a_b_query = "county == 2"
    res = _identify_who_attends_because_of_a_b_schooling(
        states,
        a_b_query=a_b_query,
        group_column="group_col",
        a_b_rhythm="weekly",
    )
    pd.testing.assert_series_equal(res, expected)


def test_identify_who_attends_because_of_a_b_schooling_daily():
    states = pd.DataFrame()
    states["group_col"] = [0, 1, 0, 1, 0]
    states["county"] = 2
    states["date"] = pd.Timestamp("2021-01-05")
    expected = states["group_col"].astype(bool)

    a_b_query = "county == 2"
    res = _identify_who_attends_because_of_a_b_schooling(
        states=states,
        a_b_query=a_b_query,
        group_column="group_col",
        a_b_rhythm="daily",
    )
    pd.testing.assert_series_equal(res, expected, check_names=False)


def test_idenfy_who_attends_because_of_a_b_schooling_daily2():
    states = pd.DataFrame()
    states["educ_worker"] = [False, False, True, False, False]
    states["county"] = 2
    states["group_col"] = [0, 1, 0, 1, 0]
    states["date"] = pd.Timestamp("2021-01-12")

    expected = ~states["group_col"].astype(bool)

    a_b_query = "county == 2"
    res = _identify_who_attends_because_of_a_b_schooling(
        states=states,
        a_b_query=a_b_query,
        group_column="group_col",
        a_b_rhythm="daily",
    )
    pd.testing.assert_series_equal(res, expected, check_names=False)


def test_emergency_care():
    states = pd.DataFrame()
    states["educ_worker"] = [True, False, True, False, False]
    states["always_attend"] = [False, False, True, True, False]
    states["school_group_id_0"] = [1, 1, 2, 2, -1]
    contacts = pd.Series([True, True, True, True, False], index=states.index)
    res = mixed_educ_policy(
        contacts=contacts,
        states=states,
        seed=333,
        group_id_column="school_group_id_0",
        always_attend_query="always_attend",
        a_b_query=False,
        non_a_b_attend=False,
        hygiene_multiplier=1.0,
        params=None,
    )
    # educ_worker without class, child not in emergency care, educ_worker with class,
    # attends, outside educ system
    expected = pd.Series([True, False, True, True, False])
    pd.testing.assert_series_equal(res, expected)
