import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal

from src.policies.single_policy_functions import _interpolate_activity_level
from src.policies.single_policy_functions import apply_multiplier_to_recurrent_contacts
from src.policies.single_policy_functions import (
    close_educ_facilities_until_reopening_date,
)
from src.policies.single_policy_functions import implement_a_b_school_system_above_age
from src.policies.single_policy_functions import (
    reduce_other_contacts_in_gradual_opening_or_closing,
)
from src.policies.single_policy_functions import reduce_to_systemically_relevant
from src.policies.single_policy_functions import (
    reduce_work_contacts_in_gradual_opening_or_closing,
)
from src.policies.single_policy_functions import (
    reduce_work_contacts_to_share_of_active_non_essential_workers,
)
from src.policies.single_policy_functions import set_to_zero


@pytest.fixture
def fake_states():
    states = pd.DataFrame(index=np.arange(10))
    states["state"] = ["Bayern", "Berlin"] * 5
    # date at which schools are open in Berlin but closed in Bavaria
    # date with uneven week number, i.e. where group a attends school
    states["date"] = pd.Timestamp("2020-04-23")
    states["school_group_a"] = [0, 1] * 5
    states["age"] = np.arange(10)
    states["systemically_relevant"] = [True, False] * 5
    return states


def test_set_to_zero():
    contacts = pd.Series(np.arange(3))
    states = pd.DataFrame(index=["a", "b", "c"])
    calculated = set_to_zero(states, contacts, 123)
    expected = pd.Series(0, index=["a", "b", "c"])
    assert_series_equal(calculated, expected)


def test_close_educ_facilities_until_reopening_date_multiplier_1(fake_states):
    # date at which Berlin is open but Bavaria not
    calculated = close_educ_facilities_until_reopening_date(
        states=fake_states,
        contacts=pd.Series([1] * 10),
        seed=123,
        multiplier=1,
    )

    expected = pd.Series([0, 1] * 5)
    assert_series_equal(calculated, expected)


def test_close_educ_facilities_until_reopening_date_multiplier_0(fake_states):
    calculated = close_educ_facilities_until_reopening_date(
        states=fake_states,
        contacts=pd.Series([1] * 10),
        seed=123,
        multiplier=0,
    )

    expected = pd.Series([0] * 10)
    assert_series_equal(calculated, expected)


def test_apply_multiplier_to_recurrent_contacts():
    n_obs = 10_000
    states = pd.DataFrame(index=np.arange(n_obs))
    contacts = pd.Series([1, 0] * int(n_obs / 2))
    calculated = apply_multiplier_to_recurrent_contacts(
        states=states, contacts=contacts, seed=1234, multiplier=0.25
    )

    # check that we get expected number of contacts right
    calculated_mean = calculated.mean()
    expected_mean = 0.125
    assert np.allclose(calculated_mean, expected_mean, rtol=0.005, atol=0.005)

    # check that people who stayed home before policy still stay home
    assert (calculated[contacts == 0]).sum() == 0


def test_a_b_school_system_above_age_0(fake_states):

    calculated = implement_a_b_school_system_above_age(
        states=fake_states,
        contacts=pd.Series(1, index=fake_states.index),
        seed=123,
        age_cutoff=0,
    )

    expected = pd.Series([0, 1] * 5)
    assert_series_equal(calculated, expected)


def test_a_b_school_system_above_age_5(fake_states):

    calculated = implement_a_b_school_system_above_age(
        states=fake_states,
        contacts=pd.Series(1, index=fake_states.index),
        seed=123,
        age_cutoff=5,
    )

    expected = pd.Series([1] * 6 + [0, 1] * 2)
    assert_series_equal(calculated, expected)


def test_reduce_to_systemically_relevant(fake_states):
    contacts = pd.Series([0] * 5 + [1] * 5)
    calculated = reduce_to_systemically_relevant(fake_states, contacts, 123)
    expected = pd.Series([0] * 6 + [1, 0, 1, 0])
    assert_series_equal(calculated, expected)


def test_reduce_work_contacts_to_share_of_active_non_essential_workers(fake_states):
    fake_states["work_contact_priority"] = np.arange(10)[::-1] / 10
    contacts = pd.Series([0, 1] * 5)

    calculated = reduce_work_contacts_to_share_of_active_non_essential_workers(
        states=fake_states,
        contacts=contacts,
        seed=123,
        share=0.5,
    )

    expected = pd.Series([0, 1, 0, 1] + [0] * 6)
    assert_series_equal(calculated, expected)


def test_interpolate_activity_level():
    calculated = _interpolate_activity_level(
        date="2020-03-20",
        start_level=0.5,
        end_level=1,
        start_date="2020-03-15",
        end_date="2020-03-25",
    )

    assert calculated == 0.75


def test_reduce_other_contacts_in_gradual_opening_or_closing():
    calculated = reduce_other_contacts_in_gradual_opening_or_closing(
        states=pd.DataFrame([pd.Timestamp("2020-03-20")], columns=["date"]),
        contacts=pd.Series(np.arange(10)),
        seed=1234,
        start_level=0.5,
        end_level=1,
        start_date="2020-03-15",
        end_date="2020-03-25",
    )

    expected = pd.Series(np.arange(10) * 0.75)

    assert_series_equal(calculated, expected)


def test_reduce_work_contacts_in_gradual_opening_or_closing(fake_states):
    fake_states["date"] = pd.Timestamp("2020-03-20")
    fake_states["work_contact_priority"] = np.arange(10)[::-1] / 10

    calculated = reduce_work_contacts_in_gradual_opening_or_closing(
        states=fake_states,
        contacts=pd.Series([1] * 10),
        seed=1234,
        start_level=0.5,
        end_level=1,
        start_date="2020-03-15",
        end_date="2020-03-25",
    )

    expected = pd.Series([1] * 7 + [0] * 3)
    assert_series_equal(calculated, expected)
