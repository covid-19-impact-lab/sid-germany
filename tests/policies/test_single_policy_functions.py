import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal

from src.policies.single_policy_functions import _interpolate_activity_level
from src.policies.single_policy_functions import reduce_recurrent_model
from src.policies.single_policy_functions import reduce_work_model
from src.policies.single_policy_functions import reopen_other_model
from src.policies.single_policy_functions import shut_down_model


@pytest.fixture
def fake_states():
    states = pd.DataFrame(index=np.arange(10))
    states["state"] = ["Bayern", "Berlin"] * 5
    # date at which schools are open in Berlin but closed in Bavaria
    # date with uneven week number, i.e. where group a attends school
    states["date"] = pd.Timestamp("2020-04-23")
    states["school_group_a"] = [0, 1] * 5
    states["occupation"] = pd.Categorical(
        ["school"] * 8 + ["preschool_teacher", "school_teacher"]
    )
    states["educ_worker"] = [False] * 8 + [True] * 2
    states["age"] = np.arange(10)
    return states


def test_shut_down_model_non_recurrent():
    contacts = pd.Series(np.arange(3))
    states = pd.DataFrame(index=["a", "b", "c"])
    calculated = shut_down_model(states, contacts, 123, is_recurrent=False)
    expected = pd.Series(0, index=["a", "b", "c"])
    assert_series_equal(calculated, expected)


def test_shut_down_model_recurrent():
    contacts = pd.Series(np.arange(3))
    states = pd.DataFrame(index=["a", "b", "c"])
    calculated = shut_down_model(states, contacts, 123, is_recurrent=True)
    expected = pd.Series(False, index=["a", "b", "c"])
    assert_series_equal(calculated, expected)


def test_reduce_recurrent_model_set_zero():
    states = pd.DataFrame(index=[0, 1, 2, 3])
    contacts = pd.Series([True, True, False, False])
    calculated = reduce_recurrent_model(states, contacts, 333, multiplier=0.0)
    assert (calculated == 0).all()


def test_reduce_recurrent_model_no_change():
    states = pd.DataFrame(index=[0, 1, 2, 3])
    contacts = pd.Series([True, True, False, False])
    calculated = reduce_recurrent_model(states, contacts, 333, multiplier=1.0)
    assert np.allclose(contacts, calculated)


def test_reduce_recurrent_model_one_in_four():
    n_obs = 10_000
    states = pd.DataFrame(index=np.arange(n_obs))
    contacts = pd.Series([True, False] * int(n_obs / 2))
    calculated = reduce_recurrent_model(
        states=states, contacts=contacts, seed=1234, multiplier=0.25
    )

    # check that we get expected number of contacts right
    calculated_mean = calculated.mean()
    expected_mean = 0.125
    assert np.allclose(calculated_mean, expected_mean, rtol=0.005, atol=0.005)

    # check that people who stayed home before policy still stay home
    assert not calculated[~contacts].any()


def test_reduce_work_model(fake_states):
    fake_states["work_contact_priority"] = np.arange(10)[::-1] / 10
    contacts = pd.Series(1, index=fake_states.index)
    contacts[2] = 0

    calculated = reduce_work_model(
        states=fake_states,
        contacts=contacts,
        seed=123,
        attend_multiplier=0.5,
        hygiene_multiplier=1.0,
        is_recurrent=False,
    )
    expected = pd.Series(
        [1, 1, 0, 1, 0, 0, 0, 0, 0, 0], index=fake_states.index, dtype=float
    )
    assert_series_equal(calculated, expected)


def test_reduce_work_model_with_hygiene_multiplier(fake_states):
    fake_states["work_contact_priority"] = np.arange(10)[::-1] / 10
    contacts = pd.Series(2, index=fake_states.index)
    contacts[2] = 0
    contacts[3] = 5

    calculated = reduce_work_model(
        states=fake_states,
        contacts=contacts,
        seed=123,
        attend_multiplier=0.5,
        hygiene_multiplier=0.5,
        is_recurrent=False,
    )
    expected = pd.Series(
        [1, 1, 0, 2.5, 0, 0, 0, 0, 0, 0], index=fake_states.index, dtype=float
    )
    assert_series_equal(calculated, expected)


def test_reduce_work_model_multiplier_series(fake_states):
    fake_states["work_contact_priority"] = np.arange(10)[::-1] / 10
    contacts = pd.Series(True, index=fake_states.index)
    contacts[2] = False

    calculated = reduce_work_model(
        states=fake_states,
        contacts=contacts,
        seed=123,
        attend_multiplier=pd.Series([0.5], index=[pd.Timestamp("2020-04-23")]),
        hygiene_multiplier=1.0,
        is_recurrent=True,
    )
    expected = pd.Series(
        [True, True, False, True, False, False, False, False, False, False],
        index=fake_states.index,
    )
    assert_series_equal(calculated, expected)


def test_reduce_work_model_multiplier_frame_missing_state(fake_states):
    fake_states["work_contact_priority"] = np.arange(10)[::-1] / 10
    fake_states["state"] = ["A", "B"] * 5
    contacts = pd.Series(6, index=fake_states.index)
    contacts[2] = 0
    multiplier = pd.DataFrame(data={"A": [0.5]}, index=[pd.Timestamp("2020-04-23")])

    with pytest.raises(AssertionError):
        reduce_work_model(
            states=fake_states,
            contacts=contacts,
            seed=123,
            attend_multiplier=multiplier,
            hygiene_multiplier=1.0,
            is_recurrent=False,
        )


def test_reduce_work_model_multiplier_frame(fake_states):
    fake_states["work_contact_priority"] = np.arange(10)[::-1] / 10
    fake_states["state"] = ["A", "B"] * 5
    contacts = pd.Series(1, index=fake_states.index)
    contacts[2] = 0
    multiplier = pd.DataFrame(
        data={"A": [0.55], "B": [0.85]}, index=[pd.Timestamp("2020-04-23")]
    )
    calculated = reduce_work_model(
        states=fake_states,
        contacts=contacts,
        seed=123,
        attend_multiplier=multiplier,
        hygiene_multiplier=1.0,
        is_recurrent=False,
    )
    expected = pd.Series(
        [1, 1, 0, 1, 1, 1, 0, 1, 0, 0], index=fake_states.index, dtype=float
    )
    assert_series_equal(calculated, expected)


def test_interpolate_activity_level():
    calculated = _interpolate_activity_level(
        date="2020-03-20",
        start_multiplier=0.5,
        end_multiplier=1,
        start_date="2020-03-15",
        end_date="2020-03-25",
    )

    assert calculated == 0.75


def test_reopen_other_model():
    calculated = reopen_other_model(
        states=pd.DataFrame([pd.Timestamp("2020-03-20")], columns=["date"]),
        contacts=pd.Series(np.arange(10)),
        seed=1234,
        start_multiplier=0.5,
        end_multiplier=1,
        start_date="2020-03-15",
        end_date="2020-03-25",
        is_recurrent=False,
    )

    expected = pd.Series(np.arange(10) * 0.75)

    assert_series_equal(calculated, expected)
