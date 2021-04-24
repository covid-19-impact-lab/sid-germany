import pandas as pd
import pandas.testing as pdt
import pytest

from src.contact_models.task_create_contact_params import _create_n_contacts
from src.contact_models.task_create_contact_params import _get_relevant_contacts_subset
from src.contact_models.task_create_contact_params import _make_decreasing
from src.contact_models.task_create_contact_params import (
    _reduce_empirical_distribution_to_max_contacts,
)


@pytest.fixture
def mostly_worker():
    df = pd.DataFrame()
    df["id"] = [0, 0, 1, 1, 1, 2]
    df["participant_occupation"] = ["working"] * 5 + ["not working"]
    df["place"] = ["work"] + ["leisure"] * 5
    df["phys_contact"] = True
    df["duration"] = "1-4h"
    df["recurrent"] = [True, False] * 3
    return df


def test_get_relevant_contacts_subset(mostly_worker):
    res = _get_relevant_contacts_subset(
        contacts=mostly_worker,
        places=["work", "leisure"],
        recurrent=True,
        frequency=None,
        weekend=None,
    )
    expected = mostly_worker.loc[[0, 2, 4]]
    pdt.assert_frame_equal(res, expected)


def test_create_n_contacts_work(mostly_worker):
    res = _create_n_contacts(
        mostly_worker,
        places=["work"],
        recurrent=True,
        frequency=None,
        weekend=None,
    )
    expected = pd.Series([1, 0], index=[0, 1])
    pdt.assert_series_equal(res, expected, check_names=False, check_dtype=False)


def test_create_n_contacts_leisure(mostly_worker):
    res = _create_n_contacts(
        mostly_worker,
        places=["leisure"],
        recurrent=False,
        frequency=None,
        weekend=None,
    )
    expected = pd.Series([1, 1, 1], index=[0, 1, 2])
    pdt.assert_series_equal(res, expected, check_names=False, check_dtype=False)


def test_make_decreasing_unchanged():
    already_decreasing = pd.Series([4, 3, 2, 1], name="value")
    res = _make_decreasing(already_decreasing)
    pdt.assert_series_equal(res, already_decreasing)


def test_make_decreasing_needs_change():
    already_decreasing = pd.Series([4, 3, 1, 2])
    res = _make_decreasing(already_decreasing)
    expected = pd.Series([4, 4, 1, 1], name="value")
    pdt.assert_series_equal(res, expected)


def test_reduce_empirical_distribution_to_max_contacts_no_restriction():
    emp_dist = pd.Series([5, 4, 3, 2], index=[0, 1, 2, 3], name="value")
    max_contacts = 3
    res = _reduce_empirical_distribution_to_max_contacts(emp_dist, max_contacts, 1e10)
    pdt.assert_series_equal(res, emp_dist, check_dtype=False)


def test_reduce_empirical_distribution_to_max_contacts_restriction():
    emp_dist = pd.Series([8, 6, 3, 2, 1], index=[0, 1, 2, 3, 4], name="value")
    # nobs with non-zero contacts: 12
    # total desired: 22
    # 6 * 1 + 6 * 2 is the solution (18)
    max_contacts = 2
    res = _reduce_empirical_distribution_to_max_contacts(emp_dist, max_contacts, 1e10)
    expected = pd.Series([8, 6, 6], index=[0, 1, 2], name="value")
    pdt.assert_series_equal(res, expected, check_dtype=False)
