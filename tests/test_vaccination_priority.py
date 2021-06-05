import numpy as np
import pandas as pd
import pytest

from src.create_initial_states.create_vaccination_priority import (
    _get_educators_of_young_children,
)
from src.create_initial_states.create_vaccination_priority import (
    _get_second_priority_people_acc_to_stiko,
)
from src.create_initial_states.create_vaccination_priority import _get_third_priority
from src.create_initial_states.create_vaccination_priority import (
    create_vaccination_rank,
)


def test_create_vaccination_rank_without_refusals():
    vaccination_group = pd.Series([3, 1, 1, 2, 4, 4])
    # ranks 3, 0, 1, 2, 4, 5
    expected = pd.Series([0.6, 0.0, 0.2, 0.4, 0.8, 1.0])
    res = create_vaccination_rank(vaccination_group, share_refuser=0.0, seed=333)
    res_sorted_index = res.sort_index()
    pd.testing.assert_series_equal(res_sorted_index, expected, check_index_type=False)


def test_create_vaccination_rank_lln():
    vaccination_group = pd.Series([1, 2, 3] * 500_000)
    # this does not necessarily work for other seeds
    # That is because np.random.choice does not always fit exactly the share of
    # refusers leading to the mis-identification of non_refusers / refusers
    # through the res[res < 0.5] specification
    res = create_vaccination_rank(vaccination_group, share_refuser=0.5, seed=1114)

    non_refusers = res[res < 0.5]

    comply_group1 = non_refusers[vaccination_group == 1]
    comply_group2 = non_refusers[vaccination_group == 2]
    comply_group3 = non_refusers[vaccination_group == 3]
    assert comply_group1.max() <= comply_group2.min()
    assert comply_group2.max() <= comply_group3.min()
    assert comply_group3.max() == pytest.approx(0.5, rel=1e-4, abs=1e-4)

    refuser_groups = vaccination_group[res > 0.5]
    refuser_group_shares = refuser_groups.value_counts(normalize=True).sort_index()
    expected_group_shares = pd.Series(1 / 3, index=[1, 2, 3])
    share_diff = np.abs(refuser_group_shares - expected_group_shares)
    assert share_diff.max() < 0.002


def test_get_second_priority_people_acc_to_stiko():
    states = pd.DataFrame()
    states["age"] = [75, 30] + [75] * 3 + [30] * 15 + [14] * 3 + [4] * 2

    vaccination_group = pd.Series(
        [1, np.nan] + [np.nan] * 3 + [2] * 15 + [5] * 3 + [np.nan] * 2
    )
    res = _get_second_priority_people_acc_to_stiko(
        states=states, vaccination_group=vaccination_group
    )
    expected = pd.Series(
        [
            False,  # already has vaccination group
            False,  # wrong age
        ]
        + [True] * 3  # elderly
        + [False] * 20  # too young
    )
    pd.testing.assert_series_equal(res, expected)


def test_get_second_priority_people_acc_to_stiko_no_elderly():
    states = pd.DataFrame()
    states["age"] = [15] * 5 + [58] * 6666 + [28] * 3334 + [75] * 5

    vaccination_group = pd.Series(np.nan, index=states.index)

    np.random.seed(8899)
    res = _get_second_priority_people_acc_to_stiko(
        states=states, vaccination_group=vaccination_group
    )

    # children should remain NaN
    assert vaccination_group[:5].isnull().all()

    adult_vaccination_group = res[5:]
    # among non-elderly 13.5% chosen to be in 2nd or 3rd group
    assert 0.13 < adult_vaccination_group[:-5].mean() < 0.14
    # elderly are all in this group
    assert adult_vaccination_group[-5:].all()


def test_get_educators_of_young_children():
    states = pd.DataFrame()
    states["age"] = (
        [55, 6, 6, 6]  # class primary
        + [55, 6, 6, 6]  # class primary
        + [60, 14, 14, 16]  # class secondary
        + [-1]  # non educ participant
        + [1, 1, 1, 50]  # nursery
        + [4, 4, 3, 30]  # preschool
    )
    states["educ_worker"] = (
        [True, False, False, False]  # class primary
        + [True, False, False, False]  # class primary
        + [True, False, False, False]  # class secondary
        + [False]  # non educ participant
        + [False, False, False, True]  # nursery
        + [False, False, False, True]  # preschool
    )
    states["occupation"] = (
        ["school_teacher", "school", "school", "school"] * 3
        + ["retired"]
        + ["nursery", "nursery", "nursery", "nursery_teacher"]
        + ["preschool", "preschool", "preschool", "preschool_teacher"]
    )
    states["school_group_id_0"] = [0] * 4 + [1] * 4 + [2] * 4 + [-1] * 9

    vaccination_group = pd.Series(np.nan, index=states.index)
    vaccination_group[0] = 1
    res = _get_educators_of_young_children(states, vaccination_group)
    expected = pd.Series(
        [False] * 4 + [True] + [False] * 11 + [True, False, False, False, True]
    )
    pd.testing.assert_series_equal(res, expected)


def test_get_third_priority():
    states = pd.DataFrame()
    states["age"] = [63, 50, 80, 50] + [40] * 1000 + [50] * 1000
    states["educ_worker"] = [False, True, False, True] + [False] * 2000
    states["work_contact_priority"] = [-1, -1, 0.9, -1] + [-1, 0.3] * 1000

    vaccination_group = pd.Series(np.nan, index=states.index)
    vaccination_group[3] = 2
    np.random.seed(484)
    res = _get_third_priority(states, vaccination_group)

    # age right, educ_worker, high priority, already has priority
    assert (res[:4] == [True, True, True, False]).all()
    assert 0.025 < res[4:].mean() < 0.035
    # share above is 0.33, so young_to_old should be approximately 2
    # with equally sized groups
    young_to_old = res[states["age"] == 40].mean() / res[states["age"] == 50].mean()
    assert 1.99 < young_to_old < 2.01
