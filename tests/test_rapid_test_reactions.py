import numpy as np
import pandas as pd

from src.testing.rapid_test_reactions import rapid_test_reactions


def test_rapid_test_reactions():
    states = pd.DataFrame()
    states["quarantine_compliance"] = [0.0, 0.2, 0.4, 0.6, 0.8]
    states["cd_received_rapid_test"] = 0
    states["is_tested_positive_by_rapid_test"] = True

    contacts = pd.DataFrame()
    contacts["households"] = [True, True, True, True, True]
    contacts["other_recurrent"] = [False, True, False, True, False]
    contacts["other_non_recurrent"] = [5, 2, 2, 2, 2]

    expected = pd.DataFrame()
    expected["households"] = [True, True, True, True, 0]
    expected["other_recurrent"] = [False, False, False, False, False]
    expected["other_non_recurrent"] = [5, 0, 0, 0, 0]

    params = pd.DataFrame(
        data=[0.7, 0.15],
        columns=["value"],
        index=pd.MultiIndex.from_tuples(
            [
                ("rapid_test_demand", "reaction", "hh_contacts_multiplier"),
                ("rapid_test_demand", "reaction", "not_hh_contacts_multiplier"),
            ]
        ),
    )
    res = rapid_test_reactions(states, contacts, params, None)

    pd.testing.assert_frame_equal(res, expected, check_dtype=False)


def test_rapid_test_reactions_lln():
    np.random.seed(38484)
    states = pd.DataFrame()
    states["quarantine_compliance"] = np.random.uniform(0, 1, size=10000)
    states["cd_received_rapid_test"] = [0] * 9900 + [-3] * 90 + [-99] * 10
    states["is_tested_positive_by_rapid_test"] = (
        [True] * 9980 + [False] * 10 + [True] * 10
    )

    contacts = pd.DataFrame()
    contacts["households"] = [True] * 10000
    contacts["other"] = True

    params = pd.DataFrame(
        data=[0.7, 0.15],
        columns=["value"],
        index=pd.MultiIndex.from_tuples(
            [
                ("rapid_test_demand", "reaction", "hh_contacts_multiplier"),
                ("rapid_test_demand", "reaction", "not_hh_contacts_multiplier"),
            ]
        ),
    )

    res = rapid_test_reactions(states, contacts, params, None)

    quarantine_pool = res.loc[:9979]
    share_meet_other = quarantine_pool["other"].mean()
    share_meet_hh = quarantine_pool["households"].mean()
    assert 0.145 < share_meet_other < 0.155
    assert 0.695 < share_meet_hh < 0.705
    assert (res.loc[9980:] == contacts.loc[9980:]).all().all()
