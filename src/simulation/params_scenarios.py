import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


def baseline(params):
    return params


def no_rapid_tests_at_schools(params):
    params = params.copy(deep=True)
    params.loc[("rapid_test_demand", "educ_worker_shares"), "value"] = 0.0
    params.loc[("rapid_test_demand", "student_shares"), "value"] = 0.0
    return params


def no_rapid_tests_at_work(params):
    params = params.copy(deep=True)
    params.loc[("rapid_test_demand", "work", "share_accepting_offer"), "value"] = 0.0
    params.loc[("rapid_test_demand", "share_workers_receiving_offer"), "value"] = 0.0
    return params


def rapid_tests_at_school_every_other_day_after_april_5(params):
    params = params.copy(deep=True)
    params.loc[("rapid_test_demand", "educ_frequency", "after_easter"), "value"] = 2
    return params


def rapid_tests_at_school_every_day_after_april_5(params):
    params = params.copy(deep=True)
    params.loc[("rapid_test_demand", "educ_frequency", "after_easter"), "value"] = 1
    return params
