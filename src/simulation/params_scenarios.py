import warnings

import pandas as pd

from src.testing.shared import get_piecewise_linear_interpolation


def reduce_rapid_test_demand_after_may_17(params):
    """Reduce rapid tests of households and workers.

    Since only the offer share of workers' rapid tests is time variant we change the
    offer parameters rather than the demand even though the more intuitive
    interpretation would likely be that workers demand less tests.

    The share of workers being offered twice weekly tests is 0.7 in May.
    The share of household members seeking a rapid test after a symptom or positive test
    event was 0.4 before the estimation. This needs to be updated afterwards.

    """
    change_date = pd.Timestamp("2021-05-17")

    new_hh_val = 0.25
    warnings.warn(
        "The demand reduction for tests after household events must be updated "
        "after the estimation."
    )
    new_work_offer_share = 0.35

    params = change_date_params_after_date(
        params=params,
        loc=("rapid_test_demand", "hh_member_demand"),
        change_date=change_date,
        new_val=new_hh_val,
    )
    params = change_date_params_after_date(
        params=params,
        loc=("rapid_test_demand", "share_workers_receiving_offer"),
        change_date=change_date,
        new_val=new_work_offer_share,
    )

    return params


def change_date_params_after_date(params, loc, change_date, new_val):
    """Change time variant params after a certain date.

    This function can be used to change any params that are fed to
    get_piecewise_linear_interpolation.

    The resulting piecewise linear interpolation is the same until change_date
    and then falls on change_date to the new value and stays there.

    """
    change_date = pd.Timestamp(change_date)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="indexing past lexsort depth may impact performance."
        )
        before_params_slice = params.loc[loc]
    new_params_slice = _build_new_date_params(before_params_slice, change_date, new_val)

    new_params_slice.index = [str(x.date()) for x in new_params_slice.index]
    # The order of prepending the index levels is important to have the correct result.
    new_params_slice = pd.concat({loc[1]: new_params_slice}, names=["subcategory"])
    new_params_slice = pd.concat({loc[0]: new_params_slice}, names=["category"])

    new_params = pd.concat([params.drop(index=loc), new_params_slice])
    return new_params


def _build_new_date_params(before_params_slice, change_date, new_val):
    before = get_piecewise_linear_interpolation(before_params_slice)
    day_before = change_date - pd.Timedelta(days=1)
    if day_before in before:
        val_before_change = before.loc[day_before]
    else:
        val_before_change = before.iloc[-1]

    new_params_slice = before_params_slice.copy(deep=True)
    new_params_slice.index = pd.DatetimeIndex(new_params_slice.index)

    # remove old change points after change date
    new_params_slice = new_params_slice[:day_before]
    # implement the switch on the change date
    new_params_slice.loc[day_before] = val_before_change
    new_params_slice.loc[change_date] = new_val
    # maintain the new value over time
    new_params_slice.loc[pd.Timestamp("2025-12-31")] = new_val
    new_params_slice = new_params_slice.sort_index()
    return new_params_slice


# -------------------------------------------------------------------------------------


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
