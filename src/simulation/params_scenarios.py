import warnings

import pandas as pd

from src.config import SUMMER_SCENARIO_START
from src.testing.shared import get_piecewise_linear_interpolation


def reduce_rapid_test_demand_after_summer_scenario_start(params):
    """Reduce rapid tests of households and workers.

    Since only the offer share of workers' rapid tests is time variant we change the
    offer parameters rather than the demand even though the more intuitive
    interpretation would likely be that workers demand less tests.

    The share of workers being offered twice weekly tests is 0.7 in May.
    The share of household members seeking a rapid test after a symptom or positive test
    event was 0.4 before the estimation. This needs to be updated afterwards.

    """
    change_date = pd.Timestamp(SUMMER_SCENARIO_START)

    new_hh_val = 0.25
    warnings.warn(
        "The demand reduction for tests after household events must be updated "
        "after the estimation."
    )
    new_work_offer_share = 0.35

    params = _change_date_params_after_date(
        params=params,
        loc=("rapid_test_demand", "hh_member_demand"),
        change_date=change_date,
        new_val=new_hh_val,
    )
    params = _change_date_params_after_date(
        params=params,
        loc=("rapid_test_demand", "share_workers_receiving_offer"),
        change_date=change_date,
        new_val=new_work_offer_share,
    )

    return params


def _change_date_params_after_date(params, loc, change_date, new_val):
    """Change time variant params after a certain date.

    This function can be used to change any params that are fed to
    get_piecewise_linear_interpolation.

    The resulting piecewise linear interpolation is the same until change_date
    and then falls on change_date to the new value and stays there.

    Args:
        params (pandas.DataFrame)
        loc (tuple): tuple of length two, identifying the slice of time variant
            parameters to be changed.
        change_date (str or pandas.Timestamp): date from which on the parameter will
            take the new value.
        new_val (float): the new value which the parameter will take after change_date.

    Returns:
        params (pandas.DataFrame): the full params that were passed to this function
            with the loc exchanged for the new time variant parameters.

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


def obligatory_rapid_tests_for_employees(params):
    """Assume every worker who is offered a test accepts the offer.

    Note that this does not mean that every worker gets regularly tested as only up to
    70% of employers offer tests after the ordinance was passed.

    """
    params = params.copy(deep=True)
    params.loc[("rapid_test_demand", "work", "share_accepting_offer"), "value"] = 1.0
    return params


def rapid_test_with_90pct_compliance_after_summer_scenario_start(params):
    return _rapid_test_with_fixed_compliance_after_date(
        params, change_date=SUMMER_SCENARIO_START, new_val=0.9
    )


def _rapid_test_with_fixed_compliance_after_date(params, change_date, new_val):
    """Implement a rapid test scheme where a certain share of workers get tested."""
    params = params.copy(deep=True)
    params.loc[("rapid_test_demand", "work", "share_accepting_offer"), "value"] = 1.0
    loc = ("rapid_test_demand", "share_workers_receiving_offer")
    params = _change_date_params_after_date(
        params=params, change_date=change_date, new_val=new_val, loc=loc
    )
    return params
