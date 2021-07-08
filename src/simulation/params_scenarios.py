import warnings

import numpy as np
import pandas as pd

from src.config import AFTER_EASTER
from src.testing.shared import get_piecewise_linear_interpolation
from src.testing.shared import get_piecewise_linear_interpolation_for_one_day


def baseline(params):
    return params


def no_rapid_tests_at_schools(params):
    params = params.copy(deep=True)
    params.loc[("rapid_test_demand", "educ_worker_shares"), "value"] = 0.0
    params.loc[("rapid_test_demand", "student_shares"), "value"] = 0.0
    return params


def no_rapid_tests_at_work(params):
    params = params.copy(deep=True)
    params.loc[("rapid_test_demand", "share_accepting_work_offer"), "value"] = 0.0
    params.loc[("rapid_test_demand", "share_workers_receiving_offer"), "value"] = 0.0
    return params


def no_private_rapid_test_demand(params):
    params = params.copy(deep=True)
    params.loc[("rapid_test_demand", "private_demand"), "value"] = 0.0
    return params


def no_rapid_tests_at_schools_and_work(params):
    params = no_rapid_tests_at_schools(params)
    params = no_rapid_tests_at_work(params)
    return params


def no_rapid_tests_at_schools_and_private(params):
    params = no_rapid_tests_at_schools(params)
    params = no_private_rapid_test_demand(params)
    return params


def no_rapid_tests_at_work_and_private(params):
    params = no_rapid_tests_at_work(params)
    params = no_private_rapid_test_demand(params)
    return params


def no_rapid_tests_at_schools_after_easter(params):
    params = params.copy(deep=True)
    params.loc[("rapid_test_demand", "educ_frequency", "after_easter"), "value"] = 1000
    return params


def rapid_tests_at_school_every_other_day_after_april_5(params):
    params = params.copy(deep=True)
    params.loc[("rapid_test_demand", "educ_frequency", "after_easter"), "value"] = 2
    return params


def rapid_tests_at_school_every_day_after_april_5(params):
    params = params.copy(deep=True)
    params.loc[("rapid_test_demand", "educ_frequency", "after_easter"), "value"] = 1
    return params


def no_seasonality(params):
    """Set the seasonality to 1 everywhere.

    This induces a jump in the seasonality compared to scenarios with seasonality almost
    everywhere.

    """
    params = params.copy(deep=True)
    params.loc[("seasonality_effect", "seasonality_effect", "weak"), "value"] = 0.0
    params.loc[("seasonality_effect", "seasonality_effect", "strong"), "value"] = 0.0
    return params


def start_all_rapid_tests_after_easter(params):
    """Start all rapid tests with full force after Easter instead of fading them in."""
    to_replace = [
        "share_workers_receiving_offer",
        "educ_worker_shares",
        "student_shares",
        "private_demand",
        "share_accepting_work_offer",
    ]
    new = params.copy(deep=True)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="indexing past lexsort depth may impact performance."
        )
        for subcat in to_replace:
            max_val = params.loc[("rapid_test_demand", subcat), "value"].max()
            new = new.drop(("rapid_test_demand", subcat))

            new.loc[("rapid_test_demand", subcat, "2020-01-01"), "value"] = 0.0
            new.loc[("rapid_test_demand", subcat, "2021-04-05"), "value"] = 0.0
            new.loc[("rapid_test_demand", subcat, "2021-04-06"), "value"] = max_val
            new.loc[("rapid_test_demand", subcat, "2025-12-31"), "value"] = max_val

    return new


def keep_work_offer_share_at_23_pct_after_easter(params):
    """Set work offer share to 23 percent (March 17th value) after Easter."""
    work_offer_loc = ("rapid_test_demand", "share_workers_receiving_offer")
    new_params = _change_piecewise_linear_parameter_to_fixed_value_after_date(
        params, loc=work_offer_loc, change_date=AFTER_EASTER, new_val=0.23
    )
    return new_params


def mandatory_work_rapid_tests_after_easter(params):
    """Assume work rapid tests are nearly universal after Easter.

    We assume both 5% refusers on side of firms and 5% on the side of employees. Thus,
    effectively ~90% of workers get tested.

    """
    work_offer_loc = ("rapid_test_demand", "share_workers_receiving_offer")
    new_params = _change_piecewise_linear_parameter_to_fixed_value_after_date(
        params, loc=work_offer_loc, change_date=AFTER_EASTER, new_val=0.95
    )

    work_accept_loc = ("rapid_test_demand", "share_accepting_work_offer")
    new_params = _change_piecewise_linear_parameter_to_fixed_value_after_date(
        params=new_params,
        loc=work_accept_loc,
        change_date=AFTER_EASTER,
        new_val=0.95,
    )
    return new_params


def robustness_check_params_mid_may(params):
    return _robustness_check_params(params, "2021-05-15")


def robustness_check_params_start_may(params):
    return _robustness_check_params(params, "2021-05-01")


def robustness_check_params_end_may(params):
    return _robustness_check_params(params, "2021-06-01")


def _robustness_check_params(params, date):
    """Remove drop in share_known_cases for Easter and simplify rapid test demand."""
    params = params.query("category != 'rapid_test_demand'")

    private_loc = ("rapid_test_demand", "private_demand")
    params.loc[(*private_loc, "2020-01-01"), "value"] = 0
    params.loc[(*private_loc, "2021-02-28"), "value"] = 0
    params.loc[(*private_loc, "2021-02-28"), "value"] = 0
    params.loc[(*private_loc, date), "value"] = 0.63
    params.loc[(*private_loc, "2025-12-31"), "value"] = 0.63

    accept_loc = ("rapid_test_demand", "share_accepting_work_offer")
    params.loc[(*accept_loc, "2020-01-01"), "value"] = 0.6
    params.loc[(*accept_loc, "2025-12-31"), "value"] = 0.6

    offer_loc = ("rapid_test_demand", "share_workers_receiving_offer")
    params.loc[(*offer_loc, "2020-01-01"), "value"] = 0.0
    params.loc[(*offer_loc, "2021-01-01"), "value"] = 0.0
    params.loc[(*offer_loc, date), "value"] = 0.833
    params.loc[(*offer_loc, "2025-12-31"), "value"] = 0.833

    teacher_loc = ("rapid_test_demand", "educ_worker_shares")
    params.loc[(*teacher_loc, "2020-01-01"), "value"] = 0.0
    params.loc[(*teacher_loc, "2021-01-01"), "value"] = 0.0
    params.loc[(*teacher_loc, "2021-03-01"), "value"] = 0.3
    params.loc[(*teacher_loc, "2021-04-07"), "value"] = 0.95
    params.loc[(*teacher_loc, "2025-12-31"), "value"] = 0.95

    student_loc = ("rapid_test_demand", "student_shares")
    params.loc[(*student_loc, "2020-01-01"), "value"] = 0.0
    params.loc[(*student_loc, "2021-03-01"), "value"] = 0.0
    params.loc[(*student_loc, "2021-04-07"), "value"] = 1.0
    params.loc[(*student_loc, "2025-12-31"), "value"] = 1.0

    params = params.drop(
        index=[
            ("share_known_cases", "share_known_cases", "2021-04-01"),
            ("share_known_cases", "share_known_cases", "2021-04-05"),
        ]
    )
    return params


# ======================================================================================


def _rapid_test_with_fixed_compliance_after_date(params, change_date, new_val):
    """Implement a rapid test scheme where a certain share of workers get tested."""
    params = params.copy(deep=True)
    params.loc[("rapid_test_demand", "share_accepting_work_offer"), "value"] = 1
    loc = ("rapid_test_demand", "share_workers_receiving_offer")
    params = _change_piecewise_linear_parameter_to_fixed_value_after_date(
        params=params, change_date=change_date, new_val=new_val, loc=loc
    )
    return params


def _set_to_multiple_of_work_rapid_test_demand_after_date_start(
    params, date, multiplier
):
    """Have a constant *multiplier* x old work rapid test demand after *date*.

    Since only the offer share of workers' rapid tests is time variant we change the
    offer parameters rather than the demand even though the more intuitive
    interpretation would be that workers demand less tests. Where the reduction takes
    place is irrelevant because offer and demand shares are multiplied to get the
    overall work rapid share multiplier.

    Args:
        params (pandas.DataFrame): params DataFrame with ("rapid_test_demand",
            "share_workers_receiving_offer") entries.
        date (str or pandas.Timestamp): date after which the workers' rapid tests are
            reduced.
        multiplier (float): multiplier with which the old work rapid test share will be
            reduced. For example, to reduce the work rapid tests by 25%, set the
            multiplier to 0.75.

    """
    change_date = pd.Timestamp(date)

    work_offer_loc = ("rapid_test_demand", "share_workers_receiving_offer")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="indexing past lexsort depth may impact performance."
        )
        old_work_offer_share = get_piecewise_linear_interpolation_for_one_day(
            change_date, params.loc[work_offer_loc]
        )

    new_work_offer_share = np.clip(multiplier * old_work_offer_share, 0, 1)

    params = _change_piecewise_linear_parameter_to_fixed_value_after_date(
        params=params,
        loc=work_offer_loc,
        change_date=change_date,
        new_val=new_work_offer_share,
    )

    return params


def _change_piecewise_linear_parameter_to_fixed_value_after_date(
    params, loc, change_date, new_val
):
    """Change piecewise linear params to be constant at a new value after a certain date.

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
