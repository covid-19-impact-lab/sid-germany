import pandas as pd

from src.config import BLD
from src.policies.full_policy_blocks import get_lockdown_with_multipliers
from src.policies.full_policy_blocks import (
    get_lockdown_with_multipliers_with_a_b_schooling_above_age_cutoff,
)
from src.policies.full_policy_blocks import (
    get_lockdown_with_multipliers_with_a_b_schooling_below_age_cutoff,
)
from src.policies.policy_tools import combine_dictionaries


def create_scenario_policies(
    contact_models,
    prefix,
    start_date,
    end_date,
    school_mode="a_b_below",
    educ_multiplier=0.5,
    age_cutoff=None,
    other_multiplier=0.45,
    work_multiplier=None,
    work_fill_value=0.68,  # level between 10th of Jan and carnival
):
    """Create hypothetical policies.

    Args:
        contact_models (dict)
        prefix (str): Name of the policy.
        start_date (str)
        end_date (str)
        school_mode (str): one of "open", "a_b_below" and "a_b_above".
        educ_multiplier (float): Function of this depends on the
            school_mode.
        age_cutoff (float, optional): Only
        work_multiplier (float, pandas.Series or None):
            If None use the google mobility based work multiplier.
        work_fill_value (float or None):
            If not None, used to fill missing values in the work multiplier.
        other_multiplier (float)

    Returns:
        policies (dict)

    """
    if school_mode == "open" and age_cutoff is not None:
        raise ValueError("age cutoff must be None if schools are open.")

    work_multiplier = _process_work_multiplier(
        work_multiplier, work_fill_value, start_date, end_date
    )

    if school_mode == "open":
        policies = get_lockdown_with_multipliers(
            contact_models=contact_models,
            block_info={
                "start_date": start_date,
                "end_date": end_date,
                "prefix": prefix,
            },
            multipliers={
                "educ": educ_multiplier,
                "work": work_multiplier,
                "other": other_multiplier,
            },
        )
    elif school_mode == "a_b_below":
        policies = get_lockdown_with_multipliers_with_a_b_schooling_below_age_cutoff(
            contact_models=contact_models,
            block_info={
                "start_date": start_date,
                "end_date": end_date,
                "prefix": prefix,
            },
            multipliers={
                "educ": educ_multiplier,
                "work": work_multiplier,
                "other": other_multiplier,
            },
            age_cutoff=age_cutoff,
        )
    elif school_mode == "a_b_above":
        policies = get_lockdown_with_multipliers_with_a_b_schooling_above_age_cutoff(
            contact_models=contact_models,
            block_info={
                "start_date": start_date,
                "end_date": end_date,
                "prefix": prefix,
            },
            multipliers={
                "educ": educ_multiplier,
                "work": work_multiplier,
                "other": other_multiplier,
            },
            age_cutoff=age_cutoff,
        )
    else:
        raise ValueError(
            f"Unsupported school_mode: {school_mode}. Supported are 'open', "
            "'a_b_below' and 'a_b_above."
        )
    return policies


def get_enacted_policies_of_2021(
    contact_models,
    scenario_start,
    other_multiplier=0.45,
):
    """Get enacted policies of 2021.

    This will be updated continuously.

    Args:
        contact_models (dict)
        scenario_start (str): date until which the policies should run.
            Should be of format yyyy-mm-dd.

    Returns:
        policies (dict)

    """
    assert pd.Timestamp(scenario_start) <= pd.Timestamp("2021-03-01"), (
        "You must update the `get_enacted_policies_of_2021` function to support "
        "such a late scenario start."
    )

    work_multiplier = _get_work_multiplier(scenario_start)
    to_combine = [
        get_lockdown_with_multipliers(
            contact_models=contact_models,
            block_info={
                "start_date": "2021-01-02",
                "end_date": "2021-01-31",
                "prefix": "january",
            },
            multipliers={
                "educ": 0.0,
                "work": work_multiplier,
                "other": other_multiplier,
            },
        ),
        get_lockdown_with_multipliers(
            contact_models=contact_models,
            block_info={
                "start_date": "2021-02-01",
                "end_date": "2021-02-21",
                "prefix": "february1",
            },
            multipliers={
                "educ": 0.0,
                "work": work_multiplier,
                "other": other_multiplier,
            },
        ),
        # very varied schooling across German states
        # very often A / B schooling, only primaries (1-4)
        # and graduating classes are open
        # we simplify to have up to grade 5 open everywhere
        # in A / B schooling and older youths stay home.
        # We ignore Notbetreuung like this.
        # sources:
        # - https://taz.de/Schulen-in-Coronazeiten/!5753515/
        # - https://tinyurl.com/2jfm4tp8
        get_lockdown_with_multipliers_with_a_b_schooling_below_age_cutoff(
            contact_models=contact_models,
            block_info={
                "start_date": "2021-02-22",
                "end_date": scenario_start,
                "prefix": "educ_reopen_spring_2021",
            },
            multipliers={
                # in fall the educ multiplier was estimated to be 0.8.
                # Here, it is 0.5 to capture additional hygiene measures.
                "educ": 0.5,
                "work": work_multiplier,
                "other": other_multiplier,
            },
            age_cutoff=12,
        ),
    ]
    return combine_dictionaries(to_combine)


def get_october_to_christmas_policies(
    contact_models,
    other_multiplier=None,
    work_multiplier=None,
    work_fill_value=None,
    educ_multiplier=0.8,
):
    """Policies from October 1st 2020 until Christmas 2020.

    Args:
        contact_models (dict)
        work_multiplier (pandas.Series, optional): Series from Oct 1st to Dec 23rd.
            values are between 0 and 1. If not given, the work_multipliers
            implied by the google mobility reports are used.
        educ_multiplier (float, optional): This is the education multiplier used
            starting 2nd of November. Default is 0.8, i.e. schools were normally
            open until December, 16th.
        work_fill_value (float, optional): If given, the work_multiplier Series will
            be set to this value after November, 1st.

    """
    dates = pd.date_range("2020-10-01", "2020-12-23")
    if work_multiplier is None:
        work_multiplier_path = BLD / "policies" / "work_multiplier.csv"
        work_multiplier = pd.read_csv(work_multiplier_path, parse_dates=["date"])
        work_multiplier = work_multiplier.set_index("date")
        work_multiplier = work_multiplier.loc[dates]
    else:
        assert work_multiplier.between(
            0, 1
        ).all(), "Work multipliers must lie in [0, 1]."
        assert (work_multiplier.index == dates).all(), ""
    if work_fill_value is not None:
        assert 0 <= work_fill_value <= 1, "work fill value must lie in [0, 1]."
        work_multiplier["2020-11-02":] = work_fill_value

    to_combine = [
        get_lockdown_with_multipliers(
            contact_models=contact_models,
            block_info={
                "start_date": "2020-10-01",
                "end_date": "2020-10-09",
                "prefix": "pre_fall_vacation",
            },
            multipliers={
                "educ": 0.8,
                "work": work_multiplier,
                "other": 0.75,
            },
        ),
        get_lockdown_with_multipliers(
            contact_models=contact_models,
            block_info={
                "start_date": "2020-10-10",
                "end_date": "2020-10-23",
                "prefix": "fall_vacation",
            },
            multipliers={
                "educ": 0.8,
                "work": work_multiplier,
                "other": 1.0,
            },
        ),
        get_lockdown_with_multipliers(
            contact_models=contact_models,
            block_info={
                "start_date": "2020-10-24",
                "end_date": "2020-11-01",
                "prefix": "post_fall_vacation",
            },
            multipliers={
                "educ": 0.8,
                "work": work_multiplier,
                "other": 0.65,
            },
        ),
        get_lockdown_with_multipliers(
            contact_models=contact_models,
            block_info={
                "start_date": "2020-11-02",
                "end_date": "2020-11-22",
                "prefix": "lockdown_light",
            },
            multipliers={
                "educ": educ_multiplier,
                "work": work_multiplier,
                "other": 0.45 if other_multiplier is None else other_multiplier,
            },
        ),
        get_lockdown_with_multipliers(
            contact_models=contact_models,
            block_info={
                "start_date": "2020-11-23",
                "end_date": "2020-12-15",
                "prefix": "lockdown_light_with_fatigue",
            },
            multipliers={
                "educ": educ_multiplier,
                "work": work_multiplier,
                "other": 0.55 if other_multiplier is None else other_multiplier,
            },
        ),
        # Until start of Christmas vacation
        get_lockdown_with_multipliers(
            contact_models=contact_models,
            block_info={
                "start_date": "2020-12-16",
                "end_date": "2020-12-20",
                "prefix": "pre-christmas-lockdown-first-half",
            },
            multipliers={
                "educ": 0.0,
                "work": work_multiplier,
                "other": 0.55 if other_multiplier is None else other_multiplier,
            },
        ),
        # Until Christmas
        get_lockdown_with_multipliers(
            contact_models=contact_models,
            block_info={
                "start_date": "2020-12-21",
                "end_date": "2020-12-23",
                "prefix": "pre-christmas-lockdown-second-half",
            },
            multipliers={
                "educ": 0.0,
                "work": work_multiplier,
                "other": 0.55 if other_multiplier is None else other_multiplier,
            },
        ),
    ]
    return combine_dictionaries(to_combine)


def _process_work_multiplier(work_multiplier, fill_value, start_date, end_date):
    dates = pd.date_range(start_date, end_date)
    assert (
        fill_value is None or work_multiplier is None
    ), "fill_value may only be supplied if work_multiplier is None or vice versa"

    if isinstance(work_multiplier, float):
        return pd.Series(data=work_multiplier, index=dates)
    elif isinstance(work_multiplier, pd.Series):
        assert (
            work_multiplier.index == dates
        ).all(), f"Index is not consecutive from {start_date} to {end_date}"
    elif work_multiplier is None:
        default_path = BLD / "policies" / "work_multiplier.csv"
        default = pd.read_csv(default_path, parse_dates=["date"], index_col="date")
        expanded = default.reindex(index=dates)
        expanded = expanded.fillna(fill_value)
    return expanded


def _get_work_multiplier(scenario_start):
    work_multiplier_path = BLD / "policies" / "work_multiplier.csv"
    work_multiplier = pd.read_csv(work_multiplier_path, parse_dates=["date"])
    work_multiplier = work_multiplier.set_index("date")
    dates = pd.date_range(pd.Timestamp("2021-01-02"), pd.Timestamp(scenario_start))
    if set(dates).issubset(work_multiplier.index):
        work_multiplier = work_multiplier.loc[dates]
    else:
        work_multiplier = work_multiplier.reindex(dates).fillna(method="ffill")
    return work_multiplier
