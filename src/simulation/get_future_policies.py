from src.policies.full_policy_blocks import get_soft_lockdown
from src.policies.policy_tools import combine_dictionaries


def get_future_policies(
    contact_models, work_multiplier, other_multiplier, schools_open
):
    """Get future policy scenario.

    Args:
        contact_models (dict)
        work_multiplier (float): work multiplier used starting January 12th
        other_multiplier (float): other multiplier used starting January 12th
        schools_open (bool)

    Returns:
        policies (dict):

    """
    to_combine = [
        get_soft_lockdown(
            contact_models=contact_models,
            block_info={
                "start_date": "2021-01-04",
                "end_date": "2021-01-11",
                "prefix": "after-christmas-vacation",
            },
            multipliers={
                "educ": 0.0,
                # google mobility data says work mobility -40%
                "work": 0.95 * 0.4,
                "other": other_multiplier,
            },
        ),
        # schools reopen 1st of February
        # BW: https://tinyurl.com/y2clplul
        # BY: https://tinyurl.com/y49q2uys
        # NRW: https://tinyurl.com/y4rlx37z
        get_soft_lockdown(
            contact_models=contact_models,
            block_info={
                "start_date": "2021-01-12",
                "end_date": "2021-01-23",
                "prefix": "mid_of_january",
            },
            multipliers={
                "educ": 0.0,
                # google mobility data from autumn vacation.
                "work": 0.95 * 0.55,
                "other": other_multiplier,
            },
        ),
        get_soft_lockdown(
            contact_models=contact_models,
            block_info={
                "start_date": "2021-01-24",
                "end_date": "2021-01-31",
                "prefix": "last_january_week",
            },
            multipliers={
                "educ": 0.0,
                "work": 0.95 * min(0.55, work_multiplier),
                "other": other_multiplier,
            },
        ),
    ]
    if schools_open:
        to_combine.append(
            get_soft_lockdown(
                contact_models=contact_models,
                block_info={
                    "start_date": "2021-02-01",
                    "end_date": "2021-05-01",
                    "prefix": "from_feb_onward",
                },
                multipliers={
                    "educ": 0.6,
                    "work": 0.95 * work_multiplier,
                    "other": other_multiplier,
                },
            ),
        )
    else:
        to_combine.append(
            get_soft_lockdown(
                contact_models=contact_models,
                block_info={
                    "start_date": "2021-02-01",
                    "end_date": "2021-05-01",
                    "prefix": "from_feb_onward",
                },
                multipliers={
                    "educ": 0.0,
                    "work": 0.95 * min(0.55, work_multiplier),
                    "other": other_multiplier,
                },
            ),
        )

    return combine_dictionaries(to_combine)
