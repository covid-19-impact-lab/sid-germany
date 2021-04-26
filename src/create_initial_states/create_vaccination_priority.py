"""Create the vaccination groups and the vaccination rank.

Vaccination Groups
==================

1 = Highest Priority
--------------------

- overall 8.6 Mio individuals = ~10% of the population
- 1% live in nursing homes (https://bit.ly/3vFsByz) and not covered in our data

=> target of 9%

- over 80 year olds -> 4% of our synthetic population

- individuals working in nursing homes and outpatient nursing
    - 796 489 in nursing homes
    - 421 550 in outpaiton nursing
    - source: https://bit.ly/3vzGLBj

    => 1.5% of the population.

    => We increase this to 4.6% of the population to include other
    groups such as ICU staff. To achieve this share for the overall
    population we set the work_contact_priority to 0.9.
    With this we also reach the 9% target for the highest priority group.


2 = Very High Priority (2nd and 3rd group acc. to STIKO)
--------------------------------------------------------

- approx. 14% of the population acc. to RKI without educators.
    => 15% abstracting 1% nursing home population.

- 70 to 80 year olds
- close contacts of very high risk individuals
- individuals with other dangerous preconditions
- more medical workers

=> we model this as age group 50-70 gets 2/3 of the spots and
   1/3 goes to age group 20-50.

In addition nursery, preschool and primary teachers were moved to this group.
They are about 1% of our synthetic population.

=> target share of 16%


3 = High Priority (4th+5th category acc. to STIKO)
--------------------------------------------------

- 6.9 mio in 4th group + 9 mio in 5th group (~19%)

  = ~18% of population without the already vaccinated teachers
  => ~19% abstracting 1% nursing home population.

- 60 to 70 year olds
- other teachers
- many essential workers (police, fire fighters ...)
- people with preconditions that make them more susceptible to covid.
- close contacts of people with dangerous preconditions

Preconditions in this group include diabetes, hypertension, cancer, asthma, auto-immune
disease

=> We expect a higher share among older individuals.


4 = The Rest
-------------

Approximately 45 mio people ~ 56% of the population.
=> 57% abstracting 1% nursing home population


References
----------

- https://bit.ly/3rekfdL (RKI Stiko Empfehlung)
- https://bit.ly/3tNF01G
- https://www.tagesschau.de/inland/impfungen-lehrer-101.html
- shares of each group: https://bit.ly/3cb5uUQ

"""
import numpy as np
import pandas as pd


def create_vaccination_rank(vaccination_group, share_refuser, seed):
    """Create the order in which individuals get vaccinated, including refusers.

    Args:
        vaccination_group (pandas.Series): index is the same as that of states.
            Low values indicate individuals that have a high priority to be
            vaccinated.
        share_refuser (float): share of individuals (irrespective of their
            vaccination group) that refuse to be vaccinated.

            .. warning::
                This share must also be passed to the vaccination model!
        seed (int)

    Returns:
        vaccination_order (pandas.Series): same index as that of
            vaccination_group. Takes values between 0 and 1. Low values
            correspond to individuals that get vaccinated earlier. Refusers
            receive the highest values but cannot be distinguished from the
            rest.

    """
    np.random.seed(seed)
    sampled_to_refuse = np.random.choice(
        a=[True, False],
        size=len(vaccination_group),
        p=[share_refuser, 1 - share_refuser],
    )
    refuser_value = vaccination_group.max() + 1
    with_refusers = vaccination_group.where(~sampled_to_refuse, refuser_value)
    vaccination_order = with_refusers.rank(method="first", pct=True)
    min_at_zero = vaccination_order - vaccination_order.min()
    scaled = min_at_zero / min_at_zero.max()
    return scaled


def create_vaccination_group(states, seed):
    """Put individuals into vaccination priority groups based on age and work.

    Args:
        states (pandas.DataFrame): states DataFrame. Must contain as columns:
            "age", "work_contact_priority", "educ_worker", "school_group_id_0",
            and "occupation".
        seed (int): seed

    Returns:
        vaccination_group (pandas.Series): index is the same as states.
            Values go from 1 (highest priority) to 4 (lowest priority).
            This is irrespective of individuals refuse to get vaccinated or not.

    """
    np.random.seed(seed)
    vaccination_group = pd.Series(np.nan, index=states.index)

    first_priority = states.eval("80 <= age | work_contact_priority >= 0.90")
    vaccination_group[first_priority] = 1

    second_priority_stiko = _get_second_priority_people_acc_to_stiko(
        states, vaccination_group
    )
    assert 0.145 < second_priority_stiko.mean() < 0.155, second_priority_stiko.mean()
    vaccination_group[second_priority_stiko] = 2
    educators_of_young_children = _get_educators_of_young_children(
        states, vaccination_group
    )
    vaccination_group[educators_of_young_children] = 2

    third_priority = _get_third_priority(states, vaccination_group)
    vaccination_group[third_priority] = 3

    vaccination_group = vaccination_group.fillna(4)
    return vaccination_group


def _get_second_priority_people_acc_to_stiko(states, vaccination_group):
    """People aged 70 to 80 and people with serious preconditions."""
    elderly = states.eval("70 <= age < 80") & vaccination_group.isnull()

    share_random_2nd_priority = 0.155
    n_to_sample = share_random_2nd_priority * len(states) - elderly.sum()
    sampled_for_second_priority = _sample_from_subgroups(
        n_to_sample=n_to_sample,
        states=states,
        age_cutoff=50,
        share_to_sample_above_age_cutoff=0.67,
        vaccination_groups_so_far=vaccination_group,
    )
    second_priority = elderly | sampled_for_second_priority
    return second_priority


def _get_educators_of_young_children(states, vaccination_group):
    """nursery, preschool and primary educators."""
    # identify primary school teachers
    students = states[~states["educ_worker"]]
    mean_age_of_classes = students.groupby("school_group_id_0")["age"].mean()
    # -1 identifies people who do not belong to any school class
    mean_age_of_classes[-1] = np.nan
    primary_class_ids = mean_age_of_classes[mean_age_of_classes <= 10].index
    primary_classes = states["school_group_id_0"].isin(primary_class_ids)
    primary_teachers = states["educ_worker"] & primary_classes
    carers_for_youngsters = states["occupation"].isin(
        ["nursery_teacher", "preschool_teacher"]
    )
    eligible_educ_workers = primary_teachers | carers_for_youngsters
    second_priority_educ_workers = eligible_educ_workers & vaccination_group.isnull()
    return second_priority_educ_workers


def _get_third_priority(states, vaccination_group):
    third_priority_non_random_str = (
        "(60 <= age <= 70) | educ_worker | work_contact_priority > 0.85"
    )
    third_priority_non_random = states.eval(third_priority_non_random_str)
    n_third_priority_random = 0.075 * len(states)
    third_priority_sampled = _sample_from_subgroups(
        n_to_sample=n_third_priority_random,
        states=states,
        vaccination_groups_so_far=vaccination_group,
        age_cutoff=45,
        share_to_sample_above_age_cutoff=0.33,
    )
    third_priority = vaccination_group.isnull() & (
        third_priority_non_random | third_priority_sampled
    )
    return third_priority


def _sample_from_subgroups(
    n_to_sample,
    states,
    age_cutoff,
    share_to_sample_above_age_cutoff,
    vaccination_groups_so_far,
):
    """Sample a fixed number of adults from subgroups.

    Adults are split into those below and above *age_cutoff* and from each group a share
    of n_to_sample is drawn.

    Args:
        n_to_sample (int): number of doses to distribute. Due to rounding errors
            it might not be matched exactly.
        states (pandas.DataFrame): sid states DataFrame with an "age" column.
        age_cutoff (int): The *share_to_sample_above_age_cutoff* of the *n_to_sample*
            is randomly individuals above this cutoff. The rest is distributed among
            adults bleow the cutoff.
        share_to_sample_above_age_cutoff (float): share of *n_to_sample* that is
            distributed among individuals > age_cutoff.
        vaccination_groups_so_far (pandas.Series): Series with the same index as states
            that is NaN for individuals that have not received a vaccine priority yet.

    Returns:
        sampled (pandas.Series): Series with the same index as states and
            vaccination_group that is True for individuals that were drawn to
            receive a vaccine and False for everyone else.

    """
    assert (
        0 <= share_to_sample_above_age_cutoff <= 1
    ), "share_to_sample_above_age_cutoff must lie in [0, 1]."
    assert n_to_sample >= 0, "Only non-negative n_to_sample allowed."

    n_young = int((1 - share_to_sample_above_age_cutoff) * n_to_sample)
    n_old = int(share_to_sample_above_age_cutoff * n_to_sample)

    pool = states[states["age"] >= 19 & vaccination_groups_so_far.isnull()]
    young_pool = pool[pool["age"] < age_cutoff].index
    old_pool = pool[pool["age"] >= age_cutoff].index

    assert n_young < len(young_pool), f"{n_young}, {len(young_pool)}"
    assert n_old < len(old_pool), f"{n_old}, {len(old_pool)}"

    young_sampled_indices = np.random.choice(a=young_pool, size=n_young, replace=False)
    old_sampled_indices = np.random.choice(a=old_pool, size=n_old, replace=False)

    young_sampled = states.index.isin(young_sampled_indices)
    old_sampled = states.index.isin(old_sampled_indices)
    sampled = young_sampled | old_sampled

    return sampled
