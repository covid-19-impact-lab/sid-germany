"""Create a synthetic population that is representative of Germany."""
import itertools as it

import numpy as np
import pandas as pd
import pytask

from src.config import BLD
from src.create_initial_states.add_weekly_ids import add_weekly_ids
from src.shared import create_age_groups
from src.shared import create_age_groups_rki
from src.shared import create_groups_from_dist
from src.shared import load_dataset


DEPENDENCIES = {
    "hh_data": BLD / "data" / "mossong_2008" / "hh_sample_ger.csv",
    "hh_probabilities": BLD / "data" / "mossong_2008" / "hh_probabilities.csv",
    "working_probabilities": BLD
    / "data"
    / "population_structure"
    / "working_shares.pkl",
    "county_probabilities": BLD / "data" / "counties.parquet",
    "work_daily_dist": BLD
    / "contact_models"
    / "empirical_distributions"
    / "work_recurrent_daily.pkl",
    "work_weekly_dist": BLD
    / "contact_models"
    / "empirical_distributions"
    / "work_recurrent_weekly.pkl",
    "other_daily_dist": BLD
    / "contact_models"
    / "empirical_distributions"
    / "other_recurrent_daily.pkl",
    "other_weekly_dist": BLD
    / "contact_models"
    / "empirical_distributions"
    / "other_recurrent_weekly.pkl",
}


@pytask.mark.depends_on(DEPENDENCIES)
@pytask.mark.produces(BLD / "data" / "inital_states.pickle")
def task_create_background_characteristics(depends_on, produces):
    data = {name: load_dataset(path) for name, path in depends_on.items()}

    repeating_contact_distributions = [
        "work_daily_dist",
        "work_weekly_dist",
        "other_daily_dist",
        "other_weekly_dist",
    ]
    group_id_specs = {name: data.pop(name) for name in repeating_contact_distributions}

    df = create_background_characteristics(n_households=400_000, seed=3489, **data)
    _check_background_characteristics(df)
    model_group_ids = create_contact_model_group_ids(
        df,
        **group_id_specs,
        seed=1109,
    )
    df = df.merge(model_group_ids, left_index=True, right_index=True, validate="1:1")
    _check_group_ids(df, **group_id_specs)
    df.to_pickle(produces)


def create_contact_model_group_ids(
    df, work_daily_dist, work_weekly_dist, other_daily_dist, other_weekly_dist, seed
):
    seed = it.count(seed)
    group_ids = pd.DataFrame(index=df.index)

    # FIRST TEACHER AND STUDENTS
    # -> new occupation column!

    work_daily_group_sizes = work_daily_dist.copy(deep=True)
    work_daily_group_sizes.index += 1
    group_ids["work_daily_group_id"] = create_groups_from_dist(
        initial_states=df,
        group_distribution=work_daily_group_sizes,
        query="occupation == 'working'",
        assort_bys=["county"],
        seed=next(seed),
    )

    other_daily_group_sizes = other_daily_dist.copy(deep=True)
    other_daily_group_sizes.index += 1
    group_ids["other_daily_group_id"] = create_groups_from_dist(
        initial_states=df,
        group_distribution=other_daily_group_sizes,
        query=None,
        assort_bys=["county", "age_group"],
        seed=next(seed),
    )

    weekly_work_ids = add_weekly_ids(
        states=df,
        weekly_dist=work_weekly_dist,
        query="occupation =='working'",
        seed=9958,
        col_prefix="work_weekly_group_id",
        county_assortativeness=0.8,
    )

    group_ids = pd.merge(
        left=group_ids,
        right=weekly_work_ids,
        left_index=True,
        right_index=True,
        validate="1:1",
    )

    weekly_other_ids = add_weekly_ids(
        states=df,
        weekly_dist=other_weekly_dist,
        seed=4748,
        query=None,
        col_prefix="other_weekly_group_id",
        county_assortativeness=0.8,
    )

    group_ids = pd.merge(
        left=group_ids,
        right=weekly_other_ids,
        left_index=True,
        right_index=True,
        validate="1:1",
    )

    return group_ids


def create_background_characteristics(
    n_households,
    hh_data,
    hh_probabilities,
    county_probabilities,
    working_probabilities,
    seed,
):
    """Create the background characteristics for a synthetic population.

    Args:
        n_households (int): number of households to draw.
        hh_data (pandas.DataFrame): data from which the households will be sampled.
            Each row is an individual. Required columns are:
                - "hh_id": household identifier
                - "age": age of the individual
        hh_probabilities (pandas.DataFrame): rows are households.
            Columns are the "hh_id" and "probability", the probability with
            which the household will be sampled.
        county_probabilities (pandas.DataFrame): each row is a county.
            Columns must include:
                - "id": a unique county id
                - "name": name of the county
                - "weight": probability weight
                - "state": federal state of the county
        working_probabilities (pandas.DataFrame): the index are pandas.Intervals
            the columns contain "male" and "female" and contain the share of
            men / women that are working, respectively.
        seed (int): seed for numpy.random.

    Returns:
        df (pandas.DataFrame): DataFrame where each row represents an individual.
            Columns are:
                - hh_id
                - age
                - county
                - state
                - gender
                - occupation
                - systemically_relevant
                - age_group
                - age_group_rki

    """
    seed = it.count(seed)

    assert set(hh_probabilities["hh_id"]) == set(
        hh_data["hh_id"]
    ), "You need to specify a sampling probability for every household."

    df = _sample_hhs(
        n_households=n_households,
        hh_data=hh_data,
        hh_probabilities=hh_probabilities,
        seed=next(seed),
    )
    # age group necessary for drawing gender
    df["age_group"] = create_age_groups(df["age"])

    df["gender"] = _create_gender(df, next(seed))
    county_and_state = _draw_counties(
        hh_ids=df["hh_id"].unique(),
        county_probabilities=county_probabilities,
        seed=next(seed),
    )
    df = df.merge(county_and_state, on="hh_id", validate="m:1")
    df["occupation"] = _draw_occupation(
        df=df, working_probabilities=working_probabilities, seed=next(seed)
    )
    df["systemically_relevant"] = _draw_systemically_relevant(
        df["occupation"], seed=next(seed)
    )
    df["work_contact_priority"] = _draw_work_contact_priority(
        df["occupation"], df["systemically_relevant"], next(seed)
    )
    df["age_group_rki"] = create_age_groups_rki(df)
    df = df.astype({"age": np.uint8, "hh_id": "category"})
    df = df.sort_values("hh_id").reset_index()

    return df


def _sample_hhs(n_households, hh_data, hh_probabilities, seed):
    """
    Args:
        n_households (int): number of households to draw.
        hh_data (pandas.DataFrame): data from which the households will be sampled.
            Each row is an individual. Required columns are:
                - "hh_id": household identifier
                - "age": age of the individual
        hh_probabilities (pandas.DataFrame): rows are households.
            Columns are the "hh_id" and "probability", the probability with
            which the household will be sampled.
        seed (int): seed for numpy.random.

    Returns:
        sampled_hh (pandas.DataFrame): sampled households. Each row is an individual.

    """
    np.random.seed(seed)
    sampled_ids = np.random.choice(
        hh_probabilities.hh_id,
        p=hh_probabilities.probability,
        size=n_households,
        replace=True,
    )
    new_id_df = pd.DataFrame({"old_hh_id": sampled_ids})
    new_id_df = new_id_df.reset_index()
    new_id_df = new_id_df.rename(columns={"index": "hh_id"})
    df = new_id_df.merge(
        hh_data[["hh_id", "age"]],
        left_on="old_hh_id",
        right_on="hh_id",
        validate="m:m",
        suffixes=("", "_"),
    )
    df = df.drop(columns=["old_hh_id", "hh_id_"])
    df = df.sort_values("hh_id")
    df["hh_id"] = df["hh_id"].astype("category")
    df = df.reset_index(drop=True)

    return df


def _create_gender(df, seed):
    """Create a gender variable that replicates assortative matching.

    For children and adults not in two-adult households we draw the gender randomly.
    For older adults living alone we adjust the probability to be female upwards.

    For two adult households we draw the older adult to be more likely male and assign
    the younger adult the opposite gender.
    This is important for the spread of an infectious disease as women are less likely
    to work than men. Gender does not enter into our model otherwise.

    """
    np.random.seed(seed)
    # add helper variables
    df = df.copy(deep=True)
    hh_sizes = df.groupby("hh_id").size()
    hh_sizes.name = "hh_size"
    to_merge = hh_sizes.to_frame()
    df = df.merge(to_merge, left_on="hh_id", right_index=True, validate="m:1")
    df["underage"] = df["age"] < 18
    df["nr_children_in_hh"] = df.groupby("hh_id")["underage"].transform(np.sum)
    df["nr_adults_in_hh"] = df["hh_size"] - df["nr_children_in_hh"]

    female = pd.Series(np.nan, index=df.index)
    female[df["underage"]] = _draw_gender(size=df["underage"].sum())
    female = _add_gender_of_single_adult_hhs(
        female,
        single_adult_in_hh=df["nr_adults_in_hh"] == 1,
        age_group=df["age_group"],
    )
    female = _add_gender_of_two_adult_hhs(female, df)
    # choose remaining elderly's gender from age specific gender distribution
    elderly = df["age_group"] == "70-79"
    female[elderly] = _draw_gender(size=elderly.sum(), p_female=0.55)
    very_old = df["age_group"] == "80-100"
    female[very_old] = _draw_gender(size=very_old.sum(), p_female=0.65)
    remaining_adults = female.isnull()
    assert (
        remaining_adults.mean() < 0.2
    ), "Too many adults have not been assigned a gender."
    female[remaining_adults] = _draw_gender(size=remaining_adults.sum())

    gender_sr = female.astype(bool).replace({True: "female", False: "male"})
    gender_sr = pd.Categorical(gender_sr, categories=["male", "female"], ordered=False)
    return gender_sr


def _draw_counties(hh_ids, county_probabilities, seed):
    """Draw for each household to which county and federal state it belongs to."""
    np.random.seed(seed)
    sampled_counties = np.random.choice(
        county_probabilities.id,
        p=county_probabilities.weight,
        size=len(hh_ids),
        replace=True,
    )
    df = pd.DataFrame({"county": sampled_counties})
    df = df.reset_index()
    df = df.rename(columns={"index": "hh_id"})
    df = df.merge(
        county_probabilities[["id", "state"]], left_on="county", right_on="id"
    )
    df = df.drop(columns="id")
    df = df.astype({"state": "category", "county": "category"})
    return df


def _draw_occupation(df, working_probabilities, seed):
    """Draw whether people are in school, working, staying home or retirerd."""
    np.random.seed(seed)
    occ = pd.Series(np.nan, index=df.index)

    work_bins = [-0.5] + [x.left for x in working_probabilities.index] + [np.inf]
    df = df.copy()
    df["work_age_group"] = pd.cut(df["age"], work_bins)

    gb = df.groupby(["work_age_group", "gender"])
    for (age_bin, gender), index in gb.groups.items():
        if age_bin.left == -0.5:
            occ[index] = "school"
        elif age_bin.right == np.inf:
            occ[index] = "retired"
        else:
            p = working_probabilities.loc[age_bin, gender]
            if age_bin.right < 20:
                occ[index] = np.random.choice(
                    ["working", "school"], size=len(index), p=[p, 1 - p]
                )
            else:
                occ[index] = np.random.choice(
                    ["working", "stays home"], size=len(index), p=[p, 1 - p]
                )
    occ = occ.astype("category")

    return occ


def _draw_systemically_relevant(occupation, seed):
    """Assign each worker whether (s)he is systemically relevant.

    According to the German government around 1 in 3 German workers work
    in systemically relevant jobs
    source: https://dip21.bundestag.de/dip21/btd/19/218/1921889.pdf

    """
    np.random.seed(seed)
    values = np.random.choice(a=[True, False], size=len(occupation), p=[0.33, 0.67])
    systemically_relevant = pd.Series(
        values, index=occupation.index, name="systemically_relevant"
    )
    systemically_relevant = systemically_relevant.where(occupation == "working", False)
    return systemically_relevant


def _draw_work_contact_priority(occupation, systemically_relevant, seed):
    np.random.seed(seed)
    values = np.random.uniform(low=0, high=1, size=len(occupation))
    work_contact_priority = pd.Series(
        values, index=occupation.index, name="work_contact_priority"
    )
    work_contact_priority = work_contact_priority.where(
        occupation == "working", other=-1
    ).where(~systemically_relevant, 2)
    return work_contact_priority


def _add_gender_of_single_adult_hhs(female, single_adult_in_hh, age_group):
    """Add the gender of adults in single adult households.

    This abstracts from women being much more likely to be single parents.
    source: https://bit.ly/2ZRWrlU

    Args:
        female (pandas.Series)
        single_adult_in_hh (pandas.Series): boolean Series that's True for households
            with just one adult.
        age_group (pandas.Series):
            Series with the age group of each individual.

    Returns:
        female (pandas.Series): Input Series with ages drawn for single adult households

    """
    female = female.copy()
    single_non_elderly = single_adult_in_hh & (age_group < "70-79")
    female[single_non_elderly] = _draw_gender(size=single_non_elderly.sum())
    single_elderly = single_adult_in_hh & (age_group == "70-79")
    female[single_elderly] = _draw_gender(size=single_elderly.sum(), p_female=0.55)
    single_very_old = single_adult_in_hh & (age_group == "80-100")
    female[single_very_old] = _draw_gender(size=single_very_old.sum(), p_female=0.65)
    assert female[single_adult_in_hh].notnull().all()
    return female


def _add_gender_of_two_adult_hhs(female, hh):
    """Add gender to the female column in *hh* for households with two adults.


    We assume for households with two adults that they form a mixed sex couple as only
    1.1% of relationships are same sex. 70% of men are older in couples.
    source https://bit.ly/2XaX0FL

    """
    female = female.copy()
    # make the oldest person appear first in every household
    hh = hh.sort_values(["hh_id", "age"], ascending=False).copy(deep=True)
    two_adults = hh[hh["nr_adults_in_hh"] == 2]
    oldest_adult = two_adults.groupby("hh_id", as_index=False).nth(0)
    second_adult = two_adults.groupby("hh_id", as_index=False).nth(1)
    age_of_oldest_in_pair = _draw_gender(size=len(oldest_adult), p_female=0.3)
    age_of_2nd_oldest = ~age_of_oldest_in_pair
    female[oldest_adult.index] = age_of_oldest_in_pair
    female[second_adult.index] = age_of_2nd_oldest
    assert female[hh["nr_adults_in_hh"] == 2].notnull().all()
    return female


def _draw_gender(size, seed=None, p_female=0.5):
    if seed is not None:
        np.random.seed(seed)
    return np.random.choice([True, False], size=size, p=[p_female, 1 - p_female])


def _check_background_characteristics(df):
    """Check that the background characteristics come out right.

    Columns are:
    - systemically_relevant
    - age_group
    - age_group_rki
    """
    df = df.copy(deep=True)
    workers = df.query("occupation == 'working'")
    df["female"] = df["gender"] == "female"
    df["male"] = df["gender"] == "male"

    assert df["hh_id"].value_counts().max() <= 8
    assert df["age"].between(0, 110).all()
    assert (40 <= df["age"].median()) & (48 >= df["age"].median())
    assert df["state"].nunique() == 16

    assert df["gender"].value_counts(normalize=True).between(0.48, 0.52).all()
    young_gender_shares = df.query("age < 18")["gender"].value_counts(normalize=True)
    assert young_gender_shares.between(0.48, 0.52).all()
    df.groupby("hh_id")["gender"]
    old_women_share = (df.query("70 < age < 80")["gender"] == "female").mean()
    assert old_women_share > 0.53
    old_women_share = (df.query("age > 80")["gender"] == "female").mean()
    assert old_women_share > 0.62
    assert (df.query("occupation == 'working'")["gender"] == "female").mean() < 0.5
    # given how we assign gender,
    # men and women should appear in much more than half of households
    assert df.groupby("hh_id")["female"].any().mean() > 0.75
    assert df.groupby("hh_id")["male"].any().mean() > 0.75

    occupation_categories = ["working", "school", "stays home", "retired"]
    assert df["occupation"].isin(occupation_categories).all()
    assert 0.5 < (df["occupation"] == "working").mean()
    assert 0.6 > (df["occupation"] == "working").mean()
    assert 0.08 < (df["occupation"] == "retired").mean()
    assert 0.12 > (df["occupation"] == "retired").mean()
    assert 0.12 < (df["occupation"] == "stays home").mean()
    assert 0.18 > (df["occupation"] == "stays home").mean()
    assert (df[df["age"].between(6, 14)]["occupation"] == "school").all()
    assert (df[df["age"] > 70]["occupation"] == "retired").all()
    assert not df[df["occupation"] != "working"]["systemically_relevant"].any()
    assert workers["systemically_relevant"].mean() > 0.27
    assert workers["systemically_relevant"].mean() < 0.35


def _check_group_ids(
    df,
    work_daily_dist,
    work_weekly_dist,  # noqa: U100
    other_daily_dist,
    other_weekly_dist,  # noqa: U100
):
    df = df.copy(deep=True)

    # weekly group ids
    w_weekly_cols = [x for x in df if x.startswith("work_weekly_group")]
    o_weekly_cols = [x for x in df if x.startswith("other_weekly_group")]
    assert len(w_weekly_cols) == 14
    assert len(o_weekly_cols) == 8
    n_weekly_w_groups = df[w_weekly_cols].replace(-1, np.nan).notnull().sum(axis=1)
    df["n_weekly_w_groups"] = n_weekly_w_groups
    n_weekly_o_groups = df[o_weekly_cols].replace(-1, np.nan).notnull().sum(axis=1)
    df["n_weekly_o_groups"] = n_weekly_o_groups

    workers = df.query("occupation == 'working'")
    non_workers = df.query("occupation != 'working'")

    assert (non_workers[w_weekly_cols] == -1).all().all()
    w_weekly_size_shares = workers["n_weekly_w_groups"].value_counts(normalize=True)
    o_weekly_size_shares = df["n_weekly_o_groups"].value_counts(normalize=True)
    assert np.abs(w_weekly_size_shares - work_weekly_dist).max() < 0.04
    assert np.abs(o_weekly_size_shares - other_weekly_dist).max() < 0.08

    # daily work group ids
    w_daily_group_vc = workers["work_daily_group_id"].value_counts()
    # drop -1 category
    w_daily_group_vc = w_daily_group_vc[w_daily_group_vc > 0]
    assert w_daily_group_vc.max() <= 16
    assert (non_workers["work_daily_group_id"] == -1).all()
    assert (workers["work_daily_group_id"] != -1).all()
    # compare true and target distribution (incomplete!)
    w_daily_group_size_shares = w_daily_group_vc.value_counts(normalize=True)
    assert w_daily_group_size_shares[::-1].is_monotonic
    goal_w_daily_group_size_shares = work_daily_dist.copy(deep=True)
    goal_w_daily_group_size_shares.index += 1
    assert w_daily_group_size_shares.argmax() == goal_w_daily_group_size_shares.argmax()

    # daily other group ids
    o_daily_group_vc = df["other_daily_group_id"].value_counts()
    # drop -1 category
    o_daily_group_vc = o_daily_group_vc[o_daily_group_vc > 0]
    assert o_daily_group_vc.max() <= 6
    o_daily_group_size_shares = o_daily_group_vc.value_counts(normalize=True)
    goal_o_daily_group_size_shares = other_daily_dist.copy(deep=True)
    goal_o_daily_group_size_shares.index += 1
    diff_btw_o_shares = o_daily_group_size_shares - goal_o_daily_group_size_shares
    assert np.abs(diff_btw_o_shares).max() < 0.1
