"""Create a synthetic population that is representative of Germany."""
import itertools as it

import numpy as np
import pandas as pd
import pytask

from src.config import BLD
from src.shared import create_age_groups
from src.shared import create_age_groups_rki
from src.shared import load_dataset


@pytask.mark.depends_on(
    {
        "hh_data": BLD / "data" / "mossong_2008" / "hh_sample_ger.csv",
        "hh_probabilities": BLD / "data" / "mossong_2008" / "hh_probabilities.csv",
        "working_probabilities": BLD
        / "data"
        / "population_structure"
        / "working_shares.pkl",
        "county_probabilities": BLD / "data" / "counties.parquet",
    }
)
@pytask.mark.produces(BLD / "data" / "inital_states.parquet")
def task_create_background_characteristics(depends_on, produces):
    data = {name: load_dataset(path) for name, path in depends_on.items()}
    df = create_background_characteristics(n_households=400_000, seed=3489, **data)
    df.to_parquet(produces)


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
                - p_id (person id)
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

    df["gender"] = _create_gender(df, seed)
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

    df = df.astype({"age": np.uint8, "hh_id": "category", "p_id": "category"})

    df = df.sort_values(["hh_id", "p_id"])

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
        sampled_hh (pandas.DataFrame): sampled households.
            Each row is an individual with a person id (p_id).

    """
    np.random.seed(seed)
    sampled_ids = np.random.choice(
        hh_probabilities.hh_id,
        p=hh_probabilities.probability,
        size=n_households,
        replace=True,
    )
    new_id_from_sampled_indices = pd.DataFrame({"old_hh_id": sampled_ids})
    new_id_from_sampled_indices = new_id_from_sampled_indices.reset_index()
    new_id_from_sampled_indices = new_id_from_sampled_indices.rename(
        columns={"index": "hh_id"}
    )
    sampled_hh = new_id_from_sampled_indices.merge(
        hh_data[["hh_id", "age"]],
        left_on="old_hh_id",
        right_on="hh_id",
        validate="m:m",
        suffixes=("", "_"),
    )
    sampled_hh = sampled_hh.sort_values("hh_id")
    sampled_hh = sampled_hh.drop(columns=["old_hh_id", "hh_id_"])
    sampled_hh["hh_id"] = sampled_hh["hh_id"].astype("category")
    sampled_hh = sampled_hh.sort_values("hh_id")
    sampled_hh = sampled_hh.reset_index(drop=True).reset_index()
    sampled_hh = sampled_hh.rename(columns={"index": "p_id"})

    return sampled_hh


def _create_gender(df, seed):
    """"""
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
    female[df["underage"]] = _draw_gender(size=df["underage"].sum(), seed=next(seed))
    female = _add_gender_of_single_adult_hhs(
        female,
        single_adult_in_hh=df["nr_adults_in_hh"] == 1,
        age_group=df["age_group"],
        seed=seed,
    )
    female = _add_gender_of_two_adult_hhs(female, df, seed=next(seed))
    # choose remaining elderly's gender from age specific gender distribution
    elderly = df["age_group"] == "70-79"
    female[elderly] = _draw_gender(size=elderly.sum(), p_female=0.55, seed=next(seed))
    very_old = df["age_group"] == "80-100"
    female[very_old] = _draw_gender(size=very_old.sum(), p_female=0.65, seed=next(seed))
    remaining_adults = female.isnull()
    assert (
        remaining_adults.mean() < 0.2
    ), "Too many adults have not been assigned a gender."
    female[remaining_adults] = _draw_gender(
        size=remaining_adults.sum(), seed=next(seed)
    )

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
    sampled_counties = (
        pd.DataFrame({"county": sampled_counties})
        .reset_index()
        .rename(columns={"index": "hh_id"})
    )
    sampled_counties = (
        sampled_counties.merge(
            county_probabilities[["id", "state"]], left_on="county", right_on="id"
        )
        .drop(columns="id")
        .astype({"state": "category"})
    )
    sampled_counties["county"] = sampled_counties["county"].astype("category")
    return sampled_counties


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


def _add_gender_of_single_adult_hhs(female, single_adult_in_hh, age_group, seed):
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
    female[single_non_elderly] = _draw_gender(
        size=single_non_elderly.sum(), seed=next(seed)
    )
    single_elderly = single_adult_in_hh & (age_group == "70-79")
    female[single_elderly] = _draw_gender(
        size=single_elderly.sum(), p_female=0.55, seed=next(seed)
    )
    single_very_old = single_adult_in_hh & (age_group == "80-100")
    female[single_very_old] = _draw_gender(
        size=single_very_old.sum(), p_female=0.65, seed=next(seed)
    )
    assert female[single_adult_in_hh].notnull().all()
    return female


def _add_gender_of_two_adult_hhs(female, hh, seed):
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
    age_of_oldest_in_pair = _draw_gender(
        size=len(oldest_adult), p_female=0.3, seed=seed
    )
    age_of_2nd_oldest = ~age_of_oldest_in_pair
    female[oldest_adult.index] = age_of_oldest_in_pair
    female[second_adult.index] = age_of_2nd_oldest
    assert female[hh["nr_adults_in_hh"] == 2].notnull().all()
    return female


def _draw_gender(size, seed, p_female=0.5):
    np.random.seed(seed)
    return np.random.choice([True, False], size=size, p=[p_female, 1 - p_female])
