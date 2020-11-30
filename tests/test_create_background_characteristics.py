import pandas as pd
import pytest

from src.create_initial_states.task_create_background_characteristics import _sample_hhs
from src.create_initial_states.task_create_background_characteristics import (
    create_background_characteristics,
)


@pytest.fixture
def hh_data():
    hh_data = pd.DataFrame()
    hh_data["hh_id"] = (
        ["a1"] + ["a2"] * 2 + ["a3"] * 3 + ["b3"] * 3 + ["b1", "c1", "d1"]
    )
    hh_data["age"] = [24, 4, 36, 8, 33, 35, 33, 22, 1, 40, 60, 80]
    return hh_data


@pytest.fixture
def hh_probabilities():
    hh_probs = pd.DataFrame()
    hh_probs["hh_id"] = ["a1", "a2", "a3", "b3", "b1", "c1", "d1"]
    # 4 / 7 single-member hhs, 1 / 7 2 member hhs, 2 / 7 3 member hhs
    hh_probs["probability"] = 1 / 7
    return hh_probs


@pytest.fixture
def county_probabilities():
    county_probs = pd.DataFrame()
    county_probs["name"] = [
        "Upper North",
        "Lower North",
        "Upper South",
        "Lower South",
    ]
    county_probs["id"] = [1, 2, 3, 4]
    county_probs["state"] = ["North", "North", "South", "South"]
    county_probs["weight"] = [0.1, 0.1, 0.4, 0.4]
    return county_probs


@pytest.fixture
def working_probabilities():
    intervals = [
        pd.Interval(0, 19, "right"),
        pd.Interval(19, 64, "right"),
        pd.Interval(64, 100, "right"),
    ]
    index = pd.IntervalIndex(
        intervals,
        closed="right",
        name="interval",
        dtype="interval[float64]",
    )

    work_probs = pd.DataFrame(index=index)
    work_probs["female"] = [0.0, 0.5, 0.0]
    work_probs["male"] = [0.0, 0.8, 0.0]
    return work_probs


@pytest.fixture
def background_characteristics(
    hh_data, hh_probabilities, county_probabilities, working_probabilities
):
    df = create_background_characteristics(
        n_households=10000,
        hh_data=hh_data,
        hh_probabilities=hh_probabilities,
        county_probabilities=county_probabilities,
        working_probabilities=working_probabilities,
        seed=484,
    )
    return df


def test_create_background_characteristics(
    hh_data, hh_probabilities, county_probabilities, working_probabilities
):
    df = create_background_characteristics(
        n_households=10000,
        hh_data=hh_data,
        hh_probabilities=hh_probabilities,
        county_probabilities=county_probabilities,
        working_probabilities=working_probabilities,
        seed=930,
    )
    assert df.notnull().all().all(), "No NaN allowed in the background characteristics."
    assert not df.index.duplicated().any(), "Duplicates in index."


def test_sample_hhs(hh_data, hh_probabilities):
    df = _sample_hhs(
        n_households=10000,
        hh_data=hh_data,
        hh_probabilities=hh_probabilities,
        seed=39048,
    )
    hh_sizes = df["hh_id"].value_counts()
    assert hh_sizes.isin([1, 2, 3]).all(), "Households of the wrong size created."
    hh_size_shares = hh_sizes.value_counts(normalize=True).sort_index()
    assert hh_size_shares.between(
        [0.55, 0.12, 0.26], [0.6, 0.16, 0.3]
    ).all(), "Household sizes are not distributed as expected."


def test_create_gender(background_characteristics):
    df = background_characteristics
    assert (
        df["gender"].value_counts(normalize=True).between(0.48, 0.52).all()
    ), "Gender not approximately equally assigned"


def test_county_and_state(background_characteristics):
    df = background_characteristics
    state_shares = df["state"].value_counts(normalize=True).sort_index()
    assert state_shares.between(
        [0.18, 0.78], [0.22, 0.82]
    ).all(), "States have wrong shares."
    # counties are ordered by their id, not their name!
    county_shares = df["county"].value_counts(normalize=True).sort_index()
    assert county_shares.between(
        [0.07, 0.07, 0.37, 0.37], [0.13, 0.13, 0.43, 0.43]
    ).all(), f"County shares are wrong: {county_shares}"


def test_occupation(background_characteristics):
    df = background_characteristics
    children = df[df["age"] < 18]["occupation"]
    assert (children == "school").all(), "Not all children in school."
    retired = df[df["age"] > 65]["occupation"]
    assert (retired == "retired").all(), "Not all elderly retired."
    occ_shares = df.groupby("gender")["occupation"].value_counts(normalize=True)
    assert (
        occ_shares.loc[("male", "working")] > occ_shares.loc[("female", "working")]
    ), "Men are not working more than women."


def test_systemically_relevant(background_characteristics):
    shares = background_characteristics.groupby("occupation")[
        "systemically_relevant"
    ].mean()
    assert shares["retired"] == 0.0, "retired systemically relevant."
    assert shares["school"] == 0.0, "children systemically relevant."
    assert shares["stays home"] == 0.0, "stays home systemically relevant."
    assert (0.31 < shares["working"]) and (
        shares["working"] < 0.35
    ), "not a third of workers systemically_relevant"


def test_work_contact_priority(background_characteristics):
    df = background_characteristics
    assert (df.query("occupation != 'working'")["work_contact_priority"] == -1).all()
    assert (df.query("systemically_relevant")["work_contact_priority"] == 2).all()
    non_essential_prios = df.query("occupation == 'working' & ~ systemically_relevant")[
        "work_contact_priority"
    ]
    assert non_essential_prios.between(-0.01, 1.01).all()
    assert non_essential_prios.std() > 0.2
    assert (non_essential_prios.mean() < 0.52) & (non_essential_prios.mean() > 0.48)
