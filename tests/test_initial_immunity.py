import numpy as np
import pandas as pd
import pytest
from sid.shared import boolean_choices

from src.create_initial_states.create_initial_immunity import (
    _calculate_endog_immunity_prob,
)
from src.create_initial_states.create_initial_immunity import (
    _calculate_exog_immunity_prob,
)
from src.create_initial_states.create_initial_immunity import (
    _calculate_total_immunity_prob,
)
from src.create_initial_states.create_initial_immunity import create_initial_immunity


def date(month, day):
    return pd.Timestamp(f"2020-{month:0d}-{day:0d}")


@pytest.fixture
def empirical_data():
    df = pd.DataFrame()
    df["date"] = [date(3, 1), date(3, 1), date(3, 3), date(3, 5), date(3, 5)]
    df["county"] = ["A", "A", "A", "B", "B"]
    df["age_group_rki"] = ["young", "old", "young", "young", "old"]
    df["newly_infected"] = [1, 1, 2, 2, 2]
    return df.set_index(["date", "county", "age_group_rki"])["newly_infected"]


@pytest.fixture
def synthetic_data():
    df = pd.DataFrame()
    df["county"] = ["A"] * 5 + ["B"] * 5
    df["age_group_rki"] = ["young"] * 3 + ["old"] * 2 + ["young"] * 2 + ["old"] * 3
    return df


def test_calculate_total_immunity_prob(synthetic_data):
    total_immunity = pd.DataFrame()
    total_immunity["county"] = list("AABB")
    total_immunity["age_group_rki"] = ["old", "young"] * 2
    total_immunity["cases"] = [2, 6, 4, 4]
    total_immunity = total_immunity.set_index(["age_group_rki", "county"])["cases"]

    undetected_multiplier = 2
    population_size = 100

    expected = pd.DataFrame()
    expected["county"] = list("AABB")
    expected["age_group_rki"] = ["old", "young"] * 2
    expected["prob"] = [4 / 20, 12 / 30, 8 / 30, 8 / 20]
    expected = expected.set_index(["age_group_rki", "county"])["prob"]
    res = _calculate_total_immunity_prob(
        total_immunity=total_immunity,
        undetected_multiplier=undetected_multiplier,
        synthetic_data=synthetic_data,
        population_size=population_size,
    )
    pd.testing.assert_series_equal(
        res.sort_index(), expected.sort_index(), check_names=False
    )


def test_calculate_endog_immunity_prob(synthetic_data):
    initial_infections = pd.DataFrame()
    initial_infections["2020-03-02"] = [0, 0, 1, 0, 0, 0, 0, 0, 1, 0]
    initial_infections["2020-03-03"] = [0, 0, 0, 1, 0, 0, 0, 1, 0, 0]

    expected = pd.DataFrame()
    expected["county"] = list("AABB")
    expected["age_group_rki"] = ["old", "young", "old", "young"]
    expected["endog_immune"] = [0.5, 1 / 3, 2 / 3, 0]
    expected = expected.set_index(["age_group_rki", "county"])["endog_immune"]
    res = _calculate_endog_immunity_prob(
        synthetic_data=synthetic_data, initial_infections=initial_infections
    )
    pd.testing.assert_series_equal(expected.sort_index(), res.sort_index())


def test_calculate_exog_immunity_prob():
    total_immunity_prob = pd.DataFrame()
    total_immunity_prob["county"] = list("AABB")
    total_immunity_prob["age_group_rki"] = ["old", "young"] * 2
    total_immunity_prob["prob"] = [0.1, 0.2, 0.1, 0.0]
    total_immunity_prob = total_immunity_prob.set_index(["age_group_rki", "county"])[
        "prob"
    ]
    endog_immunity_prob = pd.DataFrame()
    endog_immunity_prob["county"] = list("AABB")
    endog_immunity_prob["age_group_rki"] = ["old", "young"] * 2
    endog_immunity_prob["prob"] = [0.05, 0.0, 0.1, 0.0]
    endog_immunity_prob = endog_immunity_prob.set_index(["age_group_rki", "county"])[
        "prob"
    ]
    expected = pd.DataFrame()
    expected["county"] = list("AABB")
    expected["age_group_rki"] = ["old", "young"] * 2
    expected["exog_immunity_prob"] = [0.052631578947368425, 0.2, 0.0, 0.0]
    expected = expected.set_index(["age_group_rki", "county"])
    expected = expected.sort_index()["exog_immunity_prob"]

    res = _calculate_exog_immunity_prob(total_immunity_prob, endog_immunity_prob)
    pd.testing.assert_series_equal(expected, res.sort_index())


def test_create_initial_immunity_lln(synthetic_data):
    full_synthetic_data = pd.concat([synthetic_data] * 50000).reset_index(drop=True)
    # synthetic data implies 20% old A, 30% young A, 30% old B, 20% young B

    np.random.seed(338)
    draw_prob = {"young": 0.2, "old": 0.1}
    individual_level_cases = pd.concat([synthetic_data] * 5000).reset_index(drop=True)
    individual_level_cases[pd.Timestamp("2020-03-03")] = boolean_choices(
        individual_level_cases["age_group_rki"].map(draw_prob.get)
    )
    empirical_group_sizes = individual_level_cases.groupby(
        ["county", "age_group_rki"]
    ).size()
    population_size = len(individual_level_cases)
    cases_by_county_and_age_group = individual_level_cases.groupby(
        ["county", "age_group_rki"]
    ).sum()
    empirical_data = cases_by_county_and_age_group.stack()
    empirical_data.index.names = ["county", "age_group_rki", "date"]
    empirical_data = empirical_data.reset_index()
    empirical_data = empirical_data.set_index(["date", "county", "age_group_rki"])
    empirical_data = empirical_data[0]
    empirical_data.name = "newly_infected"

    expected_shares = empirical_data["2020-03-03"] / empirical_group_sizes

    undetected_multiplier = 1.0

    initial_infections = pd.DataFrame(index=full_synthetic_data.index)
    to_draw = len(full_synthetic_data)
    initial_infections["2020-03-02"] = np.random.choice(
        a=[True, False], size=to_draw, p=[0.01, 0.99]
    )
    initial_infections["2020-03-03"] = np.random.choice(
        a=[True, False], size=to_draw, p=[0.01, 0.99]
    )

    full_synthetic_data["resulting_immune"] = create_initial_immunity(
        empirical_data=empirical_data,
        synthetic_data=full_synthetic_data,
        initial_infections=initial_infections,
        population_size=population_size,
        undetected_multiplier=undetected_multiplier,
        date="2020-03-04",
        seed=3399,
    )
    grouped = full_synthetic_data.groupby(["county", "age_group_rki"])
    resulting_immune_shares = grouped["resulting_immune"].mean()

    pd.testing.assert_series_equal(
        resulting_immune_shares.sort_index().round(2),
        expected_shares.sort_index().round(2),
        check_names=False,
    )
