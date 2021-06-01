"""This module holds the code to compute marginal contributions and shapley values."""
import numpy as np


def task_plot_decomposition():
    pass


def _compute_shapley_values(coalitions):
    """Compute Shapley values.

    Parameters
    ----------
    coalitions : pandas.DataFrame
        A DataFrame where each row contains one coalition and an associated payoff.

        - ``"members"`` contains an iterable with all the members participating in the
          specific coalition.
        - ``"payoff"`` contains the value associated with the coalition.

    Returns
    -------
    shapley_values: pandas.Series
        The Shapley values associated with each of the members found in all coalitions.

    """
    coalitions_w_marginal_effects = _compute_marginal_effects(coalitions)
    shapley_values = _compute_shapley_values_from_marginal_effects(
        coalitions_w_marginal_effects
    )

    return shapley_values


def _compute_marginal_effects(coalitions):
    coalitions["coalition"] = range(len(coalitions))
    coalitions["members"] = coalitions["members"].map(frozenset)
    all_members = frozenset().union(*coalitions["members"].tolist())
    set_to_code = dict(zip(coalitions["members"], coalitions["coalition"]))

    for member in all_members:
        coalitions[f"marginal_contribution_{member}"] = np.nan

    for index, row in coalitions.iterrows():
        coalition_members = row.members
        payoff = row.payoff
        for member in coalition_members:
            coalition_wo_member = coalition_members - frozenset([member])
            other_index = set_to_code[coalition_wo_member]
            other_payoff = coalitions.loc[other_index, "payoff"]
            coalitions.loc[index, f"marginal_contribution_{member}"] = (
                payoff - other_payoff
            )

    return coalitions


def _compute_shapley_values_from_marginal_effects(coalitions):
    coalitions["coalition_size"] = coalitions["members"].map(len)
    all_members = frozenset().union(*coalitions["members"].tolist())

    shapley_values = (
        coalitions.groupby("coalition_size")[
            [f"marginal_contribution_{member}" for member in all_members]
        ]
        .mean()
        .mean()
    ).rename(index=lambda x: x.replace("marginal_contribution", "shapley_value"))

    return shapley_values
