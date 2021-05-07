import pandas as pd

from src.config import BLD


def load_params(scenario):
    """Return params fitting to the given scenario.

    Note that many scenarios must be implemented through `load_simulation_inputs`.

    Args:
        scenario (str): One of ["baseline"].

    Returns:
        params (pandas.DataFrame): params adjusted to the scenario.

    """
    params = pd.read_pickle(BLD / "params.pkl")

    if scenario == "baseline":
        pass
    else:
        raise ValueError(
            "The params scenario must be one of ['baseline']. "
            f"You specified {scenario}."
        )
    return params
