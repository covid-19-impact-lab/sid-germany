def calculate_susceptibility(states, params):
    """Calculate the susceptibility factor for each individual.

    Parameters are loaded from params. The defaults in sid-germany
    are taken from https://go.nature.com/3foGBaf (extended data fig. 4).

    Args:
        states (pandas.DataFrame): sid DataFrame. It must contain an age_group
            column whose values fit the "name" index in the params of the
            susceptibility values.
        params (pandas.DataFrame): It must have a 3 level MultiIndex with
            entries whose first two values are "susceptibility" and whose
            last index value are the values age_group can take. The "value"
            column values must be floats.

    Returns:
        susceptibility (pandas.Series): Series with the same index as states.
            The values are the susceptibilities for each age group as given
            in the params.

    """
    factors = params.loc[("susceptibility", "susceptibility"), "value"]
    susceptibility = states["age_group"].replace(factors)
    susceptibility.name = "susceptibility"
    return susceptibility
