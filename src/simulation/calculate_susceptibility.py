def calculate_susceptibility(states, params):
    """Calculate the susceptibility factor for each individual.

    Parameters are loaded from params. The defaults in sid-germany
    are taken from https://go.nature.com/3foGBaf (extended data fig. 4).
    """

    factors = params.loc[("susceptibility", "susceptibility"), "value"]
    susceptibility = states["age_group"].replace(factors)
    susceptibility.name = "susceptibility"
    return susceptibility
