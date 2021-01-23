"""Build the specifiation for the base prognosis."""
from src.config import BLD
from src.config import FAST_FLAG


def build_base_prognosis_parametrization():
    n_seeds = 1 if FAST_FLAG else 15
    other_scenarios = [0.5] if FAST_FLAG else [0.3, 0.4, 0.5, 0.6]
    nested_parametrization = {}
    for other_multiplier in other_scenarios:
        nested_parametrization[other_multiplier] = []
        for i in range(n_seeds):
            seed = 700_000 * i
            name = f"{str(other_multiplier).replace('.', '_')}_{i}"
            out_path = BLD / "simulations" / "baseline_prognosis" / name / "time_series"
            nested_parametrization[other_multiplier].append((seed, out_path))
    return nested_parametrization
