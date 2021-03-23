import warnings
from pathlib import Path

FAST_FLAG = True

try:
    import pyarrow  # noqa: F401
except ImportError:
    pass
else:
    warnings.warn(
        "pyarrow is installed and used as the default parquet engine for pandas and "
        "dask if not otherwise specified with the 'engine' argument. sid relies on "
        "fastparquet and mixing both backends can have hidden consequences. Uninstall "
        "pyarrow or make sure to use fastparquet as the engine in every instance."
    )


SRC = Path(__file__).parent
BLD = SRC.parent / "bld"

POPULATION_GERMANY = 83_000_000

N_HOUSEHOLDS = 750_000

SHARE_REFUSE_VACCINATION = 0.15
"""Share of individuals refusing to be vaccinated.

    80% of Germans are somewhat or definitely willing to be vaccinated.
    12% are undecided. 8% are opposed to being vaccinated.
    We assume that 15% will refuse to be vaccinated.
    source: https://bit.ly/3c9mTgX (publication date: 2021-03-02)
"""
