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
