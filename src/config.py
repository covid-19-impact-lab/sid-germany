import warnings
from pathlib import Path

import pandas as pd

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

RKI_DATA_RANGE = pd.date_range("2020-03-01", "2020-10-31")

RELATIVE_POPULATION_PARAMETER = 1 / 100_000

POPULATION_GERMANY = 83_000_000

ONE_DAY = pd.Timedelta(days=1)


N_HOUSEHOLDS = 750_000
