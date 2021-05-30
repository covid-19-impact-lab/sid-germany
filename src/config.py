"""The global config file.

CAREFUL: THIS FILE IS NOT SPECIFIED AS THE DEPENDENCY OF ANY TASK BECAUSE OTHERWISE
CHANGING THE FAST FLAG WOULD ALWAYS TRIGGER A RERUN OF EVERY TASK.
IF YOU CHANGE ANYTHING ELSE HERE MAKE SURE TO DO A DISTCLEAN AFTERWARDS OR CAREFULLY
REMOVE EVERYTHING THAT IMPORTS THE VARIABLE YOU CHANGED.

"""
import warnings
from pathlib import Path

import pandas as pd
import sid

SUMMER_SCENARIO_START = "2021-05-17"

FAST_FLAG = "verify"
"""One of 'debug', 'verify', 'full'.

If 'debug' only the debug initial states are used and only one run of every scenario is
done. Do **not** interpret the results.

If 'verify' only 10 seeds and the base scenario are done in the fall scenarios. In the
main_predictions we use 5 seeds for each scenario. This means there 30 simulation runs
overall.

If 'full' 20 seeds are used for each scenario.

"""

SID_DEPENDENCIES = {}
for path in Path(sid.__path__[0]).iterdir():
    if path.suffix == ".py":
        SID_DEPENDENCIES[f"sid_{path.name}"] = path.resolve()


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


SRC = Path(__file__).parent.resolve()
BLD = SRC.parent / "bld"


POPULATION_GERMANY = 83_000_000

N_HOUSEHOLDS = 1_150_000

VERY_EARLY = pd.Timestamp("2020-01-01")
VERY_LATE = pd.Timestamp("2022-12-31")
