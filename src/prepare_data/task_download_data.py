from pathlib import Path

import pandas as pd
import pytask
import requests
import yaml
from tqdm import tqdm

from src.config import BLD
from src.config import SRC

PARAMETRIZED_DOWNLOADS = [
    (
        "https://www.arcgis.com/sharing/rest/content/items/"
        "f10774f1c63e40168479a1feb6c7ca74/data",
        BLD / "data" / "raw_time_series" / "rki.csv",
    ),
    (
        "https://www.rki.de/DE/Content/InfAZ/N/Neuartiges_Coronavirus/"
        "Daten/Testzahlen-gesamt.xlsx;jsessionid="
        "3E410CDC013276FC28AD711373F5D82A.internet072?__blob=publicationFile",
        BLD / "data" / "raw_time_series" / "test_statistics.xlsx",
    ),
    (
        "https://www.gstatic.com/covid19/mobility/Region_Mobility_Report_CSVs.zip",
        BLD / "data" / "raw_time_series" / "google_mobility.zip",
    ),
    (
        "https://www.rki.de/DE/Content/InfAZ/N/Neuartiges_Coronavirus/Daten/"
        "Impfquotenmonitoring.xlsx;jsessionid="
        "3E410CDC013276FC28AD711373F5D82A.internet072?__blob=publicationFile",
        BLD / "data" / "raw_time_series" / "vaccinations.xlsx",
    ),
    (
        "https://impfdashboard.de/static/data/germany_vaccinations_timeseries_v2.tsv",
        BLD / "data" / "raw_time_series" / "vaccinations_with_reason.tsv",
    ),
    (
        "https://impfdashboard.de/static/data/germany_deliveries_timeseries_v2.tsv",
        BLD / "data" / "raw_time_series" / "vaccination_deliveries.tsv",
    ),
    (
        "https://www.rki.de/DE/Content/InfAZ/N/Neuartiges_Coronavirus/Daten/"
        "Klinische_Aspekte.xlsx?__blob=publicationFile",
        BLD / "data" / "raw_time_series" / "test_distribution.xlsx",
    ),
]


def _is_download_necessary(path, response):
    """Check whether a download is necessary.

    There three criteria.

    1. If the file is missing, download it.
    2. The following two checks depend on each other.

       1. Some files have an entry in the header which specifies when the file was
          modified last. If the file has been modified, download it.
       2. If the header has no entry for the last modified date, we compare file sizes.
          If the file sizes do not match, the file is downloaded.

    """
    path_yaml = path.with_suffix(".yaml")
    if path_yaml.exists():
        last_modified_offline = pd.to_datetime(
            yaml.safe_load(path_yaml.read_text())["last_modified"]
        )
    else:
        last_modified_offline = None
    last_modified_online = pd.to_datetime(response.headers.get("last-modified", None))
    path.with_suffix(".yaml").write_text(
        yaml.dump({"last_modified": response.headers.get("last-modified", None)})
    )

    if not path.exists():
        is_necessary = True
        reason = f"The file {path.name} does not exist."
    elif (
        last_modified_online is not None
        and last_modified_online > last_modified_offline
    ):
        is_necessary = True
        reason = f"{path.name} has been modified online."
    elif last_modified_online is None:
        file_size_offline = path.stat().st_size
        file_size_online = int(response.headers.get("content-length", 0))

        if file_size_online != file_size_offline:
            is_necessary = True
            reason = f"File sizes differ for {path.name}"
        else:
            is_necessary = False
            reason = f"File {path.name} is already downloaded."
    else:
        is_necessary = False
        reason = f"File {path.name} is already downloaded."

    return is_necessary, reason


def _downloader(file: Path, url: str, response: int):
    """Download url in ``URLS[position]`` to disk with possible resumption.

    Parameters
    ----------
    file : str
        Path of file on disk
    url : str
        URL of file

    """
    # Establish connection
    r = requests.get(url, stream=True)

    # Set configuration
    block_size = 1024
    mode = "wb"

    with open(file, mode) as f:
        with tqdm(
            total=int(response.headers.get("content-length", 0)),
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=file.name,
            initial=0,
            ascii=True,
            miniters=1,
        ) as pbar:
            for chunk in r.iter_content(32 * block_size):
                f.write(chunk)
                pbar.update(len(chunk))


def download_file(url: str, path: str):
    """Execute the correct download operation.

    If offline and online filesize differ, download the file again.

    """
    # Establish connection to header of file
    response = requests.head(url, headers={"Accept-Encoding": None})

    is_necessary, reason = _is_download_necessary(path, response)

    if is_necessary:
        _downloader(path, url, response)


@pytask.mark.depends_on(
    {
        "config.py": SRC / "config.py",
    }
)
@pytask.mark.parametrize("url, produces", PARAMETRIZED_DOWNLOADS)
def task_download_file(url, produces):
    download_file(url, produces)
