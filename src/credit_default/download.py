"""Download only the accepted-loans file from the Lending Club dataset."""
import zipfile
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from src.credit_default.config import DATA_DIR

DATASET = "wordsforthewise/lending-club"
FILE_NAME = "accepted_2007_to_2018Q4.csv.gz"   # exact name from `kaggle datasets files`
RAW_DIR = DATA_DIR / "raw"


from src.credit_default.logging_utils import get_logger, run_main

log = get_logger(__name__)


def find_accepted_file() -> Path | None:
    """Return the accepted-loans file if it's already downloaded."""
    matches = sorted(RAW_DIR.rglob("accepted_2007_to_2018*.csv*"))
    gz = [m for m in matches if m.suffix == ".gz"]
    return (gz or matches or [None])[0]


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    existing = find_accepted_file()
    if existing:
        print(f"Already downloaded: {existing}")
        return

    api = KaggleApi()
    api.authenticate()
    print(f"Downloading {FILE_NAME} ...")
    api.dataset_download_file(DATASET, FILE_NAME, path=str(RAW_DIR), quiet=False)

    # Kaggle sometimes wraps single-file downloads in a .zip
    for z in RAW_DIR.glob("*.zip"):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(RAW_DIR)
        z.unlink()                              # delete the zip to save space

    found = find_accepted_file()
    if found:
        print(f"Done. Accepted loans file:\n  {found}")
    else:
        print("Download finished, but the file wasn't found. Run `ls -R data/raw`.")


if __name__ == "__main__":
    run_main(main)
