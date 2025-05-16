"""Functions for preprocessing the data."""

import os
from pathlib import Path
import re
import zipfile

from kaggle.api.kaggle_api_extended import KaggleApi
from loguru import logger
import pandas as pd

from ARISA_DSML.config import DATASET, DATASET_TEST, PROCESSED_DATA_DIR, RAW_DATA_DIR, INPUT_FILE_NAME


def get_raw_data(dataset:str=DATASET, dataset_test:str=DATASET_TEST)->None:
    api = KaggleApi()
    api.authenticate()

    download_folder = Path(RAW_DATA_DIR)
    #zip_path = download_folder / "titanic.zip"

    logger.info(f"RAW_DATA_DIR is: {RAW_DATA_DIR}")
    api.dataset_download_files(dataset_test, path=str(download_folder), unzip=True)

    # Check full path to the file
    file_path = download_folder / INPUT_FILE_NAME
    logger.info(f"File path is: {file_path}")

    # Check if file exists
    if not os.path.exists(file_path):
        logger.error(f"File {INPUT_FILE_NAME} does not exist in {file_path}")
        return


def extract_title(name:str)-> str|None:
    """Extract title from passenger name."""
    match = re.search(r",\s*([\w\s]+)\.", name)

    return match.group(1) if match else None


def preprocess_df(file:str|Path)->str|Path:
    """Preprocess datasets."""
    _, file_name = os.path.split(file)
    df_data = pd.read_csv(file)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    outfile_path = PROCESSED_DATA_DIR / file_name
    df_data.to_csv(outfile_path, index=False)

    return outfile_path


if __name__=="__main__":
    # get the train and test sets from default location
    logger.info("getting datasets")
    get_raw_data()

    # preprocess both sets
    logger.info("preprocessing train.csv")
    preprocess_df(RAW_DATA_DIR / "train.csv")
    logger.info("preprocessing test.csv")
    preprocess_df(RAW_DATA_DIR / "test.csv")
