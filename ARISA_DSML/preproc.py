"""Functions for preprocessing the data."""

import os
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi
from loguru import logger
import pandas as pd

from ARISA_DSML.config import DATASET, DATASET_TEST, PROCESSED_DATA_DIR, RAW_DATA_DIR, INPUT_FILE_NAME, quality_mapping


def get_raw_data(dataset:str=DATASET, dataset_test:str=DATASET_TEST)->None:
    api = KaggleApi()
    api.authenticate()

    download_folder = Path(RAW_DATA_DIR)

    logger.info(f"RAW_DATA_DIR is: {RAW_DATA_DIR}")
    api.dataset_download_files(dataset_test, path=str(download_folder), unzip=True)

    # Check full path to the file
    file_path = download_folder / INPUT_FILE_NAME
    logger.info(f"File path is: {file_path}")

    # Check if file exists
    if not os.path.exists(file_path):
        logger.error(f"File {INPUT_FILE_NAME} does not exist in {file_path}")
        return


def preprocess_df(file:str|Path)->str|Path:
    """Preprocess datasets."""
    _, file_name = os.path.split(file)
    df_data = pd.read_csv(file)

    df_data["quality_label"] = df_data["quality_label"].map(quality_mapping)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    outfile_path = PROCESSED_DATA_DIR / file_name
    df_data.to_csv(outfile_path, index=False)

    return outfile_path


def split_data(file:str|Path, test_size:float=0.2)->None:
    """Split the data into train and test sets."""
    df_data = pd.read_csv(file)
    df_train = df_data.sample(frac=1-test_size, random_state=42)
    df_test = df_data.drop(df_train.index)

    # Save the train and test sets
    train_file = RAW_DATA_DIR / "train.csv"
    test_file = RAW_DATA_DIR / "test.csv"
    df_train.to_csv(train_file, index=False)
    df_test.to_csv(test_file, index=False)


if __name__=="__main__":
    # get the train and test sets from default location
    logger.info("getting datasets")
    get_raw_data()

    # split the data into train and test sets
    logger.info("splitting data into train and test sets")
    split_data(RAW_DATA_DIR / INPUT_FILE_NAME)

    # preprocess both sets
    logger.info("preprocessing train.csv")
    preprocess_df(RAW_DATA_DIR / "train.csv")
    logger.info("preprocessing test.csv")
    preprocess_df(RAW_DATA_DIR / "test.csv")
