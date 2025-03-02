import os
import pandas as pd
from sklearn.model_selection import train_test_split
from smartsensor.logger import logger


def split_data(
    data: str,
    metadata: str,
    train_batches: list,
    test_batches: list,
    outdir: str,
    prefix: str,
    random_state: int = 1,
    test_size: float = 0.2,
):
    """It will combine the features and metadata, split to prepare for training and testing

    Args:
        data (str: The csv file path contains the rgb features
        metadata (str): The csv file path contains the metadata (metadata, batch)
        train_batches (list): The batch which is used for train if specify
        test_batches (list): The batch which is used for test if specify
        outdir (str): The output directory for train and test csv file
        prefix (str): The prefix to specify csv file
        random_state (int, optional): To reproduce split. Defaults to 1.
        test_size (float, optional): The test size if train, test batch is not specified. Defaults to 0.2.
    """
    data_df = pd.read_csv(data)
    meta_df = pd.read_csv(metadata)
    batches = meta_df["batch"].unique()
    has_zero_testsize = False
    if test_size == 0:
        has_zero_testsize = True
    if train_batches not in batches and has_zero_testsize:
        raise ValueError(
            f"Not found train batch {train_batches} in all valid batches {batches}"
        )
    if test_batches not in batches and has_zero_testsize:
        raise ValueError(
            f"Not found train batch {test_batches} in all valid batches {batches}"
        )
    # train test split data
    # Define train and test data holders
    train_data = pd.DataFrame()
    test_data = pd.DataFrame()
    meta_data = meta_df.merge(data_df, on="image")

    # if test size is provided
    if not has_zero_testsize:
        for dataset_name, group in meta_data.groupby("batch"):
            train_subset, test_subset = train_test_split(meta_data, test_size=test_size)
            train_data = pd.concat([train_data, train_subset])
            test_data = pd.concat([test_data, test_subset])
    # TODO: add handle cases for using batches to split data
    # reset index
    train_data = train_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)
    prefix = f"testsize_{test_size}"
    if has_zero_testsize:
        prefix = f"trainbatches_{train_batches}_testbatches_{test_batches}"
    train_path = os.path.join(outdir, f"train_{prefix}_{os.path.basename(data)}")
    test_path = os.path.join(outdir, f"test_{prefix}_{os.path.basename(data)}")
    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)
    return (train_data, test_data, prefix)
