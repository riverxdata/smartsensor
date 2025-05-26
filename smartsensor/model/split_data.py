import os
import pandas as pd
from sklearn.model_selection import train_test_split
from smartsensor.logger import logger


def split_data(
    meta_data: pd.DataFrame,
    outdir: str,
    prefix: str,
    test_size: float = 0.2,
):
    """It will combine the features and metadata, split to prepare for training and testing

    Args:
        meta_data(pd.DataFrame): The dataframe with data and metadata
        outdir (str): The output directory for train and test csv file
        prefix (str): The prefix to specify csv file
        random_state (int, optional): To reproduce split. Defaults to 1.
        test_size (float, optional): The test size if train, test batch is not specified. Defaults to 0.2.
    """
    # train test split data
    # Define train and test data holders
    train_data = pd.DataFrame()
    test_data = pd.DataFrame()
    # split dataset by batch equally
    if test_size != 0:
        for dataset_name, group in meta_data.groupby("batch"):
            train_subset, test_subset = train_test_split(meta_data, test_size=test_size)
            train_data = pd.concat([train_data, train_subset])
            test_data = pd.concat([test_data, test_subset])
        train_data = train_data.reset_index(drop=True)
        test_data = test_data.reset_index(drop=True)
        prefix = f"testsize_{test_size}"
        train_path = os.path.join(outdir, f"train_{prefix}.csv")
        test_path = os.path.join(outdir, f"test_{prefix}_.csv")
        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)
        return (train_data, test_data, prefix)
    # not split data
    meta_data.reset_index(drop=True)
    return meta_data, meta_data, "full"
