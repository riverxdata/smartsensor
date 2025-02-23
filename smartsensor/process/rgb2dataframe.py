from typing import Union
from numpy import ndarray
import pandas as pd


def rgb2dataframe(array: Union[list[list[int]], ndarray]) -> pd.DataFrame:
    """Convert RGB to dataframe. It is useful when we want to normalize the values.
    Notes: Using the normalized images causing the interval values

    Args:
        array (Union[List[List[int]], ndarray]): Just the array with 3 dimensions

    Returns:
        DataFrame: RGB dataframe
    """
    reshaped_array = array.reshape(-1, 3)
    column_names = ["R", "G", "B"]
    return pd.DataFrame(reshaped_array, columns=column_names)
