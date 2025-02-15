import re
import os
from smartsensor.logger import logger


def validate_data(data: str):
    """Validate that the provided folder contains a subfolder named 'data' with correctly formatted images."""

    if not os.path.isdir(data):
        raise logger.error(f"Provided path '{data}' is not a valid directory.")

    # The folder must contain exactly one subfolder named 'data'
    subfolder_path = os.path.join(data, "data")
    if not os.path.isdir(subfolder_path):
        raise logger.error(f"Folder '{data}' must contain a subfolder named 'data'.")

    # Check for image files inside 'data/' subfolder
    images = [f for f in os.listdir(subfolder_path) if f.lower().endswith(".jpg")]

    if not images:
        raise logger.error(
            f"Subfolder '{subfolder_path}' must contain at least one JPG image."
        )

    for img in images:
        if not IMAGE_PATTERN.match(img):
            raise logger.error(
                f"Invalid filename '{img}' in '{subfolder_path}'. Expected format: <number>-<number>-<string>.jpg"
            )

    return data
