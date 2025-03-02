from enum import Enum
import re
import math

IMAGE_PATTERN = re.compile(r"^\d+-\d+-[\w]+\.jpg$", re.IGNORECASE)
# normalize threshold
THRESHOLD_DELTA = 20
THRESHOLD_RATIO = math.log(1.2, 2)  # log to get diff fold change

# kit with relative bound color
KITS = {
    "1.1.0": [(35, 40, 40), (85, 255, 255)],  # ampiciline dataset kit
    "1.0.0": [(0, 120, 20), (150, 230, 80)],  # cuso4, fe3+
}
# resize dim
RESIZE_DIM = 1000
ROI_DIM = 50


class NormalizeMethod(str, Enum):
    non = "none"
    delta = "delta"
    ratio = "ratio"


class Degree(str, Enum):
    first = 1
    second = 2
