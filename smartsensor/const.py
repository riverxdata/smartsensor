from enum import Enum
import re

IMAGE_PATTERN = re.compile(r"^\d+-\d+-[\w]+\.jpg$", re.IGNORECASE)


class NormalizeMethod(str, Enum):
    non = "none"
    delta = "delta"
    ratio = "ratio"


class Degree(str, Enum):
    first = 1
    second = 2
