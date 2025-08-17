import re
import math

IMAGE_PATTERN = re.compile(r"^\d+-\d+-[\w]+\.jpg$", re.IGNORECASE)
# normalize threshold
THRESHOLD_DELTA = 20
THRESHOLD_RATIO = math.log(1.3, 2)  # log to get diff fold change

# kit with relative bound color
KITS = {
    "1.2.0": {
        "threshold": [(60, 60, 100), (250, 250, 200)],
        "lum": [100, 100, 100],
    },  # purple color
    "1.1.0": {
        "threshold": [(35, 40, 40), (85, 255, 255)],
        "lum": [210, 190, 185],
    },  # ampiciline dataset kit
    "1.0.0": {
        "threshold": [(0, 110, 60), (80, 220, 160)],
        "lum": [100, 100, 100],
    },  # cuso4, fe3+
}
# resize dim
RESIZE_DIM = 1000
ROI_DIM = 50
