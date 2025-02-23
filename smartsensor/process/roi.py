import cv2
import os
import glob
from smartsensor.const import RESIZE_DIM, ROI_DIM, KITS
from smartsensor.process.rgb2dataframe import rgb2dataframe


def get_r2frame(
    data: str,
    outdir: str,
    kit: str = "1.1.0",
) -> None:
    """Using the threshold to segment the images to get square frame

    Args:
        indir (str): Images path
        outdir (str): The output path
        threshold (list): Cut-off values for segment
        dim (None): Dimesion
        bg_index (None): Background position for segment
        roi_index (None): ROI position for segment
    """

    if kit not in KITS.keys():
        raise ValueError(
            f"Missing kit, the available kits are:\n{', '.join(KITS.keys())}"
        )

    # create directories for storing intermediate files and results
    r2_path = os.path.join(outdir, "squared_frame")
    os.makedirs(r2_path, exist_ok=True)

    # contour value
    threshold = KITS[kit]
    low_val = threshold[0]
    high_val = threshold[1]

    # process each image in the input directory
    # for image_location in glob.glob(os.path.join(data, "*.jpg"), recursive=True):
    #     image = cv2.imread(image_location)
    #     file_name = os.path.basename(image_location)

    #     image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #     mask = cv2.inRange(image_hsv, low_val, high_val)
    #     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     contours = sorted(contours, key=cv2.contourArea, reverse=True)
    #     if not contours:
    #         raise ValueError("Your image does not have any countours")
    #     x, y, w, h = cv2.boundingRect(contours[0])
    #     roi = image[y : y + h, x : x + w]
    #     roi = cv2.resize(roi, (RESIZE_DIM, RESIZE_DIM), interpolation=cv2.INTER_AREA)
    #     cv2.imwrite(os.path.join(r2_path, file_name), roi)
    return r2_path


def get_roi_background(
    data: str,
    outdir: str,
):
    """Get roi and background to normalize

    Args:
        data (str): The path contain frame images
        outdir (str): The outdir
    """
    roi_path = os.path.join(outdir, "raw_roi")
    bg_path = os.path.join(outdir, "background")

    os.makedirs(roi_path, exist_ok=True)
    os.makedirs(bg_path, exist_ok=True)

    center = RESIZE_DIM // 2
    half_size = ROI_DIM // 2

    # for image_location in glob.glob(os.path.join(data, "*.jpg")):
    #     image = cv2.imread(image_location)
    #     file_name = os.path.basename(image_location)
    #     # roi
    #     roi = image[
    #         center - half_size : center + half_size,
    #         center - half_size : center + half_size,
    #     ]

    #     cv2.imwrite(os.path.join(roi_path, file_name), roi)
    #     rgb2dataframe(roi).to_csv(
    #         os.path.join(roi_path, file_name.replace(".jpg", ".csv")), index=False
    #     )

    #     # background
    #     quarter_size = half_size // 2
    #     background = image[
    #         center - half_size : center + half_size, half_size : half_size * 3
    #     ]
    #     cv2.imwrite(os.path.join(bg_path, file_name), background)
    return roi_path, bg_path
