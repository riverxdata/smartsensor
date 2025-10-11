import cv2
import os
import glob
import pandas as pd
import numpy as np
import math
from numpy import ndarray
from smartsensor.process.rgb2dataframe import rgb2dataframe
from smartsensor.const import THRESHOLD_DELTA, THRESHOLD_RATIO, KITS
from smartsensor.logger import logger
import json


def normalize(
    raw_roi: str,
    background: str,
    kit: str,
    outdir: str,
    lum: list,
    auto_lum: bool = False,
) -> None:
    """Balance image

    Args:
        raw_roi (str): Path contains raw roi image
        background (str): Path contains background image
        kit (str): Kit version
        outdir (str): Output directory
        lum (list): List of luminance values
        auto_lum (bool): Whether to automatically calculate luminance

    """

    result_path = os.path.join(outdir)
    os.makedirs(result_path, exist_ok=True)
    os.makedirs(os.path.join(outdir, "ratio_normalized_roi"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "delta_normalized_roi"), exist_ok=True)
    failed_ratio_file = os.path.join(outdir, "failed_ratio.csv")
    failed_delta_file = os.path.join(outdir, "failed_delta.csv")

    # Calculate luminance as the average mean of each channel
    if auto_lum:
        # get average value of background to normalize
        channels = [[], [], []]  # B, G, R
        for path in glob.glob(os.path.join(background, "*.jpg")):
            bg_img = cv2.imread(path)
            for i in range(3):  # B=0, G=1, R=2
                channels[i].append(np.mean(bg_img[:, :, i]))
        lum = [np.mean(c) for c in channels]
    elif not lum:
        lum = KITS.get(kit, {}).get("lum")
    logger.info(f"Processing raw roi image using lum: {lum}")

    with open(os.path.join(outdir, "config.json"), "w") as f:
        json.dump({"lum": lum}, f)

    with open(failed_ratio_file, "w") as f_ratio, open(
        failed_delta_file, "w"
    ) as f_delta:
        f_ratio.write("image\n")
        f_delta.write("image\n")

        for image_location in glob.glob(os.path.join(raw_roi, "*.jpg")):
            file_name = os.path.basename(image_location)
            background_image_path = os.path.join(background, file_name)
            if not os.path.exists(background_image_path):
                raise ValueError(
                    f"Background image {background_image_path} does not exist."
                )
            raw_roi_img = cv2.imread(image_location)
            background_img = cv2.imread(background_image_path)
            logger.info(
                f"Processing raw roi image via csv file: {image_location} via background at {background_image_path}"
            )
            # normalize the image
            is_failed_delta, is_failed_ratio = run_normalize(
                roi_image=raw_roi_img,
                background=background_img,
                lum=lum,
                file_name=file_name,
                outdir=outdir,
            )

            if is_failed_ratio:
                f_ratio.write(f"{file_name}\n")
            if is_failed_delta:
                f_delta.write(f"{file_name}\n")


def run_normalize(
    roi_image: ndarray,
    background: ndarray,
    lum: list,
    file_name: str,
    outdir: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Normalized the data according to the background as standard value

        Args:
            roi_image (ndarray): The array RGB of ROI
            background (ndarray): The array RGB of background
            lum (Tuple[float, float, float]): The standard value for normalization
            file_name (str): The file name
            outdir (str): The output directory
        Returns:
            Tuple[bool,bool]: Whether the image is normalized or not in ratio and delta

    """
    is_failed_delta = normalize_delta(
        roi_image=roi_image,
        background=background,
        lum=lum,
        file_name=file_name,
        outdir=outdir,
    )

    is_failed_ratio = normalize_ratio(
        roi_image=roi_image,
        background=background,
        lum=lum,
        file_name=file_name,
        outdir=outdir,
    )
    return is_failed_delta, is_failed_ratio


def normalize_ratio(
    roi_image: ndarray,
    background: ndarray,
    lum: list,
    file_name: str,
    outdir: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Normalize ratio"""
    csv_name = file_name.replace(".jpg", ".csv")
    # take the bg color
    B = np.mean(background[:, :, 0])
    G = np.mean(background[:, :, 1])
    R = np.mean(background[:, :, 2])

    # get ratio for normalize
    ratioB = lum[0] / B
    ratioG = lum[1] / G
    ratioR = lum[2] / R

    # Normalize the ROI image using the ratios
    ratio_normalized_roi = rgb2dataframe(roi_image)
    ratio_normalized_roi["B"] *= ratioB
    ratio_normalized_roi["G"] *= ratioG
    ratio_normalized_roi["R"] *= ratioR
    ratio_normalized_roi = ratio_normalized_roi.astype(np.int32)
    ratio_normalized_roi.to_csv(
        os.path.join(outdir, "ratio_normalized_roi", csv_name),
        index=False,
    )

    tmp = roi_image
    tmp = tmp.astype(np.float64)
    tmp[:, :, 0] *= ratioB
    tmp[:, :, 1] *= ratioG
    tmp[:, :, 2] *= ratioR
    tmp = tmp.astype(np.int32)
    tmp_file = os.path.join(outdir, "ratio_normalized_roi", file_name)
    cv2.imwrite(tmp_file, tmp)

    logger.info(f"Normalize delta: {ratioB, ratioG, ratioR}")
    if (
        math.log(ratioB, 2) > THRESHOLD_RATIO
        or math.log(ratioG, 2) > THRESHOLD_RATIO
        or math.log(ratioR, 2) > THRESHOLD_RATIO
    ):
        return True

    return False


def normalize_delta(
    roi_image: ndarray,
    background: ndarray,
    lum: list,
    file_name: str,
    outdir: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Normalize detla"""
    csv_name = file_name.replace(".jpg", ".csv")
    # take the bg color
    B = np.mean(background[:, :, 0])
    G = np.mean(background[:, :, 1])
    R = np.mean(background[:, :, 2])

    # Calculate the deltas for adjustment
    deltaB = lum[0] - B
    deltaG = lum[1] - G
    deltaR = lum[2] - R

    tmp = roi_image
    tmp[:, :, 0] + deltaB
    tmp[:, :, 1] + deltaG
    tmp[:, :, 2] + deltaR

    # Adjust the ROI image using the deltas
    delta_normalized_roi = rgb2dataframe(roi_image)
    delta_normalized_roi["B"] += deltaB
    delta_normalized_roi["G"] += deltaG
    delta_normalized_roi["R"] += deltaR

    # csv
    delta_normalized_roi = delta_normalized_roi.astype(np.int32)

    delta_normalized_roi.to_csv(
        os.path.join(outdir, "delta_normalized_roi", csv_name),
        index=False,
    )

    # image
    tmp_file = os.path.join(outdir, "delta_normalized_roi", file_name)
    cv2.imwrite(tmp_file, tmp)

    logger.info(f"Normalize delta: {deltaB, deltaG, deltaR}")

    if (
        abs(deltaB) > THRESHOLD_DELTA
        or abs(deltaG) > THRESHOLD_DELTA
        or abs(deltaR) > THRESHOLD_DELTA
    ):
        return True

    return False
