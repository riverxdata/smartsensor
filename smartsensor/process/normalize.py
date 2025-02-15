def balance_image(
    indir: str,
    outdir: str,
    constant: List,
    feature_method: Callable[[Any], float],
    threshold_ratio: float = math.log(1.2, 2),
    threshold_delta: float = 20,
) -> None:
    """Image color balancing

    Args:
        indir (str): Images path
        outdir (str): Outdir path
        constant (List): constant for normalization
        feature_method (Callable[[Any], float]): methods for normalization [mean or mode]
    """

    result_path = os.path.join(outdir)
    os.makedirs(result_path, exist_ok=True)
    failed_ratio_file = os.path.join(outdir, "failed_ratio.csv")
    failed_delta_file = os.path.join(outdir, "failed_delta.csv")
    with open(failed_ratio_file, "w") as f_ratio, open(
        failed_delta_file, "w"
    ) as f_delta:
        f_ratio.write("image\n")
        f_delta.write("image\n")
        for image_location in glob.glob(os.path.join(indir, "*.jpg")):
            print(f"Processing image: {image_location}")

            file_name = os.path.splitext(os.path.basename(image_location))[0]

            roi = cv2.imread(os.path.join(result_path, "raw_roi", file_name + ".jpg"))
            background = cv2.imread(
                os.path.join(result_path, "background", file_name + ".jpg")
            )
            rgb2dataframe(roi).to_csv(
                os.path.join(outdir, "raw_roi", file_name + ".csv"), index=False
            )
            # Normalize the image
            is_failed_ratio, is_failed_delta = normalized_execute(
                roi_image=roi,
                background=background,
                lum=constant,
                feature_method=feature_method,
                file_name=file_name,
                outdir=outdir,
                threshold_ratio=threshold_ratio,
                threshold_delta=threshold_delta,
            )
            if is_failed_ratio:
                f_ratio.write(f"{file_name}\n")
            if is_failed_delta:
                f_delta.write(f"{file_name}\n")


def normalized_execute(
    roi_image: ndarray,
    background: ndarray,
    lum: Tuple[float, float, float],
    feature_method: Callable[[Any], float],
    file_name: str,
    outdir: str,
    threshold_ratio: float = math.log(1.2, 2),
    threshold_delta: float = 20,
) -> Tuple[DataFrame, DataFrame, DataFrame]:
    """Normalized the data according to the background as standard value

    Args:
        roi_image (ndarray): The array RGB of ROI
        background (ndarray): The array RGB of background
        lum (Tuple[float, float, float]): The standard value for normalization
        feature_method (Callable[[Any], float]): The mean or mode value to scale
        file_name (str): The file name
        outdir (str): The output directory
        threshold_ratio (float, optional): The threshold for ratio. Defaults to 0.1. Which means the image will be
        ignored if the ratio is greater/smaller than 0.1
        threshold_delta (float, optional): The threshold for delta. Defaults to 20. Which means the image will be
        ignored if the delta is greater/smaller than 20

    Returns:
        Tuple[bool,bool]: Whether the image is normalized or not in ratio and delta
    """

    # Take the upper region for balanced cover
    B = feature_method(background[:, :, 0])
    G = feature_method(background[:, :, 1])
    R = feature_method(background[:, :, 2])

    # Calculate the ratios for normalizationx
    ratioB = lum[0] / B
    ratioG = lum[1] / G
    ratioR = lum[2] / R
    if (
        math.log(ratioB, 2) > threshold_ratio
        or math.log(ratioG, 2) > threshold_ratio
        or math.log(ratioR, 2) > threshold_ratio
    ):
        ratio_normalized_roi = None
        is_failed_ratio = True
    else:
        # Normalize the ROI image using the ratios
        ratio_normalized_roi = rgb2dataframe(roi_image)
        ratio_normalized_roi["B"] *= ratioB
        ratio_normalized_roi["G"] *= ratioG
        ratio_normalized_roi["R"] *= ratioR
        ratio_normalized_roi.to_csv(
            os.path.join(outdir, "ratio_normalized_roi", file_name + ".csv"),
            index=False,
        )
        is_failed_ratio = False

    # Calculate the deltas for adjustment
    deltaB = lum[0] - B
    deltaG = lum[1] - G
    deltaR = lum[2] - R
    if (
        abs(deltaB) > threshold_delta
        or abs(deltaG) > threshold_delta
        or abs(deltaR) > threshold_delta
    ):
        delta_normalized_roi = False
        is_failed_delta = True
    else:
        # Adjust the ROI image using the deltas
        delta_normalized_roi = rgb2dataframe(roi_image)
        delta_normalized_roi["B"] += deltaB
        delta_normalized_roi["G"] += deltaG
        delta_normalized_roi["R"] += deltaR
        delta_normalized_roi.to_csv(
            os.path.join(outdir, "delta_normalized_roi", file_name + ".csv"),
            index=False,
        )
        is_failed_delta = False
    return (is_failed_ratio, is_failed_delta)
