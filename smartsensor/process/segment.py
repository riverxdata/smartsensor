def image_segmentations(
    indir: str,
    outdir: str,
    threshold: List = [(0, 110, 60), (80, 220, 160)],
    dim: List = [740, 740],
    bg_index: List = [50, 60, 350, 360],
    roi_index: List = [-250, -200, 345, -345],
) -> None:
    """Using the threshold to segment the images to different regions

    Args:
        indir (str): Images path
        outdir (str): The output path
        threshold (List): Cut-off values for segment
        dim (None): Dimesion
        bg_index (None): Background position for segment
        roi_index (None): ROI position for segment
    """

    # Create directories for storing intermediate files and results
    result_path = os.path.join(outdir)
    os.makedirs(result_path, exist_ok=True)
    directories = [
        "squared_frame",
        "raw_roi",
        "ratio_normalized_roi",
        "delta_normalized_roi",
        "background",
    ]
    for directory in directories:
        os.makedirs(os.path.join(result_path, directory), exist_ok=True)

    # Contour value
    low_val = threshold[0]
    high_val = threshold[1]

    # process each image in the input directory
    for image_location in glob.glob(os.path.join(indir, "*.jpg")):
        image = cv2.imread(image_location)
        file_name = os.path.splitext(os.path.basename(image_location))[0]

        mask = cv2.inRange(image, np.array(low_val), np.array(high_val))

        # Find contours in the mask
        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        # Select the largest contour
        largest_area = 0
        for cnt in contours:
            if cv2.contourArea(cnt) > largest_area:
                cont = cnt
                largest_area = cv2.contourArea(cnt)

        # Get the parameters of the bounding box
        x, y, w, h = cv2.boundingRect(cont)
        squared_frame = image[y : y + h, x : x + w]  # noqa: E203
        # Section for cover
        roi = cv2.resize(squared_frame, dim, interpolation=cv2.INTER_AREA)

        # Background image
        background = roi[
            bg_index[0] : bg_index[1], bg_index[2] : bg_index[3]  # noqa: E203
        ]

        # ROI
        # roi = roi[roi_index:-roi_index, roi_index:-roi_index]
        roi = roi[roi_index[0] : roi_index[1], roi_index[2] : roi_index[3]]  # noqa

        # File path
        squared_frame_path = os.path.join(
            result_path, "squared_frame", file_name + ".jpg"
        )
        brackground_path = os.path.join(result_path, "background", file_name + ".jpg")
        roi_path = os.path.join(result_path, "raw_roi", file_name + ".jpg")
        # Save
        cv2.imwrite(
            squared_frame_path,
            squared_frame,
        )
        cv2.imwrite(brackground_path, background)
        cv2.imwrite(roi_path, roi)
