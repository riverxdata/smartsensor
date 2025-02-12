import cv2
import glob
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestRegressor
import pickle
import os
import numpy as np
import statistics as st
import pandas as pd
from pandas import DataFrame
from numpy import ndarray
import math
from typing import Any, List, Union, Tuple, Callable
from .visualization import linear_regression_visualization
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")


# Split the images by train test split
def train_test_split_by_conv(
    conv_path: str,
    rgb_path: str,
    process_type: str,
    test_size: float,
    random_state: int,
    outdir: str,
):
    os.makedirs(outdir, exist_ok=True)
    data = get_data(rgb_path=rgb_path, concentration=conv_path, outdir=outdir)

    # List of unique concentrations in your DataFrame
    concentrations = data["concentration"].unique()
    # Initialize empty lists to store the splits
    (
        image_train_list,
        image_test_list,
        concentration_train_list,
        concentration_test_list,
    ) = ([], [], [], [])
    # Iterate through each concentration
    for concentration in concentrations:
        # Filter the DataFrame for the current concentration
        subset_conv = data[data["concentration"] == concentration]
        x_data = subset_conv.drop(columns=["concentration"])
        y_data = subset_conv["concentration"]
        if random_state is not None:
            (
                image_train,
                image_test,
                concentration_train,
                concentration_test,
            ) = train_test_split(
                x_data,
                y_data,
                test_size=test_size,
                random_state=1,
            )
        else:
            (
                image_train,
                image_test,
                concentration_train,
                concentration_test,
            ) = train_test_split(
                x_data,
                y_data,
                test_size=test_size,
            )

        # Append the splits to the lists
        image_train_list.append(image_train)
        image_test_list.append(image_test)
        concentration_train_list.append(concentration_train)
        concentration_test_list.append(concentration_test)

    # Concatenate the splits to obtain the final training and testing sets
    final_image_train = pd.concat(image_train_list)
    final_image_test = pd.concat(image_test_list)
    final_concentration_train = pd.concat(concentration_train_list)
    final_concentration_test = pd.concat(concentration_test_list)
    # Create the final training and testing sets
    final_train_data = pd.concat([final_image_train, final_concentration_train], axis=1)
    final_test_data = pd.concat([final_image_test, final_concentration_test], axis=1)
    # Save the training and testing sets
    train_path = os.path.join(outdir, f"{process_type}_train.csv")
    test_path = os.path.join(outdir, f"{process_type}_test.csv")
    final_train_data.to_csv(train_path, index=False)
    final_test_data.to_csv(test_path, index=False)
    return final_train_data, final_test_data, train_path, test_path


def rgb2dataframe(array: Union[List[List[int]], ndarray]) -> DataFrame:
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


def get_rgb(indir: str, outdir: str, threshold_stdev: float) -> str:
    """Get the mean and mode value of R, G, B channel in the images in the
    directory. Then, save it to a dataframe

    Args:
        indir (str): Input directory
        outdir (str): Output directory

    Returns:
        DataFrame: Dataframe contains RGB of mean and mode value and relative
        image id
    """
    if os.path.exists(outdir):
        pass
    else:
        os.makedirs(outdir)
    input_path = os.path.join(indir, "*.csv")
    rgb_path = os.path.join(outdir, "RGB_values.csv")
    rgb_qualification_path = os.path.join(outdir, "not_RGB_qualification.csv")
    imgs_path = [
        path
        for path in glob.glob(input_path)
        if path not in [rgb_path, rgb_qualification_path]
    ]
    if len(imgs_path) == 0:
        raise ValueError(f"The directory {indir} does not contain any images RGB csv")
    with open(rgb_path, "w") as res, open(rgb_qualification_path, "w") as res_qual:
        res.write("image,meanB,meanG,meanR,modeB,modeG,modeR,stdevB,stdevG,stdevR\n")
        res_qual.write("image\n")
        for img_path in imgs_path:
            img_id = os.path.basename(img_path.replace(".csv", ".jpg"))
            # Load image
            img = pd.read_csv(img_path)
            b, g, r = img["B"], img["G"], img["R"]
            # mean
            b_mean = np.mean(b)
            g_mean = np.mean(g)
            r_mean = np.mean(r)
            # mode
            b_mode = st.mode(b)
            g_mode = st.mode(g)
            r_mode = st.mode(r)
            # stdev
            b_stdev = np.std(b)
            g_stdev = np.std(g)
            r_stdev = np.std(r)
            # is_stdev_qualified
            is_stdev_qualified = (
                b_stdev < threshold_stdev
                and g_stdev < threshold_stdev
                and r_stdev < threshold_stdev
            )
            # write
            if is_stdev_qualified:
                res.write(
                    f"{img_id},{b_mean},{g_mean},{r_mean},{b_mode},{g_mode},{r_mode},{b_stdev},{g_stdev},{r_stdev}\n"
                )
            else:
                res_qual.write(f"{img_id}\n")

    return rgb_path


def get_data(rgb_path: str, concentration: str, outdir: str) -> DataFrame:
    """Combined the RGB features with their relative concentrations

    Args:
        rgb_path (str): The file path that contains the values for RGB features
        concentration (str): The file path that contains the values for concentration

    Returns:
        DataFrame: The combined dataframe that could be used for training and validating the
        machine learning models
    """
    df = pd.read_csv(rgb_path)
    conc = pd.read_csv(concentration)
    df = pd.merge(df, conc, on="image")
    df.to_csv(os.path.join(outdir, "data.csv"), index=False)
    return df


def custom_predict(coefficients, intercept, features):
    prediction = intercept + np.sum(np.multiply(coefficients, features))
    return prediction


def train_regression(
    train: DataFrame,
    features: List,
    degree: int,
    outdir: str,
    prefix: str,
    skip_feature_selection: bool = True,
    cv: int = 5,
) -> str:
    """Using the the training data for turning the regression model

    Args:
        train (DataFrame): dataframe with RGB values (features) and concentration (target)
        degree (int): polynomial degree
        outdir (str): the output directory
        prefix (str): the prefix name

    Returns:
        model: model path
    """
    x = train[features].values.astype(float)
    y = train["concentration"].values.astype(float)
    # cv = None
    if skip_feature_selection:
        print("Skip feature selection")
        selected_features = features
        X_selected = x
    else:
        # Using Random Forest Regressor as the estimator for RFECV
        estimator = RandomForestRegressor(random_state=1)
        print(f"Feature selection using the  the model using CV={cv}")
        rfe_selector = RFECV(estimator, step=1, cv=cv)
        rfe_selector = rfe_selector.fit(x, y)
        # Select the important features from the original feature set
        selected_features = [
            feature
            for feature, selected in zip(features, rfe_selector.support_)
            if selected
        ]

        X_selected = rfe_selector.transform(x)
        # Save the selector
        rfe_selector_path = os.path.join(outdir, f"{prefix}_rfe_selector.sav")
        with open(rfe_selector_path, "wb") as f:
            pickle.dump(rfe_selector, f)
    # Now apply Polynomial Features to the selected features
    poly = PolynomialFeatures(degree=degree)
    X_selected_poly = poly.fit_transform(X_selected)
    feature_names = poly.get_feature_names_out(selected_features)
    X_selected_poly = pd.DataFrame(data=X_selected_poly, columns=feature_names)
    # Fit the Linear Regression model with polynomial features
    clf = LinearRegression()
    clf.fit(X_selected_poly, y)

    # Save the model
    model_path = os.path.join(outdir, f"{prefix}_RGB_model.sav")
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)
    with open(os.path.join(outdir, f"{prefix}_model_infor.txt"), "w") as f:
        # Write CV
        if cv is not None:
            f.write(f"Feature selection using the  the model using CV={cv}\n")
        print("#" * 100)
        print("Your selected features are:\n")
        f.write("Selected features:\n")
        # write features
        for feature in selected_features:
            print(feature)
            f.write(f"{feature}\n")
        print("#" * 100)
        # write model
        f.write("Model:\n")
        # Create a dictionary with feature names and coefficients
        coefficients = clf.coef_
        intercept = clf.intercept_
        coefficients_dict = {
            feature: coefficient
            for feature, coefficient in zip(feature_names, coefficients)
        }
        # Add intercept to the dictionary
        # coefficients_dict["Intercept"] = intercept
        # Print the coefficients and intercept with feature names
        fomular = "y = "
        for feature, coefficient in coefficients_dict.items():
            # if feature != "Intercept":
            fomular += f" + {coefficient}x{'x'.join(feature.split())}"
            # else:
            #     fomular += f" + {coefficient}"
        fomular += f" + {intercept}"
        f.write(fomular)
        print("Your model is:\n", fomular)
    res = np.round(np.dot(X_selected_poly, coefficients) + intercept, 2)
    model_res = np.round(clf.predict(X_selected_poly), 2)
    assert sorted(list(res)) == sorted(
        list(model_res)
    ), "Incorrect model, the fomular is not correct vs the predict function"  # noqa
    # Save the selected features names
    return clf, selected_features


def evaluate_metrics(
    model: Any,
    data: DataFrame,
    features: List,
    degree: int,
) -> (DataFrame, DataFrame):
    """Simple evaluation matrics for measure the errors

    Args:
        model (Any): the model object
        transform_model (Any): the transform data object
        x (np.array): the array of RGB for training the data
        y_real (np.array): the array of  concentration

    Returns:
        List: the score matrics and, the real and predicted value
    """
    # load
    x = data[features].values.astype(float)
    poly = PolynomialFeatures(degree=degree)
    X_t = poly.fit_transform(x)
    y_real = data["concentration"].values.astype(float)
    y_pred = model.predict(X_t)

    # metrics
    rmse = np.sqrt(np.mean((y_real - y_pred) ** 2))
    mae = np.mean(np.abs(y_real - y_pred))
    r2 = r2_score(y_real, y_pred)
    metric = pd.DataFrame([[rmse, mae, r2]], columns=["rmse", "mae", "r2"])

    # result detail
    detail = pd.DataFrame([data["image"], y_real, y_pred])
    detail = detail.T
    detail.columns = ["image", "expected_concentration", "predicted_concentration"]
    detail = detail.sort_values("expected_concentration")
    detail["absolute_error"] = abs(
        detail["predicted_concentration"] - detail["expected_concentration"]
    )
    detail["error"] = (
        detail["predicted_concentration"] - detail["expected_concentration"]
    )

    return (metric, detail)


def processing_images(
    indir: str,
    outdir: str,
    threshold: List = [(0, 120, 20), (150, 230, 80)],
    dim: List = [740, 740],
    bg_index: List = [50, 60, 350, 360],
    roi_index: List = [-250, -200, 345, -345],
    constant: List = [100, 150, 40],
    feature_method: Callable[[Any], float] = np.mean,
    overwrite: bool = True,
    threshold_stdev: float = 5,
    threshold_ratio: float = math.log(1.2, 2),
    threshold_delta: float = 20,
) -> None:
    """Using the images in raw format, convert to csv, then normalize the images and save to
    csv format

    Args:
        indir (str): Input directory
        outdir (str): Output directory
        threshold (List, optional): Threshold for cutting edge
        Defaults to [(0, 120, 20), (150, 230, 80)].
        dim (List, optional): Dimension for scale the ROI. Defaults to [740, 740].
        bg_index (List, optional): The background index to cut. Defaults to [50, 60, 350, 360].
        roi_index (int, optional): The roi region boudary. Defaults to 245.
        constant (List, optional): The constant for background scale. Defaults to [60, 90, 30].
        feature_method (Callable[[Any], float], optional): feature method. Defaults to np.mean.
    """
    if os.path.exists(os.path.join(outdir, "raw_roi")) and not overwrite:
        print("Already processed, remove outdir to reprocess")
        return True
    else:
        # Step 1: Extract ROI, background, and squared_frame return RGB value
        image_segmentations(
            indir=indir,
            outdir=outdir,
            threshold=threshold,
            dim=dim,
            bg_index=bg_index,
            roi_index=roi_index,
        )
        print("Complete extract ROI")
        # Step 2: Balance images by normalizing using the background color and saving results
        balance_image(
            indir=indir,
            outdir=outdir,
            constant=constant,
            feature_method=feature_method,
            threshold_ratio=threshold_ratio,
            threshold_delta=threshold_delta,
        )
        print("Complete balance images")
        # Step3: Get the RGB value
        for feature_type in ["raw_roi", "ratio_normalized_roi", "delta_normalized_roi"]:
            get_rgb(
                indir=os.path.join(outdir, feature_type),
                outdir=os.path.join(outdir, feature_type),
                threshold_stdev=threshold_stdev,
            )


def prepare_data(
    train_concentrations: list,
    test_rgb_path: str,
    test_concentrations: list,
    train_rgb_path: str,
    outdir: str,
    prefix: str,
    random_state: int,
    test_size: float = 0.3,
):
    # Load data
    # case 1: No testing data, using all training dataset to split
    # if have more than one train datasets, split and combine each dataset
    if len(test_concentrations) == 0 or test_rgb_path is None:
        combined_train = []
        combined_test = []
        if test_size == 1:
            for train_conv in train_concentrations:
                train = get_data(
                    rgb_path=train_rgb_path, concentration=train_conv, outdir=outdir
                )
                combined_train.append(train)
            train = pd.concat(combined_train, ignore_index=True)
            test = train
        else:
            for train_conv in train_concentrations:
                train, test, train_path, test_path = train_test_split_by_conv(
                    conv_path=train_conv,
                    rgb_path=train_rgb_path,
                    process_type=prefix,
                    test_size=test_size,
                    random_state=random_state,
                    outdir=outdir,
                )
                combined_train.append(train)
                combined_test.append(test)
            train = pd.concat(combined_train, ignore_index=True)
            test = pd.concat(combined_test, ignore_index=True)
    else:
        # case 2: Require train data, test data, not use split
        combined_train = []
        combined_test = []
        for train_conv in train_concentrations:
            train = get_data(
                rgb_path=train_rgb_path, concentration=train_conv, outdir=outdir
            )
            combined_train.append(train)
        train = pd.concat(combined_train, ignore_index=True)
        for test_conv in test_concentrations:
            test = get_data(
                rgb_path=test_rgb_path, concentration=test_conv, outdir=outdir
            )
            combined_test.append(test)
        test = pd.concat(combined_test, ignore_index=True)
    train_path = os.path.join(outdir, f"{prefix}_train.csv")
    test_path = os.path.join(outdir, f"{prefix}_test.csv")
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    return train, test, train_path, test_path


def end2end_model(
    train_rgb_path: str,
    train_concentrations: list,
    test_rgb_path: str,
    test_concentrations: list,
    features: str,
    degree: int,
    outdir: str,
    prefix: str,
    skip_feature_selection: bool = True,
    test_size: float = 0.3,
    random_state: int = 1,
    cv: int = 5,
):
    train, test, train_path, test_path = prepare_data(
        train_concentrations=train_concentrations,
        test_rgb_path=test_rgb_path,
        test_concentrations=test_concentrations,
        train_rgb_path=train_rgb_path,
        outdir=outdir,
        prefix=prefix,
        test_size=test_size,
        random_state=random_state,
    )
    # Train
    features = features.split(",")
    train_model, selected_features = train_regression(
        train=train,
        features=features,
        degree=degree,
        outdir=outdir,
        prefix=prefix,
        skip_feature_selection=skip_feature_selection,
        cv=cv,
    )
    # Evaluate
    train_metric, train_detail = evaluate_metrics(
        model=train_model,
        data=train,
        features=selected_features,
        degree=degree,
    )
    test_metric, test_detail = evaluate_metrics(
        model=train_model,
        data=test,
        features=selected_features,
        degree=degree,
    )

    if test_concentrations is None or test_rgb_path is None:
        train_dataset = f"{'_'.join([os.path.basename(train_concentration).split('.')[0] for train_concentration in train_concentrations])}_train_by_split_{str(1-test_size)}"  # noqa
        test_dataset = f"{'_'.join([os.path.basename(train_concentration).split('.')[0] for train_concentration in train_concentrations])}_test_by_split_{test_size}"  # noqa
    else:
        train_dataset = f"{'+'.join([os.path.basename(train_concentration).split('.')[0] for train_concentration in train_concentrations])}"  # noqa
        test_dataset = f"{'+'.join([os.path.basename(test_concentration).split('.')[0] for test_concentration in test_concentrations])}"  # noqa
    # train res
    train_metric["train_data"] = train_dataset
    train_metric["test_data"] = train_dataset
    train_detail["train_data"] = train_dataset
    train_detail["test_data"] = train_dataset
    # test res
    test_metric["train_data"] = train_dataset
    test_metric["test_data"] = test_dataset
    test_detail["train_data"] = train_dataset
    test_detail["test_data"] = test_dataset

    train_metric["features"] = ",".join(selected_features)
    test_metric["features"] = ",".join(selected_features)
    metric = pd.concat([train_metric, test_metric], axis=0)
    detail = pd.concat([train_detail, test_detail], axis=0)
    # Export data
    metric_path = os.path.join(outdir, f"metric_{train_dataset}&{test_dataset}.csv")
    detail_path = os.path.join(outdir, f"detail_{train_dataset}&{test_dataset}.csv")
    metric.to_csv(metric_path, index=False)
    detail.to_csv(detail_path, index=False)
    return metric, detail, train_model


def visualization(
    features: List, train: DataFrame, train_concentration: str, outdir: str
):
    # visualization in train only
    for feature in features:
        linear_regression_visualization(
            df=train,
            feature=feature,
            target="concentration",
            train_dataset=train_concentration,
            outdir=outdir,
        )
