import os
import pickle
from typing import List
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from smartsensor.logger import logger


def fit(
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
        logger.info("Skip feature selection")
        selected_features = features
        X_selected = x
    else:
        # Using Random Forest Regressor as the estimator for RFECV
        estimator = RandomForestRegressor(random_state=1)
        logger.info(f"Feature selection using the  the model using CV={cv}")
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
        logger.info("Your selected features are:")
        logger.info(",".join(selected_features))

        # Write features
        f.write("Selected features:\n")
        f.write(f"{','.join(selected_features)}\n")

        # Write model
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
            if feature != "Intercept":
                fomular += f" + {coefficient}x{'x'.join(feature.split())}"
            else:
                fomular += f" + {coefficient}"
        fomular += f" + {intercept}"
        f.write(fomular)
        logger.info("Your model is:")
        logger.info(fomular)
    res = np.round(np.dot(X_selected_poly, coefficients) + intercept, 2)
    model_res = np.round(clf.predict(X_selected_poly), 2)
    assert sorted(list(res)) == sorted(list(model_res)), logger.error(
        "Incorrect model, the fomular is not correct vs the predict function"
    )
    return clf, selected_features
