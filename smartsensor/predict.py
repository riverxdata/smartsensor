import pickle
import glob
import pandas as pd
import json
import os
from sklearn.preprocessing import PolynomialFeatures
from smartsensor.process_image import process_image


def predict_new_data(
    process_dir: str,
    model_dir: str,
    new_data: str,
    outdir: str,
) -> pd.DataFrame:
    """ """
    # Load config
    with open(os.path.join(model_dir, "config.json"), "r") as f:
        model_config = json.load(f)

    with open(os.path.join(process_dir, "config.json"), "r") as f:
        process_config = json.load(f)

    # Process images
    process_image(
        data=new_data, outdir=outdir, kit=model_config["kit"], lum=process_config["lum"]
    )

    # Load to predicts
    samples = pd.read_csv(
        os.path.join(
            outdir, f"features_rgb_{model_config['normalization']}_normalized_roi.csv"
        )
    )
    X = samples[model_config["features"]].values.astype(float)

    # If selector exists, transform input
    selector_path = glob.glob(os.path.join(model_dir, "*rfe_selector.sav"))
    if len(selector_path) != 0:
        with open(selector_path[0], "rb") as f:
            selector = pickle.load(f)
        X = selector.transform(X)

    # Apply Polynomial Transformation
    poly = PolynomialFeatures(degree=model_config["degree"])
    X_poly = poly.fit_transform(X)

    # Load trained model
    model_path = glob.glob(os.path.join(model_dir, "*model.sav"))
    if len(model_path) == 0:
        raise ValueError("Missing model sav file")

    with open(model_path[0], "rb") as f:
        model = pickle.load(f)

    # Predict
    prediction = model.predict(X_poly)
    res = pd.concat(
        [samples, pd.DataFrame(prediction, columns=["predicted_concentration"])], axis=1
    )
    res.to_csv(os.path.join(outdir, "prediction.csv"))
    return res
