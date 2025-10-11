import pickle
import glob
import pandas as pd
import json
import os
from sklearn.preprocessing import PolynomialFeatures


def predict_new_data(
    processed_dir: str,
    model_dir: str,
    outdir: str,
) -> pd.DataFrame:
    """ """
    # Load config
    with open(os.path.join(model_dir, "config.json"), "r") as f:
        model_config = json.load(f)

    # Load to predicts
    samples = pd.read_csv(
        os.path.join(
            processed_dir,
            f"features_rgb_{model_config['normalization']}_normalized_roi.csv",
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
    model_path = glob.glob(os.path.join(model_dir, "full_RGB_model.sav"))
    if len(model_path) == 0:
        raise ValueError("Missing model sav file")

    with open(model_path[0], "rb") as f:
        model = pickle.load(f)

    # Predict
    prediction = model.predict(X_poly)
    res = pd.concat(
        [samples, pd.DataFrame(prediction, columns=["predicted_concentration"])], axis=1
    )
    os.makedirs(outdir, exist_ok=True)
    res = res.sort_values("predicted_concentration")
    res["error"] = res["predicted_concentration"] - res["concentration"]
    res.to_csv(os.path.join(outdir, "prediction.csv"))
    return res
