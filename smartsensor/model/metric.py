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
