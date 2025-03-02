def custom_predict(coefficients, intercept, features):
    prediction = intercept + np.sum(np.multiply(coefficients, features))
    return prediction
