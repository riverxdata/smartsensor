import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import os


def linear_regression_visualization(
    df: pd.DataFrame, feature: str, target: str, train_dataset: str, outdir: str
) -> None:
    """
    Visualize linear regression for a feature and target in a DataFrame.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - feature (str): The name of the feature variable.
    - target (str): The name of the target variable.
    - train_dataset (str): The name train_dataset
    - outdir (str): The directory to save the visualization.

    Returns:
    - None
    """
    # Fit a linear regression model
    model = LinearRegression()
    model.fit(df[[feature]], df[target])

    # Predictions
    df[f"predicted_{target}"] = model.predict(df[[feature]])

    # Calculate R-squared
    r_squared = r2_score(df[target], df[f"predicted_{target}"])

    # Plot the data with regression line and R-squared value
    plt.figure(figsize=(6, 5))
    plt.rcParams["font.family"] = "Helvetica"
    sns.scatterplot(
        x=feature,
        y=target,
        data=df,
        color="black",
        edgecolor="black",
        s=50,
    )
    sns.lineplot(
        x=feature,
        y=f"predicted_{target}",
        data=df,
        color="red",
        label=f"Regression Line (R²={r_squared:.2f})",
    )
    plt.title("Data with Linear Regression Line", fontsize=16)
    plt.ylabel(target.capitalize(), fontsize=16)
    # Adjust font size of legend
    plt.legend(fontsize=12)
    plt.ylabel(target.capitalize())
    train_dataset = os.path.basename(train_dataset).split(".")[0]
    plt.savefig(f"{outdir}/{train_dataset}_{feature}_linear_regression.pdf")
    plt.savefig(f"{outdir}/{train_dataset}_{feature}_linear_regression.png")
