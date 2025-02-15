import typer
import os
from typing_extensions import Annotated

from smartsensor.const import Degree, NormalizeMethod
from smartsensor.logger import logger
from smartsensor.e2e import end2end_pipeline

app = typer.Typer(help="Smart Optical Sensor")


@app.command()
def run(
    train_image_data: str = typer.Option(help="Image train data folder"),
    test_image_data: str = typer.Option(None, help="Image test data folder"),
    concentration: str = typer.Option(
        help="Image concentration for both train and test data"
    ),
    feature: str = typer.Option(
        "meanR,meanG,meanG,modeR,modeG,modeB",
        "--feature",
        help="Features used in model, separated by commas in string",
    ),
    test_size: float = typer.Option(
        None, "--test-size", help="Test data size for splitting to train the model"
    ),
    norm: list[NormalizeMethod] = typer.Option(
        [NormalizeMethod.non],
        "--norm",
        help="Normalization method.",
    ),
    degree: list[Degree] = typer.Option(
        [Degree.first], "--degree", help="Degree of polynomial regression"
    ),
    out: str = typer.Option(".", help="Folder to save model"),
):
    """
    Run the end-to-end model for Smart Optical Sensor.
    """
    logger.info("Starting Smart Optical Sensor model training...")
    logger.info("Your configuration is belowed")
    logger.info(f"Train data folder: {train_data}")
    logger.info(f"Features: {feature}")
    logger.info(f"Test size: {test_size}")
    logger.info(f"Normalization: {norm}")
    logger.info(f"Degree: {degree}")
    logger.info(f"Output folder: {out}")

    if not os.path.exists(out):
        logger.info(f"Output folder '{out}' does not exist!")
        os.makedirs(out)

    # Call your model function here
    end2end_model(data, batch, feature, test_size, norm, degree, out)

    logger.info("Model execution completed.")


if __name__ == "__main__":
    app()
