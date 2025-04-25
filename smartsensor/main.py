import typer
import os
from typing_extensions import Annotated

from smartsensor.const import Degree, NormalizeMethod
from smartsensor.logger import logger
from smartsensor.process_image import process_image
from smartsensor.e2e import end2end_pipeline


app = typer.Typer(
    help="Smart Optical Sensor: Using smartphone camera as sensor", add_completion=False
)


@app.command()
def model(
    data: str = typer.Option(help="Csv file, output by the process"),
    meta: str = typer.Option(
        help="The metadata file, contains image, relative concentration, relative batch",
    ),
    prefix: str = typer.Option(
        None,
        "--prefix",
        help="Output prefix",
    ),
    features: str = typer.Option(
        "meanR,meanG,meanG,modeR,modeG,modeB",
        "--features",
        help="Features used in model, separated by commas in string",
    ),
    skip_feature_selection: bool = typer.Option(
        True,
        "--skip-feature-selection",
        help="Skip feature selection",
    ),
    cv: int = typer.Option(
        5,
        "--cv",
        help="Fold for cross validation",
    ),
    test_size: float = typer.Option(
        0.2, "--test-size", help="Test data size for splitting to train the model"
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
    logger.info(f"Data folder: {data}")
    logger.info(f"Metadata: {meta}")
    logger.info(f"Features: {features}")
    logger.info(f"Skip feature selection: {skip_feature_selection}")
    logger.info(f"Cross validation: {cv}")
    logger.info(f"Test size: {test_size}")
    logger.info(f"Normalization: {norm}")
    logger.info(f"Degree: {degree}")
    logger.info(f"Output folder: {out}")

    end2end_pipeline(
        data=data,
        metadata=meta,
        features=features,
        degree=degree,
        skip_feature_selection=skip_feature_selection,
        cv=cv,
        outdir=out,
        prefix=prefix,
        test_size=test_size,
    )


@app.command()
def process(
    data: str = typer.Option(help="Path to the image or folder of raw images"),
    outdir: str = typer.Option(".", help="Folder to save processed images"),
    kit: str = typer.Option(
        "1.1.0",
        help="The kit that has been used to capture image. By default, kit is used for ampiciline dataset",
    ),
):
    """
    Normalize the images to standardize RGB features.
    """
    process_image(data=data, outdir=outdir, kit=kit)


if __name__ == "__main__":
    app()
