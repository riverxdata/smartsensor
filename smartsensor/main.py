import os
import typer
import json
from smartsensor.logger import logger
from smartsensor.process_image import process_image
from smartsensor.e2e import end2end_pipeline
from smartsensor.predict import predict_new_data
from smartsensor.process.any2jpg import heic2jpg as convert_heic_to_jpg


app = typer.Typer(
    help="Smart Optical Sensor: Using smartphone camera as sensor", add_completion=False
)


@app.command()
def model(
    data: str = typer.Option(help="Folder contain the processed data"),
    normalization: str = typer.Option("raw", "--norm", help="Normalization method"),
    kit: str = typer.Option("1.1.0", "--kit", help="Kit normalization"),
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
        False,
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
    degree: int = typer.Option([1], "--degree", help="Degree of polynomial regression"),
    replication: int = typer.Option(100, "--replication", help="Number of replication"),
    out: str = typer.Option(".", help="Folder to save model"),
):
    """
    Run the end-to-end model for Smart Optical Sensor.
    """
    logger.info("Starting Smart Optical Sensor model training...")
    logger.info("Your configuration is belowed")
    logger.info(f"Data folder: {data}")
    logger.info(f"Normalization: {normalization}")
    logger.info(f"Features: {features}")
    logger.info(f"Skip feature selection: {skip_feature_selection}")
    logger.info(f"Cross validation: {cv}")
    logger.info(f"Test size: {test_size}")
    logger.info(f"Degree: {degree}")
    logger.info(f"Output folder: {out}")
    os.makedirs(out, exist_ok=True)
    end2end_pipeline(
        data=data,
        kit=kit,
        norm=normalization,
        features=features,
        degree=degree,
        skip_feature_selection=skip_feature_selection,
        cv=cv,
        outdir=out,
        prefix=prefix,
        test_size=test_size,
        replication=replication,
    )


@app.command()
def process(
    data: str = typer.Option(
        None,
        help="Path to the image or folder of raw images",
    ),
    outdir: str = typer.Option(
        ".",
        help="Folder to save processed images",
    ),
    process_dir: str = typer.Option(
        None,
        "--process-dir",
        help="Path to the processed to get config for base background color",
    ),
    kit: str = typer.Option(
        "1.1.0",
        help="The kit that has been used to capture image. By default, kit is used for ampiciline dataset",
    ),
    auto_lum: bool = typer.Option(
        False,
        "--auto-lum",
        help="Automatically calculate luminance from background images",
    ),
):
    """
    Normalize the images to standardize RGB features.
    """
    lum = None
    if process_dir:
        with open(os.path.join(process_dir, "config.json"), "r") as f:
            process_config = json.load(f)
            lum = process_config.get("lum")

    process_image(
        data=data,
        outdir=outdir,
        kit=kit,
        auto_lum=auto_lum,
        lum=lum,
    )


@app.command()
def heic2jpg(
    data: str = typer.Option(help="Path to the image or folder of raw images"),
):
    """
    Convert HEIC images to JPG format.
    """
    convert_heic_to_jpg(folder_path=data)


@app.command()
def predict(
    processed_dir: str = typer.Option(
        "--processed-dir",
        help="Path to the processed to get config for base background color",
    ),
    model_dir: str = typer.Option("--model-dir", help="Path to the model result"),
    outdir: str = typer.Option("--outdir", help="Path to the output directory"),
):
    """
    Normalize the images to standardize RGB features.
    """
    predict_new_data(
        processed_dir=processed_dir,
        model_dir=model_dir,
        outdir=outdir,
    )


if __name__ == "__main__":
    app()
