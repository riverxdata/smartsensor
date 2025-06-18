from smartsensor.process.roi import get_r2frame, get_roi_background
from smartsensor.process.normalize import normalize
from smartsensor.process.features import get_features
from smartsensor.logger import logger


def process_image(data: str, outdir: str, kit: str = "1.1.0", lum=None) -> None:
    """Processing image

    Args:
        data (str): Path to the image or folder of raw images
        outdir (str): Folder to save processed images
        kit (str, optional): Kit version to get relative threshol. Defaults to "1.1.0".

    """
    # get frame
    logger.info("Starting normalize data")
    logger.info("Getting square frame")
    r2frame_outdir = get_r2frame(
        data=data,
        outdir=outdir,
        kit=kit,
    )
    logger.info("Complete get square frame")

    # get roi and background
    logger.info("Get roi and background")
    raw_roi_path, bg_path = get_roi_background(
        data=r2frame_outdir,
        outdir=outdir,
    )
    logger.info("Complete get the background")

    # balance
    logger.info("Balance image")
    if lum:
        logger.info(f"Use lum from argument:{lum}")
        normalize(
            raw_roi=raw_roi_path,
            background=bg_path,
            outdir=outdir,
            lum=lum,
        )
    else:
        normalize(
            raw_roi=raw_roi_path,
            background=bg_path,
            outdir=outdir,
        )

    logger.info("Complete balance image")
    # extract features
    logger.info("Extract feature")
    get_features(outdir=outdir)
    logger.info("Complete extrac feature")
