from smartsensor.process.roi import get_r2frame, get_roi_background
from smartsensor.process.normalize import normalize
from smartsensor.process.features import get_features
from smartsensor.logger import logger


def process_image(
    data: str, outdir: str, lum: list, kit: str = "1.1.0", auto_lum: bool = False
) -> None:
    """Processing image

    Args:
        data (str): Path to the image or folder of raw images
        outdir (str): Folder to save processed images
        lum (list, optional): List of luminance values. Defaults to [].
        kit (str, optional): Kit version to get relative threshol. Defaults to "1.1.0".
        auto_lum: (bool, optional): Automatically calculate luminance from background images. Defaults to False.


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
    normalize(
        raw_roi=raw_roi_path,
        background=bg_path,
        kit=kit,
        outdir=outdir,
        auto_lum=auto_lum,
        lum=lum,
    )

    logger.info("Complete balance image")
    # extract features
    logger.info("Extract feature")
    get_features(outdir=outdir)
    logger.info("Complete extrac feature")
