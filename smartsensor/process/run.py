import math
import numpy as np


def processing_images(
    data: str,
    outdir: str,
    threshold: list = [(0, 120, 20), (150, 230, 80)],
    dim: list = [740, 740],
    bg_index: list = [50, 60, 350, 360],
    roi_index: list = [-250, -200, 345, -345],
    constant: list = [100, 150, 40],
    feature_method: Callable[[Any], float] = np.mean,
    overwrite: bool = True,
    threshold_stdev: float = 5,
    threshold_ratio: float = math.log(1.2, 2),
    threshold_delta: float = 20,
) -> None:
    """Using the images in raw format, convert to csv, then normalize the images and save to
    csv format

    Args:
        indir (str): Input directory
        outdir (str): Output directory
        threshold (list, optional): Threshold for cutting edge
        Defaults to [(0, 120, 20), (150, 230, 80)].
        dim (list, optional): Dimension for scale the ROI. Defaults to [740, 740].
        bg_index (list, optional): The background index to cut. Defaults to [50, 60, 350, 360].
        roi_index (int, optional): The roi region boudary. Defaults to 245.
        constant (list, optional): The constant for background scale. Defaults to [60, 90, 30].
        feature_method (Callable[[Any], float], optional): feature method. Defaults to np.mean.
    """
    if os.path.exists(os.path.join(outdir, "raw_roi")) and not overwrite:
        print("Already processed, remove outdir to reprocess")
        return True
    else:
        # Step 1: Extract ROI, background, and squared_frame return RGB value
        image_segmentations(
            indir=indir,
            outdir=outdir,
            threshold=threshold,
            dim=dim,
            bg_index=bg_index,
            roi_index=roi_index,
        )
        print("Complete extract ROI")
        # Step 2: Balance images by normalizing using the background color and saving results
        balance_image(
            indir=indir,
            outdir=outdir,
            constant=constant,
            feature_method=feature_method,
            threshold_ratio=threshold_ratio,
            threshold_delta=threshold_delta,
        )
        print("Complete balance images")
        # Step3: Get the RGB value
        for feature_type in ["raw_roi", "ratio_normalized_roi", "delta_normalized_roi"]:
            get_rgb(
                indir=os.path.join(outdir, feature_type),
                outdir=os.path.join(outdir, feature_type),
                threshold_stdev=threshold_stdev,
            )
