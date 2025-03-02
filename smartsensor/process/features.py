import os
import glob
import numpy as np
import pandas as pd
import statistics as st


def get_features(outdir: str) -> str:
    """_summary_

    Args:
        outdir (str): _description_

    Raises:
        ValueError: _description_


    """
    # data
    data_dirs = [
        "delta_normalized_roi",
        "ratio_normalized_roi",
        "raw_roi",
    ]
    for d_dir in data_dirs:
        processed_outdir = os.path.join(outdir, d_dir)
        input_path = os.path.join(processed_outdir, "*.csv")
        rgb_path = os.path.join(outdir, f"features_rgb_{d_dir}.csv")
        imgs_path = glob.glob(input_path)
        if len(imgs_path) == 0:
            raise ValueError(
                f"The directory {input_path} does not contain any images RGB csv"
            )
        with open(rgb_path, "w") as res, open(rgb_path, "w") as res_qual:
            res.write(
                "image,meanB,meanG,meanR,modeB,modeG,modeR,stdevB,stdevG,stdevR\n"
            )
            for img_path in imgs_path:
                img_id = os.path.basename(img_path.replace(".csv", ".jpg"))
                # Load image
                img = pd.read_csv(img_path)
                b, g, r = img["B"], img["G"], img["R"]
                # mean
                b_mean = np.mean(b)
                g_mean = np.mean(g)
                r_mean = np.mean(r)
                # mode
                b_mode = st.mode(b)
                g_mode = st.mode(g)
                r_mode = st.mode(r)
                # stdev
                b_stdev = np.std(b)
                g_stdev = np.std(g)
                r_stdev = np.std(r)
                # write
                res.write(
                    f"{img_id},{b_mean},{g_mean},{r_mean},{b_mode},{g_mode},{r_mode},{b_stdev},{g_stdev},{r_stdev}\n"
                )
