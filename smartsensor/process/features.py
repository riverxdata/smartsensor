import os
import glob
import numpy as np
import pandas as pd
import statistics as st


def get_features(outdir: str) -> str:
    """
    Extracts features from RGB image CSVs and saves to a summarized CSV file for each directory.

    Args:
        outdir (str): Path to the output directory.

    Raises:
        ValueError: If no CSV files are found in the expected subdirectories.
    """
    data_dirs = [
        "delta_normalized_roi",
        "ratio_normalized_roi",
        "raw_normalized_roi",
    ]

    for d_dir in data_dirs:
        processed_outdir = os.path.join(outdir, d_dir)
        input_path = os.path.join(processed_outdir, "*.csv")
        rgb_path = os.path.join(outdir, f"features_rgb_{d_dir}.csv")
        imgs_path = glob.glob(input_path)

        if not imgs_path:
            raise ValueError(f"No CSV files found in {input_path}")

        rows = []

        for img_path in imgs_path:
            img_id = os.path.basename(img_path.replace(".csv", ".jpg"))
            img = pd.read_csv(img_path)
            b, g, r = img["B"], img["G"], img["R"]

            def safe_mode(channel):
                try:
                    return st.mode(channel)
                except:  # noqa
                    return channel.iloc[0]  # fallback to first value

            rows.append(
                {
                    "image": img_id,
                    "meanB": np.mean(b),
                    "meanG": np.mean(g),
                    "meanR": np.mean(r),
                    "modeB": safe_mode(b),
                    "modeG": safe_mode(g),
                    "modeR": safe_mode(r),
                    "stdevB": np.std(b),
                    "stdevG": np.std(g),
                    "stdevR": np.std(r),
                }
            )

        # Create DataFrame and add two extra columns
        data_df = pd.DataFrame(rows)
        data_df["batch"] = data_df["image"].apply(
            lambda x: x.split("_")[-1].split(".jpg")[0]
        )
        data_df["concentration"] = data_df["image"].apply(lambda x: x.split("-")[0])

        # Write to CSV
        data_df.to_csv(rgb_path, index=False)

    return outdir
