from smartsensor.base import (
    end2end_model,
)
import glob
import os
import pandas as pd

# params
# Summary: The data is CuSO4 focus which means the camera is focus on the CuSO4 solution to capture the image
data_path = "EDA/Fe3+"
indir = f"{data_path}/raw_data"
test_size = 0.3
batches = ["batch1", "batch2", "batch3"]

# without filter
process_outdir = f"{data_path}/process_data_not_filter"
# QUESTION: DOES THE NORMALIZATION AFFECT THE RESULT?
# 1. Combine 3 batches into one models. Using train test split for each dataset and each concentration. Due to random
# split #noqa
# Repeate by N time for measuring the variance
N = 5
# RAW
# Change here
raw_res = []
process_type = "raw_roi"
prefix = "raw_roi"
# Stable
indir_e2e = f"{process_outdir}/{process_type}"
outdir_e2e = f"{data_path}/result_without_filter_and_feature_selection_3_batches_train_test_split"
features = "meanR,meanG,meanB,modeR,modeB,modeG"
degree = 1
outdir = os.path.join(outdir_e2e, process_type, f"repeat_{N}_times")
os.makedirs(outdir, exist_ok=True)
# repeat here
for i in range(1, N + 1):
    outdir_n = os.path.join(outdir, f"repeat_{i}")
    # raw_data
    train_rgb_path = f"{indir_e2e}/RGB_values.csv"
    train_concentrations = [f"{indir}/{b}.csv" for b in batches]
    # test
    test_rgb_path = None
    test_concentrations = []
    metric, detail = end2end_model(
        train_rgb_path,
        train_concentrations,
        test_rgb_path,
        test_concentrations,
        features,
        degree,
        outdir_n,
        prefix,
        test_size=test_size,
        random_state=None,
    )
    metric = metric[metric["train_data"] != metric["test_data"]]
    metric["prefix"] = f"repeat_{i}"
    raw_res.append(metric)
# Change here
pd.DataFrame(pd.concat(raw_res)).to_csv(
    f"{outdir_e2e}/result_raw_3_batches_train_test_split.csv", index=False
)

# DELTA
raw_res = []
process_type = "delta_normalized_roi"
prefix = "delta_roi"
# Stable
indir_e2e = f"{process_outdir}/{process_type}"
outdir_e2e = f"{data_path}/result_without_filter_and_feature_selection_3_batches_train_test_split"
features = "meanR,meanG,meanB,modeR,modeB,modeG"
degree = 1
outdir = os.path.join(outdir_e2e, process_type, f"repeat_{N}_times")
os.makedirs(outdir, exist_ok=True)
# repeat here
for i in range(1, N + 1):
    outdir_n = os.path.join(outdir, f"repeat_{i}")
    # raw_data
    train_rgb_path = f"{indir_e2e}/RGB_values.csv"
    train_concentrations = [f"{indir}/{b}.csv" for b in batches]
    # test
    test_rgb_path = None
    test_concentrations = []
    metric, detail = end2end_model(
        train_rgb_path,
        train_concentrations,
        test_rgb_path,
        test_concentrations,
        features,
        degree,
        outdir_n,
        prefix,
        test_size=test_size,
        random_state=None,
    )
    metric = metric[metric["train_data"] != metric["test_data"]]
    metric["prefix"] = f"repeat_{i}"
    raw_res.append(metric)
# Change here
pd.DataFrame(pd.concat(raw_res)).to_csv(
    f"{outdir_e2e}/result_delta_3_batches_train_test_split.csv", index=False
)

# RATIO
raw_res = []
process_type = "ratio_normalized_roi"
prefix = "ratio_roi"
# Stable
indir_e2e = f"{process_outdir}/{process_type}"
outdir_e2e = f"{data_path}/result_without_filter_and_feature_selection_3_batches_train_test_split"
features = "meanR,meanG,meanB,modeR,modeB,modeG"
degree = 1
outdir = os.path.join(outdir_e2e, process_type, f"repeat_{N}_times")
os.makedirs(outdir, exist_ok=True)
# repeat here
for i in range(1, N + 1):
    outdir_n = os.path.join(outdir, f"repeat_{i}")
    # raw_data
    train_rgb_path = f"{indir_e2e}/RGB_values.csv"
    train_concentrations = [f"{indir}/{b}.csv" for b in batches]
    # test
    test_rgb_path = None
    test_concentrations = []
    metric, detail = end2end_model(
        train_rgb_path,
        train_concentrations,
        test_rgb_path,
        test_concentrations,
        features,
        degree,
        outdir_n,
        prefix,
        test_size=test_size,
        random_state=None,
    )
    metric = metric[metric["train_data"] != metric["test_data"]]
    metric["prefix"] = f"repeat_{i}"
    raw_res.append(metric)
# Change here
pd.DataFrame(pd.concat(raw_res)).to_csv(
    f"{outdir_e2e}/result_ratio_3_batches_train_test_split.csv", index=False
)

# Integrate the result
metrics = glob.glob(f"{outdir_e2e}/*3_batches_train_test_split.csv")
sum_stats = []
for m in metrics:
    prepare_stat = pd.read_csv(m)
    # only capture the test dataset
    prepare_stat = prepare_stat[prepare_stat["train_data"] != prepare_stat["test_data"]]
    stats = prepare_stat.describe().reset_index().rename(columns={"index": "metric"})
    stats["process_type"] = m.split("/")[-1].split(".")[0].split("_")[1]
    sum_stats.append(stats)
    prepare_export = pd.concat(sum_stats).sort_values("metric")
    export = prepare_export[
        prepare_export["metric"].isin(["mean", "min", "max", "std"])
    ].reset_index(drop=True)
    export.to_csv(f"{outdir_e2e}/sum_stats.csv", index=False)
print(export[export["metric"] == "mean"])
