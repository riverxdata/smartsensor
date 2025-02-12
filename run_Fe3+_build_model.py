from smartsensor.base import (
    end2end_model,
)
import os
import pandas as pd

# params
# Summary: The data is CuSO4 focus which means the camera is focus on the CuSO4 solution to capture the image
data_path = "EDA/Fe3+"
indir = f"{data_path}/raw_data"
test_size = 0.2
batches = ["batch1", "batch2", "batch3"]

# without filter
process_outdir = f"{data_path}/process_data_not_filter"
# QUESTION: DOES THE NORMALIZATION AFFECT THE RESULT?
raw_res = []
# Stable
outdir_e2e = f"{data_path}/result_turning"
features = "meanR,meanG,meanB,modeR,modeB,modeG"
# matrix
process_types = ["raw_roi", "ratio_normalized_roi", "delta_normalized_roi"]
degrees = [1, 2]
ignore_feature_selection = [True, False]
for process_type in process_types:
    for degree in degrees:
        for ignore in ignore_feature_selection:
            outdir = os.path.join(
                outdir_e2e,
                process_type + "_degree_" + str(degree) + "_ignore_" + str(ignore),
            )
            os.makedirs(outdir, exist_ok=True)
            # train will be used to test again, cv defined inside with 20 images per cross validation
            indir_e2e = f"{process_outdir}/{process_type}"
            train_rgb_path = f"{indir_e2e}/RGB_values.csv"
            train_concentrations = [f"{indir}/{b}.csv" for b in batches]
            test_rgb_path = train_rgb_path
            test_concentrations = train_concentrations
            metric, detail = end2end_model(
                train_rgb_path,
                train_concentrations,
                test_rgb_path,
                test_concentrations,
                features,
                degree,
                outdir,
                process_type,
                test_size=test_size,
                random_state=1,
                skip_feature_selection=ignore,
            )
            metric["process_type"] = process_type
            metric["degree"] = degree
            metric["ignore_feature_selection"] = ignore
            raw_res.append(metric)
# save result
df = pd.DataFrame(pd.concat(raw_res))
df.to_csv(f"{outdir_e2e}/result.csv", index=False)

print("Best parameters:")
best_param = df[df.r2 == df.r2.max()].iloc[0]
print(best_param)

# build model with the best parameters
process_type = best_param["process_type"]
degree = best_param["degree"]
ignore = best_param["ignore_feature_selection"]

outdir = os.path.join(
                outdir_e2e,
                "best_param",
            )
os.makedirs(outdir, exist_ok=True)
# train will be used to test again, cv defined inside with 20 images per cross validation
indir_e2e = f"{process_outdir}/{process_type}"
train_rgb_path = f"{indir_e2e}/RGB_values.csv"
train_concentrations = [f"{indir}/{b}.csv" for b in batches]
test_rgb_path = train_rgb_path
test_concentrations = train_concentrations
metric, detail = end2end_model(
    train_rgb_path,
    train_concentrations,
    test_rgb_path,
    test_concentrations,
    features,
    degree,
    outdir,
    process_type,
    test_size=test_size,
    random_state=1,
    skip_feature_selection=ignore,
)
