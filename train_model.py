import argparse
import os
import pandas as pd
import pickle

from smartsensor.base import end2end_model


def cross_validate(indir,
                   outdir,
                   process_outdir,
                   norm_types = ["raw_roi", "ratio_normalized_roi", "delta_normalized_roi"],
                   degrees = [1, 2],
                   ignore_feature_selection = [True, False],
                   batches = ["batch1", "batch2", "batch3", "batch4", "batch5"],
                   test_size = 0.2,
                   features = "meanR,meanG,meanB,modeR,modeB,modeG"):

    raw_res = []

    for norm in norm_types:
        for degree in degrees:
            for ignore in ignore_feature_selection:
                outdir = os.path.join(
                    outdir,
                    norm + "_degree_" + str(degree) + "_ignore_" + str(ignore),
                )
                os.makedirs(outdir, exist_ok=True)
                # train will be used to test again, cv defined inside with 20 images per cross validation
                indir_e2e = f"{process_outdir}/{norm}"
                train_rgb_path = f"{indir_e2e}/RGB_values.csv"
                train_concentrations = [f"{indir}/{b}.csv" for b in batches]
                test_rgb_path = train_rgb_path
                test_concentrations = train_concentrations
                metric, detail, _ = end2end_model(
                    train_rgb_path,
                    train_concentrations,
                    test_rgb_path,
                    test_concentrations,
                    features,
                    degree,
                    outdir,
                    norm,
                    test_size=test_size,
                    random_state=1,
                    skip_feature_selection=ignore,
                )
                metric["process_type"] = norm
                metric["degree"] = degree
                metric["ignore_feature_selection"] = ignore
                raw_res.append(metric)

    # save result
    df = pd.DataFrame(pd.concat(raw_res))
    df.to_csv(f"{outdir}/result.csv", index=False)

    # find the best parameters
    best_param = df[df.r2 == df.r2.max()].iloc[0]

    return best_param


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Smart Optical Sensor")
    parser.add_argument('--data', help='data folder', default="EDA/CuSO4")
    parser.add_argument('--batch', help='data batch', default="batch1,batch2,batch3,batch4,batch5")
    parser.add_argument('--feature', help='feature used in model', default="meanR,meanG,meanB,modeR,modeB,modeG")
    parser.add_argument('--testsize', help='test data size', default=0.2)
    parser.add_argument('--norm',help='list of data normalization',default='raw,delta,ratio')
    parser.add_argument('--degree', help='list of degree of poly-regression', default='1,2')
    parser.add_argument('--out', help='folder to save model', default='.')

    args = parser.parse_args()
    process_types = {"raw":"raw_roi", "ratio":"ratio_normalized_roi", "delta":"delta_normalized_roi"}
    batch = args.batch.split(',')
    norm_type = [process_types[k] for k in args.norm.split(',')]
    degree = [int(v) for v in args.degree.split(',')]

    data_path = args.data
    indir = f"{data_path}/raw_data"
    outdir_e2e = f"{data_path}/result_turning"
    process_outdir = f"{data_path}/process_data_not_filter"

    best_param = cross_validate(indir=indir,
                                outdir=outdir_e2e,
                                process_outdir=process_outdir,
                                norm_types=norm_type,
                                degrees=degree,
                                batches=batch,
                                test_size=float(args.testsize),
                                features=args.feature)

    print("Best parameters")
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
    train_concentrations = [f"{indir}/{b}.csv" for b in batch]
    test_rgb_path = train_rgb_path
    test_concentrations = train_concentrations

    metric, detail, model = end2end_model(
        train_rgb_path,
        train_concentrations,
        test_rgb_path,
        test_concentrations,
        args.feature,
        degree,
        outdir,
        process_type,
        test_size=float(args.testsize),
        random_state=1,
        skip_feature_selection=ignore,
    )

    # save
    with open(f'{args.out}/model.pkl', 'wb') as f:
        pickle.dump(model, f)