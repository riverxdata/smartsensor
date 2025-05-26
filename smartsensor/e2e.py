from smartsensor.logger import logger
from smartsensor.model.split_data import split_data
from smartsensor.model.train import fit
from smartsensor.model.metric import evaluate_metrics, aggregate_replication_metrics
import pandas as pd
import os
import tempfile
import warnings

warnings.filterwarnings("ignore", message="X does not have valid feature names*")


def end2end_pipeline(
    data: str,
    metadata: str,
    features: str,
    degree: int,
    skip_feature_selection: bool,
    cv: int,
    outdir: str,
    prefix: str,
    test_size: float = 0.2,
    replication: int = 100,
):
    metrics = []
    features = features.split(",")
    degree = int(degree[0].value)
    data_df = pd.read_csv(data)
    meta_df = pd.read_csv(metadata)
    batches = meta_df["batch"].unique()
    meta_data = meta_df.merge(data_df, on="image")
    meta_data.reset_index(drop=True)
    with tempfile.TemporaryDirectory() as tmpdirname:
        for rep in range(1, replication + 1):
            logger.info(f"Replication {rep}")

            # split data
            train, test, prefix = split_data(
                meta_data=meta_data,
                outdir=outdir,
                prefix=prefix,
                test_size=test_size,
            )
            # train
            train_model, selected_features = fit(
                train=train,
                features=features,
                degree=degree,
                skip_feature_selection=skip_feature_selection,
                cv=cv,
                outdir=outdir,
                prefix=prefix,
            )
            # Evaluate
            train_metric, _ = evaluate_metrics(
                model=train_model,
                data=train,
                features=selected_features,
                degree=degree,
            )
            test_metric, _ = evaluate_metrics(
                model=train_model,
                data=test,
                features=selected_features,
                degree=degree,
            )

            # train res
            train_metric["data"] = "train"
            test_metric["data"] = "test"
            train_metric["features"] = ",".join(selected_features)
            test_metric["features"] = ",".join(selected_features)
            metric = pd.concat([train_metric, test_metric], axis=0)
            metrics.append(metric)

    replications_metrics = pd.concat(metrics, axis=0)
    summary_df = aggregate_replication_metrics(replications_metrics)
    summary_df.to_csv(os.path.join(outdir, "metrics.csv"), index=False)

    # use all data for training to use in real application
    # train
    train, test, prefix = split_data(
        meta_data=meta_data,
        outdir=outdir,
        prefix=prefix,
        test_size=0,
    )
    train_model, selected_features = fit(
        train=train,
        features=features,
        degree=degree,
        skip_feature_selection=skip_feature_selection,
        cv=cv,
        outdir=outdir,
        prefix=prefix,
    )
    # train res
    train_metric["data"] = "train"
    test_metric["data"] = "test"
    train_metric["features"] = ",".join(selected_features)
    test_metric["features"] = ",".join(selected_features)
    metric = pd.concat([train_metric, test_metric], axis=0)
    metrics.append(metric)
