from smartsensor.logger import logger
from smartsensor.model.split_data import split_data
from smartsensor.model.train import fit
from smartsensor.model.metric import evaluate_metrics
import pandas as pd
import os


def end2end_pipeline(
    data: str,
    metadata: str,
    features: str,
    degree: int,
    skip_feature_selection: bool,
    cv: int,
    train_batches: list,
    test_batches: list,
    outdir: str,
    prefix: str,
    test_size: float = 0.2,
):
    # split data
    train, test, prefix = split_data(
        data=data,
        metadata=metadata,
        train_batches=train_batches,
        test_batches=test_batches,
        outdir=outdir,
        prefix=prefix,
        test_size=test_size,
    )
    # train
    features = features.split(",")
    degree = int(degree[0].value)
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
    train_metric, train_detail = evaluate_metrics(
        model=train_model,
        data=train,
        features=selected_features,
        degree=degree,
    )
    test_metric, test_detail = evaluate_metrics(
        model=train_model,
        data=test,
        features=selected_features,
        degree=degree,
    )

    # train res
    train_metric["data"] = "train"
    train_detail["data"] = "train"

    # test res
    test_metric["data"] = "test"
    test_detail["data"] = "test"

    train_metric["features"] = ",".join(selected_features)
    test_metric["features"] = ",".join(selected_features)
    metric = pd.concat([train_metric, test_metric], axis=0)
    detail = pd.concat([train_detail, test_detail], axis=0)
    # export data
    metric_path = os.path.join(outdir, f"metric_{prefix}.csv")
    detail_path = os.path.join(outdir, f"detail_{prefix}.csv")
    metric.to_csv(metric_path, index=False)
    detail.to_csv(detail_path, index=False)
    return metric, detail, train_model
