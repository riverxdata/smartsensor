from smartsensor.process.validate import validate_data
from smartsensor.process.data import prepare_data
from smartsensor.logger import logger


def end2end_pipeline(
    train_data: str,
    test_data: str,
    concentration: list,
    features: str,
    degree: int,
    outdir: str,
    prefix: str,
    skip_feature_selection: bool = True,
    test_size: float = 0.3,
    random_state: int = 1,
    cv: int = 5,
):
    # check concentration file
    if not os.path.exists(concentration):
        raise ValueError(f"Missing the concentrationn file at {""}")

    train, test, train_path, test_path = prepare_data(
        train_data=train_data,
        test_data=test_data,
        concentration=concentration,
        outdir=outdir,
        prefix=prefix,
        test_size=test_size,
        random_state=random_state,
    )
    # # Train
    # features = features.split(",")
    # train_model, selected_features = train_regression(
    #     train=train,
    #     features=features,
    #     degree=degree,
    #     outdir=outdir,
    #     prefix=prefix,
    #     skip_feature_selection=skip_feature_selection,
    #     cv=cv,
    # )
    # # Evaluate
    # train_metric, train_detail = evaluate_metrics(
    #     model=train_model,
    #     data=train,
    #     features=selected_features,
    #     degree=degree,
    # )
    # test_metric, test_detail = evaluate_metrics(
    #     model=train_model,
    #     data=test,
    #     features=selected_features,
    #     degree=degree,
    # )

    # if test_concentrations is None or test_rgb_path is None:
    #     train_dataset = f"{'_'.join([os.path.basename(train_concentration).split('.')[0] for train_concentration in train_concentrations])}_train_by_split_{str(1-test_size)}"  # noqa
    #     test_dataset = f"{'_'.join([os.path.basename(train_concentration).split('.')[0] for train_concentration in train_concentrations])}_test_by_split_{test_size}"  # noqa
    # else:
    #     train_dataset = f"{'+'.join([os.path.basename(train_concentration).split('.')[0] for train_concentration in train_concentrations])}"  # noqa
    #     test_dataset = f"{'+'.join([os.path.basename(test_concentration).split('.')[0] for test_concentration in test_concentrations])}"  # noqa
    # # train res
    # train_metric["train_data"] = train_dataset
    # train_metric["test_data"] = train_dataset
    # train_detail["train_data"] = train_dataset
    # train_detail["test_data"] = train_dataset
    # # test res
    # test_metric["train_data"] = train_dataset
    # test_metric["test_data"] = test_dataset
    # test_detail["train_data"] = train_dataset
    # test_detail["test_data"] = test_dataset

    # train_metric["features"] = ",".join(selected_features)
    # test_metric["features"] = ",".join(selected_features)
    # metric = pd.concat([train_metric, test_metric], axis=0)
    # detail = pd.concat([train_detail, test_detail], axis=0)
    # # Export data
    # metric_path = os.path.join(outdir, f"metric_{train_dataset}&{test_dataset}.csv")
    # detail_path = os.path.join(outdir, f"detail_{train_dataset}&{test_dataset}.csv")
    # metric.to_csv(metric_path, index=False)
    # detail.to_csv(detail_path, index=False)
    # return metric, detail, train_model
