def prepare_data(
    train_data: list,
    test_data: str,
    concentration: list,
    outdir: str,
    prefix: str,
    random_state: int,
    test_size: float = 0.3,
):
    #
    # Load data
    # case 1: No testing data, using all training dataset to split
    # if have more than one train datasets, split and combine each dataset
    if len(test_concentrations) == 0 or test_rgb_path is None:
        combined_train = []
        combined_test = []
        if test_size == 1:
            for train_conv in train_concentrations:
                train = get_data(
                    rgb_path=train_rgb_path, concentration=train_conv, outdir=outdir
                )
                combined_train.append(train)
            train = pd.concat(combined_train, ignore_index=True)
            test = train
        else:
            for train_conv in train_concentrations:
                train, test, train_path, test_path = train_test_split_by_conv(
                    conv_path=train_conv,
                    rgb_path=train_rgb_path,
                    process_type=prefix,
                    test_size=test_size,
                    random_state=random_state,
                    outdir=outdir,
                )
                combined_train.append(train)
                combined_test.append(test)
            train = pd.concat(combined_train, ignore_index=True)
            test = pd.concat(combined_test, ignore_index=True)
    else:
        # case 2: Require train data, test data, not use split
        combined_train = []
        combined_test = []
        for train_conv in train_concentrations:
            train = get_data(
                rgb_path=train_rgb_path, concentration=train_conv, outdir=outdir
            )
            combined_train.append(train)
        train = pd.concat(combined_train, ignore_index=True)
        for test_conv in test_concentrations:
            test = get_data(
                rgb_path=test_rgb_path, concentration=test_conv, outdir=outdir
            )
            combined_test.append(test)
        test = pd.concat(combined_test, ignore_index=True)
    train_path = os.path.join(outdir, f"{prefix}_train.csv")
    test_path = os.path.join(outdir, f"{prefix}_test.csv")
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    return train, test, train_path, test_path


def get_rgb(indir: str, outdir: str, threshold_stdev: float) -> str:
    """Get the mean and mode value of R, G, B channel in the images in the
    directory. Then, save it to a dataframe

    Args:
        indir (str): Input directory
        outdir (str): Output directory

    Returns:
        DataFrame: Dataframe contains RGB of mean and mode value and relative
        image id
    """
    if os.path.exists(outdir):
        pass
    else:
        os.makedirs(outdir)
    input_path = os.path.join(indir, "*.csv")
    rgb_path = os.path.join(outdir, "RGB_values.csv")
    rgb_qualification_path = os.path.join(outdir, "not_RGB_qualification.csv")
    imgs_path = [
        path
        for path in glob.glob(input_path)
        if path not in [rgb_path, rgb_qualification_path]
    ]
    if len(imgs_path) == 0:
        raise ValueError(f"The directory {indir} does not contain any images RGB csv")
    with open(rgb_path, "w") as res, open(rgb_qualification_path, "w") as res_qual:
        res.write("image,meanB,meanG,meanR,modeB,modeG,modeR,stdevB,stdevG,stdevR\n")
        res_qual.write("image\n")
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
            # is_stdev_qualified
            is_stdev_qualified = (
                b_stdev < threshold_stdev
                and g_stdev < threshold_stdev
                and r_stdev < threshold_stdev
            )
            # write
            if is_stdev_qualified:
                res.write(
                    f"{img_id},{b_mean},{g_mean},{r_mean},{b_mode},{g_mode},{r_mode},{b_stdev},{g_stdev},{r_stdev}\n"
                )
            else:
                res_qual.write(f"{img_id}\n")

    return rgb_path


def get_data(rgb_path: str, concentration: str, outdir: str) -> DataFrame:
    """Combined the RGB features with their relative concentrations

    Args:
        rgb_path (str): The file path that contains the values for RGB features
        concentration (str): The file path that contains the values for concentration

    Returns:
        DataFrame: The combined dataframe that could be used for training and validating the
        machine learning models
    """
    df = pd.read_csv(rgb_path)
    conc = pd.read_csv(concentration)
    df = pd.merge(df, conc, on="image")
    df.to_csv(os.path.join(outdir, "data.csv"), index=False)
    return df
