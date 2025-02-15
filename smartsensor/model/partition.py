# Split the images by train test split
def train_test_split_by_conv(
    conv_path: str,
    rgb_path: str,
    process_type: str,
    test_size: float,
    random_state: int,
    outdir: str,
):
    os.makedirs(outdir, exist_ok=True)
    data = get_data(rgb_path=rgb_path, concentration=conv_path, outdir=outdir)

    # List of unique concentrations in your DataFrame
    concentrations = data["concentration"].unique()
    # Initialize empty lists to store the splits
    (
        image_train_list,
        image_test_list,
        concentration_train_list,
        concentration_test_list,
    ) = ([], [], [], [])
    # Iterate through each concentration
    for concentration in concentrations:
        # Filter the DataFrame for the current concentration
        subset_conv = data[data["concentration"] == concentration]
        x_data = subset_conv.drop(columns=["concentration"])
        y_data = subset_conv["concentration"]
        if random_state is not None:
            (
                image_train,
                image_test,
                concentration_train,
                concentration_test,
            ) = train_test_split(
                x_data,
                y_data,
                test_size=test_size,
                random_state=1,
            )
        else:
            (
                image_train,
                image_test,
                concentration_train,
                concentration_test,
            ) = train_test_split(
                x_data,
                y_data,
                test_size=test_size,
            )

        # Append the splits to the lists
        image_train_list.append(image_train)
        image_test_list.append(image_test)
        concentration_train_list.append(concentration_train)
        concentration_test_list.append(concentration_test)

    # Concatenate the splits to obtain the final training and testing sets
    final_image_train = pd.concat(image_train_list)
    final_image_test = pd.concat(image_test_list)
    final_concentration_train = pd.concat(concentration_train_list)
    final_concentration_test = pd.concat(concentration_test_list)
    # Create the final training and testing sets
    final_train_data = pd.concat([final_image_train, final_concentration_train], axis=1)
    final_test_data = pd.concat([final_image_test, final_concentration_test], axis=1)
    # Save the training and testing sets
    train_path = os.path.join(outdir, f"{process_type}_train.csv")
    test_path = os.path.join(outdir, f"{process_type}_test.csv")
    final_train_data.to_csv(train_path, index=False)
    final_test_data.to_csv(test_path, index=False)
    return final_train_data, final_test_data, train_path, test_path
