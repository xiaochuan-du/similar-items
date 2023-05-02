import argparse
import json
import os
import tarfile
from io import StringIO

import joblib
import pandas as pd
import numpy as np
from numpy import dtype

from sklearn.model_selection import train_test_split


from config import (
    raw_feature_columns_names,
    raw_feature_columns_dtype,
    label_column,
    label_column_dtype,
    feature_columns_dtype,
    feature_columns,
    output_raw_columns,
    output_engineered_feature_columns,
)
from utils import generate_features

try:
    from sagemaker_containers.beta.framework import (
        worker,
    )
except ImportError:
    pass


def input_fn(input_data, content_type):
    """Parse input data payload

    We currently only take csv input. Since we need to process both labelled
    and unlabelled data we first determine whether the label column is present
    by looking at how many columns were provided.
    """
    if content_type == "text/csv":
        # Read the raw input data as CSV.
        df = pd.read_csv(StringIO(input_data), header=None)

        if len(df.columns) == len(raw_feature_columns_names) + 1:
            # This is a labelled example, includes the ring label
            df.columns = raw_feature_columns_names + [label_column]
        elif len(df.columns) == len(raw_feature_columns_names):
            # This is an unlabelled example.
            df.columns = raw_feature_columns_names
        else:
            raise ValueError(f"Number of columns [{len(df.columns)}] mismatch!")

        return df
    else:
        raise ValueError("{} not supported by script!".format(content_type))


def output_fn(features, accept):
    """Format prediction output

    The default accept/content-type between containers for serial inference is JSON.
    We also want to set the ContentType or mimetype as the same value as accept so the next
    container can read the response payload correctly.
    """
    if accept == "application/json":
        # choose the text field
        instances = features[feature_columns[0]].tolist()
        json_output = {"inputs": instances}
        return worker.Response(json.dumps(json_output), mimetype=accept)
    else:
        raise Exception(
            "{} accept type is not supported by this script.".format(accept)
        )


def model_fn(model_dir):
    """Deserialize fitted model"""
    return None


def predict_fn(input_data, _):
    """Preprocess input data

    We implement this because the default predict_fn uses .predict(), but our model is a preprocessor
    so we want to use .transform().

    The output is returned in the following order:

        rest of features either one hot encoded or standardized
    """
    return generate_features(input_data)


if __name__ == "__main__":
    input_dir = "/opt/ml/processing/input"
    base_dir = "/opt/ml/processing"

    # Take the set of files and read them all into a single pandas dataframe
    input_files = [
        os.path.join(input_dir, file)
        for file in os.listdir(input_dir)
        if file.endswith(".csv")
    ]
    raw_data = [
        pd.read_csv(file, dtype={**raw_feature_columns_dtype, **label_column_dtype})
        for file in input_files
    ]
    data = pd.concat(raw_data).reset_index(drop=True)

    # ignore the rows which titles are empty
    data = data[~data.title.isna()]
    data["description"] = data.description.fillna(value="")

    # cut title and decription to a predefined threshold.
    max_sequence_len = 500
    data.loc[data.title.str.len() > max_sequence_len, "title"] = data[
        data.title.str.len() > max_sequence_len
    ].title.str[0:max_sequence_len]
    data.loc[data.description.str.len() > max_sequence_len, "description"] = data[
        data.description.str.len() > max_sequence_len
    ].description.str[0:max_sequence_len]

    train_sz = int(len(data) * 0.8)
    val_sz = int(len(data) * 0.1)
    test_sz = len(data) - train_sz - val_sz

    X_train_val, X_test_raw, y_train_val, y_test = train_test_split(
        data[raw_feature_columns_names],
        data[label_column].values,
        random_state=42,
        test_size=test_sz,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, random_state=42, test_size=val_sz
    )

    print(f"X_train shape: {X_train.shape} y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape} y_val shape: {y_val.shape}")
    print(f"X_test_raw shape: {X_test_raw.shape} y_test shape: {y_test.shape}")

    X_train_raw = X_train.copy()
    X_train = generate_features(X_train)
    X_val = generate_features(X_val)

    nan_shape = X_train[X_train.text.isna()].shape
    print(f"X_train NaN shape: {nan_shape}")

    # store raw training data and feature engineered training data
    # TODO order🤦 !!!!!!!!!!!!!!!
    number_of_chunks = 100
    for idx, chunk in enumerate(
        np.array_split(
            X_train_raw.assign(**{label_column: y_train})[output_raw_columns],
            number_of_chunks,
        )
    ):
        chunk.to_csv(f"{base_dir}/train_raw/train_{idx}.csv", header=True, index=False)

    X_train.assign(**{label_column: y_train})[output_engineered_feature_columns].to_csv(
        f"{base_dir}/train/train.csv", header=True, index=False
    )

    # store feature engineered validation data
    X_val.assign(**{label_column: y_val})[output_engineered_feature_columns].to_csv(
        f"{base_dir}/validation/validation.csv", header=True, index=False
    )

    # store raw testing data for inference
    X_test_raw.assign(**{label_column: y_test})[output_raw_columns].to_csv(
        f"{base_dir}/test_raw/test.csv", header=False, index=False
    )

    with open(f"config.json", "w") as outfile:
        json.dump(
            {
                "output_engineered_feature_columns": output_engineered_feature_columns,
                "label_column": label_column,
            },
            outfile,
        )

    with tarfile.open(f"{base_dir}/prep_model/model.tar.gz", "w:gz") as tar_handle:
        tar_handle.add(f"config.json")
