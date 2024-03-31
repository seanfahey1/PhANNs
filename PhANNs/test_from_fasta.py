import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import toml
from Bio import SeqIO
from load import Data
from sklearn.metrics import classification_report
from tensorflow.keras.models import load_model
from utils import get_test_data, load_data

FORMAT = "%(asctime)-24s %(levelname)-8s | %(message)s"
logging.basicConfig(
    filename=f'test_{datetime.now().strftime("%Y_%m_%d_%H:%M")}.log',
    level=logging.INFO,
    format=FORMAT,
)


def get_args():
    parser = argparse.ArgumentParser(
        description="Tests a PhANNs model using user supplied data in fasta format."
    )
    parser.add_argument(
        "--c",
        "-config",
        type=str,
        default="config.toml",
        help="Optional config file in .toml format.",
    )

    parser.add_argument(
        "--i",
        "-input",
        type=str,
        required=True,
        help="Input .fasta file.",
    )

    args = parser.parse_args()
    return args


def load_arrays(model_dir):
    mean_arr = load_data(Path(model_dir), "mean_final.p")
    std_arr = load_data(Path(model_dir), "std_final.p")
    return mean_arr, std_arr


def z_score(mean_val, std_val, value):
    if std_val != 0:
        z_value = (value - mean_val) / std_val
        return z_value
    else:
        return 0


def predict(model_dir, model_name, test_X):
    y_hats = []
    for model_number in range(1, 11):
        model_full_name = f"{model_name}_{'{:02d}'.format(model_number)}.h5"
        model_path = model_dir / model_full_name
        logging.info(f"Running predictions on model {model_full_name}")
        model = load_model(model_path)

        y_hat = model.predict(test_X, verbose=2)
        y_hats.append(y_hat)

    y_hats_array = np.array(y_hats)
    return y_hats_array


def main():
    args = get_args()
    input_file = Path(args.i)
    config = toml.load(args.c)
    output_dir = config["test"]["output_dir"]

    mean_arr, std_arr = load_arrays(config["load"]["output_data_dir"])

    num_proteins = sum(1 for x in SeqIO.parse(input_file, format="fasta"))
    data = Data(num_proteins)

    logging.info(f"Parsing sequences from {input_file.resolve()}")
    row_counter = 0
    for record in SeqIO.parse(input_file, "fasta"):
        sequence = record.seq.__str__().upper()
        row = data.feature_extract(sequence)
        data.add_to_array(row, row_counter, cls_number=0, group=0)
        row_counter += 1

    logging.info("Calculating z-scores...")
    arr_z = np.zeros(data.arr.shape)
    for col in range(data.arr.shape[1]):
        for row in range(data.arr.shape[0]):
            mean_val = mean_arr[col]
            stdev_val = std_arr[col]
            arr_z[row, col] = z_score(mean_val, stdev_val, data.arr[row, col])

    class_numbers = config["load"]["class_number"]
    label_names = sorted(class_numbers, key=class_numbers.get)
    model_name = config["test"].get("model")
    model_dir = Path(config["main"].get("project_root_dir")) / config["train"].get(
        "model_dir"
    )
    features_dir = Path(config["main"].get("project_root_dir")) / config["load"].get(
        "output_data_dir"
    )

    y_hats = predict(model_dir, model_name, arr_z)

    predicted_Y = np.sum(y_hats, axis=0)
    predicted_Y_index = np.argmax(predicted_Y, axis=1)
    # indices as 0 indexed but class numbers start at 1
    predicted_Y_class_id = predicted_Y_index + 1

    predicted_Y_df = pd.DataFrame(predicted_Y, columns=label_names)
    predicted_Y_df["prediction_idx"] = predicted_Y_class_id
    predicted_Y_df["prediction"] = [label_names[x] for x in predicted_Y_index]

    predicted_Y_df.to_csv(
        Path(output_dir) / "predictions.csv", header=True, index=False
    )


if __name__ == "__main__":
    sys.exit(main())
