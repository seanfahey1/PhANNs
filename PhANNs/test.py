import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import toml
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
        description="Tests a PhANNs model using pre-loaded test data. Must run load.py step first."
    )
    parser.add_argument(
        "--c",
        "-config",
        type=str,
        default="config.toml",
        help="Optional config file in .toml format.",
    )

    args = parser.parse_args()
    return args


def get_data(model_dir):
    group_arr = load_data(Path(model_dir), "group_arr.p")
    class_arr = load_data(Path(model_dir), "class_arr.p")

    return group_arr, class_arr


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
    logging.info("Starting....")
    args = get_args()
    config = toml.load(args.c)

    output_dir = Path(config["test"]["output_dir"])
    output_dir.mkdir(exist_ok=True, parents=True)

    logging.info("Config:")
    for section in config.keys():
        for k, v in config[section].items():
            logging.info(f"\t{section}:{k}:{v}")

    model_name = config["test"].get("model")
    model_dir = Path(config["main"].get("project_root_dir")) / config["train"].get(
        "model_dir"
    )
    features_dir = Path(config["main"].get("project_root_dir")) / config["load"].get(
        "output_data_dir"
    )
    group_arr, class_arr = get_data(features_dir)

    class_numbers = config["load"]["class_number"]
    label_names = sorted(class_numbers, key=class_numbers.get)

    test_X, test_Y = get_test_data(features_dir, model_name, class_arr, group_arr)
    y_hats = predict(model_dir, model_name, test_X)

    predicted_Y = np.sum(y_hats, axis=0)
    predicted_Y_index = np.argmax(predicted_Y, axis=1)
    # indices as 0 indexed but class numbers start at 1
    predicted_Y_class_id = predicted_Y_index + 1

    classification = classification_report(
        test_Y, predicted_Y_class_id, target_names=label_names
    )
    logging.info(f"Classification Report:\n{classification}")

    predicted_Y_df = pd.DataFrame(predicted_Y, columns=label_names)
    predicted_Y_df["prediction_idx"] = predicted_Y_class_id
    predicted_Y_df["prediction"] = [label_names[x] for x in predicted_Y_index]

    predicted_Y_df.to_csv(output_dir / "predictions.csv", header=True, index=False)


if __name__ == "__main__":
    sys.exit(main())
