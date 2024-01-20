import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import toml
from sklearn.metrics import classification_report
from tensorflow.keras.models import load_model
from utils import get_test_data, load_data

FORMAT = "%(asctime)-24s | %(message)s"
logging.basicConfig(
    filename=f'train_{datetime.now().strftime("%Y_%m_%d_%H:%M")}.log',
    level=logging.INFO,
    format=FORMAT,
)


def get_data(model_dir):
    group_arr = load_data(Path(model_dir), "group_arr.p")
    class_arr = load_data(Path(model_dir), "class_arr.p")

    return group_arr, class_arr


def predict(model_dir, model_name, test_X):
    y_hats = []
    for model_number in range(1, 11):
        model = load_model(model_dir / f"{model_name}_{':02d'.format(model_number)}.h5")

        y_hat = model.predict(test_X, verbose=2)
        y_hats.append(y_hat)

    y_hats_array = np.array(y_hats)
    return y_hats_array


def main():
    logging.info("Starting....")
    config = toml.load("config.toml")

    logging.info("Config:")
    for section in config.keys():
        for k, v in config[section].items():
            logging.info(f"\t{section}:{k}:{v}")

    model_name = config["test"].get("model")
    model_dir = Path(config["main"].get("project_root_dir")) / config["train"].get(
        "model_dir"
    )
    group_arr, class_arr = get_data(model_dir)

    class_numbers = config["load"]["class_number"]
    label_names = sorted(class_numbers, key=class_numbers.get)

    test_X, test_Y = get_test_data(model_name, class_arr, group_arr)
    y_hats = predict(model_dir, model_name, test_X)

    predicted_Y = np.sum(y_hats, axis=0)
    predicted_Y_index = np.argmax(predicted_Y, axis=1)

    classification = classification_report(
        test_Y, predicted_Y_index, target_names=label_names
    )
    logging.info(classification)

    logging.info(predicted_Y)
    logging.info(predicted_Y_index)
    # TODO: These outputs need to be saved somehow. Also add that path to the config object.


if __name__ == "__main__":
    sys.exit(main())
