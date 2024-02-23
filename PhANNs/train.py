import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import toml
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Activation, Dense, Dropout
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from utils import dump_data, get_train_data, get_validation_data, load_data

FORMAT = "%(asctime)-24s %(levelname)-8s | %(message)s"
logging.basicConfig(
    filename=f'train_{datetime.now().strftime("%Y_%m_%d_%H:%M")}.log',
    level=logging.INFO,
    format=FORMAT,
)


def get_args():
    parser = argparse.ArgumentParser(
        description="Trains a PhANNs model using pre-loaded data. Must run load.py step first."
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


def add_to_df(df, test_Y_index, test_Y_predicted, model_name, class_numbers):
    label_names = sorted(class_numbers, key=class_numbers.get)
    df_label_names = label_names + ["weighted avg"]

    report = classification_report(
        test_Y_index, test_Y_predicted + 1, target_names=label_names, output_dict=True
    )
    for label in df_label_names:
        score_type = "precision"
        data_row = [model_name, label, score_type, report[label][score_type]]
        df = df.append(
            pd.Series(data_row, index=df.columns), sort=False, ignore_index=True
        )
        logging.info(score_type, "".join([str(x) for x in data_row]))

        score_type = "recall"
        data_row = [model_name, label, score_type, report[label][score_type]]
        df = df.append(
            pd.Series(data_row, index=df.columns), sort=False, ignore_index=True
        )
        logging.info(score_type, "".join([str(x) for x in data_row]))

        score_type = "f1-score"
        data_row = [model_name, label, score_type, report[label][score_type]]
        df = df.append(
            pd.Series(data_row, index=df.columns), sort=False, ignore_index=True
        )
        logging.info(score_type, "".join([str(x) for x in data_row]))

    return df


def train_kfold(
    model_name,
    features_dir,
    df,
    df_val,
    df_acc,
    class_numbers,
    class_arr,
    group_arr,
    out_dir,
):
    for model_number in range(1, 11):
        logging.info("-" * 80)
        logging.warning(f"Doing cross validation on {model_name} - {model_number}")

        train_X, train_Y_index = get_train_data(
            features_dir, model_name, model_number, class_arr, group_arr
        )
        test_X, test_Y_index = get_validation_data(
            features_dir, model_name, model_number, class_arr, group_arr
        )

        feature_count = train_X.shape[1]
        unique_classes = np.unique(train_Y_index)
        num_classes = len(unique_classes)
        logging.info(f"{num_classes} unique classes found.")

        # These arrays basically OHE the class to columns. Instead of a bunch of class numbers, we have an array with a
        # single `1` on each row indicating the class. I subtract 1 from the array to fit into zero indexing.
        train_Y = np.eye(num_classes)[train_Y_index - 1]
        test_Y = np.eye(num_classes)[test_Y_index - 1]

        logging.info(f"Test x shape: {test_X.shape}, test y shape: {test_Y.shape}")
        logging.info(f"Train x shape: {train_X.shape}, train y shape: {train_Y.shape}")
        logging.info(f"Train Y values look like: {train_Y.shape}\n{train_Y}")
        es = EarlyStopping(
            monitor="loss", mode="min", verbose=2, patience=5, min_delta=0.02
        )

        val_model_path = (
            (Path(out_dir) / f'{model_name}_val_{"{:02d}".format(model_number)}.h5')
            .resolve()
            .__str__()
        )
        mc = ModelCheckpoint(
            val_model_path,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        )

        acc_model_path = (
            (Path(out_dir) / f'{model_name}_acc_{"{:02d}".format(model_number)}.h5')
            .resolve()
            .__str__()
        )
        mc2 = ModelCheckpoint(
            acc_model_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        )

        class_weights = compute_class_weight(
            "balanced", range(num_classes), train_Y_index - 1
        )

        train_weights = dict(zip(range(num_classes), class_weights))
        logging.info(f"Train weights:\n{train_weights}")

        model = Sequential()
        opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.0, amsgrad=False)
        model.add(
            Dense(
                feature_count,
                input_dim=feature_count,
                kernel_initializer="random_uniform",
                activation="relu",
            )
        )
        model.add(Dropout(0.2))
        model.add(Dense(200, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(200, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(num_classes, activation="softmax"))

        model.compile(
            loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"]
        )
        logging.info("Model built. Fitting...")
        history = model.fit(
            train_X,
            train_Y,
            validation_data=(test_X, test_Y),
            epochs=120,
            batch_size=5000,
            verbose=2,
            class_weight=train_weights,
            callbacks=[es, mc, mc2],
        )

        logging.info("Fit model. Starting predictions...")
        test_Y_predicted = model.predict_classes(test_X)
        logging.info(test_Y_predicted)

        logging.info("Finished training! Saving models")
        df = add_to_df(df, test_Y_index, test_Y_predicted, model_name, class_numbers)

        model_path = (
            (Path(out_dir) / f'{model_name}_{"{:02d}".format(model_number)}.h5')
            .resolve()
            .__str__()
        )
        model.save(model_path)

        model_val = load_model(val_model_path)
        test_Y_predicted_val = model_val.predict_classes(test_X)
        df_val = add_to_df(
            df_val, test_Y_index, test_Y_predicted_val, model_name, class_numbers
        )

        model_acc = load_model(acc_model_path)
        test_Y_predicted_acc = model_acc.predict_classes(test_X)
        df_acc = add_to_df(
            df_acc, test_Y_index, test_Y_predicted_acc, model_name, class_numbers
        )

        K.clear_session()

    return df, df_val, df_acc


def main():
    logging.info("Starting....")
    args = get_args()
    config = toml.load(args.c)

    logging.info("Config:")
    for section in config.keys():
        for k, v in config[section].items():
            logging.info(f"\t{section}:{k}:{v}")

    features_dir = Path(config["main"].get("project_root_dir")) / config["load"].get(
        "output_data_dir"
    )
    out_dir = Path(config["main"].get("project_root_dir")) / config["train"].get(
        "model_dir"
    )
    out_dir.mkdir(exist_ok=True, parents=True)

    class_numbers = config["load"]["class_number"]

    columns = ["model", "class", "score_type", "value"]
    df = pd.DataFrame(columns=columns)
    df_val = pd.DataFrame(columns=columns)
    df_acc = pd.DataFrame(columns=columns)

    all_models = [
        model for model, value in config["train"].get("models").items() if value
    ]

    group_arr = load_data(features_dir, "group_arr.p")
    class_arr = load_data(features_dir, "class_arr.p")

    # TODO: select models to train based on config file
    for model_name in all_models:
        logging.info("_\|/_" * 16)
        logging.warning(f"STARTING NEW MODEL: {model_name}")
        df, df_val, df_acc = train_kfold(
            model_name,
            features_dir,
            df,
            df_val,
            df_acc,
            class_numbers,
            class_arr,
            group_arr,
            out_dir,
        )

    dump_data(df, out_dir, "all_results.p")
    dump_data(df, out_dir, "all_results_df_val.p")
    dump_data(df, out_dir, "all_results_df_acc.p")


if __name__ == "__main__":
    sys.exit(main())
