import logging
import os
import sys
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

from utils import dump_data, load_data

FORMAT = "%(asctime)-24s | %(message)s"
logging.basicConfig(filename="train.log", level=logging.INFO, format=FORMAT)


def get_train_data(model_name, model_number, class_arr, group_arr):
    train_data = []
    for prefix in range(1, 11):
        if prefix == model_number:  # mask out validation data
            continue
        train_data.append(load_data("features", f"{prefix}_{model_name}.p"))

    all_train_data = np.concatenate(train_data, axis=0)
    y_train = class_arr[(group_arr != model_number) & (group_arr != 11)]

    return all_train_data, y_train


def get_validation_data(model_name, model_number, class_arr, group_arr):
    validation_data = load_data("features", f"{model_number}_{model_name}.p")
    y = class_arr[(group_arr == model_number) & (group_arr != 11)]

    return validation_data, y


def get_test_data(model_name, class_arr, group_arr):
    test_data = load_data("features", f"11_{model_name}.p")
    y = class_arr[(group_arr == 11)]

    return test_data, y


def add_to_df(df, test_Y_index, test_Y_predicted, model_name):
    labels_names = [
        "Major capsid",
        "Minor capsid",
        "Baseplate",
        "Major tail",
        "Minor tail",
        "Portal",
        "Tail fiber",
        "Tail shaft",
        "Collar",
        "Head-Tail joining",
        "Others",
    ]
    labels_dataframe = [
        "Major capsid",
        "Minor capsid",
        "Baseplate",
        "Major tail",
        "Minor tail",
        "Portal",
        "Tail fiber",
        "Tail shaft",
        "Collar",
        "Head-Tail joining",
        "Others",
        "weighted avg",
    ]
    report = classification_report(
        test_Y_index, test_Y_predicted, target_names=labels_names, output_dict=True
    )
    for label in labels_dataframe:
        score_type = "precision"
        data_row = [model_name, label, score_type, report[label][score_type]]
        df = df.append(
            pd.Series(data_row, index=df.columns), sort=False, ignore_index=True
        )
        score_type = "recall"
        data_row = [model_name, label, score_type, report[label][score_type]]
        df = df.append(
            pd.Series(data_row, index=df.columns), sort=False, ignore_index=True
        )
        score_type = "f1-score"
        data_row = [model_name, label, score_type, report[label][score_type]]
        df = df.append(
            pd.Series(data_row, index=df.columns), sort=False, ignore_index=True
        )
    return df


def train_kfold(model_name, df, df_val, df_acc, class_arr, group_arr, out_dir):
    for model_number in range(1, 11):
        logging.info(f"Doing cross validation on {model_name} - {model_number}")

        train_X, train_Y_index = get_train_data(
            model_name, model_number, class_arr, group_arr
        )
        test_X, test_Y_index = get_validation_data(
            model_name, model_number, class_arr, group_arr
        )

        feature_count = train_X.shape[1]
        unique_classes = np.unique(train_Y_index)
        num_classes = len(unique_classes)

        # These arrays basically OHE the class to columns. Instead of a bunch of class numbers, we have an array with a
        # single `1` on each row indicating the class. I subtract 1 from the array to fit into zero indexing.
        train_Y = np.eye(num_classes)[train_Y_index - 1]
        test_Y = np.eye(num_classes)[test_Y_index - 1]

        logging.info(f"test x shape: {test_X.shape}, test y shape: {test_Y.shape}")

        es = EarlyStopping(
            monitor="loss", mode="min", verbose=2, patience=5, min_delta=0.02
        )
        mc = ModelCheckpoint(
            os.path.join(
                "models", model_name + "_val_" + "{:02d}".format(model_number) + ".h5"
            ),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        )
        mc2 = ModelCheckpoint(
            os.path.join(
                "models", model_name + "_acc_" + "{:02d}".format(model_number) + ".h5"
            ),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        )
        print(num_classes, train_Y_index)
        class_weights = compute_class_weight(
            "balanced", range(1, num_classes + 1), train_Y_index
        )

        train_weights = dict(zip(range(1, num_classes + 1), class_weights))
        logging.info(f"train weights:\n{train_weights}")

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

        logging.info("Built model...")
        test_Y_predicted = model.predict_classes(test_X)

        logging.info("Finished training! Saving models")
        df = add_to_df(df, test_Y_index, test_Y_predicted, model_name)
        model.save(Path(out_dir) / f'{model_name}_{"{:02d}".format(model_number)}.h5')

        model_val = load_model(
            Path(out_dir) / f'{model_name}_val_{"{:02d}".format(model_number)}.h5'
        )
        test_Y_predicted_val = model_val.predict_classes(test_X)
        df_val = add_to_df(df_val, test_Y_index, test_Y_predicted_val, model_name)

        model_acc = load_model(
            Path(out_dir) / f'{model_name}_acc_{"{:02d}".format(model_number)}.h5'
        )
        test_Y_predicted_acc = model_acc.predict_classes(test_X)
        df_acc = add_to_df(df_acc, test_Y_index, test_Y_predicted_acc, model_name)

        K.clear_session()

    return df, df_val, df_acc


def main():
    logging.info("Starting....")
    config = toml.load("config.toml")

    logging.info("Config:")
    for section in config.keys():
        for k, v in config[section].items():
            logging.info(f"\t{section}:{k}:{v}")

    out_dir = config["train"].get("model_dir")

    columns = ["model", "class", "score_type", "value"]
    df = pd.DataFrame(columns=columns)
    df_val = pd.DataFrame(columns=columns)
    df_acc = pd.DataFrame(columns=columns)

    all_models = [
        "di_sc",
        "di_sc_p",
        "tri_sc",
        "tri_sc_p",
        "tetra_sc",
        "tetra_sc_p",
        "di",
        "di_p",
        "tri",
        "tri_p",
        "tetra_sc_tri_p",
        "all",
    ]

    group_arr = load_data("features", "group_arr.p")
    class_arr = load_data("features", "class_arr.p")

    for model_name in all_models:
        logging.info(f"Starting model: {model_name}")
        df, df_val, df_acc = train_kfold(
            model_name, df, df_val, df_acc, class_arr, group_arr, out_dir
        )

    dump_data(df, "models", "all_results.p")
    dump_data(df, "models", "all_results_df_val.p")
    dump_data(df, "models", "all_results_df_acc.p")


if __name__ == "__main__":
    sys.exit(main())
