import pickle as p
from pathlib import Path


def dump_data(file, out_dir, array_name):
    out_path = Path(out_dir) / array_name

    with open(out_path.with_suffix(".p"), "wb") as out:
        p.dump(file, out, protocol=4)


def load_data(file_dir, file_name):
    path = Path(file_dir) / file_name

    with open(path, "rb") as file:
        contents = p.load(file)

    return contents


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
