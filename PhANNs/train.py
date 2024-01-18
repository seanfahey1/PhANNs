import os
import pickle as p

import numpy as np
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Activation, Dense, Dropout
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam

from utils import load_data


def get_train(model_name, model_number, class_arr, group_arr):
    kk = np.array(range(10))
    my_list = [
        p.load(
            open(
                os.path.join("06_features", str(x + 1) + "_" + model_name + ".p"), "rb"
            )
        )
        for x in kk[kk != model_number]
    ]
    my_cat_arr = np.concatenate(my_list, axis=0)
    del my_list
    Y_index = class_arr[(group_arr != model_number) & (group_arr != 10)]
    return (my_cat_arr, Y_index)


def get_validation(model_name, model_number, class_arr, group_arr):
    my_arr = p.load(
        open(
            os.path.join(
                "06_features", str(model_number + 1) + "_" + model_name + ".p"
            ),
            "rb",
        )
    )
    Y_index = class_arr[(group_arr == model_number) & (group_arr != 10)]
    return (my_arr, Y_index)


def get_test(model_name, class_arr, group_arr):
    my_arr = p.load(open(os.path.join("06_features", "11_" + model_name + ".p"), "rb"))
    Y_index = class_arr[(group_arr == 10)]
    return (my_arr, Y_index)
