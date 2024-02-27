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


def z_score(mean, std, data):
    pass


def main():
    args = get_args()
    input_file = Path(args.i)
    config = toml.load(args.c)

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
    for col in data.arr.T:
        pass


if __name__ == "__main__":
    sys.exit(main())
