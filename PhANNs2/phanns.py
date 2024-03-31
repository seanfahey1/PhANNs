#!/usr/bin/env python
from pathlib import Path

import pyximport
from Bio import SeqIO
from utils.config_handler import load_stored_config
from utils.data import Data
from utils.data_loading_utils import collect_files, fasta_count
from utils.logger_util import Logger

pyximport.install()
from utils.calc import zscore


def load_dataset():
    config = load_stored_config()
    logger = Logger(filename="load_{time}.log")
    logger.log_config(config)

    out_dir = Path(config["main"].get("project_root_dir")) / config[
        "file_locations"
    ].get("array_data_dir")
    out_dir.mkdir(parents=True, exist_ok=True)

    file_dict = collect_files(
        Path(config["main"].get("project_root_dir"))
        / config["file_locations"].get("fasta_data_dir"),
        config["class_info"]["group_prefixes"],
        config["group_names"],
    )

    num_proteins = fasta_count(file_dict.keys())
    logger.log(f"Found {num_proteins} protein sequences")

    data = Data(num_proteins)
    row_counter = 0

    for file_path, keys in file_dict.items():
        cls, group_number = keys["class"], keys["group"]
        cls_number = config["load"]["class_number"][cls]

        logger.log(f"file: {file_path.name}, class number: {cls_number}")

        for record in SeqIO.parse(file_path, "fasta"):
            sequence = record.seq.__str__().upper()
            row = data.feature_extract(sequence)
            data.add_to_array(row, row_counter, cls_number, group_number)
            row_counter += 1

        logger.log(f"Finished file {file_path.stem}. Current row: {row_counter}")

    mean_array, stdev_array, zscore_array = zscore(data.arr)
    print("")


def train():
    logger = Logger(filename="train_{time}.log")
    pass


def initial_test():
    logger = Logger(filename="initial_test_{time}.log")
    pass


def test_from_fasta():
    logger = Logger(filename="test_from_fasta_{time}.log")
    pass


load_dataset()
