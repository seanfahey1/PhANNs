from pathlib import Path

import numpy as np
from Bio import SeqIO


def fasta_count(all_files):
    total_seqs = 0
    for file in all_files:
        for _ in SeqIO.parse(file, "fasta"):
            total_seqs += 1
    return total_seqs


def collect_files(directory: str, prefixes: list, labels: dict):
    """Collect de-de-replicated fasta files for each class.

    Args:
        directory (str): The target directory path.
        prefixes (list): The list of prefixes for each class (typically 1_, 2_, ..., n_)
        labels (dict): A dictionary of labels in the format of {"file_stem": "class_id_code"}

    Returns:
        _type_: _description_
    """
    file_dict = {}
    for prefix in prefixes:
        for file, cls in labels.items():
            file_name = f"{prefix}{file}.fasta"
            file_dict[Path(directory) / file_name] = {
                "class": cls,
                "group": int(prefix.strip("_")),
            }
    return file_dict


def zscore(mean_array, stddev_array, data_array):
    zscore_array = np.zeros(data_array.shape)

    for i in range(data_array.shape[1]):
        mean_val, stddev_val, data_col = (
            mean_array[i],
            stddev_array[i],
            data_array[:, i],
        )

        for j in range(len(data_col)):
            zscore_val = (mean_val - data_col[j]) / stddev_val
            zscore_array[j][i] = zscore_val

    return zscore_array
