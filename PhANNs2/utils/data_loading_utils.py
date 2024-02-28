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
    file_dict = {}
    for prefix in prefixes:
        for file, cls in labels.items():
            file_name = f"{prefix}{file}.fasta"
            file_dict[Path(directory) / file_name] = {
                "class": cls,
                "group": int(prefix.strip("_")),
            }
    return file_dict


def zscore(array: np.array, axis=0):
    mean_array = np.zeros(array.shape[1])
    stdev_array = np.zeros(array.shape[1])
    zscore_array = np.zeros(array.shape)

    for i, col in enumerate(array.T):
        mean_val = np.mean(col)
        stdev_val = np.std(col)

        mean_array[i] = mean_val
        stdev_array[i] = stdev_val

        for j, value in enumerate(col):
            zscore_val = (value - mean_val) / stdev_val
            zscore_array[i][j] = zscore_val

    return mean_array, stdev_array, zscore_array
