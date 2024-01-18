import itertools
import logging
import sys
from pathlib import Path

import numpy as np
import toml
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from scipy import stats

from utils import dump_data

FORMAT = "%(asctime)-24s | %(message)s"
logging.basicConfig(filename="load.log", level=logging.INFO, format=FORMAT)


class Data:
    def __init__(self, protein_count):
        self.arr = np.empty((protein_count, 11201), dtype=np.float)
        self.class_arr = np.empty(protein_count, dtype=np.int)
        self.group_arr = np.empty(protein_count, dtype=np.int)
        self.id_arr = np.empty(protein_count, dtype=np.int)

        aa = "AILMVNQSTGPCHKRDEFWY"
        sc = "11111222233455566777"
        self.sc_translator = aa.maketrans(aa, sc)

        AA = sorted([x for x in aa])
        SC = ["1", "2", "3", "4", "5", "6", "7"]

        self.di_pep = ["".join(i) for i in itertools.product(AA, repeat=2)]
        self.tri_pep = ["".join(i) for i in itertools.product(AA, repeat=3)]
        self.di_sc = ["".join(i) for i in itertools.product(SC, repeat=2)]
        self.tri_sc = ["".join(i) for i in itertools.product(SC, repeat=3)]
        self.tetra_sc = ["".join(i) for i in itertools.product(SC, repeat=4)]

    def feature_extract(self, raw_sequence):
        sequence = (
            raw_sequence.upper()
            .replace("X", "A")
            .replace("J", "L")
            .replace("*", "A")
            .replace("Z", "E")
            .replace("B", "D")
        )
        len_seq = len(sequence)
        sequence_sc = sequence.translate(self.sc_translator)

        di_pep_count = [sequence.count(x) / (len_seq - 1) for x in self.di_pep]
        di_pep_count_n = np.asarray(di_pep_count, dtype=np.float)

        tri_pep_count = [sequence.count(x) / (len_seq - 2) for x in self.tri_pep]
        tri_pep_count_n = np.asarray(tri_pep_count, dtype=np.float)

        di_sc_count = [sequence_sc.count(x) / (len_seq - 1) for x in self.di_sc]
        di_sc_count_n = np.asarray(di_sc_count, dtype=np.float)

        tri_sc_count = [sequence_sc.count(x) / (len_seq - 2) for x in self.tri_sc]
        tri_sc_count_n = np.asarray(tri_sc_count, dtype=np.float)

        tetra_sc_count = [sequence_sc.count(x) / (len_seq - 3) for x in self.tetra_sc]
        tetra_sc_count_n = np.asarray(tetra_sc_count, dtype=np.float)

        X = ProteinAnalysis(sequence)
        additional_features = [
            X.isoelectric_point(),
            X.instability_index(),
            len_seq,
            X.aromaticity(),
            X.molar_extinction_coefficient()[0],
            X.molar_extinction_coefficient()[1],
            X.gravy(),
            X.molecular_weight(),
        ]

        additional_features_array = np.asarray(additional_features, dtype=np.float)

        row = np.concatenate(
            (
                di_pep_count_n,
                tri_pep_count_n,
                di_sc_count_n,
                tri_sc_count_n,
                tetra_sc_count_n,
                additional_features_array,
            )
        )
        row = row.reshape((1, row.shape[0]))

        return row

    def add_to_array(self, row, row_num, cls_number, group):
        self.arr[row_num, :] = row
        self.class_arr[row_num] = cls_number
        self.group_arr[row_num] = group
        self.id_arr[row_num] = row_num


def fasta_count(all_files):
    total_seqs = 0
    for file in all_files:
        for _ in SeqIO.parse(file, "fasta"):
            total_seqs += 1
    return total_seqs


def model_specific_arrays(out_dir, z_array, group_array):
    di_n = 400
    tri_n = 8000
    di_sc_n = 49
    tri_sc_n = 343
    tetra_sc_n = 2401
    p_n = 8

    di_end = di_n
    tri_end = di_end + tri_n
    di_sc_end = tri_end + di_sc_n
    tri_sc_end = di_sc_end + tri_sc_n
    tetra_sc_end = tri_sc_end + tetra_sc_n

    di_range = np.r_[:di_end]
    tri_range = np.r_[di_end:tri_end]
    di_sc_range = np.r_[tri_end:di_sc_end]
    tri_sc_range = np.r_[di_sc_end:tri_sc_end]
    tetra_sc_range = np.r_[tri_sc_end:tetra_sc_end]
    p_range = np.r_[tetra_sc_end : tetra_sc_end + p_n]
    di_sc_p_range = np.r_[di_sc_range, p_range]
    tri_sc_p_range = np.r_[tri_sc_range, p_range]
    tetra_sc_p_range = np.r_[tetra_sc_range, p_range]
    di_p_range = np.r_[di_range, p_range]
    tri_p_range = np.r_[tri_range, p_range]
    tetra_sc_tri_p_range = np.r_[tetra_sc_range, tri_range, p_range]
    all_range = np.r_[: tetra_sc_end + p_n]

    model_ranges = {
        "di_sc": di_sc_range,
        "di_sc_p": di_sc_p_range,
        "tri_sc": tri_sc_range,
        "tri_sc_p": tri_sc_p_range,
        "tetra_sc": tetra_sc_range,
        "tetra_sc_p": tetra_sc_p_range,
        "di": di_range,
        "di_p": di_p_range,
        "tri": tri_range,
        "tri_p": tri_p_range,
        "tetra_sc_tri_p": tetra_sc_tri_p_range,
        "all": all_range,
    }

    for group in np.unique(group_array):
        for name in model_ranges.keys():
            z_array_single_group = z_array[(group_array == group),]
            z_array_current_model = z_array_single_group[:, model_ranges[name]]

            dump_data(z_array_current_model, out_dir, f"{group}_{name}.p")


def main():
    logging.info("Starting....")
    config = toml.load("config.toml")
    out_dir = config["load"].get("output_data_dir")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    logging.info("Config:")
    for section in config.keys():
        for k, v in config[section].items():
            logging.info(f"\t{section}:{k}:{v}")

    file_dict = {}
    for prefix in config["load"].get("prefixes"):
        for file, cls in config["load"].get("fasta_labels").items():
            file_name = prefix + file + '.fasta'
            file_dict[Path(config["load"].get("train_data_dir")) / file_name] = {
                "class": cls,
                "number": int(prefix.strip("_")),
                "group": prefix.strip("_"),
            }

    num_proteins = fasta_count(file_dict.keys())
    logging.info(f"Found {num_proteins} protein sequences")

    data = Data(num_proteins)

    for file_path, keys in file_dict.items():
        cls, file_number = keys["class"], keys["number"]
        cls_number = config['load']['class_number'][cls]

        logging.info(f'file: {file_path.name}, class number: {cls_number}')

        row_counter = 0

        for record in SeqIO.parse(file_path, "fasta"):
            sequence = record.seq.__str__().upper()
            row = data.feature_extract(sequence)
            data.add_to_array(row, row_counter, cls_number, file_number)
            row_counter += 1

        logging.info(f"Finished file {file_path.stem}. Current row: {row_counter}")

    logging.info("Calculating z-scores...")
    arr_z = np.apply_along_axis(stats.zscore, 0, data.arr)
    logging.info("Calculating means...")
    mean_arr = np.apply_along_axis(np.mean, 0, data.arr)
    logging.info("Calculating standard deviations...")
    std_arr = np.apply_along_axis(np.std, 0, data.arr)

    logging.info("Writing arrays to disk")
    for array, name in (
        (data.class_arr, "class_arr"),
        (data.group_arr, "group_arr"),
        (mean_arr, "mean_final"),
        (std_arr, "std_final"),
        (arr_z, "all_data"),
    ):
        dump_data(array, out_dir, name)

    logging.info("Writing model specific arrays to disk")
    model_specific_arrays(out_dir, arr_z, data.group_arr)


if __name__ == "__main__":
    sys.exit(main())
