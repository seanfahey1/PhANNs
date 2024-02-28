#!/usr/bin/env python

import itertools

import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis


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
