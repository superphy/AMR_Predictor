"""
Merges the parallel splits of multi_mer_matrix.py,
saves the matrix with all possible features (leave feature selection to model)
"""

import numpy as np
import os, sys
from sklearn.feature_selection import SelectKBest

if __name__== "__main__":
    kmer_length = sys.argv[1]
    dataset = sys.argv[2]

    if dataset == 'public':
        all_genomes = [i.split('.')[0] for i in np.load("data/multi-mer/genome_names.npy")]
        features = np.load("data/genomes/top_feats/all_31mers.npy")
        split_nums = list(range(1,47))
        save_prefix = ''
    else:
        all_genomes = [i.split('.')[0] for i in np.load("data/multi-mer/grdi_genome_names.npy")]
        features = np.load("data/genomes/top_feats/grdi_all_31mers.npy")
        split_nums = list(range(47,59))
        save_prefix = 'grdi_'

    num_rows = len(all_genomes)
    num_cols = len(features)

    #full_matrix = np.zeros((num_rows,num_cols), dtype = 'uint8')
    full_matrix = []

    cols_dict = { features[i] : i for i in range(0, num_cols)}
    rows_dict = { all_genomes[i] : i for i in range(0, num_rows)}

    for split_num in split_nums:

        split_matrix = np.load("data/multi-mer/splits/{}mer_matrix{}.npy".format(kmer_length, split_num))
        split_rows_genomes = np.load("data/multi-mer/splits/{}mer_rows{}.npy".format(kmer_length, split_num))

        if dataset == 'public':
            for i, genome_id in enumerate(split_rows_genomes):
                assert genome_id == all_genomes[i+((split_num-1)*128)]
        else:
            for i, genome_id in enumerate(split_rows_genomes):
                assert genome_id == all_genomes[i+((split_num-47)*128)]

        if(split_num in [1,47]):
            full_matrix = split_matrix
        else:
            full_matrix = np.concatenate((full_matrix,split_matrix), axis=0)

        """
        for i, split_row in enumerate(split_matrix):
            for j, split_col in enumerate(split_row):
                full_matrix[i+((split_num-1)*128),j] = split_col
        """


    np.save("data/multi-mer/{}{}mer_matrix.npy".format(save_prefix, kmer_length), full_matrix)
    np.save("data/multi-mer/{}{}mer_rows.npy".format(save_prefix, kmer_length), all_genomes)
    np.save("data/multi-mer/{}{}mer_cols.npy".format(save_prefix, kmer_length), features)
