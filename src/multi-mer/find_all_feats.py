"""
takes the 5 folds of seen kmers and finds a master list
"""

import numpy as np
import pandas as pd
import os, sys
from itertools import chain

if __name__ == "__main__":
    kmer_length = sys.argv[1]
    out = sys.argv[2]
    dataset_path = sys.argv[3]

    if dataset_path == 'grdi_':
        dataset = 'grdi'
        set_nums = list(range(6,10))
    else:
        dataset = 'public'
        set_nums = list(range(1,6))

    feat_sets = {}
    for set_num in set_nums:
        feat_sets[set_num] = np.load("data/genomes/top_feats/all_{}mers{}.npy".format(kmer_length,set_num))

    all_feats = [feat_sets[i] for i in set_nums]

    master_mers = np.array(list(set(chain(*all_feats))))
    bool_mask = [len(master_mers[i]) for i in range(len(master_mers))]
    bool_mask = np.array([i == int(kmer_length) for i in bool_mask])
    master_mers = master_mers[bool_mask]

    np.save(out, master_mers)
