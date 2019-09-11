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
    feat_sets = {}
    for set_num in [str(i+1) for i in range(5)]:
        feat_sets[set_num] = np.load("data/genomes/top_feats/1000_{}mers{}.npy".format(kmer_length,set_num))

    all_feats = [feat_sets[str(i+1)] for i in range(5)]

    master_mers = np.array(list(set(chain(*all_feats))))
    bool_mask = [len(master_mers[i]) for i in range(len(master_mers))]
    bool_mask = np.array([i == int(kmer_length) for i in bool_mask])
    master_mers = master_mers[bool_mask]

    np.save(out, master_mers)
