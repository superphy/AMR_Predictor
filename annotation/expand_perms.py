"""
This program takes a drug and dataset name, extracts the top 5 features,
then returns all 15mer permuations of each top 5 11mer
"""

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
import os, sys

if __name__ == "__main__":
    drug = sys.argv[1]
    dataset = sys.argv[2]

    def find_perms(OxB_mer):
        # take an 11 mer and return all possible 15mers
        OxF_mers = []
        nts = ['A','G','C','T']

        # we are adding 4 nucleotides, each with 4 options
        # for each possible set of 4 nucleotides, there are
        # 5 possible locations the 11mer can be placed

        for nt1 in nts:
            for nt2 in nts:
                for nt3 in nts:
                    for nt4 in nts:
                        OxF_mers.append(OxB_mer+nt1+nt2+nt3+nt4)
                        OxF_mers.append(nt1+OxB_mer+nt2+nt3+nt4)
                        OxF_mers.append(nt1+nt2+OxB_mer+nt3+nt4)
                        OxF_mers.append(nt1+nt2+nt3+OxB_mer+nt4)
                        OxF_mers.append(nt1+nt2+nt3+nt4+OxB_mer)
        return OxF_mers, OxB_mer

    # find the top 5 features
    top_feats = []
    all_feats = np.load('data/features/'+drug+'_1000feats_XGBtrainedOn'+dataset+'_testedOn'+dataset+'_fold1.npy')
    top_x_feats = 5

    # until we have at least 5 features
    while(top_x_feats>0):

        # find the highest ranking score
        m = max(all_feats[1])

        # find the indeces that match the highest score
        top_indeces = [i for i, j in enumerate(all_feats[1]) if j == m]

        # make sure we dont take more than top_x total, this can be removed
        # to keep all tying features in the pipeline
        if(len(top_indeces) > top_x_feats):
            top_indeces[:top_x_feats]

        # subtract how many we found from the total we need
        top_x_feats -= len(top_indeces)

        # now add the kmers at this index to our top features
        for i in top_indeces:
            top_feats.append((all_feats[0][i]).decode('utf-8'))
            # set the score to 0 so that the next loops dont catch these kmers
            all_feats[1][i] = 0

    # double check that we have the correct number of features
    assert(len(top_feats)==5)

    # expand the top 5 11mers into 15mers
    OxF_mer_matrix = []
    OxB_labels = []

    with ProcessPoolExecutor(max_workers=5) as ppe:
        for perms, feat in ppe.map(find_perms, top_feats):
            OxF_mer_matrix.append(perms)
            OxB_labels.append(feat)

    np.save("annotation/{}_1000feats_{}_15mers.npy".format(drug,dataset), OxF_mer_matrix)
    np.save("annotation/{}_1000feats_{}_15mers_parent.npy".format(drug,dataset), OxB_labels)
