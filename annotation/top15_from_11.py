"""
For each antimicrobial, this program takes the top 5 11mers
and a list of possible 15mer expansions, and finally returns
the most important of each of the 15mers as according to a chi2.

For each 5 top-11mers, we search through the genome using the top 15mer

This script requires the outputs from expand_important.smk
"""

import numpy as np
import pandas as pd
import os, sys

drugs=["AMP","AMC","AZM","CHL","CIP","CRO","FIS","FOX","GEN","NAL","SXT","TET","TIO"]

def nan_to_zero(ele):
    if(ele=='nan'):
        return 0
    return ele

for drug in drugs:
    # 2D array, each row consists of 1280 15mer expansions
    OxF_mers = np.load("annotation/{}_1000feats_public_15mers.npy".format(drug))

    # 1D array, 11mer parent labels for each row of 15mers
    # OxF_mer_parents[0] is the 11mer parent of all 15mers in OxF_mers[0]
    OxF_mer_parents = np.load("annotation/{}_1000feats_public_15mers_parent.npy".format(drug))

    # 2D array of shape (3,6400)
    # row[0] are 15mer, row[1] is the score by xgboost, row[2] is the score by chi2
    feature_ranks = np.load("annotation/{}_public_feature_ranks.npy".format(drug))

    # set 'nan' strings to zero for max functions
    feature_ranks[2] = [nan_to_zero(i) for i in feature_ranks[2]]

    # loop throug each 11mer parent, find the highest performing 15mer
    for parent in OxF_mer_parents:

        # find the top 15mer by the chi2
        m = max(all_feats[2])

        # pull the index of all kmers tieing the top score
        top_indeces = [i for i, j in enumerate(all_feats[2]) if j == m]

        # make sure we dont take more than 1 total, takes the first in the event of a tie
        if(len(top_indeces) > 0):
            top_indeces = top_indeces[0]
        top_15mer = feature_ranks[0][top_indeces]
        top_15mer_score = feature_ranks[2][top_indeces]

        #TODO: append lists of top 15mers, their parent, drug, score, and then search through the genes for it

    sys.exit()
