import numpy as np
from Bio import SeqIO
import os, sys

drugs = ['AMP','AMC','CRO','AZM','CHL','CIP','SXT','FIS','FOX','GEN','NAL','TET','TIO']

def numpy_to_fasta(feat_list, fasta_path):
    """
    Takes numpy array and saves sequences to fasta
    """
    if isinstance(feat_list[0], bytes):
        feat_list = [i.decode('utf-8') for i in feat_list]
    with open(fasta_path,'w') as fasta:
        for feat in feat_list:
            fasta.write(">{}\n".format(feat))
            fasta.write("{}\n".format(feat))


if __name__ == "__main__":
    for drug in drugs:
        arr = np.load("predict/features/1000feats_{}.npy".format(drug))
        numpy_to_fasta(arr, "predict/features/1000feats_{}.fasta".format(drug))

    all_rel = np.load("predict/features/relevant_feats_1000.npy")
    all_rel = all_rel[[len(i)==11 for i in all_rel]]

    numpy_to_fasta(all_rel, "predict/features/relevant_feats_1000.fasta")
