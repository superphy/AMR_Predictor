#!/usr/bin/env python

"""
loads in the top features and prints out the importance of any
kmers found in a specific area.
"""

from Bio import Seq, SeqIO
import sys

import numpy as np

def find_index(arr, element):
    for i, val in enumerate(arr):
        if(val == element):
            return i
    return -1

def remove_b(name):
    name = name[2:]
    return name[:-1]

all_feats = np.load("data/features/AMP_3000feats_XGBtrainedOnpublic_testedOncv_fold1.npy")

all_feats_up = [i.decode('utf-8') for i in all_feats[0]]


print(all_feats_up[0])
print(len(all_feats_up), all_feats.shape)
#all_feats_new = all_feats[:,find_index(all_feats_up,'ACGCTGAAAAT')]
#all_feats_up = [['ACGCTGAAAAT'],[0.022727273]]
#all_feats_up = [i.decode('utf-8') for i in all_feats_new[0]]

all_feats_up = np.vstack((all_feats_up,all_feats[1]))

for record in SeqIO.parse("data/genomes/clean/SRR2567008.fasta", "fasta"):
    # will only search in the specific region, so for a gene in region 257367:258836 search record.seq[257367:258836]
    beta_range = record.seq[257367:258836]
    miss_count = 0
    for i in range(len(beta_range)-10):
        elemer = beta_range[i:i+11]
        eleindx = find_index(all_feats_up[0], elemer)

        if(eleindx!=-1):
            #print('we found a index at ', eleindx)
            print("{} has an importance of {}".format(elemer,all_feats_up[1][eleindx]))
        else:
            miss_count+=1

    print("there were {} misses in contig {}".format(miss_count, record.id))
