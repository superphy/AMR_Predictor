#!/usr/bin/env python

import numpy as np
from sklearn.externals import joblib
import pandas as pd
import collections

drug = 'AMP'
feats = '3000'
fold = '1'
top_x_feats = 5
train ='public'
test = 'cv'

def find_index(arr, element):
    for i, val in enumerate(arr):
        if(val == element):
            return i
    raise Exception("Index not found")

def remove_b(name):
    name = name[2:]
    return name[:-1]


kmer_matrix = np.load("data/"+drug+"/kmer_matrix.npy")

kmer_cols = np.load("data/"+drug+"/kmer_cols.npy")
kmer_cols = [i.decode('utf-8') for i in kmer_cols]

kmer_rows_mic = np.load("data/"+drug+"/kmer_rows_mic.npy")
kmer_rows_mic = [i.decode('utf-8') for i in kmer_rows_mic]

kmer_rows_genomes = np.load("data/"+drug+"/kmer_rows_genomes.npy")
kmer_rows_genomes = [i.decode('utf-8') for i in kmer_rows_genomes]

mic_class_dict = joblib.load("data/public_mic_class_order_dict.pkl")[drug]

top_feats = []
all_feats = np.load('data/features/'+drug+'_'+feats+'feats_XGBtrainedOn'+train+'_testedOn'+test+'_fold'+fold+'.npy')
while(top_x_feats>0):
    m = max(all_feats[1])
    top_indeces = [i for i, j in enumerate(all_feats[1]) if j == m]
    top_x_feats -= len(top_indeces)
    for i in top_indeces:
        top_feats.append((all_feats[0][i]).decode('utf-8'))
        all_feats[1][i] = 0

#make pandas dataframe of all mic bins on y axis, important features on x axis, and the counts of each

feat_counts = np.zeros((len(mic_class_dict),len(top_feats)))

top_feats_indeces = [find_index(kmer_cols, i) for i in top_feats]

for i in top_feats_indeces:
    for j, mic in enumerate(kmer_rows_mic):
        feat_counts[find_index(mic_class_dict, mic)][find_index(top_feats, kmer_cols[i])] += kmer_matrix[j][i]

counts_df = pd.DataFrame(data = feat_counts, index = mic_class_dict, columns = top_feats)

pd.set_option('display.max_columns', 15)

mic_counts = collections.Counter(kmer_rows_mic)
mic_sums = [ mic_counts[i] for i in mic_class_dict]

for i in range(len(mic_sums)):
    counts_df.values[i] = [j/mic_sums[i] for j in counts_df.values[i]]

print(counts_df)
counts_df.to_pickle('data/features/'+drug+'_'+feats+'feats_XGBtrainedOn'+train+'_testedOn'+test+'_fold'+fold+'.pkl')

errors_df = pd.read_pickle('data/errors/'+drug+'_'+feats+'feats_XGBtrainedOn'+train+'_testedOn'+test+'_fold'+fold+'.pkl')

kmer_counts = np.zeros((errors_df.values.shape[0],len(top_feats)))

for i, genome in enumerate(errors_df['Genome']):
    for j, kmer_index in enumerate(top_feats_indeces):
        # count how many times that kmer is in genome
        #print(kmer_matrix.shape)
        #print("****INDEX: ", remove_b(genome))
        kmer_counts[i][j] = kmer_matrix[find_index(kmer_rows_genomes, remove_b(genome))][kmer_index]

#load the new hit information
for i, kmer in enumerate(top_feats):
    errors_df[kmer]=kmer_counts[:,i]

print(errors_df)
