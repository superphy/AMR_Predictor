#!/usr/bin/env python

"""
This program calculates the diversity of all datasets in
public and grdi datasets
"""

import numpy as np
import pandas as pd
import skbio
from collections import Counter
import pickle
import sys

# this functions is necessary because b'string' is being recognized as a string instead of a byte
def remove_b(name):
    name = name[2:]
    return name[:-1]

drugs = ['AMC','AMP','AZM','CHL','CIP','CRO','FIS','FOX','GEN','NAL','SXT','TET','TIO']

# getting the correct order of MIC classes,
# e.g. pub_dict['AMP'] returns ['<=1.0000', '2.0000', '4.0000', '8.0000', '16.0000', '>=32.0000']
with open('data/public_mic_class_order_dict.pkl','rb') as fh:
    pub_dict = pickle.load(fh)
with open('data/grdi_mic_class_order_dict.pkl','rb') as fh:
    grdi_dict = pickle.load(fh)

# create a dataframe to load the results into
df = pd.DataFrame(data = np.zeros((13,3), dtype = 'float'), columns = ['public','grdi','kh'],index = drugs)

for drug in drugs:
    for set in ['grdi', 'public', 'kh']:
        if(drug == 'FIS' and set =="grdi"):
            continue

        path = ''
        order = []
        if set == 'grdi':
            path = 'grdi_'
            order = grdi_dict[drug]
        elif set == 'kh':
            path = 'kh_'
            order = pub_dict[drug]
        else:
            order = pub_dict[drug]

        # load and remove bytes recognizes as strings
        rows_mic = np.load('data/'+path+drug+'/kmer_rows_mic.npy')
        for i, ele in enumerate(rows_mic):
            if ele[0]=='b':
                rows_mic[i] = remove_b(ele)

        if set =='public' or set =='kh':
            rows_mic = [i.decode('utf-8') for i in rows_mic]

        # count occurances of each MIC value for each drug&set combination
        counts = Counter(rows_mic)

        # make dict of counts using {set}_dict as keys and counts as values
        ordered_counts = {}
        for mic in order:
            ordered_counts[str(mic)] = counts[mic]

        df[set][drug] = skbio.diversity.alpha.simpson(list(ordered_counts.values()))

df.to_pickle("data/simpsons_diversity_dataframe.pkl")
